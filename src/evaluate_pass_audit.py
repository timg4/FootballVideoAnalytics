"""Pruefen, wie viele manuelle Stichproben fuer Team-Passquoten noetig sind.

Die bereits vollstaendig geprueften Ereignisse dienen nur als Ground Truth fuer
eine Monte-Carlo-Simulation. Fuer jede Wiederholung wird eine geschichtete
Teilmenge gezogen und so getan, als waere nur diese manuell bekannt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["ball_tracklet", "abgabe_frame", "annahme_frame"]


def allocation(counts: pd.Series, requested: int) -> dict[str, int]:
    requested = min(requested, int(counts.sum()))
    raw = counts / counts.sum() * requested
    allocated = {key: min(int(counts[key]), max(1, int(np.floor(raw[key]))))
                 for key in counts.index}
    while sum(allocated.values()) < requested:
        candidates = [key for key in counts.index if allocated[key] < counts[key]]
        key = max(candidates, key=lambda value: raw[value] - allocated[value])
        allocated[key] += 1
    while sum(allocated.values()) > requested:
        candidates = [key for key in counts.index if allocated[key] > 1]
        key = min(candidates, key=lambda value: raw[value] - allocated[value])
        allocated[key] -= 1
    return allocated


def estimate(population: pd.DataFrame, sampled: pd.DataFrame):
    totals = {team: {"attempts": 0.0, "complete": 0.0} for team in (0, 1)}
    for stratum, rows in population.groupby("stratum"):
        audit = sampled.loc[sampled["stratum"].eq(stratum)]
        if audit.empty:
            continue
        weight = len(rows) / len(audit)
        for team in (0, 1):
            attempts = (audit["review_status"].eq("pass") &
                        audit["review_von_team"].eq(team))
            complete = attempts & audit["review_zu_team"].eq(team)
            totals[team]["attempts"] += weight * attempts.sum()
            totals[team]["complete"] += weight * complete.sum()
    for team in (0, 1):
        attempts = totals[team]["attempts"]
        totals[team]["quote"] = (100 * totals[team]["complete"] / attempts
                                  if attempts else np.nan)
    return totals


def main():
    parser = argparse.ArgumentParser(description="Pass-Audit simulieren")
    parser.add_argument("auto_candidates_csv")
    parser.add_argument("pass_review_csv")
    parser.add_argument("--sizes", default="16,24,32")
    parser.add_argument("--repetitions", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output", default="data/output/video_project_pass_audit_evaluation.json")
    args = parser.parse_args()

    auto = pd.read_csv(args.auto_candidates_csv)
    review = pd.read_csv(args.pass_review_csv)
    truth_columns = review[KEY + ["review_status", "review_von_team",
                                  "review_zu_team"]].rename(columns={
        "review_status": "truth_status",
        "review_von_team": "truth_von_team",
        "review_zu_team": "truth_zu_team",
    })
    merged = auto.merge(truth_columns, on=KEY, how="left")
    # Unklare und noch nicht gepruefte Ereignisse eignen sich nicht als Ground Truth.
    population = merged.loc[merged["truth_status"].isin(["pass", "no", "shot"])].copy()
    population["review_status"] = population["truth_status"]
    population["review_von_team"] = population["truth_von_team"]
    population["review_zu_team"] = population["truth_zu_team"]
    population["same_auto"] = (population["von_team"] == population["zu_team"]).astype(int)
    population["stratum"] = (population["auto_status"].astype(str) + "|" +
                              population["von_team"].astype(str) + "|" +
                              population["same_auto"].astype(str))
    counts = population.groupby("stratum").size()
    truth = estimate(population, population)
    rng = np.random.default_rng(args.seed)
    simulations = {}
    for size in [int(value) for value in args.sizes.split(",")]:
        alloc = allocation(counts, size)
        errors = {team: {"attempts": [], "quote": []} for team in (0, 1)}
        for _ in range(args.repetitions):
            pieces = []
            for stratum, rows in population.groupby("stratum"):
                chosen = rng.choice(rows.index.to_numpy(), size=alloc[stratum], replace=False)
                pieces.append(population.loc[chosen])
            sampled = pd.concat(pieces)
            result = estimate(population, sampled)
            for team in (0, 1):
                errors[team]["attempts"].append(result[team]["attempts"] -
                                                 truth[team]["attempts"])
                errors[team]["quote"].append(result[team]["quote"] -
                                              truth[team]["quote"])
        summary = {}
        for team in (0, 1):
            summary[str(team)] = {}
            for metric in ("attempts", "quote"):
                values = np.asarray(errors[team][metric])
                summary[str(team)][metric] = {
                    "median_abs_error": float(np.nanmedian(np.abs(values))),
                    "p90_abs_error": float(np.nanpercentile(np.abs(values), 90)),
                    "p95_low": float(np.nanpercentile(values, 2.5)),
                    "p95_high": float(np.nanpercentile(values, 97.5)),
                }
        simulations[str(size)] = {"allocation": alloc, "errors": summary}

    report = {"population": len(population), "truth": truth,
              "strata": counts.to_dict(), "simulations": simulations}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Ground Truth aus {len(population)} eindeutigen Ereignissen: {truth}")
    for size, result in simulations.items():
        print(f"Stichprobe {size}:")
        for team in (0, 1):
            error = result["errors"][str(team)]
            print(f"  Team {team}: Quote median |Fehler| "
                  f"{error['quote']['median_abs_error']:.1f} pp, "
                  f"90% <= {error['quote']['p90_abs_error']:.1f} pp")
    print(f"Auswertung: {args.output}")


if __name__ == "__main__":
    main()
