"""Zeitliche Kontaktzuordnung aus Ball- und Spielerbewegung evaluieren.

Statt nur im Endframe den naechsten Spieler zu nehmen, sammelt dieses
Diagnosewerkzeug die Ballnaehe jedes Tracklets vor der Abgabe bzw. nach der
Annahme. Ausgegeben werden nur ehrlich gegen das manuelle Pass-Review gemessene
Varianten; das beste Ergebnis wird nicht automatisch produktiv geschaltet.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Temporale Kontaktregeln evaluieren")
    parser.add_argument("pass_review_csv")
    parser.add_argument("ball_track_csv")
    parser.add_argument("positions_csv")
    parser.add_argument("team_assignments_csv")
    parser.add_argument("--output", default="data/output/video_project_contact_motion_evaluation.json")
    args = parser.parse_args()

    review = pd.read_csv(args.pass_review_csv)
    passes = review.loc[review["review_status"].eq("pass")]
    contacts = []
    for event, (_, row) in enumerate(passes.iterrows()):
        for side, frame_column, team_column in (
            ("von", "abgabe_frame", "review_von_team"),
            ("zu", "annahme_frame", "review_zu_team"),
        ):
            contacts.append({
                "event": event, "side": side, "frame": int(row[frame_column]),
                "tracklet": int(row["ball_tracklet"]),
                "truth": int(float(row[team_column])),
            })

    ball = pd.read_csv(args.ball_track_csv,
                       usecols=["frame", "ball_tracklet", "x_m", "y_m"])
    ball_by_key = {(int(row.ball_tracklet), int(row.frame)):
                   (float(row.x_m), float(row.y_m))
                   for row in ball.itertuples(index=False)}
    teams = pd.read_csv(args.team_assignments_csv,
                        usecols=["tracker_id", "team"]).set_index("tracker_id")["team"]
    positions = pd.read_csv(args.positions_csv,
                            usecols=["frame", "tracker_id", "x_m", "y_m", "on_pitch"])
    positions = positions.loc[positions["on_pitch"].eq(1)]
    positions["team"] = positions["tracker_id"].map(teams)
    positions = positions.dropna(subset=["team"])
    players_by_frame = defaultdict(list)
    for row in positions.itertuples(index=False):
        players_by_frame[int(row.frame)].append(
            (int(row.tracker_id), int(row.team), float(row.x_m), float(row.y_m)))

    variants = []
    for window in (4, 8, 12, 20, 30):
        for scale in (0.6, 1.0, 1.5, 2.5):
            predictions, truths, confidence = [], [], []
            for contact in contacts:
                offsets = (range(-window, 1) if contact["side"] == "von" else
                           range(0, window + 1))
                tracker_scores = defaultdict(float)
                for offset in offsets:
                    frame = contact["frame"] + offset
                    ball_xy = ball_by_key.get((contact["tracklet"], frame))
                    if ball_xy is None:
                        continue
                    # Frames direkt am Kontakt sind aussagekraeftiger.
                    temporal = np.exp(-abs(offset) / max(2.0, window / 2))
                    for tracker, team, x_m, y_m in players_by_frame.get(frame, []):
                        distance = np.hypot(x_m - ball_xy[0], y_m - ball_xy[1])
                        tracker_scores[(tracker, team)] += float(
                            temporal * np.exp(-0.5 * (distance / scale) ** 2))
                team_best = {}
                for (_, team), score in tracker_scores.items():
                    team_best[team] = max(team_best.get(team, 0.0), score)
                if not team_best:
                    prediction, margin = -1, 0.0
                else:
                    ordered = sorted(team_best.items(), key=lambda item: item[1], reverse=True)
                    prediction = ordered[0][0]
                    first, second = ordered[0][1], ordered[1][1] if len(ordered) > 1 else 0.0
                    margin = (first - second) / (first + second + 1e-9)
                predictions.append(prediction)
                truths.append(contact["truth"])
                confidence.append(margin)

            predictions, truths = np.asarray(predictions), np.asarray(truths)
            confidence = np.asarray(confidence)
            valid = predictions >= 0
            recalls = [float((predictions[truths == team] == team).mean()) for team in (0, 1)]
            subsets = []
            for threshold in (0.25, 0.50, 0.75):
                mask = valid & (confidence >= threshold)
                subsets.append({
                    "threshold": threshold, "n": int(mask.sum()),
                    "coverage": float(mask.mean()),
                    "accuracy": float((predictions[mask] == truths[mask]).mean())
                    if mask.any() else None,
                })
            variants.append({
                "window": window, "scale_m": scale,
                "accuracy": float((predictions[valid] == truths[valid]).mean()),
                "balanced_accuracy": float(np.mean(recalls)),
                "recall_green": recalls[0], "recall_blue": recalls[1],
                "coverage": float(valid.mean()), "confidence_subsets": subsets,
            })

    variants.sort(key=lambda row: (row["balanced_accuracy"], row["accuracy"]), reverse=True)
    output = {"contacts": len(contacts), "variants": variants}
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("Beste Varianten:")
    for row in variants[:8]:
        print(f"  window={row['window']:2d}, scale={row['scale_m']:.1f}m: "
              f"acc={row['accuracy']:.3f}, bal={row['balanced_accuracy']:.3f}, "
              f"green={row['recall_green']:.3f}, blue={row['recall_blue']:.3f}")
    print(f"Auswertung: {args.output}")


if __name__ == "__main__":
    main()
