"""Tracklets konservativ zu anonymen Spieler-Kandidaten verbinden.

Automatische Links brauchen gleichzeitig:
  - gleiches Team,
  - zeitlich/räumlich plausible Fortsetzung,
  - hohe Person-ReID-Ähnlichkeit,
  - einen eindeutigen Abstand zum zweitbesten Kandidaten.

Das Resultat ist absichtlich kein erzwungenes Clustering auf eine geratene
Spielerzahl. Lange/mehrdeutige Wiedereintritte bleiben getrennte Kandidaten und
können später in der Mapping-CSV manuell demselben Spieler zugeordnet werden.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Konservatives Tracklet-Stitching")
    parser.add_argument("positions_csv")
    parser.add_argument("assignments_csv")
    parser.add_argument("distances_csv")
    parser.add_argument("embeddings_npz")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-gap", type=float, default=5.0,
                        help="Maximale zeitliche Lücke eines Auto-Links in Sekunden")
    parser.add_argument("--min-similarity", type=float, default=0.85)
    parser.add_argument("--similarity-margin", type=float, default=0.04,
                        help="Mindestvorsprung vor dem zweitbesten Link")
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    default_prefix = Path(args.positions_csv).stem.replace("_positionen", "")
    out_dir = Path(__file__).resolve().parent.parent / "data" / "output"
    prefix = Path(args.output_prefix) if args.output_prefix else out_dir / default_prefix

    team_of = {}
    with open(args.assignments_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["team"] != "":
                team_of[int(row["tracker_id"])] = int(row["team"])

    points = defaultdict(list)
    with open(args.positions_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            tid = int(row["tracker_id"])
            if int(row["auf_platz"]) and tid in team_of:
                points[tid].append((int(row["frame"]),
                                    float(row["x_m"]), float(row["y_m"])))

    distance_of = defaultdict(lambda: (0.0, 0.0, 0))
    with open(args.distances_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            distance_of[int(row["tracker_id"])] = (
                float(row["distanz_m"]), float(row["sichtbare_dauer_s"]),
                int(row["positionspunkte"]))

    with np.load(args.embeddings_npz) as data:
        emb_ids = data["track_ids"].astype(int)
        embeddings = data["embeddings"].astype(float)
        consistency = dict(zip(emb_ids, data["consistency"].astype(float)))
    embedding_of = dict(zip(emb_ids, embeddings))

    summary = {}
    for tid, pts in points.items():
        pts.sort()
        k = min(10, len(pts))
        start_xy = np.median(np.asarray(pts[:k])[:, 1:3], axis=0)
        end_xy = np.median(np.asarray(pts[-k:])[:, 1:3], axis=0)
        summary[tid] = {
            "start": pts[0][0], "end": pts[-1][0],
            "start_xy": start_xy, "end_xy": end_xy,
            "team": team_of[tid],
        }

    max_gap_frames = int(round(args.max_gap * args.fps))
    candidates = []
    by_source = defaultdict(list)
    by_target = defaultdict(list)
    tids = sorted(summary)
    for source in tids:
        a = summary[source]
        if source not in embedding_of:
            continue
        for target in tids:
            if source == target or target not in embedding_of:
                continue
            b = summary[target]
            gap = b["start"] - a["end"]
            if a["team"] != b["team"] or not (-10 <= gap <= max_gap_frames):
                continue
            distance = float(np.linalg.norm(a["end_xy"] - b["start_xy"]))
            # Bis 1,5 m Grundtoleranz plus maximal 7 m/s während der Lücke.
            max_distance = 1.5 + 7.0 * max(gap, 0) / args.fps
            if distance > max_distance:
                continue
            similarity = float(embedding_of[source] @ embedding_of[target])
            if similarity < args.min_similarity:
                continue
            candidate = {
                "source": source, "target": target, "gap": gap,
                "distance": distance, "similarity": similarity,
                "score": similarity - 0.05 * distance / max_distance,
            }
            candidates.append(candidate)
            by_source[source].append(candidate)
            by_target[target].append(candidate)

    # Nur wechselseitig beste, hinreichend eindeutige Links zulassen.
    accepted = []
    for candidate in candidates:
        source_options = sorted(by_source[candidate["source"]],
                                key=lambda item: -item["score"])
        target_options = sorted(by_target[candidate["target"]],
                                key=lambda item: -item["score"])
        if candidate is not source_options[0] or candidate is not target_options[0]:
            continue
        source_margin = (candidate["score"] - source_options[1]["score"]
                         if len(source_options) > 1 else 1.0)
        target_margin = (candidate["score"] - target_options[1]["score"]
                         if len(target_options) > 1 else 1.0)
        if min(source_margin, target_margin) < args.similarity_margin:
            continue
        candidate["margin"] = min(source_margin, target_margin)
        accepted.append(candidate)

    # Konflikte und Zyklen verhindern; Zeitrichtung macht Zyklen praktisch
    # unmöglich, die explizite Prüfung schützt auch bei kurzem Überlapp.
    successor = {}
    predecessor = {}
    final_links = []
    for candidate in sorted(accepted, key=lambda item: -item["score"]):
        source, target = candidate["source"], candidate["target"]
        if source in successor or target in predecessor:
            continue
        cursor = target
        cycle = False
        while cursor in successor:
            cursor = successor[cursor]
            if cursor == source:
                cycle = True
                break
        if cycle:
            continue
        successor[source] = target
        predecessor[target] = source
        final_links.append(candidate)

    chains = []
    for tid in tids:
        if tid in predecessor:
            continue
        chain = []
        cursor = tid
        while cursor not in chain:
            chain.append(cursor)
            if cursor not in successor:
                break
            cursor = successor[cursor]
        chains.append(chain)
    chains.sort(key=lambda chain: (team_of[chain[0]], summary[chain[0]]["start"]))

    links_path = Path(f"{prefix}_tracklet_links.csv")
    with open(links_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_tracklet", "target_tracklet", "team", "gap_s",
                    "distanz_m", "reid_similarity", "eindeutigkeits_margin"])
        for link in sorted(final_links, key=lambda item: summary[item["source"]]["end"]):
            w.writerow([link["source"], link["target"], team_of[link["source"]],
                        f"{link['gap'] / args.fps:.3f}", f"{link['distance']:.3f}",
                        f"{link['similarity']:.4f}", f"{link['margin']:.4f}"])

    candidates_path = Path(f"{prefix}_spieler_kandidaten.csv")
    mapping_path = Path(f"{prefix}_spieler_mapping.csv")
    with open(candidates_path, "w", newline="") as candidates_file, \
            open(mapping_path, "w", newline="") as mapping_file:
        cw = csv.writer(candidates_file)
        mw = csv.writer(mapping_file)
        cw.writerow(["kandidat_id", "team", "tracklets", "distanz_m",
                     "sichtbare_dauer_s", "start_s", "ende_s",
                     "min_reid_konsistenz"])
        mw.writerow(["kandidat_id", "team", "spieler_name_oder_nummer",
                     "tracklets", "automatisch_verbunden"])
        counters = defaultdict(int)
        link_by_source = {item["source"]: item for item in final_links}
        for chain in chains:
            team = team_of[chain[0]]
            counters[team] += 1
            candidate_id = f"T{team}-K{counters[team]:03d}"
            distance = sum(distance_of[tid][0] for tid in chain)
            duration = sum(distance_of[tid][1] for tid in chain)
            start = min(summary[tid]["start"] for tid in chain) / args.fps
            end = max(summary[tid]["end"] for tid in chain) / args.fps
            min_consistency = min(consistency.get(tid, 0.0) for tid in chain)
            joined = ";".join(map(str, chain))
            cw.writerow([candidate_id, team, joined, f"{distance:.2f}",
                         f"{duration:.2f}", f"{start:.2f}", f"{end:.2f}",
                         f"{min_consistency:.3f}"])
            mw.writerow([candidate_id, team, "", joined,
                         int(any(tid in link_by_source for tid in chain))])

    print(f"Auto-Links: {len(final_links)} -> {links_path}")
    print(f"Spieler-Kandidaten: {len(chains)} -> {candidates_path}")
    print(f"Manuelles Mapping: {mapping_path}")
    for team in sorted(set(team_of.values())):
        count = sum(team_of[chain[0]] == team for chain in chains)
        print(f"  Team {team}: {count} Kandidaten")


if __name__ == "__main__":
    main()
