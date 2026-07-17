"""Wenige manuelle Spieler-Referenzen per Re-ID auf Tracklets übertragen.

Die Nutzerin/der Nutzer markiert nur 2–3 eindeutige Tracklets pro Spieler.
Aus deren Appearance-Embeddings wird je Spieler ein Prototyp gebildet. Nur
Tracklets mit hoher Ähnlichkeit, klarem Vorsprung und ohne zeitlichen Konflikt
werden automatisch übernommen; der Rest bleibt explizit unzugeordnet.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def normalize(vector):
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def main():
    parser = argparse.ArgumentParser(description="Spielerlabels aus Referenzen verbreiten")
    parser.add_argument("seed_mapping_csv")
    parser.add_argument("positions_csv")
    parser.add_argument("distances_csv")
    parser.add_argument("embeddings_npz")
    parser.add_argument("--min-similarity", type=float, default=0.75)
    parser.add_argument("--min-margin", type=float, default=0.06)
    parser.add_argument("--max-overlap", type=int, default=15,
                        help="Erlaubte Überlappung zweier Tracklets desselben Spielers in Frames")
    parser.add_argument("--team", type=int, default=None,
                        help="Optional nur dieses Team propagieren (z.B. 1 für eigenes Team)")
    parser.add_argument("--ambiguous", action="append", default=[],
                        help="Kommagetrennte, schwer trennbare Spieler; Auto-Treffer werden gruppiert")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    seeds = []
    with open(args.seed_mapping_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = row["spieler_name_oder_nummer"].strip()
            team = int(row["team"])
            if name and (args.team is None or team == args.team):
                seeds.append((int(row["tracker_id"]), team, name))
    if not seeds:
        raise SystemExit("Die Referenz-CSV enthält noch keine Spielerlabels.")

    with np.load(args.embeddings_npz) as data:
        ids = data["track_ids"].astype(int)
        embeddings = data["embeddings"].astype(float)
        teams = data["teams"].astype(int)
    embedding_of = dict(zip(ids, embeddings))
    team_of = dict(zip(ids, teams))

    spans = {}
    with open(args.positions_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if not int(row["on_pitch"]):
                continue
            tid, frame = int(row["tracker_id"]), int(row["frame"])
            if tid not in embedding_of:
                continue
            if tid not in spans:
                spans[tid] = [frame, frame]
            else:
                spans[tid][1] = frame

    stats = defaultdict(lambda: (0.0, 0.0))
    with open(args.distances_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            distance = row.get("distanz_m", row.get("distance_m"))
            duration = row.get("sichtbare_dauer_s", row.get("visible_seconds"))
            if distance is None or duration is None:
                raise SystemExit(
                    "Distance CSV needs distanz_m/sichtbare_dauer_s or "
                    "distance_m/visible_seconds columns")
            stats[int(row["tracker_id"])] = (
                float(distance), float(duration))

    player_seeds = defaultdict(list)
    seed_ids = set()
    for tid, team, name in seeds:
        if tid not in embedding_of:
            continue
        player_seeds[(team, name)].append(tid)
        seed_ids.add(tid)
    prototypes = {
        player: normalize(np.mean([embedding_of[tid] for tid in tids], axis=0))
        for player, tids in player_seeds.items()
    }

    assigned = {}
    intervals = defaultdict(list)
    for player, tids in player_seeds.items():
        for tid in tids:
            assigned[tid] = (player, "manuell", 1.0, 1.0)
            if tid in spans:
                intervals[player].append(tuple(spans[tid]))

    candidates = []
    for tid, embedding in embedding_of.items():
        if tid in seed_ids or tid not in spans:
            continue
        options = []
        for player, prototype in prototypes.items():
            if player[0] != int(team_of[tid]):
                continue
            similarity = float(embedding @ prototype)
            options.append((similarity, player))
        options.sort(reverse=True, key=lambda item: item[0])
        if not options:
            continue
        best_similarity, best_player = options[0]
        second_similarity = options[1][0] if len(options) > 1 else -1.0
        margin = best_similarity - second_similarity
        if best_similarity >= args.min_similarity and margin >= args.min_margin:
            candidates.append((best_similarity, margin, stats[tid][0], tid, best_player))

    # Die sichersten/längsten zuerst. Bereits zugewiesene Zeitintervalle
    # verhindern, dass zwei gleichzeitig sichtbare Personen denselben Namen
    # erhalten. Prototypen bleiben auf den manuellen Seeds fixiert (kein Drift).
    for similarity, margin, _, tid, player in sorted(candidates, reverse=True):
        start, end = spans[tid]
        conflict = any(min(end, old_end) - max(start, old_start) + 1 > args.max_overlap
                       for old_start, old_end in intervals[player])
        if conflict:
            continue
        assigned[tid] = (player, "automatisch", similarity, margin)
        intervals[player].append((start, end))

    output = (Path(args.output) if args.output else
              Path(args.seed_mapping_csv).with_name(
                  Path(args.seed_mapping_csv).stem.replace("manuelles", "auto") + ".csv"))
    ambiguous_groups = []
    for value in args.ambiguous:
        names = {name.strip() for name in value.split(",") if name.strip()}
        if len(names) >= 2:
            ambiguous_groups.append(names)
    with open(output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tracker_id", "team", "spieler_name_oder_nummer",
                    "distanz_m", "sichtbare_dauer_s", "quelle",
                    "reid_similarity", "eindeutigkeits_margin"])
        for tid, (player, source, similarity, margin) in sorted(assigned.items()):
            distance, duration = stats[tid]
            output_name = player[1]
            if source == "automatisch":
                for names in ambiguous_groups:
                    if output_name in names:
                        output_name = "/".join(sorted(names)) + " unklar"
                        break
            w.writerow([tid, player[0], output_name, f"{distance:.2f}",
                        f"{duration:.2f}", source, f"{similarity:.4f}",
                        f"{margin:.4f}"])

    total_distance = defaultdict(float)
    assigned_distance = defaultdict(float)
    for tid, (distance, _) in stats.items():
        if tid in team_of:
            total_distance[int(team_of[tid])] += distance
    for tid, (player, _, _, _) in assigned.items():
        assigned_distance[player[0]] += stats[tid][0]

    print(f"Automatisches Mapping: {output}")
    print(f"{len(seed_ids)} manuelle Seeds -> {len(assigned)} zugeordnete Tracklets")
    for team in sorted(total_distance):
        share = assigned_distance[team] / max(total_distance[team], 1e-9)
        print(f"  Team {team}: {assigned_distance[team] / 1000:.3f} km "
              f"zugeordnet ({share:.1%} der sichtbaren Distanz)")


if __name__ == "__main__":
    main()
