"""Aus einer konservativen Ballspur Ballbesitz- und Passkandidaten ableiten.

Das Ergebnis ist eine Review-Liste, noch keine endgültige Passstatistik. Besitz
wird nur vergeben, wenn ein teamzugeordneter Spieler mehrere Frames stabil in
Ballnähe bleibt. Wechsel innerhalb desselben Balltracklets werden als mögliche
Pässe bzw. Ballverluste ausgegeben und müssen visuell validiert werden.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_mapping(path, column):
    result = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get(column, "") != "":
                result[int(row["tracker_id"])] = row[column]
    return result


def runs(values):
    if not values:
        return []
    result = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            result.append((start, index, values[start]))
            start = index
    return result


def main():
    parser = argparse.ArgumentParser(description="Ballbesitz + Passkandidaten")
    parser.add_argument("ball_track_csv")
    parser.add_argument("positions_csv")
    parser.add_argument("team_assignments_csv")
    parser.add_argument("player_mapping_csv")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--possession-distance-m", type=float, default=2.5)
    parser.add_argument("--min-possession-frames", type=int, default=6)
    parser.add_argument("--fill-gap-frames", type=int, default=6)
    parser.add_argument("--max-transfer-frames", type=int, default=60)
    parser.add_argument("--min-transfer-m", type=float, default=2.0)
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    ball_path = Path(args.ball_track_csv)
    prefix = args.output_prefix or ball_path.stem.replace("_ball_track", "")
    out_dir = ball_path.parent
    possession_path = out_dir / f"{prefix}_ballbesitz_episoden.csv"
    passes_path = out_dir / f"{prefix}_pass_kandidaten.csv"
    stats_path = out_dir / f"{prefix}_pass_statistik_vorlaeufig.csv"

    team_of = {tid: int(team) for tid, team in
               load_mapping(args.team_assignments_csv, "team").items()}
    player_of = load_mapping(args.player_mapping_csv, "spieler_name_oder_nummer")

    players_by_frame = defaultdict(list)
    with open(args.positions_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            tid = int(row["tracker_id"])
            if int(row["on_pitch"]) and tid in team_of:
                players_by_frame[int(row["frame"])].append(
                    (tid, float(row["x_m"]), float(row["y_m"])))

    ball_by_track = defaultdict(list)
    with open(ball_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row["frame"] = int(row["frame"])
            row["ball_tracklet"] = int(row["ball_tracklet"])
            row["x_m"] = float(row["x_m"])
            row["y_m"] = float(row["y_m"])
            ball_by_track[row["ball_tracklet"]].append(row)

    episodes = []
    for ball_tracklet, points in ball_by_track.items():
        points.sort(key=lambda row: row["frame"])
        labels = []
        details = []
        for point in points:
            nearest = None
            for tid, x_m, y_m in players_by_frame.get(point["frame"], []):
                distance = float(np.hypot(point["x_m"] - x_m, point["y_m"] - y_m))
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, tid)
            if nearest is not None and nearest[0] <= args.possession_distance_m:
                distance, tid = nearest
                identity = player_of.get(tid, f"track:{tid}")
                labels.append(identity)
                details.append((tid, team_of[tid], player_of.get(tid, ""), distance))
            else:
                labels.append(None)
                details.append(None)

        # Kurze Aussetzer zwischen demselben Besitzer füllen.
        for start, end, value in runs(labels):
            if (value is None and end - start <= args.fill_gap_frames and
                    start > 0 and end < len(labels) and
                    labels[start - 1] == labels[end]):
                labels[start:end] = [labels[start - 1]] * (end - start)
                details[start:end] = [details[start - 1]] * (end - start)

        # Kurze Fremd-/Rauschbesitze entfernen, danach gleiche Besitzer mergen.
        for start, end, value in runs(labels):
            if value is not None and end - start < args.min_possession_frames:
                labels[start:end] = [None] * (end - start)
                details[start:end] = [None] * (end - start)
        for start, end, value in runs(labels):
            if (value is None and end - start <= args.fill_gap_frames and
                    start > 0 and end < len(labels) and
                    labels[start - 1] == labels[end]):
                labels[start:end] = [labels[start - 1]] * (end - start)
                details[start:end] = [details[start - 1]] * (end - start)

        for start, end, identity in runs(labels):
            if identity is None or end - start < args.min_possession_frames:
                continue
            tid, team, player, _ = details[start]
            distances = [item[3] for item in details[start:end] if item is not None]
            episodes.append({
                "ball_tracklet": ball_tracklet,
                "start_frame": points[start]["frame"],
                "end_frame": points[end - 1]["frame"],
                "dauer_s": (points[end - 1]["frame"] - points[start]["frame"] + 1) /
                           args.fps,
                "identity": identity,
                "tracker_id": tid,
                "spieler": player,
                "team": team,
                "median_abstand_m": float(np.median(distances)),
                "start_x_m": points[start]["x_m"],
                "start_y_m": points[start]["y_m"],
                "end_x_m": points[end - 1]["x_m"],
                "end_y_m": points[end - 1]["y_m"],
            })

    episodes.sort(key=lambda row: row["start_frame"])
    episode_fields = list(episodes[0]) if episodes else []
    with open(possession_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=episode_fields)
        writer.writeheader()
        for row in episodes:
            writer.writerow({key: (f"{value:.3f}" if isinstance(value, float) else value)
                             for key, value in row.items()})

    pass_rows = []
    by_ball_track = defaultdict(list)
    for episode in episodes:
        by_ball_track[episode["ball_tracklet"]].append(episode)
    for ball_tracklet, track_episodes in by_ball_track.items():
        track_episodes.sort(key=lambda row: row["start_frame"])
        for source, target in zip(track_episodes, track_episodes[1:]):
            gap = target["start_frame"] - source["end_frame"] - 1
            transfer = float(np.hypot(target["start_x_m"] - source["end_x_m"],
                                      target["start_y_m"] - source["end_y_m"]))
            if (source["identity"] == target["identity"] or gap < 0 or
                    gap > args.max_transfer_frames or transfer < args.min_transfer_m):
                continue
            same_team = source["team"] == target["team"]
            pass_rows.append({
                "ball_tracklet": ball_tracklet,
                "abgabe_frame": source["end_frame"],
                "annahme_frame": target["start_frame"],
                "dauer_s": (target["start_frame"] - source["end_frame"]) / args.fps,
                "distanz_m": transfer,
                "von_team": source["team"],
                "zu_team": target["team"],
                "von_tracker_id": source["tracker_id"],
                "zu_tracker_id": target["tracker_id"],
                "von_spieler": source["spieler"],
                "zu_spieler": target["spieler"],
                "ergebnis": "angekommen" if same_team else "ballverlust",
                "review_status": "ungeprüft",
            })

    pass_fields = list(pass_rows[0]) if pass_rows else []
    with open(passes_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pass_fields)
        writer.writeheader()
        for row in pass_rows:
            writer.writerow({key: (f"{value:.3f}" if isinstance(value, float) else value)
                             for key, value in row.items()})

    stats = []
    for team in sorted(set(row["von_team"] for row in pass_rows)):
        team_rows = [row for row in pass_rows if row["von_team"] == team]
        completed = sum(row["ergebnis"] == "angekommen" for row in team_rows)
        stats.append({
            "team": team,
            "sichtbare_passkandidaten": len(team_rows),
            "davon_angekommen": completed,
            "davon_ballverlust": len(team_rows) - completed,
            "vorlaeufige_quote_pct": 100 * completed / len(team_rows),
        })
    stat_fields = list(stats[0]) if stats else []
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=stat_fields)
        writer.writeheader()
        for row in stats:
            writer.writerow({key: (f"{value:.1f}" if isinstance(value, float) else value)
                             for key, value in row.items()})

    print(f"Besitzepisoden: {len(episodes)} -> {possession_path}")
    print(f"Passkandidaten: {len(pass_rows)} -> {passes_path}")
    for row in stats:
        print(f"  Team {row['team']}: {row['sichtbare_passkandidaten']} Kandidaten, "
              f"{row['vorlaeufige_quote_pct']:.1f} % gleiches Team")
    print("Noch ungeprüft: Diese Quote nicht als Endergebnis verwenden.")


if __name__ == "__main__":
    main()
