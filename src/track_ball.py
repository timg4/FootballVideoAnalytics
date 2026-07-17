"""Aus gefilterten YOLO-Kandidaten eine konservative Ballspur bilden.

Die Ausgabe ist bewusst in kurze Tracklets geteilt. Eine Lücke wird nur dann
überbrückt, wenn Position, vergangene Zeit und eine großzügige maximale
Ballgeschwindigkeit zusammenpassen. Interpolation erfolgt höchstens über eine
halbe Sekunde; längere Unsicherheit bleibt sichtbar statt erfunden zu werden.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


NUMERIC_FIELDS = ("x1", "y1", "x2", "y2", "x_ref", "y_ref", "x_m", "y_m",
                  "conf", "nearest_distance_m")


def parsed(row):
    result = dict(row)
    result["frame"] = int(row["frame"])
    result["source_frame"] = int(row["source_frame"])
    for field in NUMERIC_FIELDS:
        result[field] = float(row[field]) if row.get(field, "") != "" else np.nan
    return result


def main():
    parser = argparse.ArgumentParser(description="Konservative Ballspur")
    parser.add_argument("filtered_candidates_csv")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min-conf", type=float, default=0.05)
    parser.add_argument("--restart-conf", type=float, default=0.18)
    parser.add_argument("--hard-restart-conf", type=float, default=0.40)
    parser.add_argument("--max-gap", type=int, default=15)
    parser.add_argument("--max-speed-m-s", type=float, default=40.0)
    parser.add_argument("--position-tolerance-m", type=float, default=0.4)
    parser.add_argument("--max-linked-speed-m-s", type=float, default=50.0)
    parser.add_argument("--min-track-hits", type=int, default=3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_path = Path(args.filtered_candidates_csv)
    output_path = (Path(args.output) if args.output else
                   input_path.with_name(input_path.stem.replace("_candidates_filtered", "")
                                        + "_track.csv"))

    candidates_by_frame = defaultdict(list)
    with open(input_path, newline="", encoding="utf-8-sig") as f:
        for raw in csv.DictReader(f):
            if int(raw["usable"]) and float(raw["conf"]) >= args.min_conf:
                row = parsed(raw)
                candidates_by_frame[row["frame"]].append(row)

    tracks = defaultdict(list)
    current_track = -1
    last = None
    for frame in sorted(candidates_by_frame):
        candidates = candidates_by_frame[frame]
        choice = None

        if last is not None and frame - last["frame"] <= args.max_gap:
            delta_frames = frame - last["frame"]
            max_distance = (args.position_tolerance_m + args.max_speed_m_s *
                            delta_frames / args.fps)
            valid = []
            for candidate in candidates:
                distance = float(np.hypot(candidate["x_m"] - last["x_m"],
                                          candidate["y_m"] - last["y_m"]))
                if distance <= max_distance:
                    near_player = (np.isfinite(candidate["nearest_distance_m"]) and
                                   candidate["nearest_distance_m"] <= 3.0)
                    score = (3.0 * candidate["conf"] - 0.025 * distance +
                             0.08 * near_player)
                    valid.append((score, candidate))
            if valid:
                choice = max(valid, key=lambda item: item[0])[1]
            elif max(candidate["conf"] for candidate in candidates) >= args.hard_restart_conf:
                choice = max(candidates, key=lambda candidate: candidate["conf"])
                current_track += 1

        if last is None or frame - last["frame"] > args.max_gap:
            restart = [candidate for candidate in candidates
                       if candidate["conf"] >= args.restart_conf]
            if restart:
                choice = max(restart, key=lambda candidate: candidate["conf"])
                current_track += 1

        if choice is not None:
            tracks[current_track].append(choice)
            last = choice

    # Kurze Phantomspuren verwerfen und physikalisch unplausible Verbindungen
    # nochmals hart trennen. Beide Anker bleiben erhalten, aber nicht verbunden.
    final_tracks = []
    current = []
    for rows in tracks.values():
        if len(rows) < args.min_track_hits:
            continue
        for row in rows:
            if current:
                previous = current[-1]
                dt = (row["frame"] - previous["frame"]) / args.fps
                speed = np.hypot(row["x_m"] - previous["x_m"],
                                 row["y_m"] - previous["y_m"]) / dt
                if speed > args.max_linked_speed_m_s:
                    if len(current) >= args.min_track_hits:
                        final_tracks.append(current)
                    current = []
            current.append(row)
        if len(current) >= args.min_track_hits:
            final_tracks.append(current)
        current = []

    fields = ["frame", "source_frame", "ball_tracklet", "x_ref", "y_ref",
              "x_m", "y_m", "conf", "interpolated", "nearest_tracker_id",
              "nearest_player", "nearest_team", "nearest_distance_m"]
    output_rows = []
    for track_id, rows in enumerate(final_tracks):
        for index, row in enumerate(rows):
            output_rows.append({
                "frame": row["frame"], "source_frame": row["source_frame"],
                "ball_tracklet": track_id, "x_ref": row["x_ref"],
                "y_ref": row["y_ref"], "x_m": row["x_m"], "y_m": row["y_m"],
                "conf": row["conf"], "interpolated": 0,
                "nearest_tracker_id": row["nearest_tracker_id"],
                "nearest_player": row["nearest_player"],
                "nearest_team": row["nearest_team"],
                "nearest_distance_m": row["nearest_distance_m"],
            })
            if index + 1 == len(rows):
                continue
            following = rows[index + 1]
            gap = following["frame"] - row["frame"]
            if not 1 < gap <= args.max_gap:
                continue
            for step in range(1, gap):
                ratio = step / gap
                output_rows.append({
                    "frame": row["frame"] + step,
                    "source_frame": row["source_frame"] + step,
                    "ball_tracklet": track_id,
                    "x_ref": row["x_ref"] + ratio * (following["x_ref"] - row["x_ref"]),
                    "y_ref": row["y_ref"] + ratio * (following["y_ref"] - row["y_ref"]),
                    "x_m": row["x_m"] + ratio * (following["x_m"] - row["x_m"]),
                    "y_m": row["y_m"] + ratio * (following["y_m"] - row["y_m"]),
                    "conf": min(row["conf"], following["conf"]),
                    "interpolated": 1,
                    "nearest_tracker_id": "", "nearest_player": "",
                    "nearest_team": "", "nearest_distance_m": "",
                })

    output_rows.sort(key=lambda row: row["frame"])
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in output_rows:
            writer.writerow({
                key: (f"{value:.3f}" if isinstance(value, float) else value)
                for key, value in row.items()
            })

    anchors = sum(not row["interpolated"] for row in output_rows)
    interpolated = len(output_rows) - anchors
    print(f"Balltracklets: {len(final_tracks)}")
    print(f"Anker: {anchors}, interpoliert: {interpolated}")
    print(f"Abgedeckte Frames: {len(output_rows)}/25150 "
          f"({len(output_rows) / 25150 * 100:.1f} %)")
    print(f"Ausgabe: {output_path}")


if __name__ == "__main__":
    main()
