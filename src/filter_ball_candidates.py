"""YOLO-Ballkandidaten geometrisch filtern und mit Spielern anreichern.

Die niedrige YOLO-Schwelle liefert absichtlich viele Kandidaten. Dieses Skript
projiziert sie in Platzmeter, entfernt Treffer außerhalb des Felds, markiert
über viele Zeitblöcke wiederkehrende statische Rasen-/Zaunmerkmale und ergänzt
den jeweils nächsten sichtbaren Spieler. Es wählt noch keine endgültige
Ballspur; alle Filterentscheidungen bleiben als Spalten nachvollziehbar.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from piecewise_pitch import transform_piecewise


def read_optional_mapping(path, value_column):
    if not path:
        return {}
    result = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get(value_column, "") != "":
                result[int(row["tracker_id"])] = row[value_column]
    return result


def main():
    parser = argparse.ArgumentParser(description="Ballkandidaten filtern")
    parser.add_argument("candidates_csv")
    parser.add_argument("localization_npz")
    parser.add_argument("calibration_json")
    parser.add_argument("positions_csv")
    parser.add_argument("--team-assignments")
    parser.add_argument("--player-mapping")
    parser.add_argument("--margin-m", type=float, default=0.5)
    parser.add_argument("--static-cell-m", type=float, default=0.25)
    parser.add_argument("--static-block-seconds", type=float, default=10.0)
    parser.add_argument("--static-min-blocks", type=int, default=8)
    parser.add_argument("--static-min-hits", type=int, default=30)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    candidates_path = Path(args.candidates_csv)
    output_path = (Path(args.output) if args.output else
                   candidates_path.with_name(candidates_path.stem + "_filtered.csv"))

    with np.load(args.localization_npz) as data:
        frames = data["frames"].astype(int)
        piecewise = "H_px_to_pitch_left" in data
        if piecewise:
            left_all = data["H_px_to_pitch_left"].copy()
            right_all = data["H_px_to_pitch_right"].copy()
            reliable_all = (data["anchor_reliable"].astype(bool)
                            if "anchor_reliable" in data
                            else np.ones(len(frames), dtype=bool))
            transforms_by_frame = {
                int(frame): (left, right, bool(reliable))
                for frame, left, right, reliable in zip(
                    frames, left_all, right_all, reliable_all)
            }
        else:
            matrices = data["H_px_to_pitch"].copy()
            H_by_frame = {
                int(frame): matrix for frame, matrix in zip(frames, matrices)
            }

    calibration = json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))
    length = float(calibration["pitch"]["laenge"])
    width = float(calibration["pitch"]["breite"])
    split_x = (float(calibration["piecewise"]["split_x_m"])
               if piecewise else None)

    team_of = read_optional_mapping(args.team_assignments, "team")
    player_of = read_optional_mapping(args.player_mapping,
                                      "spieler_name_oder_nummer")

    players_by_frame = defaultdict(list)
    with open(args.positions_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if int(row["on_pitch"]):
                players_by_frame[int(row["frame"])].append(
                    (int(row["tracker_id"]), float(row["x_m"]), float(row["y_m"])))

    projected = []
    total = 0
    with open(candidates_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            total += 1
            frame = int(row["frame"])
            point = np.array([[[float(row["x_ref"]), float(row["y_ref"])]]],
                             dtype=np.float64)
            if piecewise:
                transforms = transforms_by_frame.get(frame)
                if transforms is None or not transforms[2]:
                    continue
                mapped, _ = transform_piecewise(
                    point.reshape(-1, 2), transforms[0], transforms[1],
                    split_x, length, width)
                x_m, y_m = mapped.reshape(2)
            else:
                matrix = H_by_frame.get(frame)
                if matrix is None:
                    continue
                x_m, y_m = cv2.perspectiveTransform(point, matrix).reshape(2)
            if not (np.isfinite(x_m) and np.isfinite(y_m)):
                continue
            on_pitch = (-args.margin_m <= x_m <= length + args.margin_m and
                        -args.margin_m <= y_m <= width + args.margin_m)
            if not on_pitch:
                continue
            projected.append((row, float(x_m), float(y_m)))

    block_frames = max(1, int(round(args.static_block_seconds * args.fps)))
    cell_stats = defaultdict(lambda: [0, set()])
    cells = []
    for row, x_m, y_m in projected:
        cell = (int(round(x_m / args.static_cell_m)),
                int(round(y_m / args.static_cell_m)))
        cells.append(cell)
        stats = cell_stats[cell]
        stats[0] += 1
        stats[1].add(int(row["frame"]) // block_frames)

    static_core = {
        cell for cell, (hits, blocks) in cell_stats.items()
        if hits >= args.static_min_hits and len(blocks) >= args.static_min_blocks
    }
    # Ein Nachbarzellensaum fängt die kleine Homographie-/Box-Jitterbewegung
    # desselben festen Bildmerkmals ab. Kurze echte Balllücken sind später
    # interpolierbar und deshalb weniger schädlich als langlebige Phantomtracks.
    static_cells = {
        (cell[0] + dx, cell[1] + dy)
        for cell in static_core for dx in (-1, 0, 1) for dy in (-1, 0, 1)
    }

    fields = [
        "frame", "source_frame", "x1", "y1", "x2", "y2", "x_ref", "y_ref",
        "x_m", "y_m", "conf", "static_rejected", "shape_ok", "usable",
        "nearest_tracker_id", "nearest_player", "nearest_team",
        "nearest_distance_m",
    ]
    usable_count = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (row, x_m, y_m), cell in zip(projected, cells):
            box_width = float(row["x2"]) - float(row["x1"])
            box_height = float(row["y2"]) - float(row["y1"])
            aspect = box_width / max(box_height, 1e-6)
            shape_ok = (3 <= box_width <= 35 and 3 <= box_height <= 35 and
                        0.5 <= aspect <= 2.0)
            static_rejected = cell in static_cells
            usable = shape_ok and not static_rejected
            usable_count += usable

            nearest_tid, nearest_distance = "", float("inf")
            for tid, player_x, player_y in players_by_frame.get(int(row["frame"]), []):
                distance = float(np.hypot(x_m - player_x, y_m - player_y))
                if distance < nearest_distance:
                    nearest_tid, nearest_distance = tid, distance

            writer.writerow({
                "frame": row["frame"],
                "source_frame": row["source_frame"],
                "x1": row["x1"], "y1": row["y1"],
                "x2": row["x2"], "y2": row["y2"],
                "x_ref": row["x_ref"], "y_ref": row["y_ref"],
                "x_m": f"{x_m:.3f}", "y_m": f"{y_m:.3f}",
                "conf": row["conf"],
                "static_rejected": int(static_rejected),
                "shape_ok": int(shape_ok),
                "usable": int(usable),
                "nearest_tracker_id": nearest_tid,
                "nearest_player": player_of.get(nearest_tid, ""),
                "nearest_team": team_of.get(nearest_tid, ""),
                "nearest_distance_m": (f"{nearest_distance:.3f}"
                                       if np.isfinite(nearest_distance) else ""),
            })

    print(f"Rohkandidaten: {total}")
    print(f"Auf/nahe Platz: {len(projected)}")
    print(f"Statische Kernzellen: {len(static_core)}")
    print(f"Nutzbare Kandidaten: {usable_count}")
    print(f"Ausgabe: {output_path}")


if __name__ == "__main__":
    main()
