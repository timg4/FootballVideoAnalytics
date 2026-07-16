"""Convert image positions into pitch meters and produce heatmap/running distances.

Combines the three building blocks:
  1. tracking CSV (detect_track.py)        : who is where in the image
  2. homographies NPZ (register_frames.py) : how the camera is currently rotated
  3. calibration JSON (calibrate_pitch.py) : reference image -> meter

Output: position CSV (meters), heatmap PNG, running distances per track. The pitch
filter falls out for free: anything outside the field dimensions (the neighboring
pitches) is flagged and excluded from the stats.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from pitch_model import PitchModel

MARGIN_M = 1.5  # tolerance (m) outside the lines before a point is dropped


def main():
    parser = argparse.ArgumentParser(description="Image positions -> pitch meters")
    parser.add_argument("tracks_csv")
    parser.add_argument("homographies_npz")
    parser.add_argument("calibration_json")
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--team-assignments", default=None,
                        help="optional CSV from team_assign.py for team stats")
    parser.add_argument("--fps", type=float, default=29.97,
                        help="frame rate of the source video (default: 29.97)")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "data" / "output"
    prefix = args.out_prefix or Path(args.tracks_csv).stem.replace("_tracked", "")
    team_of = {}
    if args.team_assignments:
        with open(args.team_assignments, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row["team"] != "":
                    team_of[int(row["tracker_id"])] = int(row["team"])

    cal = json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))
    pitch = PitchModel(**cal["pitch"])
    # only needed for legacy files (global calibration + registration NPZ);
    # with direct localization (H_px_to_pitch in the NPZ) this key is absent.
    H_px_to_pitch = (np.linalg.inv(np.array(cal["H_pitch_to_px"]))
                     if "H_pitch_to_px" in cal else None)
    with np.load(args.homographies_npz) as homographies:
        direct_H = (homographies["H_px_to_pitch"].copy()
                    if "H_px_to_pitch" in homographies else None)
        H_all = (homographies["H"].copy() if "H" in homographies else None)
        # newer streaming files store the absolute video frame indices separately.
        # older clip files without `frames` stay compatible.
        registered_frames = (homographies["frames"].copy()
                             if "frames" in homographies else None)
    matrices = direct_H if direct_H is not None else H_all
    H_by_frame = ({int(frame): H for frame, H in zip(registered_frames, matrices)}
                  if registered_frames is not None else None)

    fps = args.fps

    # foot points -> meters
    per_track = defaultdict(list)
    n_total = n_on_pitch = 0
    rows_out = []
    with open(args.tracks_csv, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            frame_idx = int(r["frame"])
            if H_by_frame is not None:
                H_frame = H_by_frame.get(frame_idx)
                if H_frame is None:
                    continue
            else:
                if frame_idx >= len(H_all):
                    continue
                H_frame = H_all[frame_idx]
            foot = np.array([[[(float(r["x1"]) + float(r["x2"])) / 2,
                               float(r["y2"])]]], dtype=np.float64)
            transform = H_frame if direct_H is not None else H_px_to_pitch @ H_frame
            xy = cv2.perspectiveTransform(foot, transform)
            x_m, y_m = xy.reshape(2)
            on_pitch = (-MARGIN_M <= x_m <= pitch.laenge + MARGIN_M
                        and -MARGIN_M <= y_m <= pitch.breite + MARGIN_M)
            n_total += 1
            n_on_pitch += on_pitch
            tid = int(r["tracker_id"])
            rows_out.append([frame_idx, tid, f"{x_m:.2f}", f"{y_m:.2f}", int(on_pitch)])
            if on_pitch:
                per_track[tid].append((frame_idx, x_m, y_m))

    csv_path = out_dir / f"{prefix}_positions.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "tracker_id", "x_m", "y_m", "on_pitch"])
        w.writerows(rows_out)
    print(f"{n_on_pitch}/{n_total} detections on the pitch -> {csv_path}")

    # heatmaps for all detected people, and optionally split per team
    def write_heatmap(track_ids, path):
        scale, margin = 12, 30
        base = pitch.draw_topdown(scale=scale, margin=margin)
        heat = np.zeros(base.shape[:2], dtype=np.float32)
        for tid in track_ids:
            for _, x_m, y_m in per_track.get(tid, []):
                px = int(x_m * scale) + margin
                py = int(y_m * scale) + margin
                if 0 <= px < heat.shape[1] and 0 <= py < heat.shape[0]:
                    heat[py, px] += 1
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=scale * 1.2)
        heat = (255 * heat / max(heat.max(), 1e-6)).astype(np.uint8)
        colored = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
        alpha = (heat.astype(np.float32) / 255 * 0.75)[..., None]
        result = (base * (1 - alpha) + colored * alpha).astype(np.uint8)
        cv2.imwrite(str(path), result)
        print(f"Heatmap: {path}")

    write_heatmap(per_track.keys(), out_dir / f"{prefix}_heatmap.png")
    for team in sorted(set(team_of.values())):
        tids = [tid for tid in per_track if team_of.get(tid) == team]
        write_heatmap(tids, out_dir / f"{prefix}_team_{team}_heatmap.png")

    # running distances (smoothed, only plausible jumps). A tracklet ID is not a
    # persistent player identity, so distance and duration are reported only for
    # the actually observed, contiguous stretches.
    print("\nRunning distances (top 10, only on-pitch frames):")
    dists = {}
    for tid, pts in per_track.items():
        if len(pts) < 10:
            continue
        pts = sorted(pts)
        xs = np.array([p[1] for p in pts])
        ys = np.array([p[2] for p in pts])
        k = 7  # moving average against detection jitter
        if len(xs) > k:
            kernel = np.ones(k) / k
            xs = np.convolve(xs, kernel, mode="valid")
            ys = np.convolve(ys, kernel, mode="valid")
            frame_numbers = np.array([p[0] for p in pts])[k - 1:]
        else:
            frame_numbers = np.array([p[0] for p in pts])
        steps = np.hypot(np.diff(xs), np.diff(ys))
        dt = np.diff(frame_numbers) / fps
        # gaps up to one second may belong to the same visible stretch. Longer
        # absences count neither as distance nor as visible time. The allowed
        # step scales with the real frame spacing (matters with a stride too).
        observed = (dt > 0) & (dt <= 1.0)
        plausible = observed & (steps < 12 * dt)  # >12 m/s = tracking error
        distance = float(steps[plausible].sum())
        visible_duration = float(dt[observed].sum())
        dists[tid] = (distance, visible_duration, len(pts))

    dist_path = out_dir / f"{prefix}_distances.csv"
    with open(dist_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tracker_id", "distance_m", "visible_seconds",
                    "avg_kmh", "position_samples"])
        for tid, (distance, duration, n_points) in sorted(
                dists.items(), key=lambda kv: -kv[1][0]):
            speed = distance / duration * 3.6 if duration > 0 else 0.0
            w.writerow([tid, f"{distance:.2f}", f"{duration:.2f}",
                        f"{speed:.2f}", n_points])
    print(f"Distance report: {dist_path}")

    if team_of:
        team_path = out_dir / f"{prefix}_team_stats.csv"
        with open(team_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["team", "visible_distance_km", "tracklets_with_distance",
                        "on_pitch_detections", "summed_visible_tracklet_hours"])
            for team in sorted(set(team_of.values())):
                tids = [tid for tid in per_track if team_of.get(tid) == team]
                values = [dists[tid] for tid in tids if tid in dists]
                distance = sum(value[0] for value in values)
                duration = sum(value[1] for value in values)
                detections = sum(len(per_track[tid]) for tid in tids)
                w.writerow([team, f"{distance / 1000:.3f}", len(values),
                            detections, f"{duration / 3600:.3f}"])
        print(f"Team stats: {team_path}")

    for tid, (distance, duration, _) in sorted(
            dists.items(), key=lambda kv: -kv[1][0])[:10]:
        speed = distance / duration * 3.6 if duration > 0 else 0.0
        print(f"  Track #{tid}: {distance:5.1f} m in {duration:.1f} s visible "
              f"(avg {speed:.1f} km/h)")


if __name__ == "__main__":
    main()
