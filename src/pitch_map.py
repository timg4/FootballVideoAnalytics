"""Phase 2c: Spielerpositionen in Spielfeld-Metern + Heatmap + Laufdistanzen.

Kombiniert die drei Bausteine:
  1. Tracking-CSV (detect_track.py)        — wo ist wer im Bild
  2. Homographien-NPZ (register_frames.py) — wie ist die Kamera gerade gedreht
  3. Kalibrierung-JSON (calibrate_pitch.py)— Referenzbild -> Meter

Ergebnis: Positions-CSV (Meter), Heatmap-PNG, Laufdistanzen pro Track.
Der Platz-Filter fällt gratis ab: Wer außerhalb der Platzmaße steht
(Nachbarfelder), wird markiert und aus den Statistiken ausgeschlossen.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from pitch_model import PitchModel

RAND_M = 1.5  # Toleranz (m) außerhalb der Linien, bevor ein Punkt rausfliegt


def main():
    parser = argparse.ArgumentParser(description="Bildpositionen -> Spielfeld-Meter")
    parser.add_argument("tracks_csv")
    parser.add_argument("homographies_npz")
    parser.add_argument("calibration_json")
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--fps", type=float, default=29.97,
                        help="Bildrate des Originalvideos (Standard: 29.97)")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "data" / "output"
    prefix = args.out_prefix or Path(args.tracks_csv).stem.replace("_tracked", "")

    cal = json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))
    pitch = PitchModel(**cal["pitch"])
    H_px_to_pitch = np.linalg.inv(np.array(cal["H_pitch_to_px"]))
    with np.load(args.homographies_npz) as homographies:
        H_all = homographies["H"].copy()
        # Neue Streaming-Dateien speichern absolute Video-Frameindizes separat.
        # Alte Clip-Dateien ohne `frames` bleiben weiterhin kompatibel.
        registered_frames = (homographies["frames"].copy()
                             if "frames" in homographies else None)
    H_by_frame = ({int(frame): H for frame, H in zip(registered_frames, H_all)}
                  if registered_frames is not None else None)

    fps = args.fps

    # Fußpunkte -> Meter
    per_track = defaultdict(list)
    n_total = n_auf_platz = 0
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
            xy = cv2.perspectiveTransform(foot, H_px_to_pitch @ H_frame)
            x_m, y_m = xy.reshape(2)
            auf_platz = (-RAND_M <= x_m <= pitch.laenge + RAND_M
                         and -RAND_M <= y_m <= pitch.breite + RAND_M)
            n_total += 1
            n_auf_platz += auf_platz
            tid = int(r["tracker_id"])
            rows_out.append([frame_idx, tid, f"{x_m:.2f}", f"{y_m:.2f}", int(auf_platz)])
            if auf_platz:
                per_track[tid].append((frame_idx, x_m, y_m))

    csv_path = out_dir / f"{prefix}_positionen.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "tracker_id", "x_m", "y_m", "auf_platz"])
        w.writerows(rows_out)
    print(f"{n_auf_platz}/{n_total} Detektionen auf dem Platz -> {csv_path}")

    # Heatmap (alle Spieler auf dem Platz)
    scale, margin = 12, 30
    base = pitch.draw_topdown(scale=scale, margin=margin)
    heat = np.zeros(base.shape[:2], dtype=np.float32)
    for pts in per_track.values():
        for _, x_m, y_m in pts:
            px = int(x_m * scale) + margin
            py = int(y_m * scale) + margin
            if 0 <= px < heat.shape[1] and 0 <= py < heat.shape[0]:
                heat[py, px] += 1
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=scale * 1.2)
    heat = (255 * heat / max(heat.max(), 1e-6)).astype(np.uint8)
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    alpha = (heat.astype(np.float32) / 255 * 0.75)[..., None]
    heatmap = (base * (1 - alpha) + colored * alpha).astype(np.uint8)
    heat_path = out_dir / f"{prefix}_heatmap.png"
    cv2.imwrite(str(heat_path), heatmap)
    print(f"Heatmap: {heat_path}")

    # Laufdistanzen (geglättet, nur plausible Sprünge)
    print("\nLaufdistanzen (Top 10, nur Frames auf dem Platz):")
    dists = {}
    for tid, pts in per_track.items():
        if len(pts) < 10:
            continue
        pts = sorted(pts)
        xs = np.array([p[1] for p in pts])
        ys = np.array([p[2] for p in pts])
        k = 7  # gleitender Mittelwert gegen Detektionszittern
        if len(xs) > k:
            kernel = np.ones(k) / k
            xs = np.convolve(xs, kernel, mode="valid")
            ys = np.convolve(ys, kernel, mode="valid")
            frame_numbers = np.array([p[0] for p in pts])[k - 1:]
        else:
            frame_numbers = np.array([p[0] for p in pts])
        steps = np.hypot(np.diff(xs), np.diff(ys))
        dt = np.diff(frame_numbers) / fps
        # Der zulässige Weg skaliert mit dem echten Frame-Abstand (auch bei Stride).
        steps = steps[steps < 12 * dt]  # >12 m/s ist Tracking-Fehler, kein Sprint
        dists[tid] = steps.sum()
    if registered_frames is not None and len(registered_frames):
        typische_luecke = np.median(np.diff(registered_frames)) if len(registered_frames) > 1 else 1
        dauer = (registered_frames[-1] - registered_frames[0] + typische_luecke) / fps
    else:
        dauer = len(H_all) / fps
    dauer = max(dauer, 1e-6)
    for tid, d in sorted(dists.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  Track #{tid}: {d:5.1f} m in {dauer:.1f} s "
              f"(Schnitt {d / dauer * 3.6:.1f} km/h)")


if __name__ == "__main__":
    main()
