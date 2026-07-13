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
    parser.add_argument("--team-assignments", default=None,
                        help="Optionale CSV aus team_assign.py für Teamstatistiken")
    parser.add_argument("--fps", type=float, default=29.97,
                        help="Bildrate des Originalvideos (Standard: 29.97)")
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
    # Nur für Legacy-Dateien nötig (globale Kalibrierung + Registrierungs-NPZ);
    # bei direkter Lokalisierung (H_px_to_pitch im NPZ) entfällt der Schlüssel.
    H_px_to_pitch = (np.linalg.inv(np.array(cal["H_pitch_to_px"]))
                     if "H_pitch_to_px" in cal else None)
    with np.load(args.homographies_npz) as homographies:
        direct_H = (homographies["H_px_to_pitch"].copy()
                    if "H_px_to_pitch" in homographies else None)
        H_all = (homographies["H"].copy() if "H" in homographies else None)
        # Neue Streaming-Dateien speichern absolute Video-Frameindizes separat.
        # Alte Clip-Dateien ohne `frames` bleiben weiterhin kompatibel.
        registered_frames = (homographies["frames"].copy()
                             if "frames" in homographies else None)
    matrices = direct_H if direct_H is not None else H_all
    H_by_frame = ({int(frame): H for frame, H in zip(registered_frames, matrices)}
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
            transform = H_frame if direct_H is not None else H_px_to_pitch @ H_frame
            xy = cv2.perspectiveTransform(foot, transform)
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

    # Heatmaps für alle erkannten Personen und optional getrennt pro Team.
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

    # Laufdistanzen (geglättet, nur plausible Sprünge). Eine Tracklet-ID
    # ist keine dauerhafte Spieleridentität; deshalb werden Distanz und Dauer
    # nur für tatsächlich beobachtete, zusammenhängende Abschnitte angegeben.
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
        # Lücken bis eine Sekunde dürfen zum selben sichtbaren Abschnitt
        # gehören. Längere Abwesenheiten werden weder als Weg noch als
        # sichtbare Zeit gezählt. Der zulässige Weg skaliert mit dem echten
        # Frame-Abstand (wichtig auch bei Stride).
        observed = (dt > 0) & (dt <= 1.0)
        plausible = observed & (steps < 12 * dt)  # >12 m/s = Tracking-Fehler
        distanz = float(steps[plausible].sum())
        sichtbare_dauer = float(dt[observed].sum())
        dists[tid] = (distanz, sichtbare_dauer, len(pts))

    dist_path = out_dir / f"{prefix}_distanzen.csv"
    with open(dist_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tracker_id", "distanz_m", "sichtbare_dauer_s",
                    "durchschnitt_kmh", "positionspunkte"])
        for tid, (distanz, dauer, n_punkte) in sorted(
                dists.items(), key=lambda kv: -kv[1][0]):
            speed = distanz / dauer * 3.6 if dauer > 0 else 0.0
            w.writerow([tid, f"{distanz:.2f}", f"{dauer:.2f}",
                        f"{speed:.2f}", n_punkte])
    print(f"Distanzreport: {dist_path}")

    if team_of:
        team_path = out_dir / f"{prefix}_team_statistiken.csv"
        with open(team_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["team", "sichtbare_distanz_km", "tracklets_mit_distanz",
                        "on_pitch_detektionen", "summierte_sichtbare_trackletzeit_h"])
            for team in sorted(set(team_of.values())):
                tids = [tid for tid in per_track if team_of.get(tid) == team]
                values = [dists[tid] for tid in tids if tid in dists]
                distance = sum(value[0] for value in values)
                duration = sum(value[1] for value in values)
                detections = sum(len(per_track[tid]) for tid in tids)
                w.writerow([team, f"{distance / 1000:.3f}", len(values),
                            detections, f"{duration / 3600:.3f}"])
        print(f"Teamstatistiken: {team_path}")

    for tid, (distanz, dauer, _) in sorted(
            dists.items(), key=lambda kv: -kv[1][0])[:10]:
        speed = distanz / dauer * 3.6 if dauer > 0 else 0.0
        print(f"  Track #{tid}: {distanz:5.1f} m in {dauer:.1f} s sichtbar "
              f"(Schnitt {speed:.1f} km/h)")


if __name__ == "__main__":
    main()
