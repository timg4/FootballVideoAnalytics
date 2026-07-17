"""Spieler-Heatmaps und robuste Lauf-/Sprintkennzahlen erzeugen.

Alle Werte sind sichtbare Mindestwerte. Geschwindigkeit wird segmentweise aus
geglätteten Meterpositionen berechnet; extreme Messsprünge werden verworfen.
"""

import argparse
import csv
import html
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from pitch_model import PitchModel


def slugify(value):
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    return value.strip("_") or "spieler"


def split_segments(points, max_gap=2):
    points = sorted(points)
    if not points:
        return []
    result, current = [], [points[0]]
    for point in points[1:]:
        if point[0] - current[-1][0] > max_gap:
            result.append(current)
            current = [point]
        else:
            current.append(point)
    result.append(current)
    return result


def speed_samples(points, fps, smooth_frames=15, delta_frames=15,
                  max_speed_kmh=35.0):
    """(frame, km/h, smoothed_xy) aus zusammenhängenden Trackabschnitten."""
    samples = []
    kernel = np.ones(smooth_frames) / smooth_frames
    for segment in split_segments(points):
        if len(segment) < smooth_frames + delta_frames + 2:
            continue
        frames = np.asarray([point[0] for point in segment])
        xy = np.asarray([[point[1], point[2]] for point in segment])
        smoothed = np.column_stack([
            np.convolve(xy[:, axis], kernel, mode="valid") for axis in range(2)
        ])
        smooth_frames_abs = frames[smooth_frames - 1:]
        frame_delta = (smooth_frames_abs[delta_frames:] -
                       smooth_frames_abs[:-delta_frames])
        dt = frame_delta / fps
        distance = np.linalg.norm(
            smoothed[delta_frames:] - smoothed[:-delta_frames], axis=1)
        speed = distance / dt * 3.6
        valid = ((frame_delta >= delta_frames) &
                 (frame_delta <= delta_frames + 2) &
                 np.isfinite(speed) & (speed < max_speed_kmh))
        for index in np.flatnonzero(valid):
            samples.append((int(smooth_frames_abs[index]), float(speed[index]),
                            smoothed[index]))
    return samples


def count_events(samples, threshold, fps, min_duration_s=0.5):
    """Zusammenhängende Geschwindigkeitsphasen oberhalb einer Schwelle."""
    if not samples:
        return 0, 0.0, 0.0
    min_frames = int(round(min_duration_s * fps))
    events = 0
    duration = distance = 0.0
    index = 0
    while index < len(samples):
        if samples[index][1] < threshold:
            index += 1
            continue
        end = index + 1
        while (end < len(samples) and samples[end][1] >= threshold and
               samples[end][0] - samples[end - 1][0] <= 2):
            end += 1
        frame_duration = samples[end - 1][0] - samples[index][0] + 1
        if frame_duration >= min_frames:
            events += 1
            duration += frame_duration / fps
            distance += float(np.linalg.norm(
                samples[end - 1][2] - samples[index][2]))
        index = end
    return events, duration, distance


def draw_heatmap(pitch, points, path, title):
    scale, margin = 12, 30
    base = pitch.draw_topdown(scale=scale, margin=margin)
    heat = np.zeros(base.shape[:2], dtype=np.float32)
    for _, x_m, y_m in points:
        px, py = int(x_m * scale) + margin, int(y_m * scale) + margin
        if 0 <= px < heat.shape[1] and 0 <= py < heat.shape[0]:
            heat[py, px] += 1
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=scale * 1.2)
    heat_u8 = (255 * heat / max(float(heat.max()), 1e-6)).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
    alpha = (heat_u8.astype(np.float32) / 255 * 0.75)[..., None]
    result = (base * (1 - alpha) + colored * alpha).astype(np.uint8)
    cv2.putText(result, title, (margin, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), result)


def main():
    parser = argparse.ArgumentParser(description="Spielerleistung + Heatmaps")
    parser.add_argument("positions_csv")
    parser.add_argument("mapping_csv")
    parser.add_argument("calibration_json")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--high-intensity", type=float, default=15.0,
                        help="Schwelle Hochintensitätslauf in km/h")
    parser.add_argument("--sprint", type=float, default=20.0,
                        help="Sprintschwelle in km/h")
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "data" / "output"
    prefix = args.output_prefix or "video_project"
    heat_dir = out_dir / f"{prefix}_spieler_heatmaps"
    heat_dir.mkdir(parents=True, exist_ok=True)

    mapping = {}
    mapping_rows = []
    with open(args.mapping_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            tid = int(row["tracker_id"])
            mapping[tid] = row["spieler_name_oder_nummer"]
            mapping_rows.append(row)

    points_by_track = defaultdict(list)
    with open(args.positions_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            tid = int(row["tracker_id"])
            if int(row["on_pitch"]) and tid in mapping:
                points_by_track[tid].append((int(row["frame"]),
                                             float(row["x_m"]), float(row["y_m"])))

    points_by_player = defaultdict(list)
    samples_by_player = defaultdict(list)
    for tid, points in points_by_track.items():
        player = mapping[tid]
        points_by_player[player].extend(points)
        samples_by_player[player].extend(speed_samples(points, args.fps))
    for player in samples_by_player:
        samples_by_player[player].sort(key=lambda item: item[0])

    distance_by_player = defaultdict(float)
    duration_by_player = defaultdict(float)
    tracks_by_player = defaultdict(set)
    for row in mapping_rows:
        player = row["spieler_name_oder_nummer"]
        distance_by_player[player] += float(row["distanz_m"])
        duration_by_player[player] += float(row["sichtbare_dauer_s"])
        tracks_by_player[player].add(int(row["tracker_id"]))

    pitch_data = json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))
    pitch = PitchModel(**pitch_data["pitch"])
    result_rows = []
    for player in sorted(points_by_player, key=lambda name: -distance_by_player[name]):
        samples = samples_by_player[player]
        speeds = np.asarray([sample[1] for sample in samples])
        robust_peak = float(np.percentile(speeds, 99)) if len(speeds) else 0.0
        high_events, high_seconds, high_distance = count_events(
            samples, args.high_intensity, args.fps)
        sprint_events, sprint_seconds, sprint_distance = count_events(
            samples, args.sprint, args.fps)
        heat_name = f"{slugify(player)}.png"
        draw_heatmap(pitch, points_by_player[player], heat_dir / heat_name, player)
        result_rows.append({
            "spieler": player,
            "sichtbare_distanz_km": distance_by_player[player] / 1000,
            "sichtbare_dauer_min": duration_by_player[player] / 60,
            "tracklets": len(tracks_by_player[player]),
            "robustes_spitzentempo_kmh": robust_peak,
            "high_intensity_events": high_events,
            "high_intensity_sekunden": high_seconds,
            "high_intensity_distanz_m": high_distance,
            "sprints": sprint_events,
            "sprint_sekunden": sprint_seconds,
            "sprint_distanz_m": sprint_distance,
            "heatmap": f"{heat_dir.name}/{heat_name}",
        })

    csv_path = out_dir / f"{prefix}_spieler_leistungsdaten.csv"
    fields = [key for key in result_rows[0] if key != "heatmap"] if result_rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in result_rows:
            writer.writerow({key: (f"{value:.3f}" if isinstance(value, float) else value)
                             for key, value in row.items() if key in fields})

    cards = json.dumps(result_rows, ensure_ascii=False)
    dashboard_path = out_dir / f"{prefix}_spieler_dashboard.html"
    dashboard = f"""<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Spieler-Leistungsdaten</title><style>
body{{font:15px system-ui;background:#0f172a;color:#f8fafc;margin:0}}header{{padding:18px 22px;background:#111827;position:sticky;top:0;z-index:2}}h1{{margin:0 0 5px}}header p{{margin:0;color:#cbd5e1}}main{{padding:16px;display:grid;grid-template-columns:repeat(auto-fill,minmax(350px,1fr));gap:14px}}article{{background:#1e293b;border:1px solid #334155;border-radius:10px;overflow:hidden}}article h2{{margin:12px 14px 6px}}img{{display:block;width:100%;background:#132}}table{{width:calc(100% - 28px);margin:8px 14px 14px;border-collapse:collapse}}td{{padding:5px;border-bottom:1px solid #334155}}td:last-child{{text-align:right;font-weight:600}}.warn{{color:#fde68a}}</style></head><body><header><h1>Spieler-Leistungsdaten</h1><p>Sichtbare Mindestwerte · Spitzentempo = robustes 99. Perzentil · Hochintensiv ≥ {args.high_intensity:g} km/h · Sprint ≥ {args.sprint:g} km/h, jeweils mindestens 0,5 s. <span class="warn">Mehrdeutige automatische Zuordnungen bleiben gruppiert.</span></p></header><main id="root"></main><script>
const DATA={cards};const root=document.querySelector('#root');for(const x of DATA){{const a=document.createElement('article');a.innerHTML=`<h2>${{x.spieler}}</h2><img src="${{x.heatmap}}"><table><tr><td>Sichtbare Distanz</td><td>${{x.sichtbare_distanz_km.toFixed(3)}} km</td></tr><tr><td>Sichtbare Zeit</td><td>${{x.sichtbare_dauer_min.toFixed(1)}} min</td></tr><tr><td>Robustes Spitzentempo</td><td>${{x.robustes_spitzentempo_kmh.toFixed(1)}} km/h</td></tr><tr><td>Hochintensitätsläufe</td><td>${{x.high_intensity_events}} (${{x.high_intensity_distanz_m.toFixed(0)}} m)</td></tr><tr><td>Sprints</td><td>${{x.sprints}} (${{x.sprint_distanz_m.toFixed(0)}} m)</td></tr><tr><td>Zugeordnete Tracklets</td><td>${{x.tracklets}}</td></tr></table>`;root.appendChild(a);}}
</script></body></html>"""
    dashboard_path.write_text(dashboard, encoding="utf-8")

    print(f"Leistungsdaten: {csv_path}")
    print(f"Dashboard: {dashboard_path}")
    print(f"Heatmaps: {heat_dir}")
    for row in result_rows:
        print(f"  {row['spieler']}: {row['sichtbare_distanz_km']:.3f} km, "
              f"Peak99 {row['robustes_spitzentempo_kmh']:.1f} km/h, "
              f"{row['sprints']} Sprints")


if __name__ == "__main__":
    main()
