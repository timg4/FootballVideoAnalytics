"""Build a small visual review page for a full-video pitch calibration.

The review samples the whole recording, draws the exact pitch boundary, maps
player foot points, and optionally adds the tracked ball.  It is deliberately a
human acceptance check: aggregate counts alone cannot reveal a boundary that
cuts through the active game in one camera direction.
"""

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from piecewise_pitch import draw_piecewise_overlay, transform_piecewise


def parse_frames(value):
    return sorted({int(item.strip()) for item in value.split(",") if item.strip()})


def selected_rows(path, selected):
    rows = defaultdict(list)
    if not path:
        return rows
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            if frame in selected:
                rows[frame].append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Visual full-video QA for a piecewise pitch calibration")
    parser.add_argument("video")
    parser.add_argument("calibration_json")
    parser.add_argument("localization_npz")
    parser.add_argument("tracks_csv")
    parser.add_argument("--ball-track-csv")
    parser.add_argument("--frames",
                        help="comma-separated frames; default samples the video")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output",
                        default="data/output/video_project_pitch_qa.html")
    args = parser.parse_args()

    calibration = json.loads(
        Path(args.calibration_json).read_text(encoding="utf-8"))
    piecewise = calibration.get("piecewise")
    if not piecewise:
        raise SystemExit("QA currently requires a piecewise calibration")
    pitch = calibration["pitch"]
    length = float(pitch["laenge"])
    width = float(pitch["breite"])
    split = float(piecewise["split_x_m"])

    with np.load(args.localization_npz) as localization:
        all_frames = localization["frames"].astype(int)
        H_left_all = localization["H_px_to_pitch_left"].astype(float)
        H_right_all = localization["H_px_to_pitch_right"].astype(float)
        reliable_all = (localization["anchor_reliable"].astype(bool)
                        if "anchor_reliable" in localization
                        else np.ones(len(all_frames), dtype=bool))

    index_of = {int(frame): index for index, frame in enumerate(all_frames)}
    if args.frames:
        frames = parse_frames(args.frames)
    else:
        targets = np.linspace(all_frames.min(), all_frames.max(),
                              max(2, args.samples)).round().astype(int)
        reliable_frames = all_frames[reliable_all]
        sampled = [int(reliable_frames[np.argmin(abs(reliable_frames - target))])
                   for target in targets]
        anchors = [int(item["frame"]) for item in piecewise["anchors"]]
        frames = sorted(set(sampled + anchors))
    missing = [frame for frame in frames if frame not in index_of]
    if missing:
        raise SystemExit(f"Frames not in localization: {missing}")

    selected = set(frames)
    tracks = selected_rows(args.tracks_csv, selected)
    balls = selected_rows(args.ball_track_csv, selected)

    output = Path(args.output).resolve()
    image_dir = output.with_name(output.stem + "_frames")
    image_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cards = []
    for frame_index in frames:
        index = index_of[frame_index]
        H_left = H_left_all[index]
        H_right = H_right_all[index]
        reliable = bool(reliable_all[index])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"Could not read video frame {frame_index}")
        result = draw_piecewise_overlay(
            frame, H_left, H_right, split, length, width, thickness=3)

        inside = outside = 0
        for row in tracks.get(frame_index, []):
            foot = np.array([[(float(row["x1"]) + float(row["x2"])) / 2,
                              float(row["y2"])]], dtype=float)
            mapped, _ = transform_piecewise(
                foot, H_left, H_right, split, length, width)
            x_m, y_m = mapped[0]
            on_pitch = 0 <= x_m <= length and 0 <= y_m <= width
            inside += int(on_pitch)
            outside += int(not on_pitch)
            color = (40, 205, 70) if on_pitch else (35, 45, 235)
            point = tuple(np.rint(foot[0]).astype(int))
            cv2.circle(result, point, 6, color, -1, cv2.LINE_AA)
            cv2.putText(result, f"#{row['tracker_id']} {x_m:.0f},{y_m:.0f}",
                        (point[0] + 7, point[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)

        ball_state = "kein Balltrack"
        for row in balls.get(frame_index, [])[:1]:
            point_float = np.array([[float(row["x_ref"]),
                                     float(row["y_ref"])]], dtype=float)
            mapped, _ = transform_piecewise(
                point_float, H_left, H_right, split, length, width)
            ball_x, ball_y = mapped[0]
            ball_inside = 0 <= ball_x <= length and 0 <= ball_y <= width
            ball_state = "Ball innen" if ball_inside else "BALL AUSSEN"
            point = tuple(np.rint(point_float[0]).astype(int))
            color = (255, 185, 0) if ball_inside else (0, 0, 255)
            cv2.circle(result, point, 12, color, 3, cv2.LINE_AA)
            cv2.putText(result, f"BALL {ball_x:.1f},{ball_y:.1f}",
                        (point[0] + 14, point[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

        minutes = int(frame_index / fps // 60)
        seconds = int(frame_index / fps % 60)
        reliability = "metrisch freigegeben" if reliable else "herausgefiltert"
        banner = (f"Frame {frame_index} | {minutes:02d}:{seconds:02d} | "
                  f"{reliability} | Spieler {inside} innen / {outside} aussen | "
                  f"{ball_state}")
        cv2.rectangle(result, (0, 0), (result.shape[1], 34), (20, 25, 34), -1)
        cv2.putText(result, banner, (9, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.49, (245, 245, 245), 1, cv2.LINE_AA)

        image_path = image_dir / f"frame_{frame_index:05d}.jpg"
        cv2.imwrite(str(image_path), result, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cards.append({
            "frame": frame_index,
            "time": f"{minutes:02d}:{seconds:02d}",
            "inside": inside,
            "outside": outside,
            "reliable": reliable,
            "ball": ball_state,
            "image": image_path.relative_to(output.parent).as_posix(),
        })
    cap.release()

    cards_json = json.dumps(cards, ensure_ascii=False)
    title = "Pitch-Kalibrierung prüfen"
    document = f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
body{{margin:0;background:#f4f4f2;color:#202225;font:15px/1.45 system-ui,sans-serif}}
header{{max-width:1120px;margin:auto;padding:28px 20px 18px}}
h1{{font-size:24px;margin:0 0 8px}} p{{margin:6px 0}} .legend{{color:#565b61}}
main{{max-width:1120px;margin:auto;padding:0 20px 36px;display:grid;gap:20px}}
article{{background:#fff;border:1px solid #d7d9dc;border-radius:6px;overflow:hidden}}
img{{display:block;width:100%;height:auto;background:#111}}
.meta{{padding:12px 14px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
.meta strong{{margin-right:auto}} button{{font:inherit;padding:6px 10px;border:1px solid #aeb3b8;border-radius:4px;background:#fff;cursor:pointer}}
button.selected{{background:#253449;color:#fff;border-color:#253449}}
input{{font:inherit;min-width:220px;flex:1;padding:7px;border:1px solid #aeb3b8;border-radius:4px}}
#export{{margin-top:12px}} .filtered{{color:#8b3b13}}
@media(max-width:600px){{header,main{{padding-left:10px;padding-right:10px}} input{{min-width:100%}}}}
</style></head><body>
<header><h1>{title}</h1>
<p>Gelb = echte Feldgrenze, Cyan = virtuelle Naht zwischen den beiden Kameramodellen, Grün = im Feld, Rot = außerhalb, Blau = Ball.</p>
<p class="legend">Als auffällig markieren, wenn Gelb durch das aktive Spiel schneidet, ein Spieler des Nachbarfelds grün wird oder der sichtbare Ball außerhalb liegt.</p>
<button id="export">Bewertung als CSV exportieren</button></header>
<main id="review"></main>
<script>
const DATA={cards_json}; const KEY='football-pitch-qa-v1';
let saved=JSON.parse(localStorage.getItem(KEY)||'{{}}');
const root=document.getElementById('review');
function persist(){{localStorage.setItem(KEY,JSON.stringify(saved));}}
for(const item of DATA){{
 const state=saved[item.frame]||{{status:'',note:''}}; saved[item.frame]=state;
 const card=document.createElement('article');
 card.innerHTML=`<img loading="lazy" src="${{item.image}}" alt="Kontrollframe ${{item.frame}}"><div class="meta"><strong class="${{item.reliable?'':'filtered'}}">Frame ${{item.frame}} · ${{item.time}} · ${{item.reliable?'freigegeben':'gefiltert'}} · ${{item.inside}}/${{item.outside}} innen/außen</strong><button data-value="ok">passt</button><button data-value="bad">auffällig</button><input aria-label="Notiz zu Frame ${{item.frame}}" placeholder="optionale Notiz" value="${{state.note.replaceAll('&','&amp;').replaceAll('"','&quot;')}}"></div>`;
 const buttons=[...card.querySelectorAll('button')];
 function paint(){{buttons.forEach(b=>b.classList.toggle('selected',b.dataset.value===state.status));}}
 buttons.forEach(button=>button.onclick=()=>{{state.status=button.dataset.value;persist();paint();}});
 card.querySelector('input').oninput=e=>{{state.note=e.target.value;persist();}}; paint(); root.appendChild(card);
}}
document.getElementById('export').onclick=()=>{{
 const rows=[['frame','zeit','metrisch_freigegeben','spieler_innen','spieler_aussen','ball','bewertung','notiz']];
 for(const item of DATA){{const s=saved[item.frame]||{{}};rows.push([item.frame,item.time,item.reliable?1:0,item.inside,item.outside,item.ball,s.status||'',s.note||'']);}}
 const csv=rows.map(r=>r.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\\r\\n');
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv;charset=utf-8'}}));a.download='video_project_pitch_qa.csv';a.click();URL.revokeObjectURL(a.href);
}};
</script></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(f"QA page: {output}")
    print(f"Frames: {', '.join(map(str, frames))}")


if __name__ == "__main__":
    main()
