"""Klickbares Review der tatsächlichen Kontaktperson an bestätigten Pässen.

Pro Abgabe/Annahme werden drei Frames und die räumlich nächsten Spieler
angezeigt. Die alte Näherungszuordnung ist vorausgewählt. Das Review erzeugt
Ground Truth für einen späteren Ball-Person-Interaction-Classifier.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


COLORS = [(56, 189, 248), (34, 197, 94), (251, 146, 60),
          (244, 114, 182), (250, 204, 21), (167, 139, 250)]


def load_mapping(path: str, value_column: str, cast=str):
    result = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get(value_column, "") != "":
                result[int(row["tracker_id"])] = cast(row[value_column])
    return result


def clock(frame: int, fps: float) -> str:
    seconds = frame / fps
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def main():
    parser = argparse.ArgumentParser(description="Touch-/Kontaktreview erzeugen")
    parser.add_argument("video_1080")
    parser.add_argument("pass_review_csv")
    parser.add_argument("ball_track_csv")
    parser.add_argument("positions_csv")
    parser.add_argument("player_tracks_csv")
    parser.add_argument("team_assignments_csv")
    parser.add_argument("player_mapping_csv")
    parser.add_argument("--frame-offset", type=int, default=10)
    parser.add_argument("--frame-step", type=int, default=6)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output", default="data/output/video_project_contact_review.html")
    args = parser.parse_args()

    with open(args.pass_review_csv, newline="", encoding="utf-8-sig") as handle:
        pass_rows = [row for row in csv.DictReader(handle)
                     if row.get("review_status") == "pass"]

    ball = {}
    with open(args.ball_track_csv, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            ball[(int(row["ball_tracklet"]), int(row["frame"]))] = (
                float(row["x_ref"]), float(row["y_ref"]),
                float(row["x_m"]), float(row["y_m"]),
            )

    team_of = load_mapping(args.team_assignments_csv, "team", int)
    player_of = load_mapping(args.player_mapping_csv,
                             "spieler_name_oder_nummer", str)

    contact_specs = []
    requested_position_frames = set()
    requested_box_frames = set()
    for event_index, row in enumerate(pass_rows):
        for side, frame_column, tracker_column, team_column in (
            ("Abgabe", "abgabe_frame", "von_tracker_id", "review_von_team"),
            ("Annahme", "annahme_frame", "zu_tracker_id", "review_zu_team"),
        ):
            frame = int(row[frame_column])
            spec = {
                "event_index": event_index,
                "ball_tracklet": int(row["ball_tracklet"]),
                "side": side,
                "frame": frame,
                "auto_tracker_id": int(row[tracker_column]),
                "review_team": (int(float(row[team_column]))
                                if row.get(team_column, "") != "" else None),
            }
            contact_specs.append(spec)
            requested_position_frames.add(frame)
            requested_box_frames.update((frame - args.frame_step, frame,
                                         frame + args.frame_step))

    positions_by_frame = defaultdict(list)
    with open(args.positions_csv, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            if frame in requested_position_frames and int(row["on_pitch"]):
                tracker = int(row["tracker_id"])
                if tracker in team_of:
                    positions_by_frame[frame].append(
                        (tracker, float(row["x_m"]), float(row["y_m"])))

    boxes = {}
    with open(args.player_tracks_csv, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            if frame in requested_box_frames:
                boxes[(frame, int(row["tracker_id"]))] = tuple(
                    float(row[name]) for name in ("x1", "y1", "x2", "y2"))

    for spec in contact_specs:
        _, _, ball_x, ball_y = ball[(spec["ball_tracklet"], spec["frame"])]
        distances = []
        for tracker, x_m, y_m in positions_by_frame[spec["frame"]]:
            distance = float(np.hypot(x_m - ball_x, y_m - ball_y))
            if (spec["frame"], tracker) in boxes:
                distances.append((distance, tracker))
        distances.sort()
        candidates = distances[:args.max_candidates]
        auto_tracker = spec["auto_tracker_id"]
        if auto_tracker not in [tracker for _, tracker in candidates]:
            auto_distance = next((distance for distance, tracker in distances
                                  if tracker == auto_tracker), None)
            if auto_distance is not None:
                candidates[-1:] = [(auto_distance, auto_tracker)]
                candidates.sort()
        spec["candidates"] = [
            {
                "key": chr(65 + index), "tracker_id": tracker,
                "team": team_of.get(tracker),
                "team_name": "Blau" if team_of.get(tracker) == 1 else "Grün",
                "player": player_of.get(tracker, ""),
                "distance_m": round(distance, 2),
                "auto": tracker == auto_tracker,
            }
            for index, (distance, tracker) in enumerate(candidates)
        ]

    output = Path(args.output)
    image_dir = output.with_name(output.stem + "_images")
    image_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video_1080)
    if not cap.isOpened():
        raise SystemExit(f"Video konnte nicht geöffnet werden: {args.video_1080}")

    cards = []
    for index, spec in enumerate(contact_specs):
        panels = []
        for relative, label in ((-args.frame_step, "vorher"), (0, "Kontakt"),
                                (args.frame_step, "nachher")):
            frame = spec["frame"] + relative
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame + args.frame_offset)
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f"Video-Frame fehlt: {frame + args.frame_offset}")
            scale_x, scale_y = image.shape[1] / 1280, image.shape[0] / 720
            for candidate_index, candidate in enumerate(spec["candidates"]):
                box = boxes.get((frame, candidate["tracker_id"]))
                if box is None:
                    continue
                color = COLORS[candidate_index]
                x1, y1, x2, y2 = [int(value * scale_x if n % 2 == 0 else value * scale_y)
                                  for n, value in enumerate(box)]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                cv2.rectangle(image, (x1, max(0, y1 - 30)), (x1 + 32, y1), color, -1)
                cv2.putText(image, candidate["key"], (x1 + 7, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (15, 23, 42), 2,
                            cv2.LINE_AA)
            ball_row = ball.get((spec["ball_tracklet"], frame))
            if ball_row:
                center = (int(ball_row[0] * scale_x), int(ball_row[1] * scale_y))
                cv2.circle(image, center, 18, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.circle(image, center, 3, (0, 0, 255), -1, cv2.LINE_AA)
            panel = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
            cv2.rectangle(panel, (0, 0), (639, 30), (15, 23, 42), -1)
            cv2.putText(panel, f"{label}  F{frame}", (8, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)
            panels.append(panel)
        name = f"contact_{index:03d}.jpg"
        cv2.imwrite(str(image_dir / name), cv2.hconcat(panels),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        cards.append({
            **spec, "id": index, "time": clock(spec["frame"], args.fps),
            "image": f"{image_dir.name}/{name}",
        })
        if (index + 1) % 10 == 0 or index + 1 == len(contact_specs):
            print(f"Kontaktkarten: {index + 1}/{len(contact_specs)}", flush=True)
    cap.release()

    data = json.dumps(cards, ensure_ascii=False)
    key = f"contact-review:{Path(args.pass_review_csv).name}:v1"
    document = f"""<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Kontaktperson-Review</title><style>
body{{font:15px system-ui;background:#0f172a;color:#f8fafc;margin:0}}header{{position:sticky;top:0;z-index:3;background:#111827;padding:12px 18px;border-bottom:1px solid #334155}}h1{{margin:0 0 5px;font-size:21px}}header p{{margin:0;color:#cbd5e1}}button{{border:0;border-radius:7px;padding:8px 11px;cursor:pointer;font-weight:700}}#export{{background:#38bdf8;margin-left:10px}}main{{padding:12px;display:grid;grid-template-columns:repeat(auto-fill,minmax(700px,1fr));gap:11px}}article{{background:#1e293b;border:2px solid #334155;border-radius:10px;overflow:hidden}}article.done{{border-color:#22c55e}}article.unclear{{border-color:#f59e0b}}img{{display:block;width:100%}}.meta{{padding:9px 11px}}h2{{font-size:16px;margin:0 0 5px}}.hint{{color:#cbd5e1}}.actions{{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}}.candidate{{color:#0f172a}}.auto{{outline:3px solid white}}.accept{{background:#22c55e}}.unknown{{background:#f59e0b}}.small{{font-size:12px;opacity:.82}}@media(max-width:800px){{main{{grid-template-columns:1fr}}}}</style></head><body><header><h1>Kontaktperson bestätigen</h1><p>Rot = Ball. A–F = mögliche Spieler. Weiß umrandet = bisherige Automatik. Meist reicht „Auto passt“. <span id="count"></span><button id="export">CSV exportieren</button></p></header><main id="root"></main><script>
const DATA={data},KEY={json.dumps(key)};let state=JSON.parse(localStorage.getItem(KEY)||'{{}}');const root=document.querySelector('#root');
function entry(x){{return state[x.id]||(state[x.id]={{status:'',tracker_id:'',team:'',player:''}})}}
function save(){{localStorage.setItem(KEY,JSON.stringify(state));count()}}function count(){{let v=Object.values(state);document.querySelector('#count').textContent=`Erledigt ${{v.filter(x=>x.status).length}}/${{DATA.length}} · korrigiert ${{v.filter(x=>x.status==='corrected').length}} · unklar ${{v.filter(x=>x.status==='unclear').length}}`;}}
function choose(article,x,c,status){{let e=entry(x);e.status=status;e.tracker_id=String(c?.tracker_id??'');e.team=String(c?.team??'');e.player=String(c?.player??'');article.className=status==='unclear'?'unclear':'done';save();}}
for(const x of DATA){{let a=document.createElement('article'),e=entry(x);if(e.status)a.className=e.status==='unclear'?'unclear':'done';let auto=x.candidates.find(c=>c.auto);a.innerHTML=`<img src="${{x.image}}"><div class="meta"><h2>#${{x.id+1}} · ${{x.side}} · ${{x.time}} · erwartetes Team: ${{x.review_team===1?'Blau':'Grün'}}</h2><div class="hint">Wähle den Spieler, der den Ball tatsächlich abgibt/annimmt – nicht bloß den nächststehenden Gegner.</div><div class="actions"><button class="accept">✓ Auto passt (${{auto?.key||'?'}})</button>${{x.candidates.map((c,i)=>`<button class="candidate ${{c.auto?'auto':''}}" style="background:rgb(${{[248,94,60,182,21,250][i]}},${{[189,197,146,114,204,139][i]}},${{[56,34,251,244,250,167][i]}})">${{c.key}} · ${{c.team_name}} ${{c.player||'#'+c.tracker_id}} <span class="small">${{c.distance_m}}m</span></button>`).join('')}}<button class="unknown">unklar / nicht dabei</button></div></div>`;a.querySelector('.accept').onclick=()=>choose(a,x,auto,'confirmed');a.querySelectorAll('.candidate').forEach((b,i)=>b.onclick=()=>choose(a,x,x.candidates[i],x.candidates[i].auto?'confirmed':'corrected'));a.querySelector('.unknown').onclick=()=>choose(a,x,null,'unclear');root.appendChild(a);}}
document.querySelector('#export').onclick=()=>{{let cols=['event_index','ball_tracklet','side','frame','review_team','auto_tracker_id','selected_tracker_id','selected_team','selected_player','contact_status'];let esc=v=>'"'+String(v??'').replaceAll('"','""')+'"',lines=[cols.map(esc).join(',')];for(const x of DATA){{let e=entry(x),row={{...x,selected_tracker_id:e.tracker_id,selected_team:e.team,selected_player:e.player,contact_status:e.status}};lines.push(cols.map(k=>esc(row[k])).join(','));}}let blob=new Blob(['\\ufeff'+lines.join('\\n')],{{type:'text/csv'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='video_project_contact_review.csv';a.click();URL.revokeObjectURL(a.href);}};save();
</script></body></html>"""
    output.write_text(document, encoding="utf-8")
    print(f"Review: {output}")
    print(f"Bilder: {image_dir}")


if __name__ == "__main__":
    main()
