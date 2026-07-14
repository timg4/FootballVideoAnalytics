"""Scrollbare Offline-Reviewseite zur manuellen Spielerzuordnung erzeugen.

Die Seite zeigt pro Tracklet mehrere gute Ganzkörper-Crops sowie Zeit und
sichtbare Distanz. Eingaben werden im Browser-LocalStorage gehalten und als
CSV exportiert; es ist kein lokaler Webserver erforderlich.
"""

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Offline-Spieler-Review erzeugen")
    parser.add_argument("video")
    parser.add_argument("tracks_csv")
    parser.add_argument("assignments_csv")
    parser.add_argument("distances_csv")
    parser.add_argument("embeddings_npz")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--images", type=int, default=3,
                        help="Maximale Vorschaubilder pro Tracklet")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    prefix = Path(args.tracks_csv).stem.replace("_tracked", "")
    out_dir = Path(__file__).resolve().parent.parent / "data" / "output"
    output = Path(args.output) if args.output else out_dir / f"{prefix}_spieler_review.html"
    image_dir = output.parent / f"{output.stem}_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    team_of = {}
    with open(args.assignments_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["team"] != "":
                team_of[int(row["tracker_id"])] = int(row["team"])

    stats = {}
    with open(args.distances_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            tid = int(row["tracker_id"])
            if tid in team_of:
                stats[tid] = {
                    "distance": float(row["distanz_m"]),
                    "duration": float(row["sichtbare_dauer_s"]),
                    "points": int(row["positionspunkte"]),
                }

    with np.load(args.embeddings_npz) as data:
        sample_tids = data["sample_track_ids"].astype(int)
        sample_frames = data["sample_frames"].astype(int)

    frames_by_tid = defaultdict(list)
    for tid, frame in zip(sample_tids, sample_frames):
        if tid in stats and frame not in frames_by_tid[tid]:
            frames_by_tid[tid].append(frame)
    for tid in frames_by_tid:
        choices = frames_by_tid[tid]
        if len(choices) > args.images:
            indices = np.linspace(0, len(choices) - 1, args.images).round().astype(int)
            frames_by_tid[tid] = [choices[index] for index in indices]

    wanted = {(frame, tid) for tid, frames in frames_by_tid.items() for frame in frames}
    boxes = {}
    spans = {}
    with open(args.tracks_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            tid = int(row["tracker_id"])
            if tid in stats:
                if tid not in spans:
                    spans[tid] = [frame, frame]
                else:
                    spans[tid][1] = frame
            if (frame, tid) in wanted:
                boxes[(frame, tid)] = tuple(float(row[key])
                                            for key in ("x1", "y1", "x2", "y2"))

    by_frame = defaultdict(list)
    for (frame, tid), box in boxes.items():
        by_frame[frame].append((tid, box))

    images_by_tid = defaultdict(list)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Video nicht lesbar: {args.video}")
    last_frame = max(by_frame) if by_frame else -1
    frame_idx = 0
    while frame_idx <= last_frame:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, box in by_frame.get(frame_idx, []):
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            x1 = max(0, int(x1 - 0.08 * width))
            x2 = min(frame.shape[1], int(x2 + 0.08 * width))
            y1 = max(0, int(y1 - 0.03 * height))
            y2 = min(frame.shape[0], int(y2 + 0.03 * height))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            preview = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_CUBIC)
            name = f"track_{tid}_frame_{frame_idx}.jpg"
            cv2.imwrite(str(image_dir / name), preview,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            images_by_tid[tid].append(f"{image_dir.name}/{name}")
        frame_idx += 1
    cap.release()

    cards = []
    for tid, stat in stats.items():
        if tid not in spans:
            continue
        cards.append({
            "tracker_id": tid,
            "team": team_of[tid],
            "distance": round(stat["distance"], 2),
            "duration": round(stat["duration"], 2),
            "start": round(spans[tid][0] / args.fps, 2),
            "end": round(spans[tid][1] / args.fps, 2),
            "images": images_by_tid.get(tid, []),
        })
    cards.sort(key=lambda card: -card["distance"])

    title = f"{prefix}: Spieler-Tracklets zuordnen"
    data_json = json.dumps(cards, ensure_ascii=False)
    storage_key = json.dumps(f"footballanalytics:{prefix}:player-mapping")
    page = f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body{{font:15px system-ui;background:#111827;color:#f3f4f6;margin:0}}
header{{position:sticky;top:0;z-index:2;background:#111827ee;padding:14px 20px;border-bottom:1px solid #374151}}
h1{{font-size:20px;margin:0 0 6px}} .hint{{margin:0 0 10px;color:#fde68a}} .controls{{display:flex;gap:14px;flex-wrap:wrap;align-items:center}}
input,select,button{{font:inherit;padding:7px 9px;border-radius:6px;border:1px solid #4b5563;background:#1f2937;color:#fff}}
button{{cursor:pointer;background:#2563eb}} #summary{{color:#93c5fd}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:12px;padding:16px}}
.card{{background:#1f2937;border:1px solid #374151;border-radius:9px;padding:11px}}
.meta{{display:flex;justify-content:space-between;margin-bottom:8px}} .team0{{color:#86efac}} .team1{{color:#93c5fd}}
.images{{display:flex;gap:5px;height:190px;justify-content:center;background:#0b1020;border-radius:6px;overflow:hidden}}
.images img{{height:190px;max-width:32%;object-fit:contain}} .mapping{{width:100%;box-sizing:border-box;margin-top:9px}}
.small{{font-size:12px;color:#9ca3af;margin-top:6px}}
</style></head><body>
<header><h1>{html.escape(title)}</h1><p class="hint"><b>Nicht alles mappen:</b> Nur 2–3 eindeutig erkennbare Referenzen pro realem Spieler markieren. Den Rest ordnet Re-ID automatisch zu.</p><div class="controls">
<label>Team <select id="team"><option value="all">alle</option><option value="0">Grün</option><option value="1">Blau</option></select></label>
<label>mindestens <input id="minDistance" type="number" value="50" min="0" step="5" style="width:65px"> m</label>
<label>Suche <input id="search" placeholder="ID oder Spieler"></label>
<button id="download">Referenzen als CSV herunterladen</button><span id="summary"></span>
</div></header><main id="cards"></main>
<script>
const DATA={data_json}; const KEY={storage_key};
const saved=JSON.parse(localStorage.getItem(KEY)||'{{}}');
const teamNames={{0:'Grün',1:'Blau'}};
function save(id,value){{saved[id]=value;localStorage.setItem(KEY,JSON.stringify(saved));renderSummary();}}
function visible(c){{const t=document.querySelector('#team').value, min=+document.querySelector('#minDistance').value||0;
 const q=document.querySelector('#search').value.toLowerCase(); const label=(saved[c.tracker_id]||'').toLowerCase();
 return (t==='all'||+t===c.team)&&c.distance>=min&&(!q||String(c.tracker_id).includes(q)||label.includes(q));}}
function render(){{const root=document.querySelector('#cards');root.innerHTML='';
 for(const c of DATA.filter(visible)){{const el=document.createElement('article');el.className='card';
  el.innerHTML=`<div class="meta"><b class="team${{c.team}}">${{teamNames[c.team]}} · Track #${{c.tracker_id}}</b><b>${{c.distance.toFixed(1)}} m</b></div>
  <div class="images">${{c.images.map(x=>`<img loading="lazy" src="${{x}}">`).join('')}}</div>
  <input class="mapping" placeholder="Spielername oder Rückennummer" value="${{saved[c.tracker_id]||''}}">
  <div class="small">${{c.start.toFixed(1)}}–${{c.end.toFixed(1)}} s · ${{c.duration.toFixed(1)}} s sichtbar</div>`;
  el.querySelector('input').addEventListener('input',e=>save(c.tracker_id,e.target.value.trim()));root.appendChild(el);}}
 renderSummary();}}
function renderSummary(){{const mapped=DATA.filter(c=>(saved[c.tracker_id]||'').trim());
 const dist=mapped.reduce((s,c)=>s+c.distance,0), players=new Set(mapped.map(c=>`${{c.team}}:${{saved[c.tracker_id].toLowerCase()}}`));
 document.querySelector('#summary').textContent=`${{players.size}} Spieler · ${{mapped.length}} Referenzen · ${{(dist/1000).toFixed(2)}} km manuell`;}}
for(const id of ['team','minDistance','search'])document.querySelector('#'+id).addEventListener('input',render);
document.querySelector('#download').addEventListener('click',()=>{{let rows=[['tracker_id','team','spieler_name_oder_nummer','distanz_m','sichtbare_dauer_s']];
 for(const c of DATA){{const name=(saved[c.tracker_id]||'').trim();if(name)rows.push([c.tracker_id,c.team,name,c.distance,c.duration]);}}
 const esc=x=>'"'+String(x).replaceAll('"','""')+'"';const csv=rows.map(r=>r.map(esc).join(',')).join('\\r\\n');
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv;charset=utf-8'}}));a.download='{prefix}_manuelles_spieler_mapping.csv';a.click();}});
render();
</script></body></html>"""
    output.write_text(page, encoding="utf-8")
    print(f"Reviewseite: {output}")
    print(f"Vorschaubilder: {image_dir} ({sum(map(len, images_by_tid.values()))})")
    print("Nur 2–3 sichere Referenzen pro Spieler markieren; Re-ID verteilt den Rest.")


if __name__ == "__main__":
    main()
