"""Klickbare visuelle Prüfung der automatisch abgeleiteten Passkandidaten."""

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

import cv2


def clock(frame, fps):
    seconds = frame / fps
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def transfer_reviews(pass_rows, seed_path, tolerance):
    """Transfer only unique, close old reviews to newly inferred events."""
    if not seed_path:
        return {}
    with open(seed_path, newline="", encoding="utf-8-sig") as handle:
        seeds = [row for row in csv.DictReader(handle)
                 if row.get("review_status", "") not in ("", "ungeprüft")]
    candidates = []
    for new_index, new in enumerate(pass_rows):
        new_start = int(new["abgabe_frame"])
        new_end = int(new["annahme_frame"])
        for seed_index, seed in enumerate(seeds):
            start_delta = abs(new_start - int(seed["abgabe_frame"]))
            end_delta = abs(new_end - int(seed["annahme_frame"]))
            if start_delta <= tolerance and end_delta <= tolerance:
                candidates.append((start_delta + end_delta,
                                   max(start_delta, end_delta),
                                   new_index, seed_index))
    transferred = {}
    used_new = set()
    used_seed = set()
    for _cost, _maximum, new_index, seed_index in sorted(candidates):
        if new_index in used_new or seed_index in used_seed:
            continue
        seed = seeds[seed_index]
        transferred[new_index] = {
            "status": seed["review_status"],
            "von_team": seed.get("review_von_team", ""),
            "zu_team": seed.get("review_zu_team", ""),
            "von_spieler": seed.get("review_von_spieler", ""),
            "zu_spieler": seed.get("review_zu_spieler", ""),
        }
        used_new.add(new_index)
        used_seed.add(seed_index)
    return transferred


def main():
    parser = argparse.ArgumentParser(description="Pass-Review-HTML")
    parser.add_argument("video_1080")
    parser.add_argument("passes_csv")
    parser.add_argument("ball_track_csv")
    parser.add_argument("player_tracks_csv")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--frame-offset", type=int, default=10)
    parser.add_argument("--seed-review",
                        help="older reviewed CSV for conservative pre-filling")
    parser.add_argument("--transfer-tolerance", type=int, default=10,
                        help="maximum start/end frame difference for transfer")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    passes_path = Path(args.passes_csv)
    output_path = (Path(args.output) if args.output else
                   passes_path.with_name(passes_path.stem + "_review.html"))
    image_dir = output_path.with_name(output_path.stem + "_images")
    image_dir.mkdir(parents=True, exist_ok=True)

    with open(passes_path, newline="", encoding="utf-8-sig") as f:
        pass_rows = list(csv.DictReader(f))
    initial_state = transfer_reviews(
        pass_rows, args.seed_review, args.transfer_tolerance)

    ball_by_frame = {}
    with open(args.ball_track_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ball_by_frame[int(row["frame"])] = (float(row["x_ref"]),
                                                 float(row["y_ref"]))

    requested_boxes = set()
    requested_frames = set()
    for row in pass_rows:
        start, end = int(row["abgabe_frame"]), int(row["annahme_frame"])
        middle = (start + end) // 2
        requested_frames.update((start, middle, end))
        requested_boxes.add((start, int(row["von_tracker_id"])))
        requested_boxes.add((end, int(row["zu_tracker_id"])))

    boxes = {}
    with open(args.player_tracks_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = (int(row["frame"]), int(row["tracker_id"]))
            if key in requested_boxes:
                boxes[key] = tuple(float(row[name]) for name in ("x1", "y1", "x2", "y2"))

    cap = cv2.VideoCapture(args.video_1080)
    if not cap.isOpened():
        raise SystemExit(f"Video konnte nicht geöffnet werden: {args.video_1080}")
    cache = {}
    for frame in sorted(requested_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame + args.frame_offset)
        ok, image = cap.read()
        if not ok:
            raise SystemExit(f"Frame fehlt: {frame + args.frame_offset}")
        cache[frame] = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
    cap.release()

    cards = []
    for index, row in enumerate(pass_rows):
        start, end = int(row["abgabe_frame"]), int(row["annahme_frame"])
        middle = (start + end) // 2
        panels = []
        for label, frame in (("Abgabe", start), ("Flug", middle), ("Annahme", end)):
            panel = cache[frame].copy()
            ball = ball_by_frame.get(frame)
            if ball:
                # Referenzpixel sind 1280×720; das Panel ist exakt halb so groß.
                center = (int(ball[0] * 0.5), int(ball[1] * 0.5))
                cv2.circle(panel, center, 9, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.circle(panel, center, 2, (0, 0, 255), -1, cv2.LINE_AA)
            tid = (int(row["von_tracker_id"]) if label == "Abgabe" else
                   int(row["zu_tracker_id"]) if label == "Annahme" else None)
            if tid is not None and (frame, tid) in boxes:
                x1, y1, x2, y2 = boxes[(frame, tid)]
                cv2.rectangle(panel, (int(x1 * 0.5), int(y1 * 0.5)),
                              (int(x2 * 0.5), int(y2 * 0.5)), (255, 180, 0), 2)
            cv2.rectangle(panel, (0, 0), (639, 28), (18, 24, 35), -1)
            cv2.putText(panel, f"{label}  {clock(frame, args.fps)}  F{frame}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(panel)
        image_name = f"pass_{index:03d}.jpg"
        cv2.imwrite(str(image_dir / image_name), cv2.hconcat(panels),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        cards.append({
            **row,
            "id": index,
            "image": f"{image_dir.name}/{image_name}",
            "zeit": clock(start, args.fps),
        })

    data = json.dumps(cards, ensure_ascii=False)
    initial = json.dumps(initial_state, ensure_ascii=False)
    storage_key = f"pass-review:{passes_path.name}"
    download_name = passes_path.stem + "_review.csv"
    document = f"""<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pass-Review</title><style>
body{{font:15px system-ui;background:#0f172a;color:#f8fafc;margin:0}}header{{position:sticky;top:0;z-index:2;background:#111827;padding:14px 18px;border-bottom:1px solid #334155}}h1{{margin:0 0 5px}}header p{{margin:0;color:#cbd5e1}}button{{border:0;border-radius:6px;padding:8px 11px;cursor:pointer;font-weight:650}}#export{{background:#38bdf8;margin-left:8px}}main{{padding:14px;display:grid;grid-template-columns:repeat(auto-fill,minmax(570px,1fr));gap:12px}}article{{background:#1e293b;border:2px solid #334155;border-radius:10px;overflow:hidden}}article.pass{{border-color:#22c55e}}article.no{{border-color:#ef4444}}article.unclear{{border-color:#f59e0b}}article.shot{{border-color:#a78bfa}}img{{display:block;width:100%}}.meta{{padding:10px 12px}}h2{{font-size:17px;margin:0 0 5px}}.small{{color:#cbd5e1}}.actions{{display:flex;gap:7px;margin-top:9px;flex-wrap:wrap}}.yes{{background:#22c55e}}.nope{{background:#ef4444;color:white}}.maybe{{background:#f59e0b}}.shotbtn{{background:#a78bfa}}.corrections{{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:10px}}label{{color:#cbd5e1;font-size:13px}}select,input{{display:block;width:100%;box-sizing:border-box;margin-top:3px;padding:6px;border:1px solid #475569;border-radius:5px;background:#0f172a;color:#f8fafc}}#count{{font-weight:700}}</style></head><body><header><h1>Passkandidaten prüfen</h1><p>Rot = Ballspur, blau = vermuteter Abgeber/Empfänger. Schüsse separat markieren; falsche Teams/Namen unten korrigieren. <span id="count"></span><button id="export">CSV exportieren</button></p></header><main id="root"></main><script>
const DATA={data}, INITIAL={initial}, KEY={json.dumps(storage_key)};let state=JSON.parse(localStorage.getItem(KEY)||'null')||INITIAL,onlyOpen=true;const root=document.querySelector('#root');const filterOpen=document.createElement('button');filterOpen.textContent='alle anzeigen';document.querySelector('header p').appendChild(filterOpen);filterOpen.onclick=()=>{{onlyOpen=!onlyOpen;applyFilter();}};
function label(x,side){{const name=x[side+'_spieler'];return name||('Track #'+x[side+'_tracker_id']);}}
function entry(x){{let e=state[x.id];if(typeof e==='string')e={{status:e}};if(!e)e={{}};e.status=e.status||'';e.von_team=e.von_team??String(x.von_team??'');e.zu_team=e.zu_team??String(x.zu_team??'');e.von_spieler=e.von_spieler??String(x.von_spieler??'');e.zu_spieler=e.zu_spieler??String(x.zu_spieler??'');state[x.id]=e;return e;}}
function applyFilter(){{for(const x of DATA){{const card=document.getElementById('p'+x.id);if(card)card.hidden=onlyOpen&&Boolean(entry(x).status);}}filterOpen.textContent=onlyOpen?'alle anzeigen':'nur offene';}}
function save(){{localStorage.setItem(KEY,JSON.stringify(state));renderCount();applyFilter();}}
function renderCount(){{const v=Object.values(state).map(x=>typeof x==='string'?{{status:x}}:x);document.querySelector('#count').textContent=`Geprüft ${{v.filter(x=>x.status).length}}/${{DATA.length}} · Pass ${{v.filter(x=>x.status==='pass').length}} · Schuss ${{v.filter(x=>x.status==='shot').length}} · kein Pass ${{v.filter(x=>x.status==='no').length}} · unklar ${{v.filter(x=>x.status==='unclear').length}}`;}}
for(const x of DATA){{const a=document.createElement('article');a.id='p'+x.id;const e=entry(x);if(e.status)a.classList.add(e.status);a.innerHTML=`<img src="${{x.image}}"><div class="meta"><h2>#${{x.id+1}} · ${{x.zeit}} · Auto: Team ${{x.von_team}} → Team ${{x.zu_team}} · ${{x.ergebnis}}</h2><div>${{label(x,'von')}} → ${{label(x,'zu')}}</div><div class="small">${{Number(x.distanz_m).toFixed(1)}} m · ${{Number(x.dauer_s).toFixed(2)}} s · Frames ${{x.abgabe_frame}}–${{x.annahme_frame}}</div><div class="actions"><button class="yes" data-s="pass">Pass</button><button class="shotbtn" data-s="shot">Schuss</button><button class="nope" data-s="no">kein Pass</button><button class="maybe" data-s="unclear">unklar</button></div><div class="corrections"><label>Abgeber-Team<select data-f="von_team"><option value="1">Blau</option><option value="0">Grün</option><option value="">unklar</option></select></label><label>Empfänger-Team<select data-f="zu_team"><option value="1">Blau</option><option value="0">Grün</option><option value="">unklar</option></select></label><label>Abgeber (optional)<input data-f="von_spieler" placeholder="Name oder leer"></label><label>Empfänger (optional)<input data-f="zu_spieler" placeholder="Name oder leer"></label></div></div>`;for(const b of a.querySelectorAll('[data-s]'))b.onclick=()=>{{e.status=b.dataset.s;a.className=b.dataset.s;save();}};for(const el of a.querySelectorAll('[data-f]')){{el.value=e[el.dataset.f]??'';el.onchange=()=>{{e[el.dataset.f]=el.value;save();}};}}root.appendChild(a);}}
document.querySelector('#export').onclick=()=>{{const extras=['review_status','review_von_team','review_zu_team','review_von_spieler','review_zu_spieler'];const cols=Object.keys(DATA[0]).filter(k=>!['id','image','zeit','review_status'].includes(k)).concat(extras);const esc=v=>'"'+String(v??'').replaceAll('"','""')+'"';const lines=[cols.map(esc).join(',')];for(const x of DATA){{const e=entry(x),values={{review_status:e.status||'ungeprüft',review_von_team:e.von_team,review_zu_team:e.zu_team,review_von_spieler:e.von_spieler,review_zu_spieler:e.zu_spieler}};lines.push(cols.map(k=>esc(k in values?values[k]:x[k])).join(','));}}const blob=new Blob(['\\ufeff'+lines.join('\\n')],{{type:'text/csv'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download={json.dumps(download_name)};a.click();URL.revokeObjectURL(a.href);}};save();
</script></body></html>"""
    output_path.write_text(document, encoding="utf-8")
    print(f"Review: {output_path}")
    print(f"Bilder: {image_dir} ({len(cards)} Kandidaten)")
    if args.seed_review:
        print(f"Übertragene Bewertungen: {len(initial_state)}/{len(cards)}")


if __name__ == "__main__":
    main()
