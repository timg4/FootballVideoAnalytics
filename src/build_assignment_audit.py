"""Kompakte visuelle QA-Seite für automatische Spielerzuordnungen."""

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Audit-Seite für Spieler-ReID")
    parser.add_argument("mapping_csv")
    parser.add_argument("review_images_dir")
    parser.add_argument("--output", default=None)
    parser.add_argument("--risky", type=int, default=5,
                        help="Niedrigste Re-ID-Sicherheiten pro Spieler")
    parser.add_argument("--largest", type=int, default=3,
                        help="Größte Auto-Distanzbeiträge pro Spieler")
    args = parser.parse_args()

    mapping_path = Path(args.mapping_csv)
    image_dir = Path(args.review_images_dir)
    output = (Path(args.output) if args.output else
              mapping_path.with_name("video_project_spieler_audit.html"))

    groups = defaultdict(list)
    with mapping_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row["tracker_id"] = int(row["tracker_id"])
            row["distanz_m"] = float(row["distanz_m"])
            row["reid_similarity"] = float(row.get("reid_similarity") or 1.0)
            row["eindeutigkeits_margin"] = float(
                row.get("eindeutigkeits_margin") or 1.0)
            groups[row["spieler_name_oder_nummer"]].append(row)

    sections = []
    for player, rows in groups.items():
        manual = sorted((row for row in rows if row.get("quelle") == "manuell"),
                        key=lambda row: -row["distanz_m"])
        automatic = [row for row in rows if row.get("quelle") == "automatisch"]
        risky = sorted(automatic,
                       key=lambda row: (row["reid_similarity"],
                                        row["eindeutigkeits_margin"]))[:args.risky]
        largest = sorted(automatic, key=lambda row: -row["distanz_m"])[:args.largest]
        selected = []
        seen = set()
        for row in risky + largest:
            if row["tracker_id"] not in seen:
                selected.append(row)
                seen.add(row["tracker_id"])

        def add_images(row):
            paths = sorted(image_dir.glob(f"track_{row['tracker_id']}_frame_*.jpg"))
            copy = dict(row)
            copy["images"] = [f"{image_dir.name}/{path.name}" for path in paths[:3]]
            return copy

        sections.append({
            "player": player,
            "distance": round(sum(row["distanz_m"] for row in rows), 2),
            "manual_count": len(manual),
            "auto_count": len(automatic),
            "manual": [add_images(row) for row in manual],
            "audit": [add_images(row) for row in selected],
        })
    sections.sort(key=lambda section: -section["distance"])

    data = json.dumps(sections, ensure_ascii=False)
    title = "Spieler-ReID: kompakte Qualitätsprüfung"
    page = f"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>
body{{font:15px system-ui;background:#0f172a;color:#f8fafc;margin:0}} header{{position:sticky;top:0;z-index:3;background:#0f172aee;padding:14px 20px;border-bottom:1px solid #334155}}
h1{{margin:0 0 6px;font-size:21px}} header p{{margin:0;color:#cbd5e1}} main{{padding:16px;max-width:1500px;margin:auto}}
section{{margin:0 0 24px;background:#172033;border:1px solid #334155;border-radius:10px;padding:14px}} h2{{margin:0 0 4px}}
.summary{{color:#93c5fd;margin-bottom:12px}} h3{{font-size:15px;color:#fde68a;margin:13px 0 7px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(245px,1fr));gap:9px}} .track{{background:#1e293b;border:2px solid transparent;border-radius:8px;padding:8px;cursor:pointer}}
.track.flagged{{border-color:#ef4444;background:#3b1c24}} .meta{{display:flex;justify-content:space-between;gap:6px;margin-bottom:6px}}
.imgs{{display:flex;height:150px;gap:3px;justify-content:center;background:#090f1d;border-radius:5px;overflow:hidden}} .imgs img{{height:150px;max-width:33%;object-fit:contain}}
.small{{font-size:12px;color:#94a3b8;margin-top:5px}} .manual{{border-color:#166534}} button{{font:inherit;padding:7px 10px;border:0;border-radius:6px;background:#2563eb;color:#fff;cursor:pointer;margin-top:9px}}
</style></head><body><header><h1>{html.escape(title)}</h1><p>Grüne Karten = manuelle Referenzen. Bei den Auto-Karten werden absichtlich die unsichersten und distanzstärksten Fälle gezeigt. Falsche Karte anklicken und am Ende exportieren.</p><button id="export">Markierte Fehler als CSV exportieren</button></header><main id="root"></main>
<script>const DATA={data}; const KEY='footballanalytics:player-audit:flags'; const flags=JSON.parse(localStorage.getItem(KEY)||'{{}}');
function card(row,manual){{const el=document.createElement('article');el.className='track '+(manual?'manual ':'')+(flags[row.tracker_id]?'flagged':'');el.dataset.id=row.tracker_id;
 el.innerHTML=`<div class="meta"><b>#${{row.tracker_id}}</b><b>${{row.distanz_m.toFixed(1)}} m</b></div><div class="imgs">${{row.images.map(src=>`<img loading="lazy" src="${{src}}">`).join('')}}</div><div class="small">${{manual?'MANUELL':`AUTO · Similarity ${{row.reid_similarity.toFixed(3)}} · Margin ${{row.eindeutigkeits_margin.toFixed(3)}}`}}</div>`;
 if(!manual)el.onclick=()=>{{flags[row.tracker_id]=!flags[row.tracker_id];localStorage.setItem(KEY,JSON.stringify(flags));el.classList.toggle('flagged');}};return el;}}
const root=document.querySelector('#root');for(const s of DATA){{const sec=document.createElement('section');sec.innerHTML=`<h2>${{s.player}}</h2><div class="summary">${{(s.distance/1000).toFixed(3)}} km · ${{s.manual_count}} manuell · ${{s.auto_count}} automatisch</div><h3>Manuelle Referenzen</h3><div class="grid refs"></div><h3>Kritische automatische Stichprobe</h3><div class="grid audit"></div>`;for(const r of s.manual)sec.querySelector('.refs').appendChild(card(r,true));for(const r of s.audit)sec.querySelector('.audit').appendChild(card(r,false));root.appendChild(sec);}}
document.querySelector('#export').onclick=()=>{{const rows=[['tracker_id','status']];for(const [id,value] of Object.entries(flags))if(value)rows.push([id,'falsch']);const csv=rows.map(r=>r.join(',')).join('\\r\\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv'}}));a.download='video_project_reid_fehler.csv';a.click();}};
</script></body></html>"""
    output.write_text(page, encoding="utf-8")
    print(f"Audit-Seite: {output}")
    print(f"{len(sections)} Spieler/Gruppen, "
          f"{sum(len(section['audit']) for section in sections)} kritische Auto-Stichproben")


if __name__ == "__main__":
    main()
