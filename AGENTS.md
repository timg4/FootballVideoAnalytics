# FootballAnalytics — Projektwissen

Automatische Fußball-Statistiken (Laufdaten, Heatmaps, Teams, später Pässe/Schüsse)
aus Veo-Kameraaufnahmen von Tims Hobby-Spielen. Tim ist kein CV/ML-Experte —
Konzepte anschaulich erklären, Deutsch, lockerer Ton.

## Umgebung

- Windows 11, Python 3.13 in `.venv` (`.\.venv\Scripts\python.exe`)
- Abhängigkeiten: `requirements.txt` (ultralytics, supervision **<0.30** — ByteTrack
  wird in 0.30 entfernt, Migration aufs `trackers`-Paket steht aus)
- **Keine NVIDIA-GPU** (i7-1270P, Intel iGPU): kurze Clips lokal testen,
  ganze Spiele später auf Google Colab (noch nicht eingerichtet)
- Faustregel CPU: yolo11s @ imgsz 1280 ≈ 40× Echtzeit (8s Clip ≈ 5 min)

## Pipeline (Reihenfolge & Datenfluss)

```
Video ──1──> Tracking-CSV + annotiertes Video
       ──2──> Homographien-NPZ (Kameraschwenk pro Frame)
  1+3 ──────> Teams (Farb-Clustering)
1+2+Kalibrierung ──4──> Meter-Positionen, Platz-Filter, Heatmap, Laufdistanzen
```

1. **`src/detect_track.py`** — YOLO-Personenerkennung + ByteTrack.
   `python src\detect_track.py VIDEO --start 30 --seconds 15 --model yolo11s.pt --imgsz 1280`
   → `data/output/<name>_tracked.mp4` + `.csv` (frame, tracker_id, x1..y2, conf).
   **Wichtig:** `--imgsz 1280` + yolo11s ist Pflicht, sonst fehlen ferne Spieler.
   **Achtung:** Bei `--start` sind die Frame-Indizes in der CSV relativ zum
   Ausschnitt (0-basiert), nicht zum Video!
2. **`src/register_frames.py`** — Kameraschwenk-Kompensation: ORB-Features am
   statischen Hintergrund, Homographie pro Frame zum Referenzframe.
   `python src\register_frames.py VIDEO --ref 120 --end 239 --check`
   → `data/output/<name>_homographies.npz`. `--check` erzeugt Warp-Kontrollbilder.
   **Achtung:** lädt alle Frames in den RAM — nur für Clips bis ~2-3 Minuten!
   Streaming-Umbau + `--start`-Option stehen aus. Der `--ref`-Frame muss dem
   Kalibrier-Referenzframe entsprechen.
3. **`src/team_assign.py`** — Teams über Trikotfarben + Platz-Filter in einem:
   `python src\team_assign.py VIDEO TRACKS_CSV [--debug]`
   Helligkeits-normalisierte Farbmerkmale (Rot-/Grünanteil + Sättigung, Median,
   nur Boxen ≥45px), K-Means k=5, alle "großen" Farbgruppen als Team-Kandidaten,
   farblich ähnliche verschmelzen bis 2 Teams (Teams zerfallen im Dämmerlicht in
   hell/abgeschattet!), dann Größen-Plausibilität (< 0.62×Team-Median → raus).
   ~90% korrekt. `--debug` speichert Trikot-Kachelübersichten pro Farbgruppe.
4. **`src/pitch_map.py`** — Bild → Meter:
   `python src\pitch_map.py TRACKS_CSV HOMOGRAPHIEN_NPZ KALIBRIERUNG_JSON`
   → Positions-CSV (x_m, y_m, auf_platz), Heatmap-PNG, Laufdistanzen-Report.
   Filtert alles außerhalb der Platzmaße (= Nachbarspiele, ~46% der Detektionen).

Hilfsmodule: **`src/pitch_model.py`** (parametrisiertes Spielfeld, Zeichenfunktionen),
**`src/calibrate_pitch.py`** (interaktives Klick-Kalibrier-Tool für Tim, matplotlib).

## Der Spielort (wichtig!)

- Großes Kunstrasenfeld, darauf **3 Querfelder** (gespielt über die Breite des
  Normalfelds): je **~68 m lang × ~34 m breit**, ~1 m Abstand dazwischen.
  Jedes Querfeld hat eine eigene Veo-Kamera.
- Die Kamera steht erhöht hinter der Torauslinie des Großfelds und blickt über
  die 3 Streifen: **Tims Feld = der vorderste Streifen**; die Spiele dahinter
  müssen raus (macht der Platz-Filter in pitch_map geometrisch).
- **Mehrere Markierungs-Sets überlagern sich** auf dem Rasen: die Querfeld-Linien
  UND die Längsfeld-Linien des Großfelds (dessen 40m-Strafraum liegt mitten im
  Bild und gehört NICHT zu Tims Feld). Anstoßpunkt-Dot vorhanden; vom
  Mittelkreis ist nur der obere Bogen sichtbar (unterer abgenutzt).
- Die Veo-Follow-Cam ist ein digitaler Schwenk aus einem Panorama → Frames sind
  durch reine Homographien verbunden (deshalb funktioniert Schritt 2 so gut).
- **Kalibrierung gilt pro Kamera-Standort.** `data/calibration/platz_vorlaeufig.json`
  = Kamera der Highlight-Clips (31.05.). Von Codex per Median-Panorama abgelesen
  (Skripte im Session-Scratchpad; Vorgehen: Frames per Homographie auf eine
  Leinwand warpen, Median über die Zeit lässt Spieler verschwinden, Linien bleiben).
  Strafraummaße/Torbreite/Kreisradius des Querfelds sind noch geschätzt.

## Datenlage

- `data/videos/highlights/` — 66 Veo-Highlight-Clips (62 Schüsse, 4 Tore) vom
  Spiel 31.05.2026, je ~15-20s, 1080p. Veo-Labels stecken in den Dateinamen
  (Zeitstempel + Schuss/Tor) → geschenkte Labels für spätere Schuss-Statistiken!
- `data/videos/Video Project.mp4` — **14 min** am Stück (anderes Spiel, gleiche
  Anlage, ANDERER Kamera-Standort → braucht eigene Kalibrierung!), 720p/30fps.
  Das ist das nächste Analyse-Ziel.
- `data/output/` — alle Ergebnisse zum Tor-Clip 03 (Referenz-Testclip):
  Tracking-CSV/-Video, Homographien, Teams-Video, Positionen, Heatmap.
- Videos/Modelle/Outputs sind gitignored.

## Stand & Roadmap

| Phase | Status |
|---|---|
| 1 Erkennung+Tracking | ✅ (Feintuning: imgsz 1280, yolo11s) |
| 2 Kalibrierung/Meter | ✅ End-to-End am Tor-Clip validiert (Laufdaten plausibel: Spitze 35m/8s ≈ 16 km/h) |
| 3 Teams/Heatmaps/Laufdaten | 🚧 Teams ~90%, Heatmap + Distanzen für 1 Clip |
| 4 Ballbesitz/Pässe/Schüsse | offen (Schuss-Labels aus Clip-Namen nutzen!) |
| 5 Spieler-Identität (Re-ID) | offen (härtestes Problem, ggf. manuelle Zuordnung) |

**Nächste Schritte:**
1. `Video Project.mp4` analysieren: Tracking läuft; Registrierung braucht vorher
   den Streaming-Umbau + `--start` (RAM!); dann neue Kalibrierung für diese Kamera
   (Median-Panorama-Trick wiederholen), dann Heatmaps/Laufdaten über 14 min.
2. Google Colab-Notebook für die Batch-Verarbeitung (66 Clips ≈ 10h CPU lokal).
3. Team-Zuordnung: Restfehler über Platz-Filter eliminieren (Filter VOR Clustering).
4. Schusskarte aus den 66 gelabelten Highlight-Clips.

## Bekannte Fallstricke

- Detektions-CSV-Frameindizes sind slice-relativ (siehe oben).
- Panorama-Ränder (extreme Schwenks) haben Registrierungs-Drift — Meter-Werte
  an den Platz-Enden sind etwas ungenauer (Veo-Dewarp ist keine perfekte
  Lochkamera; eine globale Homographie stimmt nur näherungsweise über 68 m).
- Dämmerlicht macht Trikotfarben unzuverlässig; uneindeutige Kits (Streifen,
  einzelne Ausreißer-Shirts) bleiben Team-Wackelkandidaten.
- Veo-Full-Game-Download braucht Coach/Admin-Rolle + Abo; Tim hat aktuell nur
  Highlight-Downloads. 14-min-Export kam über einen manuellen Schnitt.
