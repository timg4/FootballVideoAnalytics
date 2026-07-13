# FootballAnalytics ⚽

Automatische Statistiken aus Veo-Aufnahmen unserer Hobby-Fußballspiele:
Spieler-Tracking, Laufdaten, Heatmaps — und später Pässe, Schüsse und
Einzelspieler-Profile.

## Roadmap

| Phase | Ziel | Status |
|-------|------|--------|
| 1 | Spieler-Erkennung + Tracking auf einem Clip, annotiertes Video als Ergebnis | ✅ (`--imgsz 1280` nötig für ferne Spieler) |
| 2 | Spielfeld-Kalibrierung (Homographie): Pixel → Meter, 2D-Spielfeldkarte | ✅ Direkte, driftfreie Anker-Lokalisierung für Vollvideo validiert |
| 3 | Laufdaten & Heatmaps pro Track, Team-Zuordnung über Trikotfarben | 🚧 Vollvideo ausgewertet; sichtbare Tracklet-Distanzen + gefilterte Teams |
| 4 | Ballbesitz, Passnetzwerke, Schuss-Erkennung | geplant |
| 5 | Spieler-Identifikation (Re-ID) für saubere Einzelspieler-Profile über ganze Spiele | geplant |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Verwendung

Video (z.B. Veo-Export "Download full game" aus app.veo.co) nach `data/videos/` legen, dann:

```powershell
.\.venv\Scripts\python.exe src\detect_track.py data\videos\spiel.mp4 --seconds 10
```

Das annotierte Ergebnis landet in `data/output/`, zusammen mit einer CSV der
Tracking-Rohdaten (Spielerpositionen pro Frame). Alle weiteren Schritte lesen
die CSV und brauchen die teure Erkennung nicht zu wiederholen:

```powershell
# Team-Zuordnung nach dem geometrischen Platzfilter
.\.venv\Scripts\python.exe src\team_assign.py data\videos\spiel.mp4 `
    data\output\spiel_tracked.csv `
    --positions-csv data\output\spiel_positionen.csv --no-video
```

**Besonderheit unseres Spielorts:** Mehrere Felder und Markierungs-Sets liegen
im Kamerabild. Der geometrische Filter aus `pitch_map.py` entscheidet zuerst,
welche Detektionen auf unserem Feld stehen. `team_assign.py` erhält diese
Positionen mit `--positions-csv` und clustert erst danach die Trikotfarben:

1. Farbmerkmale pro Track: helligkeits-normalisiert (Rot-/Grün-Anteil +
   Sättigung, Median statt Mittelwert), gesammelt nur von Boxen ≥ 45px
2. K-Means in 5 Farbgruppen, mehrere Starts
3. Alle Gruppen nahe der maximalen Spielergröße sind Team-Kandidaten;
   anhand normalisierter BGR-Farbanteile ähnliche Kandidaten werden verschmolzen (ein Team zerfällt im
   Dämmerlicht gern in helle + abgeschattete Trikots), bis 2 Teams übrig sind
4. Größen-Plausibilität: Tracks deutlich kleiner als ihr Team-Median fliegen
   raus (ferne Spieler entsättigen zu Grau und ähneln sonst dem weißen Team)

`--debug` speichert pro Farbgruppe eine Trikot-Kachelübersicht nach
`data/output/` zur Kontrolle. Stand: ~90% korrekt; Restfehler sind einzelne
entfernte Nachbarplatz-Spieler im Weiß-Team und Spieler mit uneindeutigen
Kits. Der saubere Fix ist die Spielfeld-Kalibrierung (Phase 2).

## Phase-2-Pipeline (Bild → Meter)

```powershell
# 1. Kameraschwenks kompensieren (einmal pro Clip)
.\.venv\Scripts\python.exe src\register_frames.py data\videos\clip.mp4 --ref 120 --check

# 2. Kalibrierung: einmal pro Platz (liegt in data\calibration\platz_vorlaeufig.json;
#    Feinschliff per Klick-Tool: src\calibrate_pitch.py)

# 3. Positionen in Meter + Heatmap + Laufdistanzen + Platz-Filter
.\.venv\Scripts\python.exe src\pitch_map.py data\output\clip_tracked.csv `
    data\output\clip_homographies.npz data\calibration\platz_vorlaeufig.json
```

**Erkenntnisse zum Platz:** Auf dem Kunstrasen liegen mehrere Markierungs-Sets
übereinander (unser Längsfeld + Quermarkierungen mit eigenem Strafraum). Unsere
Merkpunkte: Anstoßpunkt + oberer Mittelkreisbogen (der untere ist abgenutzt),
ferne Seitenlinie = Grenze zu den Spielen dahinter, nahe Seitenlinie = die
unterste durchgehende Linie. Die Quer-Strafraumlinien gehören NICHT zu unserem
Feld. Platzmaße aktuell geschätzt (60×40 m) — echte Maße noch nachtragen.

## Veo-Videos herunterladen

1. Im [Veo Clubhouse](https://app.veo.co) einloggen
2. Spiel öffnen → **Download** → **Download full game** (MP4, Follow-Cam-Ansicht)
3. Datei nach `data/videos/` verschieben

## Hardware-Hinweis

Dieser Rechner hat keine NVIDIA-GPU. YOLO-Erkennung und Tracking für ganze
Spiele laufen deshalb auf Google Colab/T4. Die direkte Platzlokalisierung mit
OpenCV ORB/BFMatcher ist dagegen CPU-lastig und lief für 25.150 Frames lokal
in rund 52 Minuten; eine T4 bringt diesem Schritt kaum etwas.

## Volles Video auf Google Colab

[![In Colab öffnen](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timg4/FootballAnalytics/blob/main/notebooks/full_video_colab.ipynb)

Das Notebook [`notebooks/full_video_colab.ipynb`](notebooks/full_video_colab.ipynb)
führt das teure YOLO-Tracking auf einer Colab-GPU aus. Für bessere, länger
stabile IDs nutzt der aktuelle Modus BoT-SORT mit Kamerabewegungskompensation,
fünf Sekunden Track-Buffer und Appearance-ReID. Das Video wird aus Google Drive
auf die schnelle Laufzeit-SSD kopiert; Tracking-Video und CSV werden danach
wieder in Drive gesichert. Die alte globale Registrierung ist nur noch ein
deaktivierter Diagnoseschritt.

1. `Video Project.mp4` in Google Drive nach
   `Meine Ablage/FootballAnalytics/input/` hochladen.
2. Notebook in Colab öffnen und als Laufzeit eine GPU auswählen.
3. Zellen von oben nach unten ausführen. `STRIDE = 1` ist die Qualitätsvariante;
   `STRIDE = 2` spart Zeit, kann bei ByteTrack aber zusätzliche ID-Wechsel erzeugen.

Für `Video Project.mp4` sind Tracking und die anschließende Meter-Auswertung
inzwischen abgeschlossen. Die lange globale Homographiekette und ein einzelnes
Panorama sind für die Meter-Auswertung verworfen, weil sie über 14 Minuten
driften. Verbindlich ist die direkte Lokalisierung jedes Frames gegen drei
kalibrierte Ankeransichten. Die Feldmaße 55,75×27,25 m stammen aus dem
Orthofoto, die Kalibrierung liegt in `video_project_ortho.json`.

```powershell
# Direkte Vollvideo-Lokalisierung (lokal etwa 52 Minuten)
.\.venv\Scripts\python.exe src\localize_pitch.py `
    "data\videos\Video Project.mp4" `
    data\calibration\video_project_ortho.json `
    --output data\output\video_project_pitch_localization.npz

# Positionen, Platzfilter, Heatmap und sichtbare Tracklet-Distanzen
.\.venv\Scripts\python.exe src\pitch_map.py `
    data\output\video_project_tracked.csv `
    data\output\video_project_pitch_localization.npz `
    data\calibration\video_project_ortho.json --fps 30

# Teams nur aus On-Pitch-Detektionen bestimmen
.\.venv\Scripts\python.exe src\team_assign.py `
    "data\videos\Video Project.mp4" `
    data\output\video_project_tracked.csv `
    --positions-csv data\output\video_project_positionen.csv `
    --no-video --debug
```
