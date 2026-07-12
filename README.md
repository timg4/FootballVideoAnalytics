# FootballAnalytics ⚽

Automatische Statistiken aus Veo-Aufnahmen unserer Hobby-Fußballspiele:
Spieler-Tracking, Laufdaten, Heatmaps — und später Pässe, Schüsse und
Einzelspieler-Profile.

## Roadmap

| Phase | Ziel | Status |
|-------|------|--------|
| 1 | Spieler-Erkennung + Tracking auf einem Clip, annotiertes Video als Ergebnis | ✅ (`--imgsz 1280` nötig für ferne Spieler) |
| 2 | Spielfeld-Kalibrierung (Homographie): Pixel → Meter, 2D-Spielfeldkarte | ✅ End-to-End (`register_frames.py` → `calibrate_pitch.py`/vorläufige Kalibrierung → `pitch_map.py`) |
| 3 | Laufdaten & Heatmaps pro Track, Team-Zuordnung über Trikotfarben | 🚧 Team-Clustering v2 in `team_assign.py`, ~90% korrekt |
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
# Team-Zuordnung + Platz-Filter (liest Original-Video + Tracking-CSV)
.\.venv\Scripts\python.exe src\team_assign.py data\videos\spiel.mp4 data\output\spiel_tracked.csv
```

**Besonderheit unseres Spielorts:** Drei Spielfelder liegen hintereinander vor
der Kamera, alle drei Spiele landen in den Detektionen. `team_assign.py` löst
Platz-Filter und Team-Zuordnung deshalb gemeinsam:

1. Farbmerkmale pro Track: helligkeits-normalisiert (Rot-/Grün-Anteil +
   Sättigung, Median statt Mittelwert), gesammelt nur von Boxen ≥ 45px
2. K-Means in 5 Farbgruppen, mehrere Starts
3. Alle Gruppen nahe der maximalen Spielergröße sind Team-Kandidaten;
   farblich ähnliche Kandidaten werden verschmolzen (ein Team zerfällt im
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

Dieser Rechner hat keine NVIDIA-GPU — kurze Clips zum Entwickeln laufen lokal
auf der CPU, ganze Spiele verarbeiten wir später auf Google Colab (kostenlose GPU).
