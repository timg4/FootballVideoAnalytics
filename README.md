# FootballAnalytics ⚽

Automatische Statistiken aus Veo-Aufnahmen unserer Hobby-Fußballspiele:
Spieler-Tracking, Laufdaten, Heatmaps — und später Pässe, Schüsse und
Einzelspieler-Profile.

## Roadmap

| Phase | Ziel | Status |
|-------|------|--------|
| 1 | Spieler-Erkennung + Tracking auf einem Clip, annotiertes Video als Ergebnis | 🚧 in Arbeit |
| 2 | Spielfeld-Kalibrierung (Homographie): Pixel → Meter, 2D-Spielfeldkarte | geplant |
| 3 | Laufdaten & Heatmaps pro Track, Team-Zuordnung über Trikotfarben | geplant |
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

Das annotierte Ergebnis landet in `data/output/`.

## Veo-Videos herunterladen

1. Im [Veo Clubhouse](https://app.veo.co) einloggen
2. Spiel öffnen → **Download** → **Download full game** (MP4, Follow-Cam-Ansicht)
3. Datei nach `data/videos/` verschieben

## Hardware-Hinweis

Dieser Rechner hat keine NVIDIA-GPU — kurze Clips zum Entwickeln laufen lokal
auf der CPU, ganze Spiele verarbeiten wir später auf Google Colab (kostenlose GPU).
