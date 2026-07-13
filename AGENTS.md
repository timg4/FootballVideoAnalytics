# FootballAnalytics — Projektwissen

## Übergabestand 13.07.2026 — VOLLVIDEO AUSGEWERTET ✅ (NEUESTE SEKTION, HAT VORRANG)

### Verbindlicher Ergebnisstand für `Video Project.mp4`

1. **Vollvideo-Lokalisierung lokal abgeschlossen.** Auf diesem Rechner dauerte
   der direkte ORB-Ankerlauf **52 Minuten**. Ergebnis:
   `data/output/video_project_pitch_localization.npz` mit allen Frames
   0–25.149, **25.150/25.150 lokalisiert, 0 Aussetzer**. Alle Matrizen sind
   endlich und nicht degeneriert. Inlier: Minimum 78, Median 625, Mittel 670,
   95. Perzentil 1.323. Ankernutzung: Frame 23.150 = 11.227 Frames,
   23.320 = 5.825, 23.620 = 8.098.
2. **Lokal statt Colab ist für diesen Schritt sinnvoll.** `localize_pitch.py`
   arbeitet mit OpenCV ORB/BFMatcher auf der CPU; eine Colab-T4 beschleunigt
   diesen Teil nicht wesentlich. Colab/GPU bleibt für YOLO-Detektion und
   Tracking sinnvoll. Lokaler 300-Frame-Test: 21,4 s; kompletter Lauf: 52 min.
3. **`pitch_map.py` erfolgreich auf der finalen Ortho-Kalibrierung gelaufen:**
   223.026/469.665 Detektionen auf Tims 55,75×27,25-m-Feld (47,5 %, im
   Mittel 8,9 sichtbare Personen/Frame; nur 5 Frames ohne On-Pitch-Detektion).
   Verbindliche Ausgaben (gitignored):
   - `data/output/video_project_positionen.csv`
   - `data/output/video_project_heatmap.png`
   - `data/output/video_project_distanzen.csv`
   - `data/output/video_project_team_statistiken.csv`
   - `data/output/video_project_team_0_heatmap.png` (Grün)
   - `data/output/video_project_team_1_heatmap.png` (Blau)
   Sichtbare Team-Mindestdistanzen: **Grün 7,643 km**, **Blau 8,179 km**.
4. **Distanzreport korrigiert:** Früher wurde jedem Track fälschlich die
   komplette Videodauer 838,3 s zugeordnet. Jetzt enthält der Report pro
   Tracklet nur seine beobachteten Abschnitte (Lücken bis 1 s), verwirft
   Sprünge >12 m/s und speichert Distanz, sichtbare Dauer, Durchschnitt und
   Punktzahl als CSV. Beispiel: Tracklet #3536 = 249,1 m in 93,3 s sichtbar.
   Das sind weiterhin **sichtbare Tracklet-Mindestdistanzen**, keine echten
   Gesamtkilometer eines identifizierten Spielers.
5. **Team-Zuordnung nach dem geometrischen Platzfilter abgeschlossen.**
   `team_assign.py` akzeptiert nun `--positions-csv`, verwendet schon bei der
   Farbsammlung ausschließlich Zeilen mit `auf_platz=1`, schreibt eine
   Zuordnungs-CSV und kann mit `--no-video` das teure Ausgabevideo auslassen.
   Ergebnis: 329 grüne und 319 blaue Tracklets; 112.616 bzw. 107.803
   On-Pitch-Detektionen. 372 kurze/kleine/Sonderfarben-Tracklets bzw. nur
   2.607 On-Pitch-Detektionen bleiben unzugeordnet. Datei:
   `data/output/video_project_team_assignments.csv`.
6. **Farbcluster visuell validiert und Merge korrigiert:** Gruppe 1 = grün,
   Gruppen 0+2 = hell-/dunkelblau. Die alte Merge-Distanz verband Gruppe 2
   fälschlich mit Grün, weil Sättigung zu stark wirkte. Das Verschmelzen
   nutzt jetzt normalisierte BGR-Farbanteile. Debug-Mosaike:
   `data/output/debug_farbgruppe_0.jpg` bis `_4.jpg`.
7. **Post-hoc Person-ReID getestet, aber bewusst nicht als fertige
   Spielerstatistik verkauft.** `extract_reid.py` extrahiert mit Intels
   `person-reidentification-retail-0288` Appearance-Vektoren: 648/648
   Team-Tracklets, 2.196 Crops, mittlere interne Konsistenz 0,886. Bei
   Similarity >=0,85 liegt die gemessene False-Match-Rate unter gleichzeitig
   sichtbaren Mitspielern bei ca. 0,2 %, aber der Recall gleicher-ID-Crops nur
   bei ca. 11 %. `stitch_tracklets.py` verbindet daher konservativ nur 36
   eindeutige Links; 612 Kandidaten bleiben. Diese sind **keine 612 Spieler**,
   sondern zeigen, dass die ByteTrack-Rohdaten für vollautomatische
   Spielerkilometer zu fragmentiert sind.
8. **Besserer GPU-Neulauf vorbereitet:** `detect_track.py` hat jetzt optional
   `--tracker botsort-reid`; Konfiguration `configs/botsort_reid.yaml` nutzt
   BoT-SORT, sparseOptFlow-Kamerakompensation, Appearance-ReID und 150 Frames
   Track-Buffer. 60-Frame-Smoke-Test erfolgreich. Das Colab-Notebook schreibt
   getrennt `video_project_botsort_reid_tracked.*` und überspringt standardmäßig
   die nicht mehr benötigte globale Legacy-Registrierung.

### Reproduktionsbefehle

```powershell
# 1. Direkte, driftfreie Vollvideo-Lokalisierung (lokal ca. 52 min)
.\.venv\Scripts\python.exe src\localize_pitch.py `
  "data\videos\Video Project.mp4" `
  data\calibration\video_project_ortho.json `
  --output data\output\video_project_pitch_localization.npz

# 2. Meterpositionen, Platzfilter, Heatmap, sichtbare Tracklet-Distanzen
.\.venv\Scripts\python.exe src\pitch_map.py `
  data\output\video_project_tracked.csv `
  data\output\video_project_pitch_localization.npz `
  data\calibration\video_project_ortho.json --fps 30

# 3. Teams nur auf geometrisch gefilterten Tracks; ohne neues Video
.\.venv\Scripts\python.exe src\team_assign.py `
  "data\videos\Video Project.mp4" `
  data\output\video_project_tracked.csv `
  --positions-csv data\output\video_project_positionen.csv `
  --no-video --debug

# 4. Teamwerte + getrennte Heatmaps ergänzen
.\.venv\Scripts\python.exe src\pitch_map.py `
  data\output\video_project_tracked.csv `
  data\output\video_project_pitch_localization.npz `
  data\calibration\video_project_ortho.json --fps 30 `
  --team-assignments data\output\video_project_team_assignments.csv
```

### Nächste sinnvolle Schritte

- Code committen/pushen und im aktualisierten Colab-Notebook den
  BoT-SORT/ReID-Vollvideo-Trackinglauf starten. Danach nur die neue CSV/MP4
  aus Drive holen; Pitch-Lokalisierung muss nicht neu gerechnet werden.
- Neue CSV durch `pitch_map.py` und `team_assign.py` schicken und zuerst die
  Fragmentierung (Anzahl/Medianlänge der Tracklets) gegen ByteTrack vergleichen.
- Erst dann post-hoc Re-ID erneut anwenden. Für echte Namen/Nummern bleibt
  voraussichtlich eine kleine manuelle Mapping-Schicht mit Tims Roster nötig.
- Optional ein gefiltertes Team-Kontrollvideo rendern (denselben Team-Befehl
  ohne `--no-video`; kostet einen zweiten Video-Pass und viel Speicher).

## Übergabestand 12.07.2026 SPÄTABENDS — PLATZERKENNUNG GELÖST ✅ (durch Sektion oben ergänzt)

### Lösung & validierter Stand

1. **Echte Feldmaße aus dem Orthofoto vermessen:** Tims Turnierfeld
   (Video Project.mp4) = **westlicher Streifen des östlichen Kunstrasens,
   55,75 × 27,25 m** (Torlinien bei Ortho-y 49,0/104,75, Seitenlinien
   x 47,25/74,5, Teiler-Spalt 1,5 m; Vermessung: `measure_precise.py` im
   Session-Scratchpad, Feinwinkel 44,51°). Die alte 68×34-Annahme war falsch
   und Hauptursache der Klick-Inkonsistenzen.
2. **Kamera-Standort verstanden:** im Spalt zwischen den beiden Ost-Streifen,
   hinter einem mobilen Tor, Blick nach W/NW (Donauturm). Beide Tore von Tims
   Feld erscheinen deshalb weit links/rechts im Horizontband.
3. **Kalibrier-Degeneration gebrochen:** Tims Klicks pro Ansicht sind
   kollinear (alles Torlinienpunkte). Fix: (a) Mittelfeld-Punkte über die
   LOKALE Registrierungskette in die Anker brücken (nur benachbarte Ansichten —
   die Brücke 23150↔23620 über den vollen Schwenk driftet!), (b) **virtueller
   Nahpunkt** = Schnitt der im Frame abgelesenen nahen Seitenlinie mit der
   Mittellinie → (L/2, B). Danach RANSAC; „Mittelpunkt“-Klick fliegt überall
   raus (war vermutlich ein Punkt eines überlagerten Markierungs-Sets).
4. **Ergebnis:** `data/calibration/video_project_ortho.json` (19 Punkte,
   localize_pitch-kompatibel; Anker-Fits 0,9–2,4 px, det > 0).
   Frame-0-Debug: Feldumriss liegt sauber, 6/29 Detektionen innen =
   exakt die sichtbaren Feldspieler vor Anpfiff. 60-Frame-Smoke-Test:
   309/1494 innen (~5/Frame, plausibel), pitch_map läuft.
5. `pitch_map.py`: `H_pitch_to_px` wird nur noch für Legacy-Kalibrierungen
   gelesen (Fix für direkte Lokalisierungs-NPZ).
6. **Am 13.07. abgeschlossen (siehe neueste Sektion):** Vollvideo-Lokalisierung
   (25.150 Frames) nach
   `data/output/video_project_pitch_localization.npz`. Danach:
   `python src\pitch_map.py data\output\video_project_tracked.csv
   data\output\video_project_pitch_localization.npz
   data\calibration\video_project_ortho.json --fps 30`
   → dann Teams (team_assign auf platzgefilterte Tracks), Heatmaps,
   sichtbare Tracklet-Distanzen.

### Merksätze für künftige Kalibrierungen

- Immer zuerst das Orthofoto (`data/calibration/ortho/`) für Maße + Layout.
- Klicks müssen NAH- und FERN-Punkte enthalten; reine Torlinien-Klicks sind
  kollinear und ergeben gefaltete Homographien (det < 0 / Bowtie-Overlays).
- Punkt-Dots auf dem Rasen nie blind als Anstoßpunkt interpretieren
  (überlagerte Markierungs-Sets!).
- Brücken zwischen Kalibrier-Ansichten nur über kurze Kettenabschnitte.

## Übergabestand 12.07.2026 ABENDS — Platzerkennung diagnostiziert (überholt durch Sektion oben)

### Diagnose: warum der Platzfilter versagte (94/1494 innen)

1. **Anker-Lokalisierung (`localize_pitch.py`) ist die RICHTIGE Architektur** —
   driftfrei, ORB-Matching liefert 100-600 Inlier pro Frame, behalten!
2. **Die Anker-Kalibrierung selbst ist kaputt.** Frame-0-Debugbild
   (Feldumriss + Fußpunkte) zeigt einen zu einem Splitter entarteten Umriss.
   Ursache: Alle 6 Klicks pro Anker liegen in einem nur ~50 px hohen
   Horizontband (die Kamera dieses Platzes hängt sehr niedrig; alle
   klassischen Landmarks — Ecken, Tore, Strafräume — sind 30+ m entfernt).
   Eine Homographie aus so einem Band ist in Tiefenrichtung unbestimmt.
3. **Verketteter Fit über alle 3 Ansichten (ORB) schlug ebenfalls fehl:**
   RANSAC findet keinen Konsens >6/14 Punkte — die Klicks sind in sich
   inkonsistent. Systematischer Grund: die Modellmaße (68×34, Tor 5 m,
   Strafraum 9×24, Kreis 5 m) sind GERATEN; mit falschen Maßen kann kein
   Klickset konsistent sein. Zusätzlich vermutlich einzelne Klicks auf dem
   falschen Linien-Set ("Ecke rechts-nah" überall schlechtester Punkt).

### Neue Grundlage: georeferenziertes Orthofoto (Durchbruch)

- Anlage identifiziert: Sportanlage bei Donauturm/Alte Donau (Wien 22),
  ca. lat 48.2373, lon 16.4185. Beide Kunstrasenplätze, Traglufthalle,
  Laufbahn im Bild.
- `data/calibration/ortho/`: `ortho_platz_komplett.jpg` (z=20,
  **0.0994 m/px**, Kachelursprung siehe `ortho_georef.json`),
  `ortho_rotiert.jpg` (Hauptplatz achsparallel, Metergitter),
  `anlage_uebersicht.jpg` (z=18). Quelle basemap.at (Stadt Wien, 2024).
- Darauf sind BEIDE Markierungs-Sets beider Plätze pixelgenau sichtbar
  (Längsfeld + 3 Querfelder mit Bögen).
- Plätze-Zuordnung (vorläufig, im nächsten Schritt verifizieren):
  Highlights-Video = Hauptplatz (westlich, an der Parkwiese);
  Video Project.mp4 = östlicher Platz (Richtung Laufbahn); Blickrichtung
  Donauturm passt zu Kamera an dessen SO-Seite.

### Empfohlener Weg (ersetzt Klick-Kalibrierung mit geratenen Maßen)

**Ortho-Referenz-Kalibrierung:** Anker-Frames direkt gegen das Orthofoto
kalibrieren. Korrespondenzen = beliebige eindeutige Merkmale (Kreuzungen
beider Linien-Sets, Bögen, Tore, Zaunpfosten-Fußpunkte) — insbesondere auch
KAMERANAHE Punkte (Quermarkierungs-Box unten im Bild), die dem Fit bisher
fehlten. Meter = Ortho-Pixel × 0.0994 in einem gewählten Platz-Koordinatensystem.
Danach bleibt `localize_pitch.py` unverändert (nur die Anker-Homographien
werden gegen Ortho-Meter gefittet), `pitch_map.py` filtert per Polygon des
eigenen Querfelds (aus dem Ortho abgelesen).

Konkrete nächste Schritte:
1. Im Ortho das Querfeld von Tims Spiel (Video Project) identifizieren und
   sein Polygon + Koordinatensystem festlegen (Verifikation: Traglufthalle,
   Tore, Zaunverlauf mit Video-Panorama abgleichen).
2. Pro Anker-Frame 6-10 Korrespondenzen Video↔Ortho sammeln (verteilt über
   NAH und FERN!), Homographie fitten, Umriss-Overlay auf Anker + Frame 0
   prüfen (Debug-Skript: Scratchpad `debug_frame0.py` der Session).
3. `localize_pitch.py`-Anker aus diesen Fits speisen, 60-Frame-Test,
   Platzfilter-Plausibilität (sollte ~5-15 Spieler/Frame innen zeigen).
4. Dann Vollvideo-Lokalisierung + `pitch_map.py`.

Frühere Sektionen unten bleiben als Historie gültig, wo sie nicht
widersprechen.

## Übergabestand 12.07.2026 (ältere Session)

### Was in dieser Session erledigt wurde

1. **Vollvideo auf Google Colab verarbeitet.**
   - Notebook: `notebooks/full_video_colab.ipynb` (von Tim bereits committed/gepusht).
   - Video: `data/videos/Video Project.mp4`, 25.150 Frames, 30 fps, 1280×720,
     knapp 14 Minuten, ca. 1 GB.
   - Tracking lief mit `yolo11s.pt`, `imgsz=1280`, `stride=1`, CUDA/T4.
   - Heruntergeladene Ergebnisse:
     - `data/output/video_project_tracked.csv`: 469.665 Zeilen, Frames 0–25.149,
       4.114 verschiedene **Tracklet-IDs**.
     - `data/output/video_project_homographies.npz`: 25.150 Matrizen,
       absolute `frames` 0–25.149, Referenzframe 0.
   - Das große Tracking-MP4 liegt in Google Drive. Es wurde nicht erneut lokal
     heruntergeladen.

2. **Tracking-IDs richtig eingeordnet.** ByteTrack-IDs sind nur Tracklets, keine
   dauerhaften Spieleridentitäten. Verlässt ein Spieler wegen des Follow-Cam-Schwenks
   das Bild, bekommt er beim Wiedereintritt meist eine neue ID. Daher sind vorerst
   nur sichtbare Mindestdistanzen/Abschnitte, keine vollständigen individuellen
   Kilometer oder Spieler-Heatmaps zulässig. Später: Tracklet-Stitching/Re-ID mit
   Teamfarbe, Meterposition, Bewegungsrichtung und ggf. manueller Bestätigung.

3. **Streaming-Registrierung und Stride-Datenfluss erweitert.**
   - `detect_track.py`: absolute Frameindizes, `--stride`, `--device`.
   - `register_frames.py`: Streaming, `--start`, `--stride`, `--output`, speichert
     `H`, `frames`, `ref`.
   - `pitch_map.py`: kann absolute `frames` und Stride korrekt zuordnen; Zeit- und
     Geschwindigkeitsberechnung berücksichtigt echte Frameabstände.

4. **Wichtige Erkenntnis: eine globale Homographiekette über 14 Minuten ist NICHT
   meterstabil.** Einzelne Kontrollbilder sehen im überlappenden Rasenbereich gut
   aus, aber die Kette driftet/entartet über lange Zeit. Rechte Bildbereiche werden
   in Referenzframe 0 teilweise auf extrem große Koordinaten abgebildet. Die Datei
   `video_project_homographies.npz` ist deshalb für Diagnose/relative kurze Abschnitte
   nützlich, aber **nicht direkt für die Meter-Auswertung des Vollvideos verwenden**.

5. **Panorama-Versuche dokumentiert und verworfen.**
   - Neues Skript: `src/build_panorama.py` (Median, schwenkbasierte Samples,
     breite Leinwand, optional kurzer Ankerbereich).
   - Dateien: `data/calibration/video_project_panorama*`.
   - Gleichmäßige Zeit-Samples verpassten kurze Extremschwenks; schwenkbasierte
     Samples deckten sie ab, aber Langzeitdrift erzeugte Inseln/Verzerrung.
   - Ein kurzer Sweep (Frames 23.100–23.700, Anker 23.300) war schärfer, zeigte
     je nach Referenz aber weiterhin eine Seite besser als die andere.
   - Fazit: **Kein einzelnes Panorama für diese Kalibrierung verwenden.** Veos
     digitaler Dewarp ist über die gesamte Breite nicht durch eine einzige saubere
     globale Homographie beschreibbar.

6. **Kalibrierwerkzeug auf Mehrbildmodus umgebaut.**
   - `src/calibrate_pitch.py` verwendet OpenCV-GUI statt Matplotlib/Tk. Grund:
     lokale Python-3.13-Installation findet `init.tcl` nicht.
   - Mehrbildmodus: `A`/`D` wechselt Originalframes, Linksklick setzt Punkt,
     Rechts-/Mittelklick oder `S` überspringt, `Esc` bricht ab.
   - Verwendeter Aufruf:

     ```powershell
     .\.venv\Scripts\python.exe src\calibrate_pitch.py `
       "data\videos\Video Project.mp4" `
       --homographies data\output\video_project_homographies.npz `
       --frames 23150,23320,23620 `
       --laenge 68 --breite 34 --name video_project
     ```

   - Frames: 23.150 = links, 23.320 = Mitte, 23.620 = rechts.
   - Ergebnis: `data/calibration/video_project.json` plus drei Kontrollbilder.
   - **Achtung:** Das globale `H_pitch_to_px` in dieser JSON ist ungültig, weil
     die Klicks über die driftende Langzeitkette nach Frame 0 transformiert wurden.
     Globaler Reprojektionsfehler: Mittel ~455 px, Median ~222 px, Maximum ~2.174 px.
     Die gelben Linien der globalen Kontrollbilder liegen deutlich falsch.
   - Die Rohklicks (`clicked_px`, `clicked_view`) sind dagegen wertvoll und lokal
     plausibel. Separat gefittete Anker:
     - Frame 23.150 (linke Seite): 6 Punkte, mittlerer Fehler **5,6 px**.
     - Frame 23.620 (rechte Seite): 6 Punkte, mittlerer Fehler **4,9 px**.
     - Frame 23.320: nur 2 Punkte, reicht nicht als eigener Anker.
   - `src/pitch_model.py` hat zusätzliche Mittelkreis-Punkte in `landmarks()`.

7. **Neue driftfreie Architektur begonnen: direkte Anker-Lokalisierung.**
   - Neues Skript: `src/localize_pitch.py`.
   - Es baut aus den Rohklicks für Frame 23.150 und 23.620 je eine lokale
     Pixel→Meter-Kalibrierung. Für jeden Videoframe werden ORB-Features nur einmal
     berechnet, direkt gegen beide Anker gematcht und der bessere Anker gewählt:

     ```text
     aktueller Frame -> direkt zum linken/rechten Anker -> Platzmeter
     ```

   - Dadurch gibt es keine über 14 Minuten akkumulierte Drift.
   - Tests:
     - Stichproben über das ganze Video: beste direkte Matches meist mehrere
       hundert bis >1.500 Inlier; schlechteste geprüfte Stichprobe 102 (im
       Vorabtest mit drei Ansichten).
     - Frames 0–59: 60/60 direkt lokalisiert, 0 Aussetzer, 164–209 Inlier.
     - Visuelle QA bei Frames 0, 5.000, 10.000, 15.000, 20.000, 25.000:
       projizierte End-/Strafraumlinien liegen stabil auf den weißen Linien.
       Kontaktbogen: `data/output/localization_qa.jpg`.
   - `pitch_map.py` erkennt jetzt NPZ-Dateien mit `H_px_to_pitch` und verwendet
     diese Matrizen direkt, statt globale Homographie × globale Kalibrierung.

### Exakter Haltepunkt / noch ungelöst

Ein kurzer End-to-End-Test mit 60 lokalisierten Frames lief technisch durch, aber
der Platzfilter war **noch nicht plausibel**:

- 94 von 1.494 Detektionen wurden als auf dem 68×34-m-Feld markiert.
- Das sind nur ca. 1–2 Spieler pro Frame; im Bild scheinen eher mehrere Spieler
  von Tims Spiel sichtbar zu sein.
- Beispiel Frame 0: Track 1 -> (18,49 m, 24,25 m) innen; weitere frühe Tracks
  lagen z.B. bei x=-8 m, x=-14 m oder weit außerhalb.
- Mögliche Ursachen:
  1. Tim hat trotz plausibler lokaler Geometrie Markierungen des falschen
     überlagerten Linien-Sets angeklickt (Querfeld vs. Großfeld).
  2. Die 94/1.494 sind für die ersten zwei Sekunden wegen vieler Nachbarfelder
     weniger falsch als angenommen; muss visuell geprüft werden.
  3. Die lokale Anker-Homographie passt End-/Strafraumlinien visuell, extrapoliert
     aber den restlichen 68×34-m-Bereich noch falsch.

Der nächste Diagnosebefehl sollte Frame 0 mit Track-Fußpunkten, Meterkoordinaten
und innen/außen-Farben erzeugen. Der Lauf wurde vom Nutzer absichtlich abgebrochen;
`%TEMP%\fa_frame0.npz` existiert nicht und es läuft kein Python-Prozess. **Diesen
Debugschritt zuerst wiederholen; noch keinen Vollvideo-Lokalisierungslauf starten.**

### Empfohlene nächste Schritte (in dieser Reihenfolge)

1. `localize_pitch.py` nur für Frame 0 (oder 0–59) ausführen und auf das Originalbild
   alle Tracking-Fußpunkte mit `(x_m,y_m)` zeichnen: grün innerhalb, rot außerhalb.
   Prüfen, welche sichtbaren Spieler Tim tatsächlich zu seinem vordersten Querfeld
   zählt. Tim sollte dieses Debugbild bestätigen.
2. Falls das falsche Linien-Set geklickt wurde: Mehrbild-Kalibrierung erneut laufen
   lassen, aber vorher ein annotiertes Linien-/Landmarken-Hilfsbild erstellen.
3. Falls die Anker lokal korrekt sind, `localize_pitch.py` vollständig auf Colab
   laufen lassen (CPU/ORB, nicht T4-beschleunigt) und die resultierende
   `video_project_pitch_localization.npz` nach Drive sichern.
4. Danach:

   ```powershell
   python src\pitch_map.py data\output\video_project_tracked.csv `
     data\output\video_project_pitch_localization.npz `
     data\calibration\video_project.json --fps 30
   ```

5. Erst wenn Platzfilter und Meterpositionen plausibel sind: Team-Clustering nach
   dem geometrischen Filter, Heatmaps und sichtbare Tracklet-Distanzen.
6. Das Colab-Notebook ist **noch nicht** um `localize_pitch.py` und den neuen
   direkten `pitch_map.py`-Lauf ergänzt. Vor dem nächsten Cloud-Lauf aktualisieren.

### Noch nicht committe Änderungen/Artefakte bei Übergabe

- Geändert: `README.md`, `src/calibrate_pitch.py`, `src/pitch_map.py`,
  `src/pitch_model.py`.
- Neu: `src/build_panorama.py`, `src/localize_pitch.py`.
- Neu in `data/calibration/`: `video_project.json`, Referenz-/Kontroll-/Panoramabilder
  sowie `highlight_ref120.jpg`. Einige Bilder sind Diagnose-Scratch und müssen
  nicht zwingend committed werden.
- `data/output/` ist gitignored; wichtige lokale Diagnose:
  `localization_qa.jpg`. Tracking-CSV und Homographien liegen ebenfalls dort.

### Korrekturen gegenüber älteren Abschnitten weiter unten

- `register_frames.py` ist inzwischen Streaming-fähig; der alte Hinweis
  „lädt alle Frames in den RAM / Umbau steht aus“ ist überholt.
- Detektions-CSV-Indizes sind inzwischen absolute Video-Frames, nicht mehr
  slice-relativ.
- Colab ist eingerichtet und das Vollvideo-Tracking ist abgeschlossen.
- Für das Vollvideo gilt nicht mehr die alte Pipeline „globale Registrierung +
  eine globale Kalibrierung“, sondern der neue direkte Ankeransatz.

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
