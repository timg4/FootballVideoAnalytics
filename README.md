# FootballAnalytics

My amateur team records its games with a Veo camera. That comes to hours of
footage nobody ever turns into numbers. This project does it automatically: it
finds the players in the video, converts their image position into real meters on
the pitch, and turns that into heatmaps, running distances, and a split into the
two teams.

The core pipeline runs from the video file all the way to the finished stats, and
I have checked it against a full 14-minute game. Ball possession, passes, and
stable per-player IDs are started but not finished yet (see
[Experimental](#experimental)).

## Result

Heatmaps of a full game, split by team. Each point is a player position in
meters, projected onto a top-down model of the pitch.

| Team green | Team blue |
|------------|-----------|
| ![Heatmap team green](assets/heatmap_team_green.png) | ![Heatmap team blue](assets/heatmap_team_blue.png) |

Both teams clearly push toward the opposing goal, and possession spreads across
the whole width. The visible running output comes to roughly 7.6 km (green) and
8.2 km (blue) over 14 minutes. What "visible" means here, and why these are not
yet true per-player kilometers, is explained under [Limitations](#limitations).

## How it works

Four steps. Each one writes its intermediate result as a CSV, so the expensive
detection runs only once and everything after it finishes in seconds.

1. **Detection and tracking** (`detect_track.py`). YOLOv11 finds the people in
   each frame, ByteTrack (or BoT-SORT with re-id) follows them over time. Distant
   players only show up at an analysis resolution of 1280 px; below that they go
   missing.
2. **Pitch localization** (`localize_pitch.py`). Each frame is matched directly
   against three calibrated anchor views, which gives the pixel-to-meter mapping.
   This was the hardest part of the project, more on that below.
3. **Metric positions** (`pitch_map.py`). Projects the foot point of every track
   onto the pitch, throws away everything outside the field dimensions (the
   neighboring games), and produces the position CSV, the heatmap, and the running
   distances.
4. **Team assignment** (`team_assign.py`). Clusters the jersey colors of the
   remaining players into two teams, using brightness-normalized color features so
   the low evening light does not wash the colors out.

## The hard part: calibration

The Veo camera is a follow-cam that pans digitally through a fixed panorama.
Chaining a single homography across 14 minutes drifts too much to get reliable
meters out of it. On top of that, several sets of lines overlap on the artificial
turf (our cross-pitch plus the markings of the full-size field), which makes
reading the field geometry ambiguous.

The way out came from outside the video. The City of Vienna publishes orthophotos
as open data (basemap.at, 10 cm per pixel). I found the facility in there and
measured the field to within ten centimeters: 55.75 by 27.25 m. Instead of
guessing, that is now the ground truth for the calibration.

![Orthophoto with meter grid](assets/orthophoto_measurement.jpg)

With that I localize each frame on its own, without drift. On the full game,
25,150 of 25,150 frames are localized, with no dropouts. As a check, here is the
projected field plus foot points across the whole match (green = on the pitch,
red = neighboring game):

![Calibration check](assets/calibration_validation.jpg)

The long version, with all the dead ends and the numbers, is in
[docs/methodology.md](docs/methodology.md).

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Tested with Python 3.13 on Windows. Tracking a whole game really wants a GPU (I
use Google Colab, see [notebooks/](notebooks/)); the pitch localization runs fine
on the CPU.

## Usage

```powershell
# 1. Detection and tracking (writes the tracking CSV + an annotated video)
.\.venv\Scripts\python.exe src\detect_track.py "data\videos\game.mp4" --imgsz 1280

# 2. Localize each frame against the ortho calibration
.\.venv\Scripts\python.exe src\localize_pitch.py "data\videos\game.mp4" `
    data\calibration\video_project_ortho.json `
    --output data\output\game_localization.npz

# 3. Metric positions, pitch filter, heatmap, running distances
.\.venv\Scripts\python.exe src\pitch_map.py data\output\game_tracked.csv `
    data\output\game_localization.npz `
    data\calibration\video_project_ortho.json --fps 30

# 4. Determine the teams from the on-pitch players
.\.venv\Scripts\python.exe src\team_assign.py "data\videos\game.mp4" `
    data\output\game_tracked.csv `
    --positions-csv data\output\game_positions.csv --no-video
```

The calibration is specific to a camera position. A new position needs a one-time
calibration through `calibrate_pitch.py` (a click tool) plus the field dimensions
measured from the orthophoto.

## Project structure

Core pipeline (`src/`):

| File | Job |
|------|-----|
| `detect_track.py` | person detection (YOLOv11) + tracking |
| `register_frames.py` | camera-pan compensation via ORB homographies |
| `localize_pitch.py` | drift-free per-frame localization against the calibration anchors |
| `calibrate_pitch.py` | interactive click tool for the calibration |
| `build_panorama.py` | median panorama as a calibration aid |
| `pitch_model.py` | parametric pitch model + drawing helpers |
| `pitch_map.py` | image positions to meters, pitch filter, heatmap, distances |
| `team_assign.py` | team assignment from jersey colors |

Configuration and notebooks live in `configs/` and `notebooks/`. The video,
model, and output files are deliberately not checked in (`.gitignore`).

### Experimental

Started, but not yet part of the checked pipeline: ball detection, pass inference,
and re-identification for stable per-player IDs (`detect_ball.py`,
`extract_reid.py`, `stitch_tracklets.py`, `player_stats.py`,
`player_performance.py`). The honest status is in
[docs/methodology.md](docs/methodology.md#open-threads).

## Limitations

- **Tracklet IDs are not player IDs.** When a player leaves the frame during a
  pan and comes back, the tracker usually gives them a new ID. The running
  distances are therefore visible minimums per tracklet, not the total kilometers
  of an identified player. Solving that cleanly is the main open problem (re-id
  plus stitching, mentioned above).
- **The follow-cam only shows a section.** On average nine or ten players are in
  frame instead of all fourteen, so the pitch filter counts fewer at times.
- **Low light.** When there is little light the jerseys desaturate, and ambiguous
  kits (white against gray) stay a weak spot of the team assignment.
- **The calibration is per camera position.** The two datasets I have come from
  two positions and each needs its own calibration.

## Data

- Footage: Veo camera, exported from the [Veo Clubhouse](https://app.veo.co).
- Orthophoto for the measurement: [basemap.at](https://basemap.at) from the City
  of Vienna (open government data, 10 cm resolution).
- Detection models: [Ultralytics YOLOv11](https://docs.ultralytics.com).
