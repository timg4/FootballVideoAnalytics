# Methodology and design decisions

These notes explain why the pipeline is built the way it is, and which
alternatives I dropped along the way. They go further than the README, because
the interesting decisions live here.

## Data flow

```
video
  --> detect_track.py    tracking CSV (frame, tracker_id, box, conf) + video
  --> localize_pitch.py  per frame: homography pixel -> meter
  --> pitch_map.py       position CSV (x_m, y_m, on_pitch), heatmap, distances
  --> team_assign.py     team per tracklet
```

Each step reads the CSV of the previous one and writes its own. That is
deliberate. Detection is by far the expensive part (minutes to hours depending on
hardware), everything after it takes seconds. When I tweak the pitch filter or the
team logic, I do not want to rerun YOLO every time. The intermediate files are
also easy to inspect when something looks off.

## Detection and tracking

**Model and resolution.** I use YOLOv11 in the small variant (`yolo11s`). The nano
variant is faster but does not reliably find the players far away. The analysis
resolution turned out to be the deciding factor: at the default 640 px the players
at the far edge of the Veo wide-angle image are simply missing, at 1280 px they
are there. That costs compute, but it is not negotiable, because otherwise half a
team disappears.

**Tracking.** ByteTrack first, because it works without an appearance model and is
fast. Later I added BoT-SORT with camera-motion compensation and appearance re-id,
because the follow-cam moves constantly and ByteTrack loses a lot of IDs on the
pans.

**The key insight here:** tracklet IDs are not player IDs. As soon as a player
walks out of frame on a pan and comes back, they are a new person to the tracker.
Over 14 minutes that produces thousands of short tracklets for a few dozen real
players. Anything I compute per tracklet (distance, team) is correct, but it does
not glue together into per-player values without extra work. Being honest about
that mattered more to me than a nice but wrong per-player kilometer count.

## Calibration: from image to real meters

For heatmaps and running paths I need a position in meters on the pitch for every
detected foot point, not in pixels. The pitch filter then falls out for free:
anything outside the known field dimensions belongs to one of the neighboring
games and gets dropped. This conversion is the genuinely hard part of the project,
and I dropped two approaches before the third one worked.

### Why the obvious approaches fail

The Veo follow-cam is not a real camera pan but a digital crop out of a fixed
panorama. Two frames are therefore related by a homography, roughly. The obvious
plan: chain a homography across the whole video, calibrate one reference frame
once, done.

It drifts. Every frame-to-frame homography carries a small error, over 25,000
frames that adds up, and by the end the projected lines sit several meters off. For
a 15-second goal highlight the chain is fine, for a 14-minute game it is not.

The second plan was a median panorama: warp every frame onto one canvas and take
the median over time so the players vanish and only the lines remain. That gives a
nice control image but inherits the same drift and warps toward the edges.

There is also a problem that has nothing to do with the camera. Several sets of
markings overlap on the artificial turf. Our cross-pitch has its own lines, and
laid across them are the markings of the full-size field with a completely
different penalty box. When you read off the field points it is easy to grab the
wrong one and mix two geometries that do not belong together.

### The orthophoto trick

The breakthrough came from outside the video. The City of Vienna provides
orthophotos as open data (basemap.at, 10 cm per pixel). I found the facility in
there, rotated the image so the field lies axis-aligned, and measured the lines
directly against a meter grid: 55.75 by 27.25 m. Before that I had worked with a
guessed 68 by 34 m, and those wrong dimensions were exactly why my click
calibrations never came out consistent. With a fixed ground truth the guessing
goes away.

### Anchor localization instead of a global chain

Instead of one running chain, I calibrate three separate anchor views (left,
center, right) against the ortho dimensions. Each video frame is then matched by
ORB feature matching directly against the best-fitting anchor, and mapped from
there into meters. Because every frame hangs off a fixed anchor rather than its
predecessor, there is no accumulated drift. A frame can match poorly on its own,
but it does not drag the others with it.

On the full game, 625 feature points match per frame at the median, the minimum is
78, and all 25,150 frames get localized without dropouts.

### The degenerate homography and the virtual near point

One detail held me up the longest. My clicked field points per view were almost
all on one line, the visible goal line, far back in the image. A homography cannot
be pinned down in the depth direction from collinear points. The fit looked good
locally (reprojection errors of a few pixels) but folded the pitch completely wrong
in depth. In the control image you can see it as a field outline that collapses
into a thin loop.

Two things fixed it. First, I bridged points from the center of the field into each
view through the local registration of the anchors, which sit only seconds apart,
but only between neighboring views, because a bridge across the full pan would
drift again. Second, I constructed a virtual near point: the intersection of the
visible near touchline with the halfway line. That point sits close to the camera
and breaks the collinearity. After that a RANSAC fit clears out the remaining
outliers. One clicked "center spot" got rejected every time, probably because it
was a point from one of the overlapping marking sets.

### Checks

Two checks convinced me the calibration was right before I started the long run.
First the geometric one: the projected field outline sits on the real lines across
the whole match, including the darkest closing phase (see
`assets/calibration_validation.jpg`). Second the plausibility one: at the median
nine or ten players per frame fall inside the field boundaries, and only 26 of
25,150 frames show more than 18. If the filter were leaky, the neighboring games
would be counted constantly.

## Team assignment

The order matters here: the geometric pitch filter first, then the color. Only
players that `pitch_map.py` places on the field go into the color analysis. Earlier
I tried to get rid of the neighboring games through color alone, which never came
out clean.

The color features are brightness-normalized, so the red and green share plus
saturation instead of raw pixel values. In low light every jersey goes dark, but
the ratios stay stable: white and gray have low saturation, a green bib has a high
green share regardless of how dark it currently is. I sample the color only from
the torso region and only from boxes above a minimum height, so grass and legs do
not leak in.

Clustering runs as K-means into five groups, not two. One team tends to split into
a bright and a shadowed group under these light conditions. The candidates are
therefore merged again by their normalized color shares until two teams remain. A
plain saturation distance first joined two wrong groups here; the normalized BGR
shares separate them more cleanly.

## Meters, heatmaps, and distances

The heatmap is a top-down model of the pitch onto which all on-pitch positions get
stamped as Gaussian kernels. Nothing fancy, but it makes the distribution readable
at a glance.

For the running distances the honest computation mattered more than a big number.
Per tracklet I smooth the meter positions with a moving average against the
detection jitter, discard jumps above 12 m/s as tracking errors, and sum the rest.
The report gives distance, visible duration, and average speed per tracklet.
Because a tracklet only covers one visible stretch of a player, these are minimums,
not total kilometers. The team sums (about 7.6 and 8.2 km) are therefore lower
bounds on the real running output.

## Open threads

What I started and why it is not finished:

**Stable player IDs.** This is the prerequisite for real per-player stats and the
hardest task. I extracted appearance vectors per tracklet (Intel
`person-reidentification`) and tried to glue the same player together through
similarity. The problem: at a threshold that produces almost no false links, it
also links only a few real pairs, because amateur kits are too similar in color.
Fully automatic is not enough; realistically this ends up as semi-automatic
stitching plus a short manual assignment against the real roster.

**Ball and passes.** The ball is small, fast, and often occluded, so the detection
is patchy. I therefore treat the ball track as a candidate list from which
possession and pass candidates can be derived, but which still needs a visual check
before it becomes a pass statistic. A later advantage: for the highlight clips the
Veo labels (shot, goal) are already in the filenames, which gives free ground truth
for a shot map.

## If I started over

Two things earlier. First, take the orthophoto as ground truth from the start
instead of guessing field dimensions. That would have saved half the calibration
odyssey. Second, separate tracklets and players clearly from the beginning instead
of hoping the tracker keeps the IDs stable.
