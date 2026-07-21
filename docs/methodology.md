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
orthophotos as open data (basemap.at, about 10 cm per pixel). I found the facility
there, rotated the image so the field lies axis-aligned, and measured the relevant
strip at 55.75 by 27.25 m. Before that I had worked with a guessed 68 by 34 m,
which was clearly the wrong scale. The orthophoto constrains the geometry, but the
video overlay still has to decide which of several painted lines belongs to this
pitch.

### Anchor localization instead of a global chain

Instead of one running chain, I calibrate three separate anchor views against the
ortho dimensions. Each video frame is then matched by ORB feature matching
directly against the best-fitting anchor and mapped from there into meters.
Because every frame hangs off a fixed anchor rather than its predecessor, there
is no accumulated drift. A frame can match poorly on its own, but it does not drag
the others with it.

On the full game, 625 feature points match per frame at the median, the minimum is
78, and all 25,150 frames get matched. These figures measure image registration,
not the physical correctness of the clicked field boundary.

### The degenerate homography and the virtual near point

One detail held me up the longest. My clicked field points per view were almost
all on one line, the visible goal line, far back in the image. A homography cannot
be pinned down in the depth direction from collinear points. The fit looked good
locally (reprojection errors of a few pixels) but folded the pitch completely wrong
in depth. In the control image you can see it as a field outline that collapses
into a thin loop.

My first fix was to bridge center-field points into the outer views and construct
a virtual near point from an apparent touchline intersection. It produced small
reprojection errors, but a later end-line frame showed that the virtual point came
from a marking of the large field. The fit was mathematically consistent and
physically wrong.

The replacement uses the orthophoto and only the markings that really exist: the
four outer lines. I click several positions along each line and switch between the
left, middle, and right views where necessary. ORB registration puts these line
samples into a common image coordinate system. Intersecting the fitted lines gives
the four corners even if a corner lies outside every individual crop. The final
full-video localization still matches every frame directly to a fixed anchor, so
the short calibration bridge does not turn into a 14-minute drift chain.

### Checks

The first aggregate checks looked plausible, but they were not strict enough. In
frame 23,620 the ball and two players in an active duel sit just beyond the
projected touchline. A generous 0.5 m filter margin hid the error in the binary
`on_pitch` count. This is now the acceptance test: before a full run, each anchor
overlay is inspected directly, and the outer line must contain the duel in that
frame without relying on the margin. The old validation image remains useful as
an example of why aggregate counts alone are insufficient.

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

For the running distances the honest computation matters more than a big number.
Per tracklet I smooth the meter positions with a moving average against the
detection jitter, discard jumps above 12 m/s as tracking errors, and sum the rest.
The report gives distance, visible duration, and average speed per tracklet.
Because a tracklet only covers one visible stretch of a player, these are minimums,
not total kilometers. The final piecewise calibration uses one homography per
half-pitch because the camera is mounted over the middle of a touchline. Six
manual depth references fit at 1.7 px mean error; strict frame checks keep the
active players and ball inside while excluding the neighboring games. In the
14-minute run, 22,872 of 25,150 frames (90.9%) remain after the conservative
anchor-transition filter. Inside a 12 px guard band around the virtual seam,
both half-pitch solutions are checked against their valid metric rectangles.
This avoids assigning a point to the wrong homography when Veo dewarping moves
the seam by a few pixels; outside that narrow band the image-side decision is
kept unchanged. The production pitch filter allows only 0.5 m outside the exact
line for foot-point and calibration noise. A previous 1.5 m tolerance retained
real throw-in positions and was removed after the visual QA.

## Open threads

What I started and why it is not finished:

**Pitch calibration.** The first global fit assigned markings from different
pitch layouts to the wrong model points. The solved version uses the measured
55.75 x 27.25 m strip, two half-pitch homographies, manual end corners and a
shared far midpoint. The invisible point below the camera is inferred instead of
clicked. Remaining uncertainty comes from Veo dewarping and the 9.1% discarded
transition frames, not from an unfitted boundary.

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
before it becomes a pass statistic. With the final piecewise geometry the track
covers 8,239 of 25,150 frames (32.8%) and still contains occasional false
detections. Manual review can remove false candidates, but it cannot recover a
pass when the ball was never detected. The reported pass results are therefore a
reviewed visible subset, not a complete or unbiased full-match pass rate. A later
advantage: for the highlight clips the
Veo labels (shot, goal) are already in the filenames, which gives free ground truth
for a shot map.

## If I started over

Two things earlier. First, take the orthophoto as ground truth from the start
instead of guessing field dimensions. That would have saved half the calibration
odyssey. Second, separate tracklets and players clearly from the beginning instead
of hoping the tracker keeps the IDs stable.
