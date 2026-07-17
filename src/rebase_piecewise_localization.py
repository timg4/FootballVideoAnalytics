"""Reuse a full old localization with a new piecewise pitch calibration.

The old NPZ already contains current-frame -> old-pitch matrices and the chosen
anchor id. By removing the old anchor calibration, the expensive ORB result
(current frame -> anchor pixels) can be recovered and composed with the new left
and right half-field calibrations.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def old_anchor_matrices(calibration):
    grouped = defaultdict(list)
    for point in calibration["points"]:
        label = point.get("clicked_view", "")
        if label.startswith("Frame ") and "clicked_px" in point:
            grouped[int(label.split()[1])].append(point)
    matrices = {}
    for frame, points in grouped.items():
        model = np.asarray([item["meter"] for item in points],
                           dtype=np.float64).reshape(-1, 1, 2)
        pixels = np.asarray([item["clicked_px"] for item in points],
                            dtype=np.float64).reshape(-1, 1, 2)
        H_pitch_to_anchor, _ = cv2.findHomography(model, pixels, 0)
        if H_pitch_to_anchor is not None:
            matrices[frame] = H_pitch_to_anchor
    return matrices


def reliability_mask(anchors, window, guard):
    """Remove one-frame anchor flicker and guard stable anchor transitions."""
    anchor_ids = np.unique(anchors)
    kernel = np.ones(window, dtype=np.int32)
    counts = np.vstack([
        np.convolve((anchors == anchor).astype(np.int32), kernel, mode="same")
        for anchor in anchor_ids
    ])
    smoothed = anchor_ids[np.argmax(counts, axis=0)]
    reliable = anchors == smoothed
    changes = np.flatnonzero(smoothed[1:] != smoothed[:-1]) + 1
    for index in changes:
        reliable[max(0, index - guard):min(len(reliable), index + guard + 1)] = False
    return reliable, smoothed, changes


def main():
    parser = argparse.ArgumentParser(
        description="Rebase old full localization onto piecewise calibration")
    parser.add_argument("old_localization_npz")
    parser.add_argument("old_calibration_json")
    parser.add_argument("piecewise_calibration_json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoothing-window", type=int, default=61)
    parser.add_argument("--transition-guard", type=int, default=18)
    args = parser.parse_args()

    if args.smoothing_window < 3 or args.smoothing_window % 2 == 0:
        raise SystemExit("--smoothing-window must be an odd number >= 3")
    old_calibration = json.loads(
        Path(args.old_calibration_json).read_text(encoding="utf-8"))
    new_calibration = json.loads(
        Path(args.piecewise_calibration_json).read_text(encoding="utf-8"))
    old_pitch_to_anchor = old_anchor_matrices(old_calibration)
    new_anchors = {
        int(item["frame"]): item
        for item in new_calibration["piecewise"]["anchors"]
    }

    with np.load(args.old_localization_npz) as source:
        old_direct = source["H_px_to_pitch"].copy()
        frames = source["frames"].copy()
        inliers = source["inliers"].copy()
        anchors = source["anchors"].copy()

    H_left = []
    H_right = []
    for old_H, anchor_frame in zip(old_direct, anchors):
        anchor_frame = int(anchor_frame)
        if anchor_frame not in old_pitch_to_anchor or anchor_frame not in new_anchors:
            raise SystemExit(f"missing old/new calibration for anchor {anchor_frame}")
        H_current_to_anchor = old_pitch_to_anchor[anchor_frame] @ old_H
        new_anchor = new_anchors[anchor_frame]
        left = (np.asarray(new_anchor["H_anchor_to_pitch_left"]) @
                H_current_to_anchor)
        right = (np.asarray(new_anchor["H_anchor_to_pitch_right"]) @
                 H_current_to_anchor)
        H_left.append(left / left[2, 2])
        H_right.append(right / right[2, 2])

    reliable, smoothed, changes = reliability_mask(
        anchors, args.smoothing_window, args.transition_guard)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        H_px_to_pitch_left=np.asarray(H_left),
        H_px_to_pitch_right=np.asarray(H_right),
        split_x_m=np.array(float(new_calibration["piecewise"]["split_x_m"])),
        frames=frames,
        inliers=inliers,
        anchors=anchors,
        smoothed_anchors=smoothed,
        anchor_reliable=reliable,
    )
    print(f"piecewise localization: {output}")
    print(f"reused frames: {len(frames)}")
    print(f"stable anchor transitions: {len(changes)}")
    print(f"reliable frames: {reliable.sum()}/{len(reliable)} "
          f"({100 * reliable.mean():.1f}%)")


if __name__ == "__main__":
    main()
