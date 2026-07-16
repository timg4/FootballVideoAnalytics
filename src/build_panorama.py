"""Build a mostly player-free median panorama from registered frames.

The frames are warped into the coordinate system of the reference frame with the
homographies. Per pixel the median over several time points is taken, which makes
moving players largely disappear. The image is computed at a reduced scale and
saved together with the transform image -> reference frame.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Median panorama for the pitch calibration")
    parser.add_argument("video")
    parser.add_argument("homographies_npz")
    parser.add_argument("--samples", type=int, default=80,
                        help="approximate number of frames across all pan ranges (default: 80)")
    parser.add_argument("--pan-bins", type=int, default=16,
                        help="number of covered camera positions from left to right")
    parser.add_argument("--start-frame", type=int, default=None,
                        help="use only homographies from this absolute frame on")
    parser.add_argument("--end-frame", type=int, default=None,
                        help="use only homographies before this absolute frame")
    parser.add_argument("--anchor", type=int, default=None,
                        help="rebase a short range onto this frame (reduces long-term drift)")
    parser.add_argument("--min-separation", type=int, default=60,
                        help="minimum spacing of selected frames within a pan range")
    parser.add_argument("--min-coverage", type=int, default=3,
                        help="minimum number of valid images per panorama pixel")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="output scale (default: 0.5)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.samples < 3:
        parser.error("--samples must be at least 3")
    if not 0 < args.scale <= 1:
        parser.error("--scale must be in (0, 1]")

    video_path = Path(args.video)
    out_dir = Path(__file__).resolve().parent.parent / "data" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = (Path(args.output) if args.output else
              out_dir / f"{video_path.stem.lower().replace(' ', '_')}_panorama.jpg")
    transform_path = output.with_name(f"{output.stem}_transform.json")

    with np.load(args.homographies_npz) as data:
        H_all = data["H"].copy()
        frames = (data["frames"].copy() if "frames" in data
                  else np.arange(len(H_all)))
        ref = int(data["ref"])

    # optionally use only one short, complete camera pan. Rebasing onto an anchor
    # frame removes the drift accumulated so far. H_original_ref_to_work_ref still
    # keeps the later calibration compatible with the coordinate system of the
    # original homography file.
    range_mask = np.ones(len(frames), dtype=bool)
    if args.start_frame is not None:
        range_mask &= frames >= args.start_frame
    if args.end_frame is not None:
        range_mask &= frames < args.end_frame
    H_original_ref_to_work_ref = np.eye(3)
    if args.anchor is not None:
        anchor_matches = np.flatnonzero(frames == args.anchor)
        if not len(anchor_matches):
            raise SystemExit(f"anchor frame {args.anchor} missing from the homography file")
        H_anchor_to_original_ref = H_all[anchor_matches[0]]
        H_original_ref_to_work_ref = np.linalg.inv(H_anchor_to_original_ref)
        H_all = np.array([H_original_ref_to_work_ref @ H for H in H_all])
    H_all = H_all[range_mask]
    frames = frames[range_mask]
    if not len(frames):
        raise SystemExit("the selected frame range is empty")

    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not width or not height:
        raise SystemExit(f"video not readable: {video_path}")

    # wide canvas for both goal ends. The reference frame deliberately does not
    # sit at the left edge, so extreme pans have room in both directions.
    offset = np.array([[1, 0, width * 2], [0, 1, height // 2], [0, 0, 1]],
                      dtype=np.float64)
    scale = np.array([[args.scale, 0, 0], [0, args.scale, 0], [0, 0, 1]],
                     dtype=np.float64)
    H_ref_to_image = scale @ offset @ H_original_ref_to_work_ref
    canvas_w = int(width * 5 * args.scale)
    canvas_h = int(height * 2 * args.scale)

    # do not sample evenly over time: short pans to the goal would easily be
    # missed. Instead determine the projected image center and pick several
    # time-separated frames from each pan range.
    normalized_H = H_all / H_all[:, 2:3, 2:3]
    center = np.array([[[width / 2, height / 2]]], dtype=np.float64)
    projected_center = np.array([
        cv2.perspectiveTransform(center, H)[0, 0] for H in normalized_H
    ])
    pan_x = projected_center[:, 0]
    det = np.linalg.det(normalized_H)
    finite = (np.isfinite(projected_center).all(axis=1)
              & (det > 0.2) & (det < 5.0)
              & (projected_center[:, 1] > -height)
              & (projected_center[:, 1] < 2 * height))
    lo, hi = np.percentile(pan_x[finite], [0.5, 99.5])
    targets = np.linspace(lo, hi, args.pan_bins)
    per_bin = max(1, int(np.ceil(args.samples / args.pan_bins)))
    chosen = []
    for target in targets:
        order = np.argsort(np.abs(pan_x - target))
        selected_here = []
        # separate in time, so different players stand at the same camera
        # position and the median can actually remove them.
        for pos in order:
            if not finite[pos] or not lo <= pan_x[pos] <= hi:
                continue
            if all(abs(int(pos) - old) >= args.min_separation for old in selected_here):
                selected_here.append(int(pos))
            if len(selected_here) == per_bin:
                break
        chosen.extend(selected_here)
    sample_positions = np.array(sorted(set(chosen)), dtype=int)
    print(f"pan range reference-x {lo:.0f} to {hi:.0f}; "
          f"{len(sample_positions)} targeted samples")
    source_frames = []
    source_H = []
    for n, pos in enumerate(sample_positions, 1):
        frame_idx = int(frames[pos])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok:
            source_frames.append(frame)
            source_H.append(H_ref_to_image @ H_all[pos])
        print(f"reading sample {n}/{len(sample_positions)}", end="\r")
    cap.release()
    print(f"\n{len(source_frames)} frames loaded; computing median ...")

    panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    tile_h = 180
    source_mask = np.full((height, width), 255, dtype=np.uint8)
    for y0 in range(0, canvas_h, tile_h):
        y1 = min(y0 + tile_h, canvas_h)
        h_tile = y1 - y0
        tile_shift = np.array([[1, 0, 0], [0, 1, -y0], [0, 0, 1]],
                              dtype=np.float64)
        # 255 marks invalid pixels and ends up last when sorting.
        stack = np.full((len(source_frames), h_tile, canvas_w, 3), 255,
                        dtype=np.uint8)
        valid_count = np.zeros((h_tile, canvas_w), dtype=np.uint16)
        for n, (frame, H) in enumerate(zip(source_frames, source_H)):
            H_tile = tile_shift @ H
            warped = cv2.warpPerspective(frame, H_tile, (canvas_w, h_tile))
            mask = cv2.warpPerspective(source_mask, H_tile, (canvas_w, h_tile),
                                       flags=cv2.INTER_NEAREST) > 0
            stack[n][mask] = warped[mask]
            valid_count += mask
        stack.sort(axis=0)
        median_index = np.maximum(valid_count.astype(np.int32) - 1, 0) // 2
        indices = np.repeat(median_index[None, ..., None], 3, axis=3)
        tile = np.take_along_axis(stack, indices, axis=0)[0]
        tile[valid_count < args.min_coverage] = 0
        panorama[y0:y1] = tile
        print(f"median rows {y1}/{canvas_h}", end="\r")

    cv2.imwrite(str(output), panorama, [cv2.IMWRITE_JPEG_QUALITY, 95])
    transform_path.write_text(json.dumps({
        "video": str(video_path),
        "ref": ref,
        "H_ref_to_image": H_ref_to_image.tolist(),
        "samples": len(source_frames),
    }, indent=2), encoding="utf-8")
    print(f"\nPanorama: {output}")
    print(f"Transform: {transform_path}")


if __name__ == "__main__":
    main()
