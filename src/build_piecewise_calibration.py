"""Build two half-field homographies for a camera on the near touchline.

The near touchline appears as two different rays because the camera stands over
its middle. This script reuses the saved manual boundary clicks: the left and
right rays meet at the camera ground point, which is assigned to half the pitch
length. Each half of the pitch then receives its own ordinary homography.
"""

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

from calibrate_from_ortho import (
    anchor_to_base_transforms,
    fit_homogeneous_line,
    intersect_lines,
)
from piecewise_pitch import draw_piecewise_overlay


def portable_path(value):
    path = Path(value).resolve()
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)


def grouped_base_points(boundary, transforms):
    grouped = {}
    for item in boundary["points"]:
        frame = int(item["frame"])
        clicked = np.asarray(item["clicked_px"], dtype=np.float64)
        base = cv2.perspectiveTransform(
            clicked.reshape(1, 1, 2), transforms[frame]
        ).reshape(2)
        grouped.setdefault(frame, []).append(base)
    return grouped


def fitted(grouped, frame):
    points = grouped.get(frame, [])
    if len(points) < 2:
        raise SystemExit(f"boundary has fewer than two clicks in frame {frame}")
    return fit_homogeneous_line(points)[0]


def main():
    parser = argparse.ArgumentParser(
        description="Piecewise calibration for a camera on boundary 4")
    parser.add_argument("calibration_json")
    parser.add_argument("--name", default="video_project_piecewise")
    parser.add_argument("--depth-references", default=None,
                        help="JSON from collect_depth_references.py")
    args = parser.parse_args()

    source_path = Path(args.calibration_json)
    calibration = json.loads(source_path.read_text(encoding="utf-8"))
    boundaries = calibration.get("boundary_line_observations")
    if not boundaries or len(boundaries) != 4:
        raise SystemExit("calibration needs four saved boundary-line observations")

    frame_indices = sorted({
        int(item["frame"])
        for boundary in boundaries
        for item in boundary["points"]
    })
    video_path = Path(calibration["video"])
    cap = cv2.VideoCapture(str(video_path))
    video_frames = []
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"video frame {frame_index} is not readable")
        video_frames.append(frame)
    cap.release()
    transforms, registration_quality = anchor_to_base_transforms(
        video_frames, frame_indices)

    groups = [grouped_base_points(boundary, transforms)
              for boundary in boundaries]
    left_frame = min(frame_indices)
    right_frame = max(frame_indices)
    far_frames = sorted(frame for frame, points in groups[1].items()
                        if len(points) >= 2)
    if len(far_frames) < 2:
        raise SystemExit("opposite touchline needs clicks in left and middle/right views")

    left_end = fitted(groups[0], left_frame)
    left_far = fitted(groups[1], far_frames[0])
    right_far = fitted(groups[1], far_frames[-1])
    right_end = fitted(groups[2], right_frame)
    near_left = fitted(groups[3], left_frame)
    near_right = fitted(groups[3], right_frame)

    corner_left_near = intersect_lines(left_end, near_left)
    corner_left_far = intersect_lines(left_end, left_far)
    corner_right_far = intersect_lines(right_end, right_far)
    corner_right_near = intersect_lines(right_end, near_right)
    camera_point = intersect_lines(near_left, near_right)
    center_far = (corner_left_far + corner_right_far) / 2

    pitch = calibration["pitch"]
    length = float(pitch["laenge"])
    width = float(pitch["breite"])
    split = length / 2
    model_left = np.array([
        [0, 0], [0, width], [split, width], [split, 0]
    ], dtype=np.float64)
    model_right = np.array([
        [split, 0], [split, width], [length, width], [length, 0]
    ], dtype=np.float64)
    base_left = np.array([
        corner_left_near, corner_left_far, center_far, camera_point
    ], dtype=np.float64)
    base_right = np.array([
        camera_point, center_far, corner_right_far, corner_right_near
    ], dtype=np.float64)

    reference_quality = None
    if args.depth_references:
        references = json.loads(
            Path(args.depth_references).read_text(encoding="utf-8"))
        H_ortho_to_pitch = np.asarray(
            calibration["H_ortho_to_pitch"], dtype=np.float64)

        def reference_arrays(items):
            model_points = []
            base_points = []
            for item in items:
                ortho_point = np.asarray(item["ortho_px"], dtype=np.float64)
                meter = cv2.perspectiveTransform(
                    ortho_point.reshape(1, 1, 2), H_ortho_to_pitch).reshape(2)
                video_point = np.asarray(item["video_px"], dtype=np.float64)
                frame = int(item["frame"])
                base = cv2.perspectiveTransform(
                    video_point.reshape(1, 1, 2), transforms[frame]).reshape(2)
                model_points.append(meter)
                base_points.append(base)
            return np.asarray(model_points), np.asarray(base_points)

        def match_controls(ref_model, ref_base, targets, side):
            """Assign manual references to the intended pitch landmarks.

            The ortho clicks contain small measurement/clicking errors.  They
            identify a known corner or the far midpoint, so use them to choose
            the landmark and then snap the model coordinate to that landmark.
            """
            if len(ref_model) < len(targets):
                raise SystemExit(
                    f"{side}: need at least {len(targets)} depth references")
            best = None
            for indices in itertools.permutations(
                    range(len(ref_model)), len(targets)):
                errors = np.array([
                    np.linalg.norm(ref_model[index] - target)
                    for index, target in zip(indices, targets)
                ])
                score = float(errors.sum())
                if best is None or score < best[0]:
                    best = (score, indices, errors)
            _, indices, errors = best
            if float(errors.max()) > 2.0:
                raise SystemExit(
                    f"{side}: a depth reference is more than 2 m from its "
                    f"expected corner/midpoint ({errors.max():.2f} m)")
            return np.asarray([ref_base[index] for index in indices]), errors

        left_ref_model, left_ref_base = reference_arrays(references["left"])
        right_ref_model, right_ref_base = reference_arrays(references["right"])

        left_manual, left_snap_errors = match_controls(
            left_ref_model, left_ref_base,
            model_left[[0, 1, 2]], "left half")
        right_manual, right_snap_errors = match_controls(
            right_ref_model, right_ref_base,
            model_right[[1, 2, 3]], "right half")

        # Both midpoint clicks refer to the same physical point.  Averaging
        # their base-frame positions makes the two homographies meet exactly.
        manual_center_gap = float(np.linalg.norm(
            left_manual[2] - right_manual[0]))
        center_far = (left_manual[2] + right_manual[0]) / 2
        corner_left_near = left_manual[0]
        corner_left_far = left_manual[1]
        corner_right_far = right_manual[1]
        corner_right_near = right_manual[2]
        base_left = np.array([
            left_manual[0], left_manual[1], center_far, camera_point
        ], dtype=np.float64)
        base_right = np.array([
            camera_point, center_far, right_manual[1], right_manual[2]
        ], dtype=np.float64)

        H_left_pitch_to_base = cv2.getPerspectiveTransform(
            model_left.astype(np.float32), base_left.astype(np.float32))
        H_right_pitch_to_base = cv2.getPerspectiveTransform(
            model_right.astype(np.float32), base_right.astype(np.float32))

        # Report the residual against the actual clicks.  Only the two
        # independently clicked far-midpoint observations can retain a small
        # error because their common location is deliberately averaged.
        left_model_fit = model_left[[0, 1, 2]]
        left_base_fit = left_manual
        right_model_fit = model_right[[1, 2, 3]]
        right_base_fit = right_manual
        left_errors = np.linalg.norm(
            cv2.perspectiveTransform(
                left_model_fit.reshape(-1, 1, 2), H_left_pitch_to_base
            ).reshape(-1, 2) - left_base_fit, axis=1)
        right_errors = np.linalg.norm(
            cv2.perspectiveTransform(
                right_model_fit.reshape(-1, 1, 2), H_right_pitch_to_base
            ).reshape(-1, 2) - right_base_fit, axis=1)
        reference_quality = {
            "left_mean_error_px": float(left_errors.mean()),
            "left_max_error_px": float(left_errors.max()),
            "right_mean_error_px": float(right_errors.mean()),
            "right_max_error_px": float(right_errors.max()),
            "far_midpoint_click_gap_px": manual_center_gap,
            "left_ortho_snap_errors_m": left_snap_errors.tolist(),
            "right_ortho_snap_errors_m": right_snap_errors.tolist(),
        }
        print(f"depth-reference errors: left {left_errors.mean():.1f}px, "
              f"right {right_errors.mean():.1f}px; midpoint gap "
              f"{manual_center_gap:.1f}px")
    else:
        H_left_pitch_to_base = cv2.getPerspectiveTransform(
            model_left.astype(np.float32), base_left.astype(np.float32))
        H_right_pitch_to_base = cv2.getPerspectiveTransform(
            model_right.astype(np.float32), base_right.astype(np.float32))

    anchors = []
    calibration_dir = Path("data/calibration")
    calibration_dir.mkdir(parents=True, exist_ok=True)
    for frame_index, frame in zip(frame_indices, video_frames):
        H_left = np.linalg.inv(H_left_pitch_to_base) @ transforms[frame_index]
        H_right = np.linalg.inv(H_right_pitch_to_base) @ transforms[frame_index]
        H_left /= H_left[2, 2]
        H_right /= H_right[2, 2]
        seam_model = np.array([
            [split, width], [split, 0]
        ], dtype=np.float64).reshape(-1, 1, 2)
        seam_px = cv2.perspectiveTransform(
            seam_model, np.linalg.inv(H_left)).reshape(-1, 2)
        anchors.append({
            "frame": frame_index,
            "H_anchor_to_pitch_left": H_left.tolist(),
            "H_anchor_to_pitch_right": H_right.tolist(),
            "seam_px": seam_px.tolist(),
        })
        overlay = draw_piecewise_overlay(
            frame, H_left, H_right, split, length, width)
        output_image = calibration_dir / (
            f"{args.name}_check_frame_{frame_index}.jpg")
        cv2.imwrite(str(output_image), overlay,
                    [cv2.IMWRITE_JPEG_QUALITY, 94])
        print(f"check: {output_image}")

    calibration["calibration_mode"] = "piecewise_camera_on_touchline"
    calibration["piecewise"] = {
        "split_x_m": split,
        "camera_boundary": 4,
        "camera_fraction": 0.5,
        "base_frame": frame_indices[0],
        "base_control_points": {
            "left_near": corner_left_near.tolist(),
            "left_far": corner_left_far.tolist(),
            "center_far": center_far.tolist(),
            "camera_point": camera_point.tolist(),
            "right_far": corner_right_far.tolist(),
            "right_near": corner_right_near.tolist(),
        },
        "anchors": anchors,
        "registration_quality": registration_quality,
        "depth_reference_quality": reference_quality,
        "depth_references": (portable_path(args.depth_references)
                             if args.depth_references else None),
    }
    calibration["piecewise_source"] = portable_path(source_path)
    output = calibration_dir / f"{args.name}.json"
    output.write_text(json.dumps(calibration, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(f"piecewise calibration: {output}")


if __name__ == "__main__":
    main()
