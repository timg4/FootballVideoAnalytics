"""Calibrate sparse video views from the four real pitch boundary lines.

The small pitch only has an outer boundary, and no single follow-cam frame shows
all four corners. The user therefore clicks several arbitrary points along each
of the four outer lines, switching between neighbouring views where necessary.
The views are registered with ORB, the lines are fitted in a shared coordinate
system, and their intersections provide even off-screen corners. No halfway line
or other non-existent landmark is invented.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pitch_model import PitchModel


def portable_path(value):
    path = Path(value).resolve()
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)


def resized(image, max_width, max_height):
    scale = min(1.0, max_width / image.shape[1], max_height / image.shape[0])
    size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


def draw_header(image, text):
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 42), (15, 23, 42), -1)
    cv2.putText(result, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (255, 255, 255), 1, cv2.LINE_AA)
    return result


def numbered_marker(image, point, number, color):
    center = tuple(np.rint(point).astype(int))
    cv2.circle(image, center, 7, color, -1, cv2.LINE_AA)
    cv2.putText(image, str(number), (center[0] + 9, center[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)


def collect_pitch_corners(ortho):
    """Collect four target-pitch corners clockwise, starting anywhere."""
    window = "1 - target pitch in orthophoto"
    display_base, scale = resized(ortho, 1400, 850)
    clicks = []

    def mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append(np.array([x / scale, y / scale], dtype=np.float64))

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, mouse)
    while True:
        display = display_base.copy()
        for index, point in enumerate(clicks, start=1):
            numbered_marker(display, point * scale, index, (0, 255, 255))
        if len(clicks) > 1:
            polygon = np.asarray(clicks) * scale
            cv2.polylines(display, [polygon.astype(np.int32)],
                          len(clicks) == 4, (0, 255, 255), 2, cv2.LINE_AA)
        display = draw_header(
            display,
            "Click 4 corners of YOUR pitch clockwise (start anywhere) | U undo | Enter accept",
        )
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("u"), ord("U")) and clicks:
            clicks.pop()
        elif key in (10, 13) and len(clicks) == 4:
            break
        elif key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("calibration aborted")
    cv2.destroyWindow(window)
    return np.asarray(clicks, dtype=np.float64)


def ortho_to_pitch_homography(corners, length, width):
    """Map clockwise corners to a rectangle; the longer first edge is length."""
    first_edge = np.linalg.norm(corners[1] - corners[0])
    second_edge = np.linalg.norm(corners[2] - corners[1])
    if first_edge >= second_edge:
        pitch_corners = np.array([[0, 0], [length, 0],
                                  [length, width], [0, width]], np.float32)
    else:
        pitch_corners = np.array([[0, 0], [0, width],
                                  [length, width], [length, 0]], np.float32)
    return cv2.getPerspectiveTransform(corners.astype(np.float32), pitch_corners)


def crop_around(points, image_shape, margin=180):
    x1 = max(0, int(np.floor(points[:, 0].min())) - margin)
    y1 = max(0, int(np.floor(points[:, 1].min())) - margin)
    x2 = min(image_shape[1], int(np.ceil(points[:, 0].max())) + margin)
    y2 = min(image_shape[0], int(np.ceil(points[:, 1].max())) + margin)
    return x1, y1, x2, y2


def collect_boundary_lines(ortho, ortho_crop_box, pitch_corners,
                           video_frames, frame_indices):
    """Collect points along each of the four real boundary lines."""
    x1, y1, x2, y2 = ortho_crop_box
    ortho_crop = ortho[y1:y2, x1:x2].copy()
    local_corners = pitch_corners - np.array([x1, y1])
    ortho_base, ortho_scale = resized(ortho_crop, 850, 760)
    resized_frames = [resized(frame, 900, 760) for frame in video_frames]
    ortho_window = "2 - highlighted boundary in orthophoto"
    video_window = "3 - click points along this boundary"
    view = {"position": 0}
    active_clicks = {"items": []}

    def video_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            position = view["position"]
            scale = resized_frames[position][1]
            point = np.array([x / scale, y / scale], dtype=np.float64)
            active_clicks["items"].append((frame_indices[position], point))

    cv2.namedWindow(ortho_window, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(video_window, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(ortho_window, 10, 60)
    cv2.moveWindow(video_window, 890, 60)
    cv2.setMouseCallback(video_window, video_mouse)
    line_observations = []
    for edge_index in range(4):
        active_clicks["items"] = []
        while True:
            position = view["position"]
            frame_index = frame_indices[position]
            video_base, video_scale = resized_frames[position]
            ortho_display = ortho_base.copy()
            polygon = local_corners * ortho_scale
            cv2.polylines(ortho_display, [polygon.astype(np.int32)], True,
                          (120, 120, 120), 2, cv2.LINE_AA)
            edge = np.asarray([
                polygon[edge_index], polygon[(edge_index + 1) % 4]
            ], dtype=np.int32)
            cv2.line(ortho_display, tuple(edge[0]), tuple(edge[1]),
                     (0, 255, 255), 5, cv2.LINE_AA)
            for corner_index, corner in enumerate(polygon):
                numbered_marker(ortho_display, corner, corner_index + 1,
                                (0, 255, 255))

            video_display = video_base.copy()
            visible_clicks = [item for item in active_clicks["items"]
                              if item[0] == frame_index]
            for number, (_source_frame, point) in enumerate(visible_clicks, start=1):
                numbered_marker(video_display, point * video_scale,
                                number, (0, 255, 255))
            if len(visible_clicks) >= 2:
                visible_points = np.asarray(
                    [item[1] * video_scale for item in visible_clicks],
                    dtype=np.float32,
                )
                vx, vy, x0_line, y0_line = cv2.fitLine(
                    visible_points, cv2.DIST_L2, 0, 0.01, 0.01).reshape(4)
                extent = 2 * max(video_display.shape[:2])
                start = (int(x0_line - extent * vx), int(y0_line - extent * vy))
                end = (int(x0_line + extent * vx), int(y0_line + extent * vy))
                cv2.line(video_display, start, end,
                         (255, 255, 0), 2, cv2.LINE_AA)
            ortho_display = draw_header(
                ortho_display,
                f"Boundary {edge_index + 1}/4: corner {edge_index + 1} -> "
                f"{(edge_index + 1) % 4 + 1}",
            )
            video_display = draw_header(
                video_display,
                f"Frame {frame_index}: click along highlighted boundary | A/D views | "
                f"U undo | Enter finish ({len(active_clicks['items'])} points)",
            )
            cv2.imshow(ortho_window, ortho_display)
            cv2.imshow(video_window, video_display)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("a"), ord("A")):
                view["position"] = max(0, position - 1)
            elif key in (ord("d"), ord("D")):
                view["position"] = min(len(video_frames) - 1, position + 1)
            elif key in (ord("u"), ord("U")) and active_clicks["items"]:
                active_clicks["items"].pop()
            elif key in (10, 13) and len(active_clicks["items"]) >= 2:
                break
            elif key == 27:
                cv2.destroyAllWindows()
                raise SystemExit("calibration aborted")
        line_observations.append(list(active_clicks["items"]))
    cv2.destroyWindow(ortho_window)
    cv2.destroyWindow(video_window)
    return line_observations


def fit_homogeneous_line(points):
    """Fit ax + by + c = 0 to 2-D points and return line plus residuals."""
    points = np.asarray(points, dtype=np.float64)
    centroid = points.mean(axis=0)
    _u, _s, vh = np.linalg.svd(points - centroid)
    direction = vh[0]
    normal = np.array([-direction[1], direction[0]])
    normal /= np.linalg.norm(normal)
    line = np.array([normal[0], normal[1], -normal @ centroid])
    residuals = np.abs(np.column_stack([points, np.ones(len(points))]) @ line)
    return line, residuals


def intersect_lines(first, second):
    point = np.cross(first, second)
    if abs(point[2]) < 1e-8:
        raise SystemExit("two neighbouring boundary lines are nearly parallel")
    return point[:2] / point[2]


def estimate_view_registration(source_frame, target_frame):
    """Estimate source-pixel -> target-pixel for two neighbouring views."""
    orb = cv2.ORB_create(nfeatures=6000)
    source_gray = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
    source_kp, source_des = orb.detectAndCompute(source_gray, None)
    target_kp, target_des = orb.detectAndCompute(target_gray, None)
    if source_des is None or target_des is None:
        return None, 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(source_des, target_des, k=2)
    good = [first for pair in matches if len(pair) == 2
            for first, second in [pair] if first.distance < 0.75 * second.distance]
    if len(good) < 20:
        return None, 0
    source = np.float32([source_kp[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
    target = np.float32([target_kp[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    return (H, int(mask.sum())) if H is not None else (None, 0)


def anchor_to_base_transforms(video_frames, frame_indices):
    """Register consecutive anchor views and express every view in the first."""
    transforms = {frame_indices[0]: np.eye(3, dtype=np.float64)}
    qualities = []
    for position in range(1, len(frame_indices)):
        previous_index = frame_indices[position - 1]
        current_index = frame_indices[position]
        H_previous_to_current, inliers = estimate_view_registration(
            video_frames[position - 1], video_frames[position])
        if H_previous_to_current is None or inliers < 30:
            raise SystemExit(
                f"views {previous_index}->{current_index}: registration failed "
                f"({inliers} inliers). Choose an additional overlapping frame between them.")
        transforms[current_index] = (
            transforms[previous_index] @ np.linalg.inv(H_previous_to_current))
        transforms[current_index] /= transforms[current_index][2, 2]
        qualities.append({
            "from_frame": previous_index,
            "to_frame": current_index,
            "inliers": inliers,
        })
        print(f"  view registration {previous_index}->{current_index}: {inliers} inliers")
    return transforms, qualities


def main():
    parser = argparse.ArgumentParser(
        description="Anchor calibration from the four real pitch boundary lines")
    parser.add_argument("video")
    parser.add_argument("orthophoto")
    parser.add_argument("--frames", required=True,
                        help="comma-separated absolute anchor frames")
    parser.add_argument("--length", type=float, required=True)
    parser.add_argument("--width", type=float, required=True)
    parser.add_argument("--goal", type=float, default=5.0)
    parser.add_argument("--name", default="pitch_ortho")
    parser.add_argument("--reuse-clicks", default=None,
                        help="reuse pitch corners and boundary clicks from a previous JSON")
    args = parser.parse_args()

    ortho = cv2.imread(args.orthophoto)
    if ortho is None:
        raise SystemExit(f"orthophoto is not readable: {args.orthophoto}")
    frame_indices = [int(value.strip()) for value in args.frames.split(",")
                     if value.strip()]
    cap = cv2.VideoCapture(args.video)
    video_frames = []
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"video frame {frame_index} is not readable")
        video_frames.append(frame)
    cap.release()

    reused_calibration = None
    if args.reuse_clicks:
        reused_calibration = json.loads(
            Path(args.reuse_clicks).read_text(encoding="utf-8"))
        pitch_corners_ortho = np.asarray(
            reused_calibration["pitch_corners_ortho_px"], dtype=np.float64)
        print(f"Step 1: reuse orthophoto corners from {args.reuse_clicks}.")
    else:
        print("Step 1: select the four corners of the target pitch in the orthophoto.")
        print("The start corner does not matter; only keep the order clockwise.")
        pitch_corners_ortho = collect_pitch_corners(ortho)
    H_ortho_to_pitch = ortho_to_pitch_homography(
        pitch_corners_ortho, args.length, args.width)
    ortho_crop_box = crop_around(pitch_corners_ortho, ortho.shape)

    pitch = PitchModel(laenge=args.length, breite=args.width,
                       tor_breite=args.goal, box_tiefe=0.0, box_breite=0.0,
                       kreis_radius=0.0, mittellinie=False)
    points = []
    anchor_quality = []
    calibration_dir = Path("data/calibration")
    calibration_dir.mkdir(parents=True, exist_ok=True)

    print("Step 2: connect neighbouring views.")
    anchor_transforms, registration_quality = anchor_to_base_transforms(
        video_frames, frame_indices)

    if reused_calibration is not None:
        print(f"Step 3: reuse boundary clicks from {args.reuse_clicks}.")
        boundary_observations = [
            [(int(item["frame"]), np.asarray(item["clicked_px"], dtype=np.float64))
             for item in line["points"]]
            for line in reused_calibration["boundary_line_observations"]
        ]
    else:
        print("Step 3: mark points along each of the four outer boundary lines.")
        print("Use A/D to change view. Add at least two, preferably 4-6 points per line.")
        boundary_observations = collect_boundary_lines(
            ortho, ortho_crop_box, pitch_corners_ortho,
            video_frames, frame_indices)

    base_lines = []
    line_fit_quality = []
    serialized_line_observations = []
    for edge_index, observations in enumerate(boundary_observations):
        base_points = []
        serialized = []
        for source_frame, source_point in observations:
            base_point = cv2.perspectiveTransform(
                np.asarray([[source_point]], dtype=np.float64),
                anchor_transforms[source_frame],
            ).reshape(2)
            base_points.append(base_point)
            serialized.append({
                "frame": source_frame,
                "clicked_px": source_point.tolist(),
                "base_px": base_point.tolist(),
            })
        line, residuals = fit_homogeneous_line(base_points)
        base_lines.append(line)
        serialized_line_observations.append({
            "edge": edge_index + 1,
            "from_corner": edge_index + 1,
            "to_corner": (edge_index + 1) % 4 + 1,
            "points": serialized,
        })
        line_fit_quality.append({
            "edge": edge_index + 1,
            "points": len(base_points),
            "mean_residual_px": float(residuals.mean()),
            "max_residual_px": float(residuals.max()),
        })
        print(f"  boundary {edge_index + 1}: {len(base_points)} points, "
              f"mean line error {residuals.mean():.1f}px")

    base_corners = np.asarray([
        intersect_lines(base_lines[(index - 1) % 4], base_lines[index])
        for index in range(4)
    ], dtype=np.float64)
    pitch_corner_meters = cv2.perspectiveTransform(
        pitch_corners_ortho.reshape(-1, 1, 2), H_ortho_to_pitch
    ).reshape(-1, 2)

    for frame_index, frame in zip(frame_indices, video_frames):
        base_to_anchor = np.linalg.inv(anchor_transforms[frame_index])
        video_corners = cv2.perspectiveTransform(
            base_corners.reshape(-1, 1, 2), base_to_anchor
        ).reshape(-1, 2)
        H_pitch_to_video, _ = cv2.findHomography(
            pitch_corner_meters.reshape(-1, 1, 2),
            video_corners.reshape(-1, 1, 2), 0)
        if H_pitch_to_video is None:
            raise SystemExit(
                f"frame {frame_index}: fit failed; re-check the boundary lines")
        projected = cv2.perspectiveTransform(
            pitch_corner_meters.reshape(-1, 1, 2), H_pitch_to_video
        ).reshape(-1, 2)
        errors = np.linalg.norm(projected - video_corners, axis=1)
        hull = cv2.convexHull(pitch_corner_meters.astype(np.float32))
        coverage = cv2.contourArea(hull) / max(args.length * args.width, 1e-6)
        anchor_quality.append({
            "frame": frame_index,
            "points": 4,
            "mean_reprojection_error_px": float(errors.mean()),
            "max_reprojection_error_px": float(errors.max()),
            "model_hull_coverage": float(coverage),
        })
        for corner_index, (video_point, meter) in enumerate(
                zip(video_corners, pitch_corner_meters)):
            points.append({
                "name": f"Pitch corner {corner_index + 1}",
                "clicked_view": f"Frame {frame_index}",
                "clicked_px": video_point.tolist(),
                "ortho_px": pitch_corners_ortho[corner_index].tolist(),
                "meter": meter.tolist(),
                "derived_from_boundary_lines": True,
            })
        overlay = pitch.draw_overlay(frame, H_pitch_to_video,
                                     color=(0, 255, 255), thickness=3)
        overlay_path = calibration_dir / f"{args.name}_check_frame_{frame_index}.jpg"
        cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 94])
        print(f"  frame {frame_index}: four line intersections, "
              f"mean error {errors.mean():.1f}px, "
              f"coverage {100 * coverage:.1f}% -> {overlay_path}")

    calibration = {
        "video": portable_path(args.video),
        "frame": frame_indices[len(frame_indices) // 2],
        "calibration_mode": "four_boundary_lines_with_registered_anchors",
        "orthophoto": portable_path(args.orthophoto),
        "pitch": pitch.to_dict(),
        "pitch_corners_ortho_px": pitch_corners_ortho.tolist(),
        "H_ortho_to_pitch": H_ortho_to_pitch.tolist(),
        "points": points,
        "anchor_quality": anchor_quality,
        "view_registration_quality": registration_quality,
        "boundary_line_observations": serialized_line_observations,
        "line_fit_quality": line_fit_quality,
        "reused_clicks_from": (portable_path(args.reuse_clicks)
                               if args.reuse_clicks else None),
    }
    output = calibration_dir / f"{args.name}.json"
    output.write_text(json.dumps(calibration, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(f"\ncalibration: {output}")
    print("Inspect all check images before running full-video localization.")


if __name__ == "__main__":
    main()
