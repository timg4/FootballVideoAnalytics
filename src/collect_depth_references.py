"""Collect ortho/video ground-point pairs for both pitch halves."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibrate_from_ortho import draw_header, numbered_marker, resized


def portable_path(value):
    path = Path(value).resolve()
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)


def collect_half(ortho, pitch_corners, H_ortho_to_pitch,
                 video_frames, frame_indices, half_name, split_x, length):
    ortho_base, ortho_scale = resized(ortho, 850, 760)
    resized_frames = [resized(frame, 900, 760) for frame in video_frames]
    ortho_window = f"depth reference - orthophoto - {half_name}"
    video_window = f"depth reference - video - {half_name}"
    state = {"view": 0, "pending": None}
    pairs = []

    def ortho_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or state["pending"] is not None:
            return
        point = np.array([x / ortho_scale, y / ortho_scale], dtype=np.float64)
        meter = cv2.perspectiveTransform(
            point.reshape(1, 1, 2), H_ortho_to_pitch).reshape(2)
        is_left = meter[0] <= split_x
        if ((half_name == "LEFT" and is_left) or
                (half_name == "RIGHT" and not is_left)):
            state["pending"] = point

    def video_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and state["pending"] is not None:
            scale = resized_frames[state["view"]][1]
            video_point = np.array([x / scale, y / scale], dtype=np.float64)
            pairs.append({
                "ortho_px": state["pending"].tolist(),
                "frame": frame_indices[state["view"]],
                "video_px": video_point.tolist(),
            })
            state["pending"] = None

    cv2.namedWindow(ortho_window, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(video_window, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(ortho_window, 10, 60)
    cv2.moveWindow(video_window, 890, 60)
    cv2.setMouseCallback(ortho_window, ortho_mouse)
    cv2.setMouseCallback(video_window, video_mouse)
    while True:
        position = state["view"]
        frame_index = frame_indices[position]
        video_base, video_scale = resized_frames[position]
        ortho_display = ortho_base.copy()
        polygon = pitch_corners * ortho_scale
        far_mid = (polygon[1] + polygon[2]) / 2
        near_mid = (polygon[0] + polygon[3]) / 2
        if half_name == "LEFT":
            half_polygon = np.array([polygon[0], polygon[1], far_mid, near_mid])
        else:
            half_polygon = np.array([near_mid, far_mid, polygon[2], polygon[3]])
        tint = ortho_display.copy()
        cv2.fillPoly(tint, [half_polygon.astype(np.int32)], (40, 150, 40))
        ortho_display = cv2.addWeighted(ortho_display, 0.72, tint, 0.28, 0)
        cv2.polylines(ortho_display, [polygon.astype(np.int32)], True,
                      (0, 255, 255), 3, cv2.LINE_AA)
        for number, pair in enumerate(pairs, start=1):
            numbered_marker(ortho_display,
                            np.asarray(pair["ortho_px"]) * ortho_scale,
                            number, (0, 255, 0))
        if state["pending"] is not None:
            numbered_marker(ortho_display, state["pending"] * ortho_scale,
                            len(pairs) + 1, (0, 165, 255))

        video_display = video_base.copy()
        visible = [pair for pair in pairs if pair["frame"] == frame_index]
        for number, pair in enumerate(visible, start=1):
            numbered_marker(video_display,
                            np.asarray(pair["video_px"]) * video_scale,
                            number, (0, 255, 0))
        phase = ("click SAME fixed ground point in VIDEO"
                 if state["pending"] is not None
                 else "click a fixed line intersection in ORTHOPHOTO")
        ortho_display = draw_header(
            ortho_display, f"{half_name} HALF: {phase}")
        video_display = draw_header(
            video_display,
            f"Frame {frame_index} | A/D views | U undo | Enter finish "
            f"({len(pairs)} pairs, minimum 2)",
        )
        cv2.imshow(ortho_window, ortho_display)
        cv2.imshow(video_window, video_display)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("a"), ord("A")) and state["pending"] is None:
            state["view"] = max(0, position - 1)
        elif key in (ord("d"), ord("D")) and state["pending"] is None:
            state["view"] = min(len(video_frames) - 1, position + 1)
        elif key in (ord("u"), ord("U")):
            if state["pending"] is not None:
                state["pending"] = None
            elif pairs:
                pairs.pop()
        elif key in (10, 13) and len(pairs) >= 2 and state["pending"] is None:
            break
        elif key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("reference collection aborted")
    cv2.destroyWindow(ortho_window)
    cv2.destroyWindow(video_window)
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Collect metric depth references")
    parser.add_argument("calibration_json")
    parser.add_argument("--output", default="data/calibration/video_project_depth_refs.json")
    args = parser.parse_args()

    calibration = json.loads(
        Path(args.calibration_json).read_text(encoding="utf-8"))
    ortho = cv2.imread(calibration["orthophoto"])
    if ortho is None:
        raise SystemExit("orthophoto is not readable")
    pitch_corners = np.asarray(calibration["pitch_corners_ortho_px"],
                               dtype=np.float64)
    H_ortho_to_pitch = np.asarray(calibration["H_ortho_to_pitch"],
                                  dtype=np.float64)
    length = float(calibration["pitch"]["laenge"])
    split_x = length / 2
    frame_indices = sorted({
        int(item["frame"])
        for boundary in calibration["boundary_line_observations"]
        for item in boundary["points"]
    })
    cap = cv2.VideoCapture(calibration["video"])
    video_frames = []
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"frame {frame_index} is not readable")
        video_frames.append(frame)
    cap.release()

    references = {
        "source_calibration": portable_path(args.calibration_json),
        "left": collect_half(
            ortho, pitch_corners, H_ortho_to_pitch,
            video_frames, frame_indices, "LEFT", split_x, length),
        "right": collect_half(
            ortho, pitch_corners, H_ortho_to_pitch,
            video_frames, frame_indices, "RIGHT", split_x, length),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(references, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(f"depth references: {output}")


if __name__ == "__main__":
    main()
