"""Visual acceptance check for one directly calibrated anchor frame.

The script deliberately uses the exact pitch boundary without the 0.5 m margin
from pitch_map.py. It is meant to catch a calibration that looks plausible in
aggregate counts but cuts through an active player or the ball.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from pitch_model import PitchModel
from piecewise_pitch import draw_piecewise_overlay, transform_piecewise


def transform_point(point, homography):
    value = np.asarray([[point]], dtype=np.float64)
    return cv2.perspectiveTransform(value, homography).reshape(2)


def main():
    parser = argparse.ArgumentParser(description="Validate one calibrated anchor frame")
    parser.add_argument("video")
    parser.add_argument("calibration_json")
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--tracks-csv")
    parser.add_argument("--ball-track-csv")
    parser.add_argument("--localization",
                        help="piecewise localization NPZ for a non-anchor frame")
    parser.add_argument("--output")
    args = parser.parse_args()

    calibration_path = Path(args.calibration_json)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    pitch = PitchModel(**calibration["pitch"])
    label = f"Frame {args.frame}"
    piecewise = calibration.get("piecewise")
    if piecewise:
        anchor = next((item for item in piecewise["anchors"]
                       if int(item["frame"]) == args.frame), None)
        if anchor is not None:
            H_left = np.asarray(
                anchor["H_anchor_to_pitch_left"], dtype=np.float64)
            H_right = np.asarray(
                anchor["H_anchor_to_pitch_right"], dtype=np.float64)
        elif args.localization:
            with np.load(args.localization) as localization:
                frames = np.asarray(localization["frames"], dtype=np.int64)
                matches = np.flatnonzero(frames == args.frame)
                if len(matches) != 1:
                    raise SystemExit(
                        f"{label} is not present exactly once in localization")
                index = int(matches[0])
                H_left = np.asarray(
                    localization["H_px_to_pitch_left"][index],
                    dtype=np.float64)
                H_right = np.asarray(
                    localization["H_px_to_pitch_right"][index],
                    dtype=np.float64)
            label += " (localized)"
        else:
            raise SystemExit(
                f"{label} is not a calibration anchor; pass --localization")
        split_x = float(piecewise["split_x_m"])
    else:
        points = [point for point in calibration["points"]
                  if point.get("clicked_view") == label]
        if len(points) < 4:
            raise SystemExit(f"{label} has only {len(points)} calibration points")
        pitch_points = np.asarray([point["meter"] for point in points],
                                  dtype=np.float64).reshape(-1, 1, 2)
        image_points = np.asarray([point["clicked_px"] for point in points],
                                  dtype=np.float64).reshape(-1, 1, 2)
        H_pitch_to_px, _ = cv2.findHomography(pitch_points, image_points, 0)
        if H_pitch_to_px is None:
            raise SystemExit("homography fit failed")
        H_px_to_pitch = np.linalg.inv(H_pitch_to_px)

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"video frame {args.frame} is not readable")
    if piecewise:
        result = draw_piecewise_overlay(
            frame, H_left, H_right, split_x,
            pitch.laenge, pitch.breite, thickness=3)
    else:
        result = pitch.draw_overlay(frame, H_pitch_to_px, color=(0, 255, 255),
                                    thickness=3)

    def map_point(point):
        if piecewise:
            mapped, _side = transform_piecewise(
                [point], H_left, H_right, split_x, pitch.laenge, pitch.breite)
            return mapped.reshape(2)
        return transform_point(point, H_px_to_pitch)

    inside_players = outside_players = 0
    if args.tracks_csv:
        with open(args.tracks_csv, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if int(row["frame"]) != args.frame:
                    continue
                foot = ((float(row["x1"]) + float(row["x2"])) / 2,
                        float(row["y2"]))
                x_m, y_m = map_point(foot)
                inside = 0 <= x_m <= pitch.laenge and 0 <= y_m <= pitch.breite
                color = (30, 210, 70) if inside else (30, 30, 240)
                inside_players += int(inside)
                outside_players += int(not inside)
                center = tuple(np.rint(foot).astype(int))
                cv2.circle(result, center, 7, color, -1, cv2.LINE_AA)
                cv2.putText(result, f"#{row['tracker_id']} {x_m:.1f},{y_m:.1f}",
                            (center[0] + 8, center[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1,
                            cv2.LINE_AA)

    ball_inside = None
    if args.ball_track_csv:
        with open(args.ball_track_csv, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if int(row["frame"]) != args.frame:
                    continue
                ball_px = (float(row["x_ref"]), float(row["y_ref"]))
                x_m, y_m = map_point(ball_px)
                ball_inside = bool(0 <= x_m <= pitch.laenge and
                                   0 <= y_m <= pitch.breite)
                color = (255, 180, 0) if ball_inside else (0, 0, 255)
                center = tuple(np.rint(ball_px).astype(int))
                cv2.circle(result, center, 13, color, 3, cv2.LINE_AA)
                cv2.putText(result, f"BALL {x_m:.1f},{y_m:.1f}",
                            (center[0] + 15, center[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                            cv2.LINE_AA)
                break

    banner = (f"Frame {args.frame} | exact boundary | players inside/outside: "
              f"{inside_players}/{outside_players}")
    if ball_inside is not None:
        banner += f" | ball: {'inside' if ball_inside else 'OUTSIDE'}"
    cv2.rectangle(result, (0, 0), (result.shape[1], 34), (15, 23, 42), -1)
    cv2.putText(result, banner, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (255, 255, 255), 1, cv2.LINE_AA)

    output = (Path(args.output) if args.output else
              Path("data/output") /
              f"{Path(args.video).stem}_calibration_frame_{args.frame}.jpg")
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), result, [cv2.IMWRITE_JPEG_QUALITY, 94])
    print(f"validation image: {output}")
    print(banner)


if __name__ == "__main__":
    main()
