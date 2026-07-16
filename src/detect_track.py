"""Detect people in a video clip and track them across the frames.

Takes a video (e.g. a Veo export), detects people with YOLO, tracks them with
either ByteTrack or BoT-SORT+ReID, and writes the tracking CSV and video.
"""

import argparse
import csv
from pathlib import Path

import supervision as sv
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO class "person"


def main():
    parser = argparse.ArgumentParser(description="Player detection and tracking")
    parser.add_argument("video", help="path to the input video")
    parser.add_argument("--start", type=float, default=0,
                        help="start time in seconds")
    parser.add_argument("--seconds", type=float, default=None,
                        help="process only N seconds from --start (default: to the end)")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="YOLO model: yolo11n.pt (fast) to yolo11m.pt (more accurate)")
    parser.add_argument("--conf", type=float, default=0.3, help="minimum confidence")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="analysis resolution; higher (e.g. 1280) finds small/distant players")
    parser.add_argument("--output", default=None, help="output path (default: data/output/)")
    parser.add_argument("--stride", type=int, default=1,
                        help="process only every Nth frame (2 = half the time, fine for stats)")
    parser.add_argument("--device", default=None,
                        help="compute device for YOLO, e.g. 0 (first GPU) or cpu (default: automatic)")
    parser.add_argument("--tracker", choices=["bytetrack", "botsort-reid"],
                        default="bytetrack",
                        help="tracker: the existing ByteTrack or BoT-SORT with ReID")
    args = parser.parse_args()

    if args.stride < 1:
        parser.error("--stride must be at least 1")

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"video not found: {video_path}")

    out_dir = Path(__file__).resolve().parent.parent / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else out_dir / f"{video_path.stem}_tracked.mp4"

    video_info = sv.VideoInfo.from_video_path(str(video_path))
    fps_orig = video_info.fps
    start_frame = int(args.start * fps_orig)
    end_frame = start_frame + int(args.seconds * fps_orig) if args.seconds \
        else video_info.total_frames
    end_frame = min(end_frame, video_info.total_frames)
    total = (end_frame - start_frame + args.stride - 1) // args.stride
    # keep the annotated video at real-time speed despite the stride
    video_info.fps = fps_orig / args.stride
    print(f"Video: {video_info.width}x{video_info.height} @ {fps_orig} fps, "
          f"processing {total} frames (stride {args.stride}, tracker {args.tracker})")

    model = YOLO(args.model)
    tracker = (sv.ByteTrack(frame_rate=fps_orig / args.stride)
               if args.tracker == "bytetrack" else None)
    botsort_config = (Path(__file__).resolve().parent.parent /
                      "configs" / "botsort_reid.yaml")
    ellipse_annotator = sv.EllipseAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.BOTTOM_CENTER,
                                        text_scale=0.4)

    # raw tracking data as CSV, so the later analysis steps (teams, heatmaps,
    # stats) don't have to repeat the expensive YOLO detection
    csv_path = output_path.with_suffix(".csv")

    frames = sv.get_video_frames_generator(str(video_path), start=start_frame,
                                           end=end_frame, stride=args.stride)
    with sv.VideoSink(str(output_path), video_info) as sink, \
            open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame", "tracker_id", "x1", "y1", "x2", "y2", "conf"])

        for i, frame in enumerate(frames):
            predict_args = dict(conf=args.conf, imgsz=args.imgsz,
                                classes=[PERSON_CLASS_ID], verbose=False)
            if args.device is not None:
                predict_args["device"] = args.device
            if args.tracker == "botsort-reid":
                result = model.track(frame, persist=True,
                                     tracker=str(botsort_config),
                                     **predict_args)[0]
                detections = sv.Detections.from_ultralytics(result)
                # in rare initialization frames the tracker has no IDs yet;
                # those boxes must not go into the CSV
                if detections.tracker_id is None:
                    detections = sv.Detections.empty()
            else:
                result = model(frame, **predict_args)[0]
                detections = sv.Detections.from_ultralytics(result)
                detections = tracker.update_with_detections(detections)

            # absolute frame number in the video, so the downstream steps
            # (teams, registration, meters) don't shift when --start is used
            for (x1, y1, x2, y2), tid, conf in zip(
                    detections.xyxy, detections.tracker_id, detections.confidence):
                writer.writerow([start_frame + i * args.stride, tid,
                                 f"{x1:.1f}", f"{y1:.1f}",
                                 f"{x2:.1f}", f"{y2:.1f}", f"{conf:.3f}"])

            labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
            annotated = ellipse_annotator.annotate(frame.copy(), detections)
            annotated = label_annotator.annotate(annotated, detections, labels=labels)
            sink.write_frame(annotated)

            if i % 25 == 0:
                print(f"Frame {i}/{total}, {len(detections)} players in frame", end="\r")

    print(f"\nDone: {output_path}")
    print(f"Tracking data: {csv_path}")


if __name__ == "__main__":
    main()
