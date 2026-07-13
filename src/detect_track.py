"""Phase 1: Spieler-Erkennung und Tracking auf einem Videoclip.

Nimmt ein Video (z.B. Veo-Export), erkennt Personen mit YOLO, verfolgt sie
wahlweise mit ByteTrack oder BoT-SORT+ReID und schreibt Tracking-CSV/Video.
"""

import argparse
import csv
from pathlib import Path

import supervision as sv
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO-Klasse "person"


def main():
    parser = argparse.ArgumentParser(description="Spieler-Erkennung + Tracking (Phase 1)")
    parser.add_argument("video", help="Pfad zum Eingabevideo")
    parser.add_argument("--start", type=float, default=0,
                        help="Startzeitpunkt in Sekunden")
    parser.add_argument("--seconds", type=float, default=None,
                        help="Nur N Sekunden ab --start verarbeiten (Standard: bis zum Ende)")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="YOLO-Modell: yolo11n.pt (schnell) bis yolo11m.pt (genauer)")
    parser.add_argument("--conf", type=float, default=0.3, help="Mindest-Konfidenz")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Analyse-Auflösung; höher (z.B. 1280) findet kleine/ferne Spieler")
    parser.add_argument("--output", default=None, help="Ausgabepfad (Standard: data/output/)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Nur jeden N-ten Frame verarbeiten (2 = halbe Zeit, für Statistiken ok)")
    parser.add_argument("--device", default=None,
                        help="Rechengerät für YOLO, z.B. 0 (erste GPU) oder cpu (Standard: automatisch)")
    parser.add_argument("--tracker", choices=["bytetrack", "botsort-reid"],
                        default="bytetrack",
                        help="Tracker: bisheriger ByteTrack oder BoT-SORT mit ReID")
    args = parser.parse_args()

    if args.stride < 1:
        parser.error("--stride muss mindestens 1 sein")

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video nicht gefunden: {video_path}")

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
    # Annotiertes Video trotz Stride in Echtzeit-Geschwindigkeit halten
    video_info.fps = fps_orig / args.stride
    print(f"Video: {video_info.width}x{video_info.height} @ {fps_orig} fps, "
          f"verarbeite {total} Frames (Stride {args.stride}, Tracker {args.tracker})")

    model = YOLO(args.model)
    tracker = (sv.ByteTrack(frame_rate=fps_orig / args.stride)
               if args.tracker == "bytetrack" else None)
    botsort_config = (Path(__file__).resolve().parent.parent /
                      "configs" / "botsort_reid.yaml")
    ellipse_annotator = sv.EllipseAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.BOTTOM_CENTER,
                                        text_scale=0.4)

    # Tracking-Rohdaten als CSV, damit spätere Analyse-Schritte (Teams, Heatmaps,
    # Statistiken) nicht die teure YOLO-Erkennung wiederholen müssen
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
                # In seltenen Initialisierungsframes kann der Tracker noch
                # keine IDs liefern; solche Boxen dürfen nicht in die CSV.
                if detections.tracker_id is None:
                    detections = sv.Detections.empty()
            else:
                result = model(frame, **predict_args)[0]
                detections = sv.Detections.from_ultralytics(result)
                detections = tracker.update_with_detections(detections)

            # Absolute Frame-Nummer im Video, damit nachgelagerte Schritte
            # (Teams, Registrierung, Meter) bei --start nicht verrutschen
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
                print(f"Frame {i}/{total} — {len(detections)} Spieler im Bild", end="\r")

    print(f"\nFertig: {output_path}")
    print(f"Tracking-Daten: {csv_path}")


if __name__ == "__main__":
    main()
