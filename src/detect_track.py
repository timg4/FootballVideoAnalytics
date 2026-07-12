"""Phase 1: Spieler-Erkennung und Tracking auf einem Videoclip.

Nimmt ein Video (z.B. Veo-Export), erkennt Personen mit YOLO, verfolgt sie
mit ByteTrack und schreibt ein annotiertes Video mit Track-IDs nach data/output/.
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
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video nicht gefunden: {video_path}")

    out_dir = Path(__file__).resolve().parent.parent / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else out_dir / f"{video_path.stem}_tracked.mp4"

    video_info = sv.VideoInfo.from_video_path(str(video_path))
    start_frame = int(args.start * video_info.fps)
    end_frame = start_frame + int(args.seconds * video_info.fps) if args.seconds \
        else video_info.total_frames
    total = end_frame - start_frame
    print(f"Video: {video_info.width}x{video_info.height} @ {video_info.fps} fps, "
          f"verarbeite {total} Frames")

    model = YOLO(args.model)
    tracker = sv.ByteTrack(frame_rate=video_info.fps)
    ellipse_annotator = sv.EllipseAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.BOTTOM_CENTER,
                                        text_scale=0.4)

    # Tracking-Rohdaten als CSV, damit spätere Analyse-Schritte (Teams, Heatmaps,
    # Statistiken) nicht die teure YOLO-Erkennung wiederholen müssen
    csv_path = output_path.with_suffix(".csv")

    frames = sv.get_video_frames_generator(str(video_path), start=start_frame,
                                           end=end_frame)
    with sv.VideoSink(str(output_path), video_info) as sink, \
            open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame", "tracker_id", "x1", "y1", "x2", "y2", "conf"])

        for i, frame in enumerate(frames):
            result = model(frame, conf=args.conf, imgsz=args.imgsz,
                           classes=[PERSON_CLASS_ID], verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections)

            # Absolute Frame-Nummer im Video, damit nachgelagerte Schritte
            # (Teams, Registrierung, Meter) bei --start nicht verrutschen
            for (x1, y1, x2, y2), tid, conf in zip(
                    detections.xyxy, detections.tracker_id, detections.confidence):
                writer.writerow([start_frame + i, tid, f"{x1:.1f}", f"{y1:.1f}",
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
