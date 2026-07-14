"""Ballkandidaten aus einem Video mit YOLO erkennen.

Das Skript ist bewusst vom Personen-Tracking getrennt: Für den winzigen Ball
verwenden wir eine niedrige Konfidenzschwelle und entfernen Fehlkandidaten erst
danach über Platzgeometrie und zeitliche Konsistenz. Die Ausgabe enthält neben
den Originalpixeln auch auf eine Referenzauflösung skalierte Mittelpunktpixel.
Damit kann das synchrone 1080p-Video direkt mit der vorhandenen 720p-
Lokalisierung kombiniert werden.
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


BALL_CLASS_ID = 32  # COCO: sports ball


def main():
    parser = argparse.ArgumentParser(description="YOLO-Ballkandidaten erkennen")
    parser.add_argument("video")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--conf", type=float, default=0.03)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default=None,
                        help="z.B. 0 für die erste GPU oder cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--start", type=int, default=0,
                        help="Erster Frame im Eingabevideo")
    parser.add_argument("--end", type=int, default=None,
                        help="Exklusiver letzter Frame im Eingabevideo")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--frame-offset", type=int, default=0,
                        help="Wird für den ausgegebenen Analyseframe addiert")
    parser.add_argument("--reference-width", type=int, default=1280)
    parser.add_argument("--reference-height", type=int, default=720)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    video_path = Path(args.video)
    output_path = (Path(args.output) if args.output else
                   Path(__file__).resolve().parent.parent / "data" / "output" /
                   f"{video_path.stem}_ball_candidates.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Video konnte nicht geöffnet werden: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    end = min(args.end if args.end is not None else frame_count, frame_count)
    if not 0 <= args.start < end:
        raise SystemExit(f"Ungültiger Framebereich: {args.start}..{end}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    model = YOLO(args.model)
    predict_args = {
        "imgsz": args.imgsz,
        "classes": [BALL_CLASS_ID],
        "conf": args.conf,
        "iou": args.iou,
        "verbose": False,
        "batch": args.batch_size,
    }
    if args.device is not None:
        predict_args["device"] = args.device

    fields = ["frame", "source_frame", "x1", "y1", "x2", "y2",
              "x_ref", "y_ref", "conf"]
    processed = detections = 0
    started = time.time()
    next_frame = args.start

    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()

        while next_frame < end:
            images, source_frames = [], []
            while len(images) < args.batch_size and next_frame < end:
                current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                ok, image = cap.read()
                if not ok:
                    next_frame = end
                    break
                if current == next_frame:
                    images.append(image)
                    source_frames.append(current)
                    next_frame += args.stride
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) < next_frame:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)

            if not images:
                break
            results = model.predict(images, **predict_args)
            for source_frame, result in zip(source_frames, results):
                processed += 1
                if result.boxes is None:
                    continue
                for box, confidence in zip(result.boxes.xyxy.cpu().numpy(),
                                           result.boxes.conf.cpu().numpy()):
                    x1, y1, x2, y2 = map(float, box)
                    writer.writerow({
                        "frame": source_frame + args.frame_offset,
                        "source_frame": source_frame,
                        "x1": f"{x1:.3f}",
                        "y1": f"{y1:.3f}",
                        "x2": f"{x2:.3f}",
                        "y2": f"{y2:.3f}",
                        "x_ref": f"{(x1 + x2) / 2 * args.reference_width / width:.3f}",
                        "y_ref": f"{(y1 + y2) / 2 * args.reference_height / height:.3f}",
                        "conf": f"{float(confidence):.6f}",
                    })
                    detections += 1

            if processed % 500 < len(images):
                elapsed = time.time() - started
                total = (end - args.start + args.stride - 1) // args.stride
                eta = elapsed / processed * max(total - processed, 0)
                print(f"{processed}/{total} Frames, {detections} Kandidaten, "
                      f"ETA {eta / 60:.1f} min", flush=True)

    cap.release()
    elapsed = time.time() - started
    print(f"Fertig: {processed} Frames, {detections} Ballkandidaten in "
          f"{elapsed / 60:.1f} min -> {output_path}")


if __name__ == "__main__":
    main()
