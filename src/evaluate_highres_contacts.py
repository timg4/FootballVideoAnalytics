"""Kontaktteam mit echter 1080p-Personenerkennung an 66 Kontaktframes testen.

Das Skript ist bewusst eine lokale Diagnose, kein Vollvideo-Tracking. YOLO wird
nur an den manuell bestaetigten Passkontakten mit imgsz=1920 ausgefuehrt. So
laesst sich vor einem teuren Colab-Lauf messen, ob schaerfere Spielerboxen die
Kontaktzuordnung tatsaechlich verbessern.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from team_assign import color_features, torso_crop


def main():
    parser = argparse.ArgumentParser(description="1080p-Kontaktboxen evaluieren")
    parser.add_argument("video_1080")
    parser.add_argument("pass_review_csv")
    parser.add_argument("ball_track_csv")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--frame-offset", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", default="data/output/video_project_highres_contact_evaluation.json")
    args = parser.parse_args()

    review = pd.read_csv(args.pass_review_csv)
    passes = review.loc[review["review_status"].eq("pass")]
    ball = pd.read_csv(args.ball_track_csv).set_index(["ball_tracklet", "frame"])
    contacts = []
    for event, (_, row) in enumerate(passes.iterrows()):
        for side, frame_column, auto_column, truth_column in (
            ("von", "abgabe_frame", "von_team", "review_von_team"),
            ("zu", "annahme_frame", "zu_team", "review_zu_team"),
        ):
            contacts.append({
                "event": event, "side": side, "frame": int(row[frame_column]),
                "tracklet": int(row["ball_tracklet"]),
                "auto": int(row[auto_column]),
                "truth": int(float(row[truth_column])),
            })

    cap = cv2.VideoCapture(args.video_1080)
    if not cap.isOpened():
        raise SystemExit(f"Video konnte nicht geoeffnet werden: {args.video_1080}")
    model = YOLO(args.model)
    detected = []
    for start in range(0, len(contacts), args.batch_size):
        batch_contacts = contacts[start:start + args.batch_size]
        images, ball_pixels = [], []
        for contact in batch_contacts:
            cap.set(cv2.CAP_PROP_POS_FRAMES, contact["frame"] + args.frame_offset)
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f"Frame fehlt: {contact['frame'] + args.frame_offset}")
            xy = ball.loc[(contact["tracklet"], contact["frame"]),
                          ["x_ref", "y_ref"]].to_numpy(float)
            ball_pixels.append((xy[0] * image.shape[1] / 1280,
                                xy[1] * image.shape[0] / 720))
            images.append(image)
        results = model.predict(images, imgsz=args.imgsz, classes=[0], conf=0.10,
                                verbose=False, device="cpu", batch=args.batch_size)
        for contact, image, ball_xy, result in zip(batch_contacts, images,
                                                    ball_pixels, results):
            candidates = []
            for box, confidence in zip(result.boxes.xyxy.cpu().numpy(),
                                       result.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = box
                height = max(y2 - y1, 1.0)
                # Horizontaler Abstand zur Box statt nur zum Boxzentrum: Der
                # Ball kann beim Kontakt seitlich direkt neben dem Fuss liegen.
                dx = max(x1 - ball_xy[0], 0.0, ball_xy[0] - x2)
                dy = abs(y2 - ball_xy[1])
                distance = float(np.hypot(dx, dy) / height)
                crop = torso_crop(image, box)
                if crop is None:
                    continue
                feature, raw = color_features(crop)
                candidates.append({
                    "distance": distance, "confidence": float(confidence),
                    "feature": feature, "raw": raw,
                    "box": [float(value) for value in box],
                })
            candidates.sort(key=lambda item: item["distance"])
            detected.append({**contact, "candidates": candidates[:5]})
        print(f"YOLO-Kontakte: {min(start + args.batch_size, len(contacts))}/"
              f"{len(contacts)}", flush=True)
    cap.release()

    # Teamfarb-Zentren nur aus anderen Ereignissen und nur aus Kontakten, bei
    # denen die bisherige Zuordnung laut Review korrekt war. Damit gelangen
    # keine Testlabels oder bekannte Fehlkontakte in den jeweiligen Fold.
    predictions = []
    for sample in detected:
        training = [row for row in detected
                    if row["event"] != sample["event"] and row["auto"] == row["truth"]
                    and row["candidates"]]
        centers = {}
        for team in (0, 1):
            values = [row["candidates"][0]["feature"] for row in training
                      if row["truth"] == team]
            centers[team] = np.median(values, axis=0)
        if not sample["candidates"]:
            prediction = -1
        else:
            nearest = sample["candidates"][0]
            prediction = min((0, 1), key=lambda team:
                             np.linalg.norm(nearest["feature"] - centers[team]))
        predictions.append(prediction)

    truth = np.asarray([row["truth"] for row in detected])
    predictions = np.asarray(predictions)
    valid = predictions >= 0
    recalls = {team: float((predictions[truth == team] == team).mean())
               for team in (0, 1)}
    report = {
        "contacts": len(detected), "imgsz": args.imgsz,
        "accuracy": float((predictions[valid] == truth[valid]).mean()),
        "balanced_accuracy": float(np.mean(list(recalls.values()))),
        "recall_green": recalls[0], "recall_blue": recalls[1],
        "no_detection": int((~valid).sum()),
        "baseline_old_accuracy": float(np.mean([row["auto"] == row["truth"]
                                                  for row in detected])),
        "rows": [
            {"event": row["event"], "side": row["side"], "frame": row["frame"],
             "truth": row["truth"], "old_auto": row["auto"],
             "highres_prediction": int(prediction),
             "nearest_normalized_distance": (row["candidates"][0]["distance"]
                                                if row["candidates"] else None)}
            for row, prediction in zip(detected, predictions)
        ],
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"},
                     indent=2))
    print(f"Auswertung: {args.output}")


if __name__ == "__main__":
    main()
