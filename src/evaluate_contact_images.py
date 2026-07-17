"""Schneller Crop-Test fuer die visuelle Teamzuordnung am Ballkontakt.

Verwendet die bereits erzeugten Pass-Review-Bilder und liest deshalb das
Vollvideo nicht erneut. Das ist ein Diagnosewerkzeug: Es prueft, wie stark die
Teamfarbe in unterschiedlich engen ballzentrierten Einzelbildern erkennbar ist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.models import ResNet18_Weights, resnet18

from contact_team_model import crop_with_padding, evaluate


def main():
    parser = argparse.ArgumentParser(description="Kontakt-Crops vergleichen")
    parser.add_argument("pass_review_csv")
    parser.add_argument("ball_track_csv")
    parser.add_argument("review_image_dir")
    parser.add_argument("--output", default="data/output/video_project_contact_crop_evaluation.json")
    parser.add_argument("--crop-sizes", default="80,112,144,192,240",
                        help="Crop-Groessen in den 640x360 Review-Panels")
    args = parser.parse_args()

    review = pd.read_csv(args.pass_review_csv)
    ball = pd.read_csv(args.ball_track_csv).set_index(["ball_tracklet", "frame"])
    confirmed = review.loc[review["review_status"].eq("pass")]
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    crop_sizes = [int(value) for value in args.crop_sizes.split(",")]
    report = {}
    for crop_size in crop_sizes:
        images, labels, groups = [], [], []
        for event, (original_index, row) in enumerate(confirmed.iterrows()):
            card = cv2.imread(str(Path(args.review_image_dir) /
                                  f"pass_{original_index:03d}.jpg"))
            if card is None:
                raise SystemExit(f"Review-Bild fehlt: pass_{original_index:03d}.jpg")
            for side, frame_column, team_column, panel_offset in (
                ("von", "abgabe_frame", "review_von_team", 0),
                ("zu", "annahme_frame", "review_zu_team", 1280),
            ):
                frame, tracklet = int(row[frame_column]), int(row["ball_tracklet"])
                xy = ball.loc[(tracklet, frame), ["x_ref", "y_ref"]].to_numpy(float)
                # Review-Panels sind halb so gross wie die 1280x720-Referenz.
                cx = panel_offset + xy[0] * 0.5
                cy = xy[1] * 0.5 - crop_size * 0.20
                crop = crop_with_padding(card, cx, cy, crop_size)
                crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
                images.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                labels.append(int(float(row[team_column])))
                groups.append(event)

        batch = torch.from_numpy(np.stack(images)).float().permute(0, 3, 1, 2) / 255.0
        batch = (batch - mean) / std
        features = []
        with torch.inference_mode():
            for start in range(0, len(batch), 16):
                features.append(model(batch[start:start + 16]).numpy())
        features = np.concatenate(features)
        _, _, metrics = evaluate(features, np.asarray(labels), np.asarray(groups))
        report[str(crop_size)] = metrics
        print(f"Crop {crop_size}: accuracy={metrics['accuracy']:.3f}, "
              f"balanced={metrics['balanced_accuracy']:.3f}", flush=True)

    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Auswertung: {args.output}")


if __name__ == "__main__":
    main()
