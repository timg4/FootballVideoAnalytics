"""Team am Ballkontakt aus kurzen, ballzentrierten Videosequenzen lernen.

Das Skript verwendet nur manuell bestaetigte Paesse als Trainingsdaten. Pro
Pass entstehen zwei Beispiele (Abgabe und Annahme). Die Sequenz wird am Ball
zentriert, damit keine automatisch vermutete Tracker-ID als Wahrheit in das
Modell einfliesst.

Die Qualitaetsmessung ist Leave-one-event-out: Beide Kontakte eines Passes
werden gemeinsam aus dem Training entfernt. Dadurch kann das Modell nicht den
zweiten Kontakt desselben Passes zum Wiedererkennen des ersten benutzen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from torchvision.models.video import R3D_18_Weights, r3d_18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporales Kontaktteam-Modell")
    parser.add_argument("video_1080")
    parser.add_argument("pass_review_csv")
    parser.add_argument("ball_track_csv")
    parser.add_argument("--output-prefix", default="data/output/video_project_contact_team")
    parser.add_argument("--frame-offset", type=int, default=10,
                        help="1080p-Frame = Analyseframe + Offset")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--window", type=int, default=24,
                        help="Frames vor/nach dem Kontakt")
    parser.add_argument("--crop", type=int, default=448,
                        help="Quadratischer Crop in 1080p-Pixeln")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def ball_lookup(ball: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for tracklet, rows in ball.groupby("ball_tracklet"):
        rows = rows.sort_values("frame")
        result[int(tracklet)] = (
            rows["frame"].to_numpy(dtype=np.int32),
            rows[["x_ref", "y_ref"]].to_numpy(dtype=np.float32),
        )
    return result


def position_at(lookup, tracklet: int, frame: int) -> np.ndarray:
    frames, xy = lookup[tracklet]
    index = int(np.searchsorted(frames, frame))
    if index == 0:
        return xy[0]
    if index == len(frames):
        return xy[-1]
    before, after = frames[index - 1], frames[index]
    if after == before:
        return xy[index]
    alpha = (frame - before) / (after - before)
    return xy[index - 1] * (1 - alpha) + xy[index] * alpha


def crop_with_padding(image: np.ndarray, cx: float, cy: float,
                      size: int) -> np.ndarray:
    half = size // 2
    x1, y1 = int(round(cx)) - half, int(round(cy)) - half
    x2, y2 = x1 + size, y1 + size
    left, top = max(0, -x1), max(0, -y1)
    right, bottom = max(0, x2 - image.shape[1]), max(0, y2 - image.shape[0])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    crop = image[y1:y2, x1:x2]
    if left or top or right or bottom:
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right,
                                  cv2.BORDER_REPLICATE)
    return crop


def read_contact(cap: cv2.VideoCapture, lookup, tracklet: int,
                 contact_frame: int, side: str, frame_offset: int,
                 count: int, window: int, crop_size: int) -> np.ndarray:
    # Abgabe: mehr Vorgeschichte; Annahme: mehr Nachgeschichte.
    relative = (np.linspace(-window, 3, count) if side == "von" else
                np.linspace(-3, window, count))
    analysis_frames = np.rint(contact_frame + relative).astype(np.int32)
    source_frames = analysis_frames + frame_offset

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(source_frames.min()))
    wanted = {int(frame): [] for frame in source_frames}
    for i, frame in enumerate(source_frames):
        wanted[int(frame)].append(i)
    decoded: dict[int, np.ndarray] = {}
    for source_frame in range(int(source_frames.min()), int(source_frames.max()) + 1):
        ok, image = cap.read()
        if not ok:
            raise RuntimeError(f"Video-Frame {source_frame} fehlt")
        if source_frame in wanted:
            decoded[source_frame] = image

    sequence = []
    scale_x = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 1280.0
    scale_y = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 720.0
    # Ball liegt am Fuss. Den Crop nach oben verschieben, damit der Oberkoerper
    # des Kontaktspielers enthalten ist und weniger Rasen unterhalb des Balls.
    vertical_shift = crop_size * 0.18
    for analysis_frame, source_frame in zip(analysis_frames, source_frames):
        xy = position_at(lookup, tracklet, int(analysis_frame))
        crop = crop_with_padding(decoded[int(source_frame)],
                                 float(xy[0] * scale_x),
                                 float(xy[1] * scale_y - vertical_shift),
                                 crop_size)
        crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
        sequence.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return np.stack(sequence)


def build_examples(video_path: str, review: pd.DataFrame, ball_lookup_,
                   args: argparse.Namespace):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Video konnte nicht geoeffnet werden: {video_path}")
    labels, groups, metadata, requests = [], [], [], []
    confirmed = review.loc[review["review_status"].eq("pass")].reset_index(drop=True)
    for event, row in confirmed.iterrows():
        for side, frame_column, team_column in (
            ("von", "abgabe_frame", "review_von_team"),
            ("zu", "annahme_frame", "review_zu_team"),
        ):
            team = row[team_column]
            if pd.isna(team) or str(team).strip() == "":
                continue
            contact_frame = int(row[frame_column])
            tracklet = int(row["ball_tracklet"])
            clip_index = len(labels)
            relative = (np.linspace(-args.window, 3, args.frames) if side == "von" else
                        np.linspace(-3, args.window, args.frames))
            analysis_frames = np.rint(contact_frame + relative).astype(np.int32)
            for sequence_index, analysis_frame in enumerate(analysis_frames):
                requests.append((int(analysis_frame + args.frame_offset), clip_index,
                                 sequence_index, int(analysis_frame), tracklet))
            labels.append(int(float(team)))
            groups.append(event)
            metadata.append({
                "event": int(event), "side": side, "frame": contact_frame,
                "ball_tracklet": tracklet, "team": int(float(team)),
            })

    clips = np.empty((len(labels), args.frames, 112, 112, 3), dtype=np.uint8)
    by_source_frame = {}
    for request in requests:
        by_source_frame.setdefault(request[0], []).append(request[1:])
    last_frame = max(by_source_frame)
    scale_x = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 1280.0
    scale_y = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 720.0
    vertical_shift = args.crop * 0.18
    # Absichtlich ein linearer Durchlauf: Random-Seek in der H.264-MP4 muss
    # je nach Keyframe vom Dateianfang dekodieren und waere 66-mal viel teurer.
    for source_frame in range(last_frame + 1):
        ok, image = cap.read()
        if not ok:
            raise RuntimeError(f"Video-Frame {source_frame} fehlt")
        for clip_index, sequence_index, analysis_frame, tracklet in by_source_frame.get(source_frame, []):
            xy = position_at(ball_lookup_, tracklet, analysis_frame)
            crop = crop_with_padding(image, float(xy[0] * scale_x),
                                     float(xy[1] * scale_y - vertical_shift),
                                     args.crop)
            crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
            clips[clip_index, sequence_index] = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        if source_frame % 2500 == 0 or source_frame == last_frame:
            print(f"Video gelesen: {source_frame}/{last_frame}", flush=True)
    cap.release()
    return clips, np.asarray(labels), np.asarray(groups), metadata


def extract_embeddings(clips: np.ndarray, batch_size: int) -> np.ndarray:
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)
    features = []
    with torch.inference_mode():
        for start in range(0, len(clips), batch_size):
            batch = torch.from_numpy(clips[start:start + batch_size]).float() / 255.0
            batch = batch.permute(0, 4, 1, 2, 3)
            batch = (batch - mean) / std
            features.append(model(batch).cpu().numpy())
            print(f"Embedding {min(start + batch_size, len(clips))}/{len(clips)}",
                  flush=True)
    return np.concatenate(features)


def fit_logistic(x: np.ndarray, y: np.ndarray, ridge: float = 1.0):
    x_aug = np.column_stack([x, np.ones(len(x))])

    def objective(beta):
        logits = np.clip(x_aug @ beta, -30, 30)
        loss = np.logaddexp(0, logits) - y * logits
        regularizer = 0.5 * ridge * np.dot(beta[:-1], beta[:-1])
        gradient = x_aug.T @ (1 / (1 + np.exp(-logits)) - y)
        gradient[:-1] += ridge * beta[:-1]
        return float(loss.sum() + regularizer), gradient

    result = minimize(objective, np.zeros(x_aug.shape[1]), jac=True,
                      method="L-BFGS-B")
    return result.x


def evaluate(features: np.ndarray, labels: np.ndarray, groups: np.ndarray):
    probabilities = np.zeros(len(labels), dtype=np.float64)
    for group in np.unique(groups):
        train, test = groups != group, groups == group
        mean, std = features[train].mean(0), features[train].std(0) + 1e-6
        train_x, test_x = (features[train] - mean) / std, (features[test] - mean) / std
        # PCA wird ausschliesslich auf dem Trainingsfold gefittet.
        _, _, vt = np.linalg.svd(train_x, full_matrices=False)
        components = vt[:min(10, len(vt))].T
        train_x, test_x = train_x @ components, test_x @ components
        beta = fit_logistic(train_x, labels[train], ridge=2.0)
        logits = np.column_stack([test_x, np.ones(len(test_x))]) @ beta
        probabilities[test] = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))

    predicted = (probabilities >= 0.5).astype(np.int32)
    accuracy = float((predicted == labels).mean())
    recalls = []
    for team in (0, 1):
        mask = labels == team
        recalls.append(float((predicted[mask] == labels[mask]).mean()))
    confidence = np.maximum(probabilities, 1 - probabilities)
    rows = []
    for threshold in (0.70, 0.80, 0.90, 0.95):
        mask = confidence >= threshold
        rows.append({
            "threshold": threshold,
            "coverage": float(mask.mean()),
            "n": int(mask.sum()),
            "accuracy": float((predicted[mask] == labels[mask]).mean()) if mask.any() else None,
        })
    report = {
        "examples": int(len(labels)),
        "events": int(len(np.unique(groups))),
        "team_counts": {str(team): int((labels == team).sum()) for team in (0, 1)},
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "recall_team_0_green": recalls[0],
        "recall_team_1_blue": recalls[1],
        "confidence_subsets": rows,
    }
    return probabilities, predicted, report


def main():
    args = parse_args()
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cache_path = prefix.with_name(prefix.name + "_embeddings.npz")
    metadata_path = prefix.with_name(prefix.name + "_examples.csv")
    report_path = prefix.with_name(prefix.name + "_evaluation.json")

    review = pd.read_csv(args.pass_review_csv)
    ball = pd.read_csv(args.ball_track_csv)
    lookup = ball_lookup(ball)

    if cache_path.exists() and not args.rebuild:
        cache = np.load(cache_path)
        features, labels, groups = cache["features"], cache["labels"], cache["groups"]
        metadata = pd.read_csv(metadata_path).to_dict("records")
        print(f"Embedding-Cache geladen: {cache_path}")
    else:
        clips, labels, groups, metadata = build_examples(
            args.video_1080, review, lookup, args,
        )
        print("Lade vortrainiertes R3D-18 und extrahiere Video-Embeddings ...", flush=True)
        features = extract_embeddings(clips, args.batch_size)
        np.savez_compressed(cache_path, features=features, labels=labels, groups=groups)
        pd.DataFrame(metadata).to_csv(metadata_path, index=False)
        print(f"Embedding-Cache: {cache_path}")

    probabilities, predicted, report = evaluate(features, labels, groups)
    examples = pd.DataFrame(metadata)
    examples["prob_team_1_blue"] = probabilities
    examples["prediction"] = predicted
    examples["correct"] = predicted == labels
    examples.to_csv(metadata_path, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Beispiele: {metadata_path}")
    print(f"Evaluation: {report_path}")


if __name__ == "__main__":
    main()
