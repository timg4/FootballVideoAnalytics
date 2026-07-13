"""Appearance-Embeddings pro Tracklet für post-hoc Spieler-Re-ID.

Verwendet ein OpenVINO-Person-ReID-Modell auf einigen großen Ganzkörper-Crops
pro Tracklet. Das Ergebnis ist noch keine Spieleridentität, sondern die
Eingabe für ein nachfolgendes, vorsichtiges Tracklet-Stitching.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
from openvino import Core

MODEL_NAME = "person-reidentification-retail-0288"
MODEL_BASE_URL = (
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/temp/"
    f"{MODEL_NAME}/FP16"
)


def l2_normalize(vector):
    norm = np.linalg.norm(vector)
    return vector / max(float(norm), 1e-12)


def main():
    parser = argparse.ArgumentParser(description="Re-ID-Embeddings pro Tracklet")
    parser.add_argument("video")
    parser.add_argument("tracks_csv")
    parser.add_argument("positions_csv")
    parser.add_argument("assignments_csv")
    parser.add_argument("model_xml", nargs="?", default=None,
                        help="OpenVINO-IR; ohne Angabe Download aus offiziellem Model Zoo")
    parser.add_argument("--samples", type=int, default=4,
                        help="Maximale Anzahl guter Crops pro Tracklet")
    parser.add_argument("--min-height", type=float, default=30,
                        help="Bevorzugte minimale Boxhöhe in Pixeln")
    parser.add_argument("--frame-distance", type=int, default=15,
                        help="Mindestabstand ausgewählter Crops in Frames")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output = (Path(args.output) if args.output else
              Path(__file__).resolve().parent.parent / "data" / "output" /
              f"{Path(args.tracks_csv).stem.replace('_tracked', '')}_reid_embeddings.npz")
    model_dir = output.parent / "models" / MODEL_NAME / "FP16"
    model_xml = Path(args.model_xml) if args.model_xml else model_dir / f"{MODEL_NAME}.xml"
    model_bin = model_xml.with_suffix(".bin")
    if not model_xml.exists() or not model_bin.exists():
        model_xml.parent.mkdir(parents=True, exist_ok=True)
        print(f"Lade {MODEL_NAME} aus dem offiziellen Open Model Zoo ...")
        urlretrieve(f"{MODEL_BASE_URL}/{MODEL_NAME}.xml", model_xml)
        urlretrieve(f"{MODEL_BASE_URL}/{MODEL_NAME}.bin", model_bin)

    team_of = {}
    with open(args.assignments_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["team"] != "":
                team_of[int(row["tracker_id"])] = int(row["team"])

    allowed = set()
    with open(args.positions_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            tid = int(row["tracker_id"])
            if int(row["auf_platz"]) and tid in team_of:
                allowed.add((int(row["frame"]), tid))

    candidates = defaultdict(list)
    with open(args.tracks_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            tid = int(row["tracker_id"])
            if (frame, tid) not in allowed:
                continue
            box = tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2"))
            height = box[3] - box[1]
            candidates[tid].append((height, frame, box))

    selected = defaultdict(list)
    for tid, choices in candidates.items():
        # Erst ausreichend große Crops; falls es davon zu wenige gibt, werden
        # die besten kleineren ergänzt. Zeitabstand verhindert vier fast
        # identische Bilder aus direkt aufeinanderfolgenden Frames.
        ordered = sorted(choices, reverse=True)
        preferred = [item for item in ordered if item[0] >= args.min_height]
        fallback = [item for item in ordered if item[0] < args.min_height]
        for item in preferred + fallback:
            _, frame, _ = item
            if all(abs(frame - old[1]) >= args.frame_distance
                   for old in selected[tid]):
                selected[tid].append(item)
            if len(selected[tid]) >= args.samples:
                break

    by_frame = defaultdict(list)
    for tid, choices in selected.items():
        for height, frame, box in choices:
            by_frame[frame].append((tid, height, box))

    core = Core()
    model = core.read_model(model_xml)
    compiled = core.compile_model(model, "CPU")
    output_layer = compiled.output(0)

    sample_embeddings = defaultdict(list)
    sample_frames = defaultdict(list)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Video nicht lesbar: {args.video}")
    last_needed = max(by_frame) if by_frame else -1
    frame_idx = 0
    while frame_idx <= last_needed:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, _, box in by_frame.get(frame_idx, []):
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            # Etwas Kontext gegen abgeschnittene Arme/Füße, aber möglichst
            # wenig Rasen und benachbarte Spieler in den Crop aufnehmen.
            x1 = max(0, int(x1 - 0.05 * width))
            x2 = min(frame.shape[1], int(x2 + 0.05 * width))
            y1 = max(0, int(y1 - 0.02 * height))
            y2 = min(frame.shape[0], int(y2 + 0.02 * height))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            image = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
            blob = image.transpose(2, 0, 1)[None].astype(np.float32)
            embedding = compiled([blob])[output_layer].reshape(-1)
            sample_embeddings[tid].append(l2_normalize(embedding))
            sample_frames[tid].append(frame_idx)
        if frame_idx and frame_idx % 3000 == 0:
            print(f"Frame {frame_idx}/{last_needed}")
        frame_idx += 1
    cap.release()

    track_ids = sorted(sample_embeddings)
    embeddings = []
    counts = []
    consistency = []
    for tid in track_ids:
        samples = np.asarray(sample_embeddings[tid])
        mean = l2_normalize(samples.mean(axis=0))
        embeddings.append(mean)
        counts.append(len(samples))
        consistency.append(float(np.mean(samples @ mean)))

    np.savez_compressed(
        output,
        track_ids=np.asarray(track_ids, dtype=np.int64),
        teams=np.asarray([team_of[tid] for tid in track_ids], dtype=np.int64),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        sample_counts=np.asarray(counts, dtype=np.int64),
        consistency=np.asarray(consistency, dtype=np.float32),
        sample_track_ids=np.asarray([
            tid for tid in track_ids for _ in sample_embeddings[tid]
        ], dtype=np.int64),
        sample_frames=np.asarray([
            frame for tid in track_ids for frame in sample_frames[tid]
        ], dtype=np.int64),
        sample_embeddings=np.asarray([
            embedding for tid in track_ids for embedding in sample_embeddings[tid]
        ], dtype=np.float32),
    )
    print(f"Re-ID-Embeddings: {output}")
    print(f"{len(track_ids)}/{len(team_of)} Team-Tracklets, "
          f"{sum(counts)} Crops, mittlere Konsistenz {np.mean(consistency):.3f}")


if __name__ == "__main__":
    main()
