"""Lokalisiert jeden Videoframe direkt gegen kalibrierte Ankeransichten.

Anders als eine über das gesamte Video verkettete Registrierung sammelt dieses
Verfahren keine Langzeitdrift an. Die im Mehrbild-Kalibrierwerkzeug geklickten
linken/rechten Ansichten werden separat kalibriert. Jeder Frame wird per ORB
direkt auf den am besten passenden Anker gematcht und sofort nach Platzmetern
abgebildet.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


MIN_MATCHES = 20
MIN_INLIERS = 30


def match_descriptors(matcher, kp, des, kp_anchor, des_anchor):
    if des is None or des_anchor is None or len(kp) < MIN_MATCHES:
        return None, 0
    pairs = matcher.knnMatch(des, des_anchor, k=2)
    good = [m for pair in pairs if len(pair) == 2
            for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < MIN_MATCHES:
        return None, 0
    src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_anchor[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return (H, int(mask.sum())) if H is not None else (None, 0)


def main():
    parser = argparse.ArgumentParser(description="Driftfreie Frame-Lokalisierung zum Spielfeld")
    parser.add_argument("video")
    parser.add_argument("calibration_json")
    parser.add_argument("--stride", type=int, default=1,
                        help="muss zum Tracking passen (Standard: 1)")
    parser.add_argument("--start", type=int, default=0, help="erster absoluter Frame")
    parser.add_argument("--end", type=int, default=None, help="Endframe exklusiv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.stride < 1:
        parser.error("--stride muss mindestens 1 sein")

    video_path = Path(args.video)
    calibration = json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for point in calibration["points"]:
        label = point.get("clicked_view", "")
        if label.startswith("Frame ") and "clicked_px" in point:
            grouped[int(label.split()[1])].append(point)

    anchors = []
    cap = cv2.VideoCapture(str(video_path))
    orb = cv2.ORB_create(nfeatures=3000)
    for frame_idx, points in sorted(grouped.items()):
        if len(points) < 4:
            print(f"Anker {frame_idx}: nur {len(points)} Punkte -> übersprungen")
            continue
        src_m = np.array([p["meter"] for p in points], np.float64).reshape(-1, 1, 2)
        dst_px = np.array([p["clicked_px"] for p in points], np.float64).reshape(-1, 1, 2)
        H_pitch_to_anchor, _ = cv2.findHomography(src_m, dst_px, 0)
        projected = cv2.perspectiveTransform(src_m, H_pitch_to_anchor)
        error = np.linalg.norm(projected - dst_px, axis=2).mean()
        if error > 20:
            print(f"Anker {frame_idx}: Klickfehler {error:.1f}px -> übersprungen")
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"Ankerframe {frame_idx} nicht lesbar")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        anchors.append({
            "frame": frame_idx,
            "kp": kp,
            "des": des,
            "H_anchor_to_pitch": np.linalg.inv(H_pitch_to_anchor),
        })
        print(f"Anker {frame_idx}: {len(points)} Punkte, mittlerer Fehler {error:.1f}px")

    if not anchors:
        raise SystemExit("Keine gültigen Anker mit mindestens 4 Klickpunkten")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(args.end or total_frames, total_frames)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    output_frames = []
    H_px_to_pitch = []
    qualities = []
    anchor_ids = []
    failed = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    for frame_idx in range(args.start, end_frame, args.stride):
        if args.stride > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        candidates = []
        for anchor in anchors:
            H_to_anchor, inliers = match_descriptors(
                matcher, kp, des, anchor["kp"], anchor["des"])
            if H_to_anchor is not None:
                H_direct = anchor["H_anchor_to_pitch"] @ H_to_anchor
                H_direct /= H_direct[2, 2]
                candidates.append((inliers, anchor["frame"], H_direct))
        if candidates:
            inliers, anchor_frame, H_direct = max(candidates, key=lambda x: x[0])
        else:
            inliers, anchor_frame, H_direct = 0, -1, None

        if H_direct is None or inliers < MIN_INLIERS:
            failed += 1
            # Kurze Aussetzer übernehmen die vorige Abbildung. Die Qualitätszahl
            # bleibt 0, sodass sie später erkennbar/filtrierbar sind.
            if H_px_to_pitch:
                H_direct = H_px_to_pitch[-1].copy()
                anchor_frame = anchor_ids[-1]
            else:
                H_direct = np.eye(3)
        output_frames.append(frame_idx)
        H_px_to_pitch.append(H_direct)
        qualities.append(inliers if inliers >= MIN_INLIERS else 0)
        anchor_ids.append(anchor_frame)
        if len(output_frames) % 100 == 0:
            print(f"Frame {frame_idx}/{total_frames}: {inliers} Inlier, "
                  f"Anker {anchor_frame}", end="\r")
    cap.release()

    out_dir = Path(__file__).resolve().parent.parent / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = (Path(args.output) if args.output else
              out_dir / f"{video_path.stem}_pitch_localization.npz")
    np.savez_compressed(output,
                        H_px_to_pitch=np.array(H_px_to_pitch),
                        frames=np.array(output_frames),
                        inliers=np.array(qualities),
                        anchors=np.array(anchor_ids))
    print(f"\nLokalisierung: {output}")
    print(f"Direkt lokalisiert: {len(output_frames) - failed}/{len(output_frames)}, "
          f"Aussetzer: {failed}")


if __name__ == "__main__":
    main()
