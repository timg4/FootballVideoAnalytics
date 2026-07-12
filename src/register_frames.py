"""Phase 2a: Frame-Registrierung — Kameraschwenks kompensieren (Streaming).

Die Veo-Follow-Cam ist ein virtueller Schwenk/Zoom aus einem Panorama, d.h.
zwischen zwei Frames liegt (näherungsweise) eine reine Rotation -> die Bilder
sind durch eine Homographie verbunden. Wir matchen ORB-Features (v.a. am
statischen Hintergrund: Zäune, Gebäude, Flutlichter) gegen einen Keyframe und
verketten die Homographien, sodass jeder Frame auf das Referenzbild abgebildet
werden kann. Frames werden gestreamt (konstanter RAM), daher auch für lange
Videos geeignet.

Ablauf: ein Vorwärtsdurchlauf ab --start. Der Referenzframe (--ref, absolut)
muss im Bereich liegen; Homographien vor dem Referenzframe werden über die
Inverse der Kette berechnet.

Ergebnis: <video>_homographies.npz mit
  H      — (n, 3, 3) Homographien Frame -> Referenzframe
  frames — (n,) absolute Frame-Indizes zu den Matrizen
  ref    — absoluter Referenz-Frame-Index

Validierung mit --check: Frames werden ins Referenz-Koordinatensystem
verzerrt und halbtransparent über das Referenzbild gelegt — Linien und
Hintergrund müssen deckungsgleich sein.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

MIN_INLIERS = 60      # darunter: Keyframe nachziehen
MIN_MATCHES = 20


def orb_match(orb, matcher, gray, kp_kf, des_kf):
    """Homographie aktueller Frame -> Keyframe, gibt (H, inlier, kp, des)."""
    kp, des = orb.detectAndCompute(gray, None)
    if des is None or des_kf is None or len(kp) < MIN_MATCHES:
        return None, 0, kp, des
    matches = matcher.knnMatch(des, des_kf, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2)
            if m.distance < 0.75 * n.distance]
    if len(good) < MIN_MATCHES:
        return None, 0, kp, des
    src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_kf[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        return None, 0, kp, des
    return H, int(mask.sum()), kp, des


def main():
    parser = argparse.ArgumentParser(description="Frame-Registrierung (Streaming)")
    parser.add_argument("video")
    parser.add_argument("--ref", type=int, default=None,
                        help="Referenzframe, absolut (Standard: erster Frame des Bereichs)")
    parser.add_argument("--start", type=int, default=0, help="erster Frame (absolut)")
    parser.add_argument("--end", type=int, default=None, help="letzter Frame (exklusiv)")
    parser.add_argument("--stride", type=int, default=1,
                        help="nur jeden N-ten Frame registrieren (muss zum Tracking passen)")
    parser.add_argument("--output", default=None,
                        help="Ausgabe-NPZ (Standard: data/output/<video>_homographies.npz)")
    parser.add_argument("--check", action="store_true",
                        help="Validierungsbilder (Warp-Überlagerungen) speichern")
    args = parser.parse_args()

    if args.stride < 1:
        parser.error("--stride muss mindestens 1 sein")

    video_path = Path(args.video)
    out_dir = Path(__file__).resolve().parent.parent / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = (Path(args.output) if args.output else
                out_dir / f"{video_path.stem}_homographies.npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    video_info = sv.VideoInfo.from_video_path(str(video_path))
    end = min(args.end or video_info.total_frames, video_info.total_frames)
    ref = args.ref if args.ref is not None else args.start
    if not (args.start <= ref < end):
        raise SystemExit(f"--ref {ref} liegt nicht im Bereich [{args.start}, {end})")

    orb = cv2.ORB_create(nfeatures=4000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    n_frames = (end - args.start + args.stride - 1) // args.stride
    print(f"Registriere {n_frames} Frames (Stride {args.stride}), "
          f"Referenz = Frame {ref} ...")

    frame_indices = []
    H_chain = []          # Homographien Frame -> erster Frame des Bereichs
    H_to_first = np.eye(3)
    kp_kf = des_kf = None
    check_frames = {}
    ref_H_to_first = None
    prev_gray = None

    frames = sv.get_video_frames_generator(str(video_path), start=args.start,
                                           end=end, stride=args.stride)
    for k, frame in enumerate(frames):
        idx = args.start + k * args.stride
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if kp_kf is None:  # allererster Frame = erster Keyframe
            kp_kf, des_kf = orb.detectAndCompute(gray, None)
            H_kf_to_first = np.eye(3)
        else:
            H, inliers, kp, des = orb_match(orb, matcher, gray, kp_kf, des_kf)
            if H is None or inliers < MIN_INLIERS:
                # Keyframe auf den vorherigen Frame nachziehen und nochmal
                if prev_gray is not None:
                    kp_kf, des_kf = orb.detectAndCompute(prev_gray, None)
                    H_kf_to_first = H_chain[-1] if H_chain else np.eye(3)
                    H, inliers, kp, des = orb_match(orb, matcher, gray, kp_kf, des_kf)
                if H is None:
                    print(f"\nFrame {idx}: Registrierung fehlgeschlagen — "
                          f"übernehme letzte Homographie")
                    H = np.eye(3)
            H_to_first = H_kf_to_first @ H
            # Keyframe regelmäßig nachziehen, damit die Basis aktuell bleibt
            if inliers and inliers < 3 * MIN_INLIERS:
                kp_kf, des_kf = kp, des
                H_kf_to_first = H_to_first

        frame_indices.append(idx)
        H_chain.append(H_to_first)
        prev_gray = gray
        if idx == ref:
            ref_H_to_first = H_to_first.copy()
        if (args.check and k % max(1, n_frames // 5) == 0) or idx == ref:
            check_frames[idx] = frame.copy()
        if k % 100 == 0:
            print(f"Frame {idx} ({k + 1}/{n_frames})", end="\r")

    if ref_H_to_first is None:
        raise SystemExit(f"Referenzframe {ref} wurde nicht verarbeitet "
                         f"(Stride-Raster prüfen)")

    # Umbasieren: Frame -> Referenzframe
    first_to_ref = np.linalg.inv(ref_H_to_first)
    H_to_ref = np.array([first_to_ref @ H for H in H_chain])

    np.savez_compressed(npz_path, H=H_to_ref,
                        frames=np.array(frame_indices), ref=ref)
    print(f"\nHomographien gespeichert: {npz_path}")

    if args.check and ref in check_frames:
        h, w = check_frames[ref].shape[:2]
        offset = np.array([[1, 0, w], [0, 1, h // 2], [0, 0, 1]], dtype=np.float64)
        canvas_size = (w * 3, h * 2)
        base = cv2.warpPerspective(check_frames[ref], offset, canvas_size)
        idx_map = {idx: n for n, idx in enumerate(frame_indices)}
        for idx, img in sorted(check_frames.items()):
            warped = cv2.warpPerspective(img, offset @ H_to_ref[idx_map[idx]],
                                         canvas_size)
            blend = cv2.addWeighted(base, 0.5, warped, 0.5, 0)
            path = out_dir / f"check_warp_f{idx}.jpg"
            cv2.imwrite(str(path), cv2.resize(blend, None, fx=0.5, fy=0.5))
            print(f"Validierung: {path}")


if __name__ == "__main__":
    main()
