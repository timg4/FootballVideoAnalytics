"""Phase 2a: Frame-Registrierung — Kameraschwenks kompensieren.

Die Veo-Follow-Cam ist ein virtueller Schwenk/Zoom aus einem Panorama, d.h.
zwischen zwei Frames liegt (näherungsweise) eine reine Rotation -> die Bilder
sind durch eine Homographie verbunden. Wir matchen ORB-Features (v.a. am
statischen Hintergrund: Zäune, Gebäude, Flutlichter) und verketten die
Homographien über Keyframes, sodass jeder Frame auf ein Referenzbild
abgebildet werden kann. Ergebnis: <video>_homographies.npz mit einer
3x3-Matrix pro Frame (Frame -> Referenzframe).

Validierung mit --check: einzelne Frames werden ins Referenz-Koordinatensystem
verzerrt und halbtransparent über das Referenzbild gelegt — Linien und
Hintergrund müssen deckungsgleich sein.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

MIN_INLIERS = 60          # darunter: neuen Keyframe setzen und nochmal versuchen
MIN_KEYFRAME_GAP = 1


def orb_homography(orb, matcher, img_a, kp_b, des_b, img_b_shape):
    """Homographie img_a -> img_b über ORB-Matches, gibt (H, inlier) zurück."""
    kp_a, des_a = orb.detectAndCompute(img_a, None)
    if des_a is None or des_b is None or len(kp_a) < 20:
        return None, 0
    matches = matcher.knnMatch(des_a, des_b, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2)
            if m.distance < 0.75 * n.distance]
    if len(good) < 20:
        return None, 0
    src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        return None, 0
    return H, int(mask.sum())


def main():
    parser = argparse.ArgumentParser(description="Frame-Registrierung gegen Referenzframe")
    parser.add_argument("video")
    parser.add_argument("--ref", type=int, default=0, help="Referenz-Frame-Index")
    parser.add_argument("--end", type=int, default=None, help="letzter Frame (exklusiv)")
    parser.add_argument("--check", action="store_true",
                        help="Validierungsbilder (Warp-Überlagerungen) speichern")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(__file__).resolve().parent.parent / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{video_path.stem}_homographies.npz"

    video_info = sv.VideoInfo.from_video_path(str(video_path))
    end = args.end or video_info.total_frames

    orb = cv2.ORB_create(nfeatures=4000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    print(f"Lese {end} Frames, Referenz = Frame {args.ref} ...")
    frames = list(sv.get_video_frames_generator(str(video_path), end=end))
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    H_to_ref = [None] * len(frames)
    H_to_ref[args.ref] = np.eye(3)

    # Vom Referenzframe aus in beide Richtungen arbeiten. Aktueller Keyframe
    # ist anfangs die Referenz; reißt das Matching ab (Schwenk zu weit),
    # wird der zuletzt registrierte Frame neuer Keyframe (Verkettung).
    for direction in (1, -1):
        kf_idx = args.ref
        kp_kf, des_kf = orb.detectAndCompute(gray[kf_idx], None)
        idx = args.ref + direction
        while 0 <= idx < len(frames):
            H, inliers = orb_homography(orb, matcher, gray[idx], kp_kf, des_kf,
                                        gray[kf_idx].shape)
            if H is None or inliers < MIN_INLIERS:
                # Keyframe nachziehen und erneut versuchen
                new_kf = idx - direction
                if new_kf == kf_idx:
                    print(f"Frame {idx}: Registrierung fehlgeschlagen "
                          f"({inliers} Inlier) — übernehme Nachbar-Homographie")
                    H_to_ref[idx] = H_to_ref[idx - direction]
                    idx += direction
                    continue
                kf_idx = new_kf
                kp_kf, des_kf = orb.detectAndCompute(gray[kf_idx], None)
                continue
            H_to_ref[idx] = H_to_ref[kf_idx] @ H
            if idx % 25 == 0:
                print(f"Frame {idx}: {inliers} Inlier (Keyframe {kf_idx})", end="\r")
            idx += direction

    np.savez_compressed(npz_path, H=np.array(H_to_ref), ref=args.ref)
    print(f"\nHomographien gespeichert: {npz_path}")

    if args.check:
        h, w = frames[0].shape[:2]
        # großzügige Leinwand um das Referenzbild herum
        offset = np.array([[1, 0, w], [0, 1, h // 2], [0, 0, 1]], dtype=np.float64)
        canvas_size = (w * 3, h * 2)
        base = cv2.warpPerspective(frames[args.ref], offset, canvas_size)
        for check_idx in np.linspace(0, len(frames) - 1, 5).astype(int):
            warped = cv2.warpPerspective(frames[check_idx],
                                         offset @ H_to_ref[check_idx], canvas_size)
            blend = cv2.addWeighted(base, 0.5, warped, 0.5, 0)
            small = cv2.resize(blend, None, fx=0.5, fy=0.5)
            path = out_dir / f"check_warp_f{check_idx}.jpg"
            cv2.imwrite(str(path), small)
            print(f"Validierung: {path}")


if __name__ == "__main__":
    main()
