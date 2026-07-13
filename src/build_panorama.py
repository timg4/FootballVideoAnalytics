"""Erzeugt ein weitgehend spielerfreies Median-Panorama aus registrierten Frames.

Die Frames werden mit den Homographien in das Koordinatensystem des
Referenzframes gewarpt. Pro Pixel wird der Median mehrerer Zeitpunkte genommen;
bewegte Spieler verschwinden dadurch größtenteils. Das Bild wird verkleinert
berechnet und zusammen mit der Transformation Bild -> Referenzframe gespeichert.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Median-Panorama für die Platzkalibrierung")
    parser.add_argument("video")
    parser.add_argument("homographies_npz")
    parser.add_argument("--samples", type=int, default=80,
                        help="ungefähre Zahl von Frames über alle Schwenkbereiche (Standard: 80)")
    parser.add_argument("--pan-bins", type=int, default=16,
                        help="Zahl abgedeckter Kamerapositionen von links bis rechts")
    parser.add_argument("--start-frame", type=int, default=None,
                        help="nur Homographien ab diesem absoluten Frame verwenden")
    parser.add_argument("--end-frame", type=int, default=None,
                        help="nur Homographien vor diesem absoluten Frame verwenden")
    parser.add_argument("--anchor", type=int, default=None,
                        help="kurzen Bereich auf diesen Frame umbasieren (reduziert Langzeitdrift)")
    parser.add_argument("--min-separation", type=int, default=60,
                        help="Mindestabstand ausgewählter Frames innerhalb eines Schwenkbereichs")
    parser.add_argument("--min-coverage", type=int, default=3,
                        help="Mindestzahl gültiger Bilder pro Panorama-Pixel")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="Ausgabeskalierung (Standard: 0.5)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.samples < 3:
        parser.error("--samples muss mindestens 3 sein")
    if not 0 < args.scale <= 1:
        parser.error("--scale muss in (0, 1] liegen")

    video_path = Path(args.video)
    out_dir = Path(__file__).resolve().parent.parent / "data" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = (Path(args.output) if args.output else
              out_dir / f"{video_path.stem.lower().replace(' ', '_')}_panorama.jpg")
    transform_path = output.with_name(f"{output.stem}_transform.json")

    with np.load(args.homographies_npz) as data:
        H_all = data["H"].copy()
        frames = (data["frames"].copy() if "frames" in data
                  else np.arange(len(H_all)))
        ref = int(data["ref"])

    # Optional nur einen kurzen vollständigen Kameraschwenk verwenden. Beim
    # Umbasieren auf einen Ankerframe wird die zuvor angesammelte Drift entfernt.
    # H_original_ref_to_work_ref hält die spätere Kalibrierung trotzdem im
    # Koordinatensystem der ursprünglichen Homographie-Datei kompatibel.
    range_mask = np.ones(len(frames), dtype=bool)
    if args.start_frame is not None:
        range_mask &= frames >= args.start_frame
    if args.end_frame is not None:
        range_mask &= frames < args.end_frame
    H_original_ref_to_work_ref = np.eye(3)
    if args.anchor is not None:
        anchor_matches = np.flatnonzero(frames == args.anchor)
        if not len(anchor_matches):
            raise SystemExit(f"Ankerframe {args.anchor} fehlt in der Homographie-Datei")
        H_anchor_to_original_ref = H_all[anchor_matches[0]]
        H_original_ref_to_work_ref = np.linalg.inv(H_anchor_to_original_ref)
        H_all = np.array([H_original_ref_to_work_ref @ H for H in H_all])
    H_all = H_all[range_mask]
    frames = frames[range_mask]
    if not len(frames):
        raise SystemExit("Der gewählte Framebereich ist leer")

    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not width or not height:
        raise SystemExit(f"Video nicht lesbar: {video_path}")

    # Breite Leinwand für beide Torseiten. Der Referenzframe liegt bewusst nicht
    # am linken Rand, damit extreme Schwenks in beide Richtungen Platz haben.
    offset = np.array([[1, 0, width * 2], [0, 1, height // 2], [0, 0, 1]],
                      dtype=np.float64)
    scale = np.array([[args.scale, 0, 0], [0, args.scale, 0], [0, 0, 1]],
                     dtype=np.float64)
    H_ref_to_image = scale @ offset @ H_original_ref_to_work_ref
    canvas_w = int(width * 5 * args.scale)
    canvas_h = int(height * 2 * args.scale)

    # Nicht gleichmäßig nach Zeit sampeln: kurze Schwenks bis ans Tor würden
    # sonst leicht fehlen. Stattdessen die projizierte Bildmitte bestimmen und
    # mehrere zeitlich getrennte Frames aus jedem Schwenkbereich auswählen.
    normalized_H = H_all / H_all[:, 2:3, 2:3]
    center = np.array([[[width / 2, height / 2]]], dtype=np.float64)
    projected_center = np.array([
        cv2.perspectiveTransform(center, H)[0, 0] for H in normalized_H
    ])
    pan_x = projected_center[:, 0]
    det = np.linalg.det(normalized_H)
    finite = (np.isfinite(projected_center).all(axis=1)
              & (det > 0.2) & (det < 5.0)
              & (projected_center[:, 1] > -height)
              & (projected_center[:, 1] < 2 * height))
    lo, hi = np.percentile(pan_x[finite], [0.5, 99.5])
    targets = np.linspace(lo, hi, args.pan_bins)
    per_bin = max(1, int(np.ceil(args.samples / args.pan_bins)))
    chosen = []
    for target in targets:
        order = np.argsort(np.abs(pan_x - target))
        selected_here = []
        # Zeitlich trennen, damit an derselben Kameraposition andere Spieler
        # stehen und der Median sie tatsächlich entfernen kann.
        for pos in order:
            if not finite[pos] or not lo <= pan_x[pos] <= hi:
                continue
            if all(abs(int(pos) - old) >= args.min_separation for old in selected_here):
                selected_here.append(int(pos))
            if len(selected_here) == per_bin:
                break
        chosen.extend(selected_here)
    sample_positions = np.array(sorted(set(chosen)), dtype=int)
    print(f"Schwenkbereich Referenz-x {lo:.0f} bis {hi:.0f}; "
          f"{len(sample_positions)} gezielte Stichproben")
    source_frames = []
    source_H = []
    for n, pos in enumerate(sample_positions, 1):
        frame_idx = int(frames[pos])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok:
            source_frames.append(frame)
            source_H.append(H_ref_to_image @ H_all[pos])
        print(f"Lese Stichprobe {n}/{len(sample_positions)}", end="\r")
    cap.release()
    print(f"\n{len(source_frames)} Frames geladen; berechne Median ...")

    panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    tile_h = 180
    source_mask = np.full((height, width), 255, dtype=np.uint8)
    for y0 in range(0, canvas_h, tile_h):
        y1 = min(y0 + tile_h, canvas_h)
        h_tile = y1 - y0
        tile_shift = np.array([[1, 0, 0], [0, 1, -y0], [0, 0, 1]],
                              dtype=np.float64)
        # 255 markiert ungültige Pixel und landet beim Sortieren am Ende.
        stack = np.full((len(source_frames), h_tile, canvas_w, 3), 255,
                        dtype=np.uint8)
        valid_count = np.zeros((h_tile, canvas_w), dtype=np.uint16)
        for n, (frame, H) in enumerate(zip(source_frames, source_H)):
            H_tile = tile_shift @ H
            warped = cv2.warpPerspective(frame, H_tile, (canvas_w, h_tile))
            mask = cv2.warpPerspective(source_mask, H_tile, (canvas_w, h_tile),
                                       flags=cv2.INTER_NEAREST) > 0
            stack[n][mask] = warped[mask]
            valid_count += mask
        stack.sort(axis=0)
        median_index = np.maximum(valid_count.astype(np.int32) - 1, 0) // 2
        indices = np.repeat(median_index[None, ..., None], 3, axis=3)
        tile = np.take_along_axis(stack, indices, axis=0)[0]
        tile[valid_count < args.min_coverage] = 0
        panorama[y0:y1] = tile
        print(f"Median-Zeilen {y1}/{canvas_h}", end="\r")

    cv2.imwrite(str(output), panorama, [cv2.IMWRITE_JPEG_QUALITY, 95])
    transform_path.write_text(json.dumps({
        "video": str(video_path),
        "ref": ref,
        "H_ref_to_image": H_ref_to_image.tolist(),
        "samples": len(source_frames),
    }, indent=2), encoding="utf-8")
    print(f"\nPanorama: {output}")
    print(f"Transformation: {transform_path}")


if __name__ == "__main__":
    main()
