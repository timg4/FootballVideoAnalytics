"""Phase 2b: Einmalige Spielfeld-Kalibrierung per Mausklick.

Zeigt einen Referenzframe des Videos. Für jeden angesagten Spielfeldpunkt
(Ecken, Torpfosten, Strafraumecken, Mittelpunkt ...):
  - Linksklick  = Punkt im Bild markieren (vorher ruhig mit der Lupe zoomen,
                  Zoom-Werkzeug danach wieder deaktivieren!)
  - Mittelklick = Punkt überspringen (wenn nicht sichtbar/unsicher)

Mindestens 4 Punkte, besser 6-8, möglichst über den ganzen Platz verteilt.
Ergebnis: data/calibration/platz.json + Kontrollbild mit projizierten
Spielfeldlinien in data/calibration/. Da die Veo-Kamera fest montiert ist,
gilt die Kalibrierung für alle Aufnahmen dieses Platzes.

Aufruf (Platzmaße in Metern anpassen, sobald bekannt):
  python src\\calibrate_pitch.py "data\\videos\\highlights\\03 000707_-_Goal.mp4" --frame 120
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from pitch_model import PitchModel


def main():
    parser = argparse.ArgumentParser(description="Spielfeld-Kalibrierung per Klick")
    parser.add_argument("video", help="Video oder Panorama-Bild")
    parser.add_argument("--frame", type=int, default=0,
                        help="Referenzframe (muss zum --ref der Registrierung passen!)")
    parser.add_argument("--homographies", default=None,
                        help="NPZ für Mehrbild-Kalibrierung")
    parser.add_argument("--frames", default=None,
                        help="Kommagetrennte Video-Frames zum Umschalten, z.B. 23150,23320,23620")
    parser.add_argument("--laenge", type=float, default=60.0)
    parser.add_argument("--breite", type=float, default=40.0)
    parser.add_argument("--tor", type=float, default=5.0)
    parser.add_argument("--box-tiefe", type=float, default=9.0)
    parser.add_argument("--box-breite", type=float, default=24.0)
    parser.add_argument("--kreis", type=float, default=5.0)
    parser.add_argument("--name", default="platz", help="Name der Kalibrierung")
    args = parser.parse_args()

    pitch = PitchModel(laenge=args.laenge, breite=args.breite, tor_breite=args.tor,
                       box_tiefe=args.box_tiefe, box_breite=args.box_breite,
                       kreis_radius=args.kreis)

    input_path = Path(args.video)
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    source_video = str(input_path)
    reference_frame = args.frame
    view_frames = []
    view_H_to_ref = []
    view_labels = []

    if args.homographies or args.frames:
        if not (args.homographies and args.frames):
            parser.error("--homographies und --frames müssen gemeinsam angegeben werden")
        requested = [int(x.strip()) for x in args.frames.split(",") if x.strip()]
        with np.load(args.homographies) as homographies:
            H_all = homographies["H"]
            registered = (homographies["frames"] if "frames" in homographies
                          else np.arange(len(H_all)))
            reference_frame = int(homographies["ref"])
            H_by_frame = {int(f): H.copy() for f, H in zip(registered, H_all)}
        cap = cv2.VideoCapture(str(input_path))
        for frame_idx in requested:
            if frame_idx not in H_by_frame:
                raise SystemExit(f"Frame {frame_idx} fehlt in der Homographie-Datei")
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, image = cap.read()
            if not ok:
                raise SystemExit(f"Frame {frame_idx} nicht aus Video lesbar")
            view_frames.append(image)
            view_H_to_ref.append(H_by_frame[frame_idx])
            view_labels.append(f"Frame {frame_idx}")
        cap.release()
        print(f"Mehrbildmodus: {len(view_frames)} Frames, A/D wechselt die Ansicht")
    elif input_path.suffix.lower() in image_suffixes:
        frame = cv2.imread(str(input_path))
        if frame is None:
            raise SystemExit(f"Bild nicht lesbar: {input_path}")
        H_image_to_ref = np.eye(3)
        transform_path = input_path.with_name(f"{input_path.stem}_transform.json")
        if transform_path.exists():
            transform = json.loads(transform_path.read_text(encoding="utf-8"))
            H_ref_to_image = np.array(transform["H_ref_to_image"], dtype=np.float64)
            H_image_to_ref = np.linalg.inv(H_ref_to_image)
            source_video = transform.get("video", source_video)
            reference_frame = int(transform.get("ref", reference_frame))
            print(f"Panorama-Transformation geladen: {transform_path}")
        view_frames = [frame]
        view_H_to_ref = [H_image_to_ref]
        view_labels = [input_path.name]
    else:
        frames = sv.get_video_frames_generator(args.video, start=args.frame,
                                               end=args.frame + 1)
        frame = next(iter(frames))
        view_frames = [frame]
        view_H_to_ref = [np.eye(3)]
        view_labels = [f"Frame {args.frame}"]
    cal_dir = Path(__file__).resolve().parent.parent / "data" / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Spielfeld-Kalibrierung ===")
    print("Linksklick = Punkt setzen | A/D = Bild wechseln")
    print("Rechts-/Mittelklick oder S = überspringen")
    print("Esc = Kalibrierung abbrechen\n")

    points = []
    marked_frames = [image.copy() for image in view_frames]
    height, width = view_frames[0].shape[:2]
    display_scale = min(1.0, 1600 / width, 850 / height)
    display_size = (int(width * display_scale), int(height * display_scale))
    window = "FootballAnalytics - Platzkalibrierung"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    click = {"value": None}
    current_view = {"index": len(view_frames) // 2}

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["value"] = (x / display_scale, y / display_scale,
                              current_view["index"])
        elif event in (cv2.EVENT_MBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            click["value"] = "skip"

    cv2.setMouseCallback(window, on_mouse)
    for name, meter in pitch.landmarks().items():
        print(f"-> {name} ... ", end="", flush=True)
        click["value"] = None
        while click["value"] is None:
            idx = current_view["index"]
            display = cv2.resize(marked_frames[idx], display_size,
                                 interpolation=cv2.INTER_AREA)
            cv2.rectangle(display, (0, 0), (display.shape[1], 48), (0, 0, 0), -1)
            title = (f"{view_labels[idx]} ({idx + 1}/{len(view_frames)}) | "
                     f"Klicke: {name} | A/D wechseln")
            title = title.encode("ascii", "replace").decode("ascii")
            cv2.putText(display, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.72, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("s"), ord("S")):
                click["value"] = "skip"
            elif key in (ord("a"), ord("A")):
                current_view["index"] = (current_view["index"] - 1) % len(view_frames)
            elif key in (ord("d"), ord("D")):
                current_view["index"] = (current_view["index"] + 1) % len(view_frames)
            elif key == 27:
                cv2.destroyAllWindows()
                raise SystemExit("Kalibrierung abgebrochen")

        if click["value"] == "skip":
            print("übersprungen")
            continue
        px = [float(click["value"][0]), float(click["value"][1])]
        clicked_view = int(click["value"][2])
        ref_px = cv2.perspectiveTransform(
            np.array([[px]], dtype=np.float64), view_H_to_ref[clicked_view]
        ).reshape(2)
        points.append({"name": name, "px": ref_px.tolist(),
                       "clicked_px": px, "clicked_view": view_labels[clicked_view],
                       "meter": list(meter)})
        center = tuple(np.round(px).astype(int))
        cv2.drawMarker(marked_frames[clicked_view], center, (0, 0, 255),
                       cv2.MARKER_CROSS, 20, 2)
        cv2.putText(marked_frames[clicked_view], name, (center[0] + 8, center[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        print(f"({px[0]:.0f}, {px[1]:.0f}) in {view_labels[clicked_view]}")

    cv2.destroyAllWindows()
    if len(points) < 4:
        raise SystemExit(f"Nur {len(points)} Punkte — mindestens 4 nötig. Abbruch.")

    src_m = np.array([p["meter"] for p in points], dtype=np.float64).reshape(-1, 1, 2)
    dst_px = np.array([p["px"] for p in points], dtype=np.float64).reshape(-1, 1, 2)
    H_pitch_to_px, _ = cv2.findHomography(src_m, dst_px, 0)

    # Reprojektionsfehler je Punkt
    proj = cv2.perspectiveTransform(src_m, H_pitch_to_px).reshape(-1, 2)
    errors = np.linalg.norm(proj - dst_px.reshape(-1, 2), axis=1)
    print("\nReprojektionsfehler (Pixel):")
    for p, e in zip(points, errors):
        print(f"  {p['name']}: {e:.1f}")
    print(f"  Mittel: {errors.mean():.1f} px")

    data = {
        "video": source_video,
        "frame": reference_frame,
        "pitch": pitch.to_dict(),
        "points": points,
        "H_pitch_to_px": H_pitch_to_px.tolist(),
    }
    json_path = cal_dir / f"{args.name}.json"
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    ref_png = cal_dir / f"{args.name}_referenzframe.png"
    cv2.imwrite(str(ref_png), view_frames[len(view_frames) // 2])

    overlay_paths = []
    for image, H_view_to_ref, label in zip(view_frames, view_H_to_ref, view_labels):
        H_pitch_to_view = np.linalg.inv(H_view_to_ref) @ H_pitch_to_px
        overlay = pitch.draw_overlay(image, H_pitch_to_view)
        safe_label = label.lower().replace(" ", "_")
        overlay_path = cal_dir / f"{args.name}_kontrolle_{safe_label}.jpg"
        cv2.imwrite(str(overlay_path), overlay)
        overlay_paths.append(overlay_path)
    print(f"\nGespeichert: {json_path}")
    print(f"Referenzframe: {ref_png}")
    print("Kontrollbilder (projizierte Linien):")
    for path in overlay_paths:
        print(f"  {path}")
    print("-> Kontrollbild anschauen: Liegen die gelben Linien auf den weißen?")


if __name__ == "__main__":
    main()
