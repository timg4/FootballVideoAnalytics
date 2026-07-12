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
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv

from pitch_model import PitchModel


def main():
    parser = argparse.ArgumentParser(description="Spielfeld-Kalibrierung per Klick")
    parser.add_argument("video")
    parser.add_argument("--frame", type=int, default=0,
                        help="Referenzframe (muss zum --ref der Registrierung passen!)")
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

    frames = sv.get_video_frames_generator(args.video, start=args.frame,
                                           end=args.frame + 1)
    frame = next(iter(frames))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cal_dir = Path(__file__).resolve().parent.parent / "data" / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(rgb)
    ax.set_title("Kalibrierung — Anweisungen in der Konsole beachten")
    plt.tight_layout()

    print("\n=== Spielfeld-Kalibrierung ===")
    print("Linksklick = Punkt setzen | Mittelklick = überspringen")
    print("Tipp: Mit dem Lupen-Werkzeug zoomen, dann Lupe DEAKTIVIEREN, dann klicken.\n")

    points = []
    for name, meter in pitch.landmarks().items():
        ax.set_title(f"Klicke: {name}   (Mittelklick = überspringen)")
        fig.canvas.draw()
        print(f"-> {name} ... ", end="", flush=True)
        clicks = plt.ginput(1, timeout=0, mouse_stop=2, mouse_pop=3)
        if not clicks:
            print("übersprungen")
            continue
        px = [float(clicks[0][0]), float(clicks[0][1])]
        points.append({"name": name, "px": px, "meter": list(meter)})
        ax.plot(px[0], px[1], "r+", markersize=14, markeredgewidth=2)
        ax.annotate(name, px, color="yellow", fontsize=8, xytext=(6, -6),
                    textcoords="offset points")
        fig.canvas.draw()
        print(f"({px[0]:.0f}, {px[1]:.0f})")

    plt.close(fig)
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
        "video": str(args.video),
        "frame": args.frame,
        "pitch": pitch.to_dict(),
        "points": points,
        "H_pitch_to_px": H_pitch_to_px.tolist(),
    }
    json_path = cal_dir / f"{args.name}.json"
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    ref_png = cal_dir / f"{args.name}_referenzframe.png"
    cv2.imwrite(str(ref_png), frame)

    overlay = pitch.draw_overlay(frame, H_pitch_to_px)
    overlay_path = cal_dir / f"{args.name}_kontrolle.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    print(f"\nGespeichert: {json_path}")
    print(f"Referenzframe: {ref_png}")
    print(f"Kontrollbild (projizierte Linien): {overlay_path}")
    print("-> Kontrollbild anschauen: Liegen die gelben Linien auf den weißen?")


if __name__ == "__main__":
    main()
