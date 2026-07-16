"""Parametric pitch model for calibration, overlays, and top-down maps.

Coordinate system: x runs along the pitch length (0 = left goal from the camera's
view), y across the width (0 = far touchline, B = near touchline at the camera).
All dimensions in meters. The defaults are a small-pitch estimate and get
corrected once the real field dimensions are known.

The German attribute and landmark names are the schema used in the calibration
JSON files (PitchModel(**cal["pitch"]) and the point labels), so they stay put as
stable identifiers.
"""

from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class PitchModel:
    # cross-pitch: played across the width of a full-size field (~68 m),
    # 3 pitches per full field with ~1 m gap, so ~34 m deep each
    laenge: float = 68.0
    breite: float = 34.0
    tor_breite: float = 5.0
    box_tiefe: float = 9.0      # penalty box: depth in front of the goal (estimated)
    box_breite: float = 24.0    # penalty box: width (estimated)
    kreis_radius: float = 5.0   # center circle radius (estimated)

    def landmarks(self):
        """Named reference points (name -> model coordinate in meters)."""
        L, B = self.laenge, self.breite
        t2 = self.tor_breite / 2
        bb2 = self.box_breite / 2
        bt = self.box_tiefe
        return {
            "Ecke links-fern": (0, 0),
            "Ecke links-nah": (0, B),
            "Ecke rechts-fern": (L, 0),
            "Ecke rechts-nah": (L, B),
            "Mittellinie x ferne Seitenlinie": (L / 2, 0),
            "Mittellinie x nahe Seitenlinie": (L / 2, B),
            "Mittelpunkt (Anstosspunkt)": (L / 2, B / 2),
            "Mittelkreis: Punkt links": (L / 2 - self.kreis_radius, B / 2),
            "Mittelkreis: Punkt rechts": (L / 2 + self.kreis_radius, B / 2),
            "Mittelkreis: Punkt fern": (L / 2, B / 2 - self.kreis_radius),
            "Mittelkreis: Punkt nah": (L / 2, B / 2 + self.kreis_radius),
            "Tor links: Pfosten fern": (0, B / 2 - t2),
            "Tor links: Pfosten nah": (0, B / 2 + t2),
            "Tor rechts: Pfosten fern": (L, B / 2 - t2),
            "Tor rechts: Pfosten nah": (L, B / 2 + t2),
            "Strafraum links: Ecke fern": (bt, B / 2 - bb2),
            "Strafraum links: Ecke nah": (bt, B / 2 + bb2),
            "Strafraum rechts: Ecke fern": (L - bt, B / 2 - bb2),
            "Strafraum rechts: Ecke nah": (L - bt, B / 2 + bb2),
        }

    def lines(self, n_circle=48):
        """All lines of the model as a list of polylines (meters)."""
        L, B = self.laenge, self.breite
        bb2, bt = self.box_breite / 2, self.box_tiefe
        out = [
            [(0, 0), (L, 0), (L, B), (0, B), (0, 0)],          # outer lines
            [(L / 2, 0), (L / 2, B)],                          # halfway line
            [(0, B / 2 - bb2), (bt, B / 2 - bb2),
             (bt, B / 2 + bb2), (0, B / 2 + bb2)],             # penalty box left
            [(L, B / 2 - bb2), (L - bt, B / 2 - bb2),
             (L - bt, B / 2 + bb2), (L, B / 2 + bb2)],         # penalty box right
        ]
        angles = np.linspace(0, 2 * np.pi, n_circle)
        out.append([(L / 2 + self.kreis_radius * np.cos(a),
                     B / 2 + self.kreis_radius * np.sin(a)) for a in angles])
        return out

    def draw_overlay(self, frame, h_pitch_to_px, color=(0, 255, 255), thickness=2):
        """Project the model lines into the image via a homography (meter -> pixel)."""
        img = frame.copy()
        for line in self.lines():
            pts = np.array(line, dtype=np.float64).reshape(-1, 1, 2)
            px = cv2.perspectiveTransform(pts, h_pitch_to_px).reshape(-1, 2)
            for a, b in zip(px, px[1:]):
                if all(np.isfinite(a)) and all(np.isfinite(b)):
                    cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)),
                             color, thickness)
        return img

    def draw_topdown(self, scale=12, margin=30):
        """Top-down view of the pitch as an image (scale = pixels per meter)."""
        w = int(self.laenge * scale) + 2 * margin
        h = int(self.breite * scale) + 2 * margin
        img = np.full((h, w, 3), (40, 90, 40), dtype=np.uint8)

        def to_px(p):
            return (int(p[0] * scale) + margin, int(p[1] * scale) + margin)

        for line in self.lines():
            for a, b in zip(line, line[1:]):
                cv2.line(img, to_px(a), to_px(b), (255, 255, 255), 2)
        return img

    def to_dict(self):
        return asdict(self)
