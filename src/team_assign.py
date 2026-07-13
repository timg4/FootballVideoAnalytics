"""Phase 3 (Vorgriff): Team-Zuordnung über Trikotfarben + Platz-Filter.

Liest die Tracking-CSV aus detect_track.py, bestimmt pro Track die Trikotfarbe
(Torso-Ausschnitt) und clustert alle Tracks in mehrere Farbgruppen.
Wenn eine Positions-CSV aus pitch_map.py übergeben wird, fließen nur
Detektionen mit auf_platz=1 in die Farbanalyse ein. Der alte Größenfilter
bleibt als zusätzliche Plausibilitätsprüfung und für Legacy-Aufrufe erhalten.

Farb-Merkmale sind helligkeits-normalisiert (Farbanteile + Sättigung), damit
Dämmerlicht die Teams nicht verwischt: Weiß/Grau hat niedrige Sättigung,
Pink hohen Rot-Anteil, Grün hohen Grün-Anteil — unabhängig davon, wie dunkel
der Spieler gerade im Bild ist.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

MIN_SAMPLE_HEIGHT_1080 = 45  # Mindest-Boxhöhe für Farbproben, bezogen auf 1080p
MIN_SAMPLES = 3          # Tracks mit weniger Farbproben bleiben unzugeordnet
OTHER = 2                # class_id für alles außerhalb unseres Spiels


def torso_crop(frame, box):
    """Torso-Ausschnitt einer Bounding Box (unterhalb Kopf, ohne Arme/Beine)."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = y2 - y1, x2 - x1
    ty1, ty2 = y1 + int(0.20 * h), y1 + int(0.50 * h)
    tx1, tx2 = x1 + int(0.25 * w), x2 - int(0.25 * w)
    crop = frame[max(ty1, 0):ty2, max(tx1, 0):tx2]
    return crop if crop.size else None


def color_features(crop):
    """Helligkeits-normalisierte Merkmale: (Rot-Anteil, Grün-Anteil, Sättigung).

    Median statt Mittelwert, damit Rasen-/Hintergrundpixel im Ausschnitt
    das Trikot nicht verfälschen. Skalierung so, dass alle drei Achsen
    im K-Means vergleichbares Gewicht haben.
    """
    med = np.median(crop.reshape(-1, 3).astype(np.float64), axis=0)  # BGR
    b, g, r = med
    total = b + g + r + 1e-6
    saturation = (med.max() - med.min()) / (med.max() + 1e-6)
    return np.array([r / total * 300, g / total * 300, saturation * 150]), med


def kmeans(points, k, restarts=10, iters=100, seed=0):
    """K-Means mit mehreren Starts, bester Lauf (kleinste Streuung) gewinnt."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        centers = points[rng.choice(len(points), k, replace=False)]
        for _ in range(iters):
            dists = ((points[:, None, :] - centers[None]) ** 2).sum(axis=-1)
            labels = dists.argmin(axis=1)
            new_centers = np.array([
                points[labels == j].mean(axis=0) if (labels == j).any() else centers[j]
                for j in range(k)])
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        inertia = ((points - centers[labels]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, labels, centers)
    return best[1], best[2]


def save_cluster_mosaics(track_crops, track_ids, labels, k, out_dir):
    """Pro Farbgruppe eine Kachelübersicht der Torso-Ausschnitte speichern."""
    tile_w, tile_h, cols = 32, 48, 12
    for j in range(k):
        tiles = []
        for tid, lab in zip(track_ids, labels):
            if lab == j:
                tiles.extend(track_crops[tid][:3])
        if not tiles:
            continue
        tiles = tiles[:cols * 6]
        rows = -(-len(tiles) // cols)
        mosaic = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
        for n, tile in enumerate(tiles):
            r, c = divmod(n, cols)
            mosaic[r * tile_h:(r + 1) * tile_h,
                   c * tile_w:(c + 1) * tile_w] = cv2.resize(tile, (tile_w, tile_h))
        path = out_dir / f"debug_farbgruppe_{j}.jpg"
        cv2.imwrite(str(path), mosaic)
        print(f"  Kachelübersicht: {path}")


def main():
    parser = argparse.ArgumentParser(description="Team-Zuordnung über Trikotfarben")
    parser.add_argument("video", help="Original-Video (nicht das annotierte!)")
    parser.add_argument("tracks_csv", help="Tracking-CSV aus detect_track.py")
    parser.add_argument("--positions-csv", default=None,
                        help="Positions-CSV aus pitch_map.py; verwendet nur auf_platz=1")
    parser.add_argument("--output", default=None)
    parser.add_argument("--assignments-output", default=None,
                        help="Ausgabe-CSV der Tracklet-Teamzuordnung")
    parser.add_argument("--clusters", type=int, default=5,
                        help="Anzahl Farbgruppen (2 Teams + Nachbarspiele/Sonstige)")
    parser.add_argument("--debug", action="store_true",
                        help="Pro Farbgruppe eine Trikot-Kachelübersicht speichern")
    parser.add_argument("--zeige-ignorierte", action="store_true",
                        help="Aussortierte Personen (Nachbarspiele) grau mit anzeigen")
    parser.add_argument("--no-video", action="store_true",
                        help="Nur Teams bestimmen und CSV schreiben, kein Video rendern")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(__file__).resolve().parent.parent / "data" / "output"
    output_path = Path(args.output) if args.output else out_dir / f"{video_path.stem}_teams.mp4"
    prefix = Path(args.tracks_csv).stem.replace("_tracked", "")
    assignments_path = (Path(args.assignments_output) if args.assignments_output
                        else out_dir / f"{prefix}_team_assignments.csv")

    allowed = None
    if args.positions_csv:
        allowed = set()
        with open(args.positions_csv, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if int(row["auf_platz"]):
                    allowed.add((int(row["frame"]), int(row["tracker_id"])))
        print(f"Platzfilter geladen: {len(allowed)} Detektionen mit auf_platz=1")

    # Tracking-Daten laden: frame -> Liste (tracker_id, box)
    per_frame = defaultdict(list)
    with open(args.tracks_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            tid = int(row["tracker_id"])
            if allowed is not None and (frame, tid) not in allowed:
                continue
            per_frame[frame].append(
                (tid,
                 (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))))
    if not per_frame:
        raise SystemExit("Keine Tracking-Detektionen nach dem Platzfilter übrig.")
    min_frame, max_frame = min(per_frame), max(per_frame)

    video_info = sv.VideoInfo.from_video_path(str(video_path))
    min_sample_height = MIN_SAMPLE_HEIGHT_1080 * video_info.height / 1080

    # Pass 1: Farbmerkmale (nur große Boxen) und Boxhöhen pro Track sammeln
    print("Pass 1: Trikotfarben sammeln ...")
    track_feats = defaultdict(list)
    track_raw = defaultdict(list)
    track_heights = defaultdict(list)
    track_crops = defaultdict(list)
    frames = sv.get_video_frames_generator(str(video_path), start=min_frame,
                                           end=max_frame + 1)
    for i, frame in enumerate(frames, start=min_frame):
        for tid, box in per_frame.get(i, []):
            height = box[3] - box[1]
            track_heights[tid].append(height)
            if height < min_sample_height:
                continue
            crop = torso_crop(frame, box)
            if crop is None:
                continue
            feat, raw = color_features(crop)
            track_feats[tid].append(feat)
            track_raw[tid].append(raw)
            if args.debug and len(track_crops[tid]) < 3 and i % 20 == 0:
                track_crops[tid].append(crop.copy())

    track_ids = sorted(t for t in track_feats if len(track_feats[t]) >= MIN_SAMPLES)
    unassigned = set(track_heights) - set(track_ids)
    print(f"{len(track_ids)} Tracks mit genug Farbproben, "
          f"{len(unassigned)} zu klein/kurz -> ignoriert")

    median_feats = np.array([np.median(track_feats[t], axis=0) for t in track_ids])
    median_heights = np.array([np.median(track_heights[t]) for t in track_ids])

    # In Farbgruppen clustern. Unser Spiel = die Gruppen mit den größten
    # Spielern (nah an der Kamera). Ein Team kann dabei in mehrere Farbgruppen
    # zerfallen (helle vs. abgeschattete Trikots), deshalb: alle ausreichend
    # großen Gruppen als Kandidaten nehmen und die farblich nächsten
    # verschmelzen, bis genau 2 Teams übrig sind.
    labels, centers = kmeans(median_feats, k=args.clusters)
    cluster_height = np.array([
        np.median(median_heights[labels == j]) if (labels == j).any() else 0
        for j in range(args.clusters)])
    # Für das Verschmelzen abgeschatteter Varianten derselben Trikotfarbe
    # sind reine Farbanteile robuster als das komplette K-Means-Merkmal:
    # dessen Sättigungsachse kann z.B. ein hellblaues Cluster fälschlich
    # näher an Grün als an ein dunkleres Blau rücken.
    track_chroma = np.array([
        np.median(track_raw[t], axis=0) /
        max(np.median(track_raw[t], axis=0).sum(), 1e-6)
        for t in track_ids])
    cluster_chroma = np.array([
        np.median(track_chroma[labels == j], axis=0)
        if (labels == j).any() else np.zeros(3)
        for j in range(args.clusters)])

    tallest = cluster_height.max()
    candidates = [j for j in range(args.clusters)
                  if cluster_height[j] >= 0.72 * tallest and (labels == j).sum() >= 2]
    if len(candidates) < 2:
        candidates = list(np.argsort(cluster_height)[-2:])
    groups = [{j} for j in candidates]
    while len(groups) > 2:
        best = None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                dist = min(np.linalg.norm(cluster_chroma[x] - cluster_chroma[y])
                           for x in groups[a] for y in groups[b])
                if best is None or dist < best[0]:
                    best = (dist, a, b)
        _, a, b = best
        groups[a] |= groups[b]
        groups.pop(b)
    groups.sort(key=lambda grp: -sum((labels == j).sum() for j in grp))

    team_of = defaultdict(lambda: OTHER)
    for tid, lab in zip(track_ids, labels):
        team_of[tid] = 0 if lab in groups[0] else 1 if lab in groups[1] else OTHER

    # Größen-Plausibilität: In der Ferne entsättigen alle Farben zu Grau und
    # ähneln dann v.a. dem weißen Team. Wer deutlich kleiner ist als der
    # Median seines Teams, steht nicht auf unserem Platz -> ignorieren.
    height_of = dict(zip(track_ids, median_heights))
    for team in (0, 1):
        member_heights = [height_of[t] for t in track_ids if team_of[t] == team]
        cutoff = 0.62 * np.median(member_heights)
        demoted = [t for t in track_ids
                   if team_of[t] == team and height_of[t] < cutoff]
        for t in demoted:
            team_of[t] = OTHER
        if demoted:
            print(f"Team {team}: {len(demoted)} zu kleine Tracks aussortiert "
                  f"(< {cutoff:.0f}px): {demoted}")

    for j in range(args.clusters):
        members = labels == j
        raw = np.median(np.vstack([np.median(track_raw[t], axis=0)
                                   for t, m in zip(track_ids, members) if m]), axis=0) \
            if members.any() else np.zeros(3)
        status = ("TEAM 0" if j in groups[0] else
                  "TEAM 1" if j in groups[1] else "ignoriert")
        print(f"Farbgruppe {j}: Rot/Grün-Anteil+Sättigung={centers[j].round(0)}, "
              f"typisch BGR={raw.astype(int)}, {members.sum():3d} Tracks, "
              f"mittlere Größe {cluster_height[j]:.0f}px -> {status}")

    if args.debug:
        save_cluster_mosaics(track_crops, track_ids, labels, args.clusters, out_dir)

    label_of = dict(zip(track_ids, labels))
    with open(assignments_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tracker_id", "team", "farbgruppe", "farbproben",
                    "median_boxhoehe_px"])
        for tid in sorted(track_heights):
            team = team_of[tid]
            w.writerow([tid, team if team != OTHER else "",
                        label_of.get(tid, ""), len(track_feats.get(tid, [])),
                        f"{np.median(track_heights[tid]):.2f}"])
    print(f"Team-Zuordnungen: {assignments_path}")

    if args.no_video:
        print("Videoausgabe mit --no-video übersprungen.")
        return

    # Pass 2: Video annotieren. Kräftige Signalfarben statt der echten
    # Trikotfarben — im Dämmerlicht sind die sonst kaum unterscheidbar
    print("Pass 2: Video annotieren ...")
    palette = sv.ColorPalette(colors=[
        sv.Color(r=230, g=40, b=40),    # Team 0: rot
        sv.Color(r=40, g=120, b=255),   # Team 1: blau
        sv.Color(r=110, g=110, b=110),  # Nachbarspiele/Sonstige: grau
    ])
    ellipse = sv.EllipseAnnotator(color=palette, color_lookup=sv.ColorLookup.CLASS)
    label_annotator = sv.LabelAnnotator(color=palette, color_lookup=sv.ColorLookup.CLASS,
                                        text_position=sv.Position.BOTTOM_CENTER, text_scale=0.4)

    frames = sv.get_video_frames_generator(str(video_path), start=min_frame,
                                           end=max_frame + 1)
    with sv.VideoSink(str(output_path), video_info) as sink:
        for i, frame in enumerate(frames, start=min_frame):
            entries = per_frame.get(i, [])
            if not args.zeige_ignorierte:
                entries = [(tid, box) for tid, box in entries
                           if team_of[tid] != OTHER]
            if entries:
                detections = sv.Detections(
                    xyxy=np.array([box for _, box in entries]),
                    class_id=np.array([team_of[tid] for tid, _ in entries]),
                    tracker_id=np.array([tid for tid, _ in entries]))
                labels_txt = [f"#{tid}" for tid, _ in entries]
                frame = ellipse.annotate(frame.copy(), detections)
                frame = label_annotator.annotate(frame, detections, labels=labels_txt)
            sink.write_frame(frame)

    print(f"Fertig: {output_path}")


if __name__ == "__main__":
    main()
