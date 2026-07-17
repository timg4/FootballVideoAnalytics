"""Passkandidaten per validierten Confidence-Regeln vorsortieren.

Die Regeln wurden auf dem ersten manuellen Review kalibriert. Sie ersetzen
nicht die Kontaktspieler-/Teamzuordnung: `auto_pass` bedeutet nur, dass der
Bewegungsablauf sehr wahrscheinlich ein Pass ist. Fälle außerhalb der sicheren
Bereiche bleiben explizit `review`.
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Passkandidaten automatisch vorsortieren")
    parser.add_argument("passes_csv")
    parser.add_argument("ball_track_csv")
    parser.add_argument("--review-csv",
                        help="Optional: manuelle Labels zur Qualitätsmessung")
    parser.add_argument("--output", default=None)
    parser.add_argument("--pass-min-duration", type=float, default=0.30)
    parser.add_argument("--pass-min-distance", type=float, default=2.0)
    parser.add_argument("--pass-max-interpolation", type=float, default=0.60)
    parser.add_argument("--no-max-duration", type=float, default=0.50)
    parser.add_argument("--no-max-distance", type=float, default=6.0)
    parser.add_argument("--no-min-interpolation", type=float, default=0.70)
    parser.add_argument("--pitch-length", type=float, default=55.75)
    parser.add_argument("--shot-goal-zone-m", type=float, default=10.0,
                        help="Goalward-Transfers in dieser Endzone bleiben Review")
    args = parser.parse_args()

    passes_path = Path(args.passes_csv)
    output_path = (Path(args.output) if args.output else
                   passes_path.with_name(passes_path.stem + "_auto.csv"))

    track_by_id = defaultdict(list)
    with open(args.ball_track_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            track_by_id[int(row["ball_tracklet"])].append(row)

    with open(passes_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    output_rows = []
    for row in rows:
        tracklet = int(row["ball_tracklet"])
        start, end = int(row["abgabe_frame"]), int(row["annahme_frame"])
        points = [point for point in track_by_id[tracklet]
                  if start <= int(point["frame"]) <= end]
        interpolation = (sum(int(point["interpolated"]) for point in points) /
                         len(points) if points else 1.0)
        max_conf = max((float(point["conf"]) for point in points), default=0.0)
        x_start = float(points[0]["x_m"]) if points else args.pitch_length / 2
        x_end = float(points[-1]["x_m"]) if points else args.pitch_length / 2
        potential_shot = (
            (x_end > x_start and
             x_end >= args.pitch_length - args.shot_goal_zone_m) or
            (x_end < x_start and x_end <= args.shot_goal_zone_m)
        )
        duration = float(row["dauer_s"])
        distance = float(row["distanz_m"])

        if potential_shot:
            auto_status = "review_shot"
            reason = "goalward in Tor-Endzone: Pass/Schuss visuell trennen"
        elif (duration >= args.pass_min_duration and
                distance >= args.pass_min_distance and
                interpolation <= args.pass_max_interpolation):
            auto_status = "auto_pass"
            reason = (f"Dauer>={args.pass_min_duration:.2f}s, "
                      f"Distanz>={args.pass_min_distance:.1f}m, "
                      f"Interpolation<={args.pass_max_interpolation:.2f}")
        elif (duration <= args.no_max_duration and
              distance <= args.no_max_distance and
              interpolation >= args.no_min_interpolation):
            auto_status = "auto_no"
            reason = (f"Dauer<={args.no_max_duration:.2f}s, "
                      f"Distanz<={args.no_max_distance:.1f}m, "
                      f"Interpolation>={args.no_min_interpolation:.2f}")
        else:
            auto_status = "review"
            reason = "außerhalb der validierten sicheren Bereiche"

        output_rows.append({
            **row,
            "interpolationsanteil": f"{interpolation:.3f}",
            "max_ball_conf": f"{max_conf:.3f}",
            "auto_status": auto_status,
            "auto_begruendung": reason,
            "potenzieller_schuss": int(potential_shot),
            "event_review_noetig": int(auto_status.startswith("review")),
            # Kontaktteam ist mit der aktuellen Näherungslogik nur ca. 74-79 %
            # korrekt und erhält deshalb bewusst ein eigenes Warnflag.
            "kontaktteam_nicht_validiert": int(auto_status == "auto_pass"),
        })

    fields = list(output_rows[0]) if output_rows else []
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    counts = Counter(row["auto_status"] for row in output_rows)
    print(f"Automatische Vorsortierung: {dict(counts)} -> {output_path}")

    if args.review_csv:
        with open(args.review_csv, newline="", encoding="utf-8-sig") as f:
            review = {(row["ball_tracklet"], row["abgabe_frame"], row["annahme_frame"]):
                      row["review_status"] for row in csv.DictReader(f)}
        measured = Counter()
        for row in output_rows:
            key = (row["ball_tracklet"], row["abgabe_frame"], row["annahme_frame"])
            label = review.get(key, "")
            if label in ("pass", "no"):
                measured[(row["auto_status"], label)] += 1
        print("Messung gegen manuelles Review:")
        for key, count in sorted(measured.items()):
            print(f"  {key[0]} / tatsächlich {key[1]}: {count}")
        print("Hinweis: In-Sample-Kalibrierung; für neue Spiele konservativ verwenden.")


if __name__ == "__main__":
    main()
