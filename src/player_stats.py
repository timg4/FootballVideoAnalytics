"""Manuell bestätigte Tracklet-Zuordnungen zu Spielerwerten aggregieren."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Spielerstatistiken aus Mapping-CSV")
    parser.add_argument("mapping_csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = []
    with open(args.mapping_csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    totals = defaultdict(lambda: {"distance": 0.0, "duration": 0.0,
                                  "tracklets": 0, "manual": 0, "auto": 0,
                                  "similarities": []})
    for row in rows:
        name = row["spieler_name_oder_nummer"].strip()
        if not name:
            continue
        key = (int(row["team"]), name)
        totals[key]["distance"] += float(row["distanz_m"])
        totals[key]["duration"] += float(row["sichtbare_dauer_s"])
        totals[key]["tracklets"] += 1
        source = row.get("quelle", "manuell")
        if source == "automatisch":
            totals[key]["auto"] += 1
            if row.get("reid_similarity"):
                totals[key]["similarities"].append(float(row["reid_similarity"]))
        else:
            totals[key]["manual"] += 1

    output = (Path(args.output) if args.output else
              Path(args.mapping_csv).with_name(
                  Path(args.mapping_csv).stem.replace("_mapping", "_statistiken") + ".csv"))
    with open(output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["team", "spieler", "sichtbare_distanz_km",
                    "sichtbare_dauer_min", "zugeordnete_tracklets",
                    "manuelle_tracklets", "automatische_tracklets",
                    "mittlere_auto_similarity"])
        for (team, name), values in sorted(
                totals.items(), key=lambda item: (item[0][0], -item[1]["distance"])):
            mean_similarity = (sum(values["similarities"]) /
                               len(values["similarities"])) \
                if values["similarities"] else 0.0
            w.writerow([team, name, f"{values['distance'] / 1000:.3f}",
                        f"{values['duration'] / 60:.2f}", values["tracklets"],
                        values["manual"], values["auto"],
                        f"{mean_similarity:.3f}" if values["auto"] else ""])
    print(f"Spielerstatistiken: {output}")
    for (team, name), values in sorted(
            totals.items(), key=lambda item: (item[0][0], -item[1]["distance"])):
        print(f"  Team {team} · {name}: {values['distance'] / 1000:.3f} km "
              f"in {values['duration'] / 60:.1f} min sichtbar")


if __name__ == "__main__":
    main()
