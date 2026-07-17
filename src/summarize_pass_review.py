"""Manuell geprüfte Pass-/Schusskarten zu sichtbaren Statistiken aggregieren."""

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path


TEAM_NAMES = {"0": "Grün", "1": "Blau", "": "Unklar"}


def normalize_player(value, team):
    value = value.strip()
    if not value:
        return f"Unbekannt {TEAM_NAMES.get(team, 'Team')}"
    key = value.casefold().replace(" ", "")
    aliases = {
        "patrick": "Patrick abe",
        "patrickabe": "Patrick abe",
        "ach/bre": "Ach/Bre unklar",
        "ach/breunklar": "Ach/Bre unklar",
    }
    return aliases.get(key, value)


def pct(numerator, denominator):
    return 100 * numerator / denominator if denominator else 0.0


def main():
    parser = argparse.ArgumentParser(description="Validierte Passstatistik")
    parser.add_argument("review_csv")
    parser.add_argument("--output-prefix", default="video_project")
    args = parser.parse_args()

    review_path = Path(args.review_csv)
    out_dir = review_path.parent
    with open(review_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    status_counts = Counter(row["review_status"] for row in rows)
    team_stats = defaultdict(lambda: Counter({
        "passversuche": 0, "angekommen": 0, "fehlpaesse": 0,
        "ergebnis_unklar": 0, "schuesse": 0,
    }))
    player_stats = defaultdict(lambda: Counter({
        "passversuche": 0, "angekommen": 0, "fehlpaesse": 0,
        "erhalten": 0, "schuesse": 0,
    }))
    player_team = {}
    event_rows = []

    for row in rows:
        status = row["review_status"]
        from_team = row["review_von_team"].strip()
        to_team = row["review_zu_team"].strip()
        passer = normalize_player(row["review_von_spieler"], from_team)
        receiver = normalize_player(row["review_zu_spieler"], to_team)

        if status == "shot":
            team_stats[from_team]["schuesse"] += 1
            player_stats[passer]["schuesse"] += 1
            player_team[passer] = from_team
            event_rows.append({
                "frame": row["abgabe_frame"], "typ": "Schuss",
                "von_team": TEAM_NAMES.get(from_team, "Unklar"),
                "zu_team": "", "von_spieler": passer, "zu_spieler": "",
                "ergebnis": "Schuss", "distanz_m": row["distanz_m"],
            })
            continue
        if status != "pass":
            continue

        team_stats[from_team]["passversuche"] += 1
        player_stats[passer]["passversuche"] += 1
        player_team[passer] = from_team
        if from_team != "" and to_team != "":
            if from_team == to_team:
                outcome = "angekommen"
                team_stats[from_team]["angekommen"] += 1
                player_stats[passer]["angekommen"] += 1
                player_stats[receiver]["erhalten"] += 1
                player_team[receiver] = to_team
            else:
                outcome = "Fehlpass/Ballverlust"
                team_stats[from_team]["fehlpaesse"] += 1
                player_stats[passer]["fehlpaesse"] += 1
        else:
            outcome = "Ergebnis unklar"
            team_stats[from_team]["ergebnis_unklar"] += 1

        event_rows.append({
            "frame": row["abgabe_frame"], "typ": "Pass",
            "von_team": TEAM_NAMES.get(from_team, "Unklar"),
            "zu_team": TEAM_NAMES.get(to_team, "Unklar"),
            "von_spieler": passer, "zu_spieler": receiver,
            "ergebnis": outcome, "distanz_m": row["distanz_m"],
        })

    events_path = out_dir / f"{args.output_prefix}_pass_events_validiert.csv"
    event_fields = ["frame", "typ", "von_team", "zu_team", "von_spieler",
                    "zu_spieler", "ergebnis", "distanz_m"]
    with open(events_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=event_fields)
        writer.writeheader()
        writer.writerows(sorted(event_rows, key=lambda row: int(row["frame"])))

    team_rows = []
    for team in sorted(team_stats, key=lambda value: (value == "", value)):
        values = team_stats[team]
        decided = values["angekommen"] + values["fehlpaesse"]
        team_rows.append({
            "team": TEAM_NAMES.get(team, "Unklar"),
            "bestaetigte_passversuche": values["passversuche"],
            "angekommen": values["angekommen"],
            "fehlpaesse": values["fehlpaesse"],
            "ergebnis_unklar": values["ergebnis_unklar"],
            "passquote_pct": pct(values["angekommen"], decided),
            "bestaetigte_schuesse": values["schuesse"],
        })
    team_path = out_dir / f"{args.output_prefix}_pass_statistik_sichtbar.csv"
    team_fields = list(team_rows[0]) if team_rows else []
    with open(team_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=team_fields)
        writer.writeheader()
        for row in team_rows:
            writer.writerow({key: (f"{value:.1f}" if isinstance(value, float) else value)
                             for key, value in row.items()})

    player_rows = []
    for player, values in sorted(player_stats.items(),
                                 key=lambda item: (-item[1]["passversuche"], item[0])):
        decided = values["angekommen"] + values["fehlpaesse"]
        player_rows.append({
            "spieler": player,
            "team": TEAM_NAMES.get(player_team.get(player, ""), "Unklar"),
            "passversuche": values["passversuche"],
            "angekommen": values["angekommen"],
            "fehlpaesse": values["fehlpaesse"],
            "passquote_pct": pct(values["angekommen"], decided),
            "angekommene_paesse_erhalten": values["erhalten"],
            "schuesse": values["schuesse"],
        })
    player_path = out_dir / f"{args.output_prefix}_spieler_pass_statistik_sichtbar.csv"
    player_fields = list(player_rows[0]) if player_rows else []
    with open(player_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=player_fields)
        writer.writeheader()
        for row in player_rows:
            writer.writerow({key: (f"{value:.1f}" if isinstance(value, float) else value)
                             for key, value in row.items()})

    def table(rows_to_render, columns):
        head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
        def display(value):
            return f"{value:.1f}" if isinstance(value, float) else str(value)
        body = "".join("<tr>" + "".join(
            f"<td>{html.escape(display(row[key]))}</td>" for key, _ in columns
        ) + "</tr>" for row in rows_to_render)
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    warning = ""
    if status_counts["ungeprüft"]:
        frames = [row["abgabe_frame"] for row in rows
                  if row["review_status"] == "ungeprüft"]
        warning = (f"<p class='warn'>{status_counts['ungeprüft']} Karte ungeprüft "
                   f"(Abgabe-Frame {', '.join(frames)}) und ausgeschlossen.</p>")
    team_table = table(team_rows, [
        ("team", "Team"), ("bestaetigte_passversuche", "Pässe"),
        ("angekommen", "Angekommen"), ("fehlpaesse", "Fehlpässe"),
        ("passquote_pct", "Quote %"), ("bestaetigte_schuesse", "Schüsse"),
    ])
    player_table = table(player_rows, [
        ("spieler", "Spieler"), ("team", "Team"),
        ("passversuche", "Pässe"), ("angekommen", "Angekommen"),
        ("fehlpaesse", "Fehlpässe"), ("passquote_pct", "Quote %"),
        ("angekommene_paesse_erhalten", "Erhalten"), ("schuesse", "Schüsse"),
    ])
    dashboard_path = out_dir / f"{args.output_prefix}_pass_dashboard.html"
    dashboard_path.write_text(f"""<!doctype html><html lang='de'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Validierte sichtbare Passstatistik</title><style>body{{font:15px system-ui;max-width:1100px;margin:25px auto;padding:0 16px;background:#0f172a;color:#f8fafc}}h1,h2{{margin-bottom:8px}}p{{color:#cbd5e1}}.warn{{color:#fde68a;font-weight:650}}table{{width:100%;border-collapse:collapse;background:#1e293b;margin-bottom:25px}}th,td{{padding:9px;border:1px solid #334155;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#111827}}</style></head><body><h1>Validierte sichtbare Passstatistik</h1><p>Nur manuell bestätigte Pass-/Schusskarten. Keine Hochrechnung auf blinde Videophasen; „kein Pass“ und „unklar“ sind ausgeschlossen.</p>{warning}<h2>Teams</h2>{team_table}<h2>Spieler</h2>{player_table}</body></html>""", encoding="utf-8")

    print(f"Review: {dict(status_counts)}")
    for row in team_rows:
        print(f"  {row['team']}: {row['bestaetigte_passversuche']} Pässe, "
              f"{row['passquote_pct']:.1f} %, {row['bestaetigte_schuesse']} Schüsse")
    print(f"Teamstatistik: {team_path}")
    print(f"Spielerstatistik: {player_path}")
    print(f"Dashboard: {dashboard_path}")


if __name__ == "__main__":
    main()
