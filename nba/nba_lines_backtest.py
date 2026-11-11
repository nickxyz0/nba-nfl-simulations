"""Backtest the simulator against real DraftKings lines (as published in ESPN's public odds feed) on this
season's finished games.
"""

import argparse
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

import nba_data
import nba_game_sim as sim

OUT = Path("model_artifacts/nba_lines_backtest")
ODDS = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{e}/competitions/{e}/odds"
TYPES = {"Total Points": "PTS", "Total Rebounds": "REB", "Total Assists": "AST", "Total Points and Rebounds": "PR",
         "Total Points and Assists": "PA", "Total Assists and Rebounds": "RA",
         "Total Points, Rebounds, and Assists": "PRA"}
BOX_PARTS = {"PTS": ["PTS"], "REB": ["REB"], "AST": ["AST"], "PR": ["PTS", "REB"], "PA": ["PTS", "AST"],
             "RA": ["REB", "AST"], "PRA": ["PTS", "REB", "AST"], "3PM": ["3PM"], "STL": ["STL"], "BLK": ["BLK"],
             "SB": ["STL", "BLK"]}
EXTRA_TYPES = {"Total 3-Point Field Goals": "3PM", "Total Steals": "STL", "Total Blocks": "BLK",
               "Total Steals and Blocks": "SB"}
EXTRA_MILESTONES = {"3-Point Field Goals Milestones": "3PM", "Steals Milestones": "STL", "Blocks Milestones": "BLK",
                    "Steals + Blocks Milestones": "SB"}


def implied(american):
    return -american / (100 - american) if american < 0 else 100 / (american + 100)


def dk_props(event, types=TYPES, milestones=None):
    """Over/under player props for one game."""
    rows, page, pages = [], 1, 1
    milestones = milestones or {}
    yes_rows = []
    while page <= pages:
        data = json.loads(nba_data._get(ODDS.format(e=event) + f"/100/propBets?limit=500&page={page}"))
        pages = data.get("pageCount", 1)
        for item in data.get("items", []):
            name = item.get("type", {}).get("name")
            athlete = re.search(r"athletes/(\d+)", item.get("athlete", {}).get("$ref", ""))
            if name in types and athlete:
                odds = item.get("odds", {})
                rows.append((athlete.group(1), types[name], odds.get("total", {}).get("value"),
                             odds.get("american", {}).get("value")))
            elif name in milestones and athlete:
                target = item.get("current", {}).get("target", {}).get("value")
                price = item.get("odds", {}).get("american", {}).get("value")
                if target is not None:
                    yes_rows.append(dict(athlete=athlete.group(1), stat=milestones[name], line=float(target) - .5,
                                         over_price=pd.to_numeric(price, errors="coerce"), under_price=np.nan,
                                         milestone=True))
        page += 1
        time.sleep(.1)
    frame = pd.DataFrame(rows, columns=["athlete", "stat", "line", "price"])
    if frame.empty:
        return pd.DataFrame(yes_rows)
    frame["line"] = pd.to_numeric(frame.line, errors="coerce")
    frame["price"] = pd.to_numeric(frame.price, errors="coerce")
    pairs = []
    for (athlete, stat), group in frame.groupby(["athlete", "stat"], sort=False):
        if len(group) == 2 and group.line.notna().all() and group.price.notna().all():
            pairs.append(dict(athlete=athlete, stat=stat, line=group.line.iloc[0], over_price=group.price.iloc[0],
                              under_price=group.price.iloc[1], milestone=False))
    return pd.DataFrame(pairs + yes_rows)


def dk_game_lines(event):
    summary = json.loads(nba_data._get(nba_data.ESPN_SUMMARY.format(event=event)))
    for p in summary.get("pickcenter", []):
        if p.get("provider", {}).get("name") == "DraftKings":
            return dict(spread=p.get("spread"), total=p.get("overUnder"),
                        home_price=(p.get("homeTeamOdds") or {}).get("spreadOdds"),
                        away_price=(p.get("awayTeamOdds") or {}).get("spreadOdds"), over_price=p.get("overOdds"),
                        under_price=p.get("underOdds"))
    return None


def run(sims=2000, season=None, seed=2026, extra=False, since=None):
    season = season or nba_data.current_season()
    events, player_games = sim.load_events(season - 1, season)
    events = sim.label_free_throws(events)
    games_file = pd.read_parquet(sim.CACHE_DIR / "espn" / f"games_{season}.parquet")
    games_file = games_file[games_file.status.eq("STATUS_FINAL")].assign(game_id=lambda f: "E" + f.event_id.astype(str)).set_index("game_id")
    box = pd.read_parquet(sim.CACHE_DIR / "espn" / f"box_{season}.parquet")
    box = box.assign(game_id="E" + box.event_id.astype(str), player="e" + box.athlete_id.astype(str))
    for column in ("PTS", "REB", "AST", "MIN", "STL", "BLK"):
        box[column] = pd.to_numeric(box[column], errors="coerce")
    box["3PM"] = pd.to_numeric(box["3PT"].astype(str).str.split("-").str[0], errors="coerce")
    current = player_games[player_games.season.eq(season) & player_games.game_id.isin(games_file.index)]
    prop_rows, game_rows = [], []
    started = time.time()
    for date in sorted(current.date.unique()):
        if since is not None and pd.Timestamp(date) < pd.Timestamp(since):
            continue
        league = sim.League(events, player_games, season, as_of=pd.Timestamp(date))
        earlier = current[current.date < date].sort_values("date")
        recent = earlier.groupby("player").minutes.apply(lambda m: m.tail(5).mean())
        for game_id in current[current.date.eq(date)].game_id.unique():
            rows = current[current.game_id.eq(game_id)]
            home, away = rows[rows.home], rows[~rows.home]
            final = games_file.loc[game_id]
            matchup = sim.Matchup(league, home.team.iloc[0], away.team.iloc[0], home.player.tolist(), away.player.tolist(),
                                  home.player[home.started].tolist(), away.player[away.started].tolist(),
                                  pd.Timestamp(date), False, final.officials if isinstance(final.officials, str) else ())
            frame, players = matchup.simulate(sims, seed)
            result = sim.SimulationResult(matchup, frame, players, seed)
            samples, info = sim.prop_samples(result)
            event = final.event_id
            try:
                lines = dk_game_lines(event)
                props = dk_props(event, EXTRA_TYPES, EXTRA_MILESTONES) if extra else dk_props(event)
            except Exception as error:
                print(f"  {game_id}: no lines ({error})", flush=True)
                continue
            margin, total = frame.home_score - frame.away_score, frame.home_score + frame.away_score
            if lines and lines["spread"] is not None and lines["total"] is not None:
                game_rows.append(dict(game_id=game_id, date=str(pd.Timestamp(date).date()), home=matchup.home, away=matchup.away,
                                      home_score=final.home_score, away_score=final.away_score, **lines,
                                      sim_margin=float(margin.mean()), sim_total=float(total.mean()),
                                      home_cover=float((margin + lines["spread"] > 0).mean()),
                                      away_cover=float((margin + lines["spread"] < 0).mean()),
                                      over=float((total > lines["total"]).mean()), under=float((total < lines["total"]).mean())))
            game_box = box[box.game_id.eq(game_id)].set_index("player")
            for p in props.itertuples():
                key = "e" + p.athlete
                if key not in samples[p.stat].columns or key not in game_box.index or pd.isna(game_box.at[key, "MIN"]) \
                        or game_box.at[key, "MIN"] <= 0:
                    continue
                values = samples[p.stat][key].to_numpy()
                actual = float(sum(game_box.at[key, c] for c in BOX_PARTS[p.stat]))
                prop_rows.append(dict(game_id=game_id, date=str(pd.Timestamp(date).date()), player=key, name=info.at[key, "name"],
                                      stat=p.stat, line=p.line, milestone=bool(getattr(p, "milestone", False)), over_price=p.over_price, under_price=p.under_price,
                                      sim_over=float((values > p.line).mean()), sim_under=float((values < p.line).mean()),
                                      sim_avg=float(values.mean()), sim_sd=float(values.std()), sim_min=float(info.at[key, "MIN"]),
                                      recent_min=float(recent.get(key, np.nan)), real_min=float(game_box.at[key, "MIN"]), actual=actual))
        print(f"{pd.Timestamp(date).date()}: {len(game_rows)} games, {len(prop_rows)} props, {time.time() - started:.0f}s", flush=True)
    return pd.DataFrame(game_rows), pd.DataFrame(prop_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=2000)
    parser.add_argument("--extra", action="store_true", help="threes, steals and blocks instead of the main props")
    parser.add_argument("--since", help="only game dates on or after this date (YYYY-MM-DD)")
    args = parser.parse_args()
    games, props = run(args.sims, extra=args.extra, since=args.since)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.extra:
        props.to_csv(OUT / "props_threes_steals_blocks.csv", index=False)
        print("saved", len(props), flush=True)
        return
    games.to_csv(OUT / "games.csv", index=False)
    props.to_csv(OUT / "props.csv", index=False)
    print("saved", len(games), len(props), flush=True)


if __name__ == "__main__":
    main()
