"""Walk-forward backtest of the NBA simulator on this season's finished games, with the defender step on
and off.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import nba_game_sim as sim


OUT = Path("model_artifacts/nba_backtest")


def actual_results(season):
    games = pd.read_parquet(sim.CACHE_DIR / "espn" / f"games_{season}.parquet")
    games = games.loc[games.status.eq("STATUS_FINAL")].assign(game_id=lambda f: "E" + f.event_id.astype(str))
    box = pd.read_parquet(sim.CACHE_DIR / "espn" / f"box_{season}.parquet")
    box = box.assign(game_id="E" + box.event_id.astype(str), player="e" + box.athlete_id.astype(str))
    for column in ("PTS", "REB", "AST", "MIN"):
        box[column] = pd.to_numeric(box[column], errors="coerce")
    box["3PM"] = pd.to_numeric(box["3PT"].astype(str).str.split("-").str[0], errors="coerce")
    return games.set_index("game_id"), box.loc[box.MIN.gt(0)]


def shot_scores(league, matchup, events):
    """Expected make chance of each real shot in this game, without and with the defender adjustment, using
    the real defenders on the floor.
    """
    shots = events.loc[events.kind.eq("fga") & events.zone.ge(0)]
    if shots.empty:
        return pd.DataFrame()
    base = league.expected_make(shots.p1, shots.zone, shots.start) + league.fg_shift
    rows = []
    for (_, shot), logit in zip(shots.iterrows(), base):
        side = 0 if shot.team == matchup.home else 1
        D = matchup.teams[1 - side]
        logit = logit + D.fg_def + matchup.home_shift[side]
        on = shot.away_on if side == 0 else shot.home_on
        defenders = [D.keys.index(k) for k in (on or "").split("|") if k in D.keys]
        group = league.group.get(shot.p1, 1)
        if defenders:
            weights = np.array([D.assign[j][group] for j in defenders])
            effect = float(np.dot(weights, [D.dlog[j] for j in defenders]) / weights.sum()) - D.def_avg
        else:
            effect = 0.
        rows.append((shot.game_id, shot.made, sim._expit(logit), sim._expit(logit + effect), effect))
    return pd.DataFrame(rows, columns=["game_id", "made", "p_off", "p_on", "effect"])


def log_loss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def paired_bootstrap(a, b, n=4000, seed=0):
    """Mean of a - b and a 95% interval, resampling games."""
    diff = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run(sims=400, seed=2026, season=None, configs=("on", "off")):
    season = season or sim.nba_data.current_season()
    events, player_games = sim.load_events(season - 1, season)
    events = sim.label_free_throws(events)
    finals, box = actual_results(season)
    current = player_games.loc[player_games.season.eq(season) & player_games.game_id.isin(finals.index)]
    dates = sorted(current.date.unique())
    game_rows, player_rows, shot_rows = [], [], []
    started = time.time()
    for date in dates:
        league = sim.League(events, player_games, season, as_of=pd.Timestamp(date))
        for game_id in current.loc[current.date.eq(date)].game_id.unique():
            rows = current.loc[current.game_id.eq(game_id)]
            home, away = rows.loc[rows.home], rows.loc[~rows.home]
            final = finals.loc[game_id]
            result = dict(game_id=game_id, date=pd.Timestamp(date).date(), home=home.team.iloc[0],
                          away=away.team.iloc[0], home_score=final.home_score, away_score=final.away_score)
            for flag, label in ((True, "on"), (False, "off")):
                if label not in configs:
                    continue
                matchup = sim.Matchup(league, result["home"], result["away"], home.player.tolist(),
                                      away.player.tolist(), home.player[home.started].tolist(),
                                      away.player[away.started].tolist(), pd.Timestamp(date), flag,
                                      final.officials if isinstance(final.officials, str) else ())
                games, players = matchup.simulate(sims, seed)
                margin = games.home_score - games.away_score
                result[f"p_home_{label}"] = float((margin > 0).mean())
                result[f"margin_{label}"] = float(margin.mean())
                result[f"total_{label}"] = float((games.home_score + games.away_score).mean())
                means = players.groupby("player")[["PTS", "OREB", "DREB", "AST", "3PM", "MIN"]].sum() / sims
                means["REB"] = means.OREB + means.DREB
                for key, line in means.iterrows():
                    player_rows.append(dict(game_id=game_id, player=key, config=label, PTS=line.PTS,
                                            REB=line.REB, AST=line.AST, **{"3PM": line["3PM"]}, MIN=line.MIN))
                if flag or "on" not in configs:
                    shot_rows.append(shot_scores(league, matchup, events.loc[events.game_id.eq(game_id)]))
            game_rows.append(result)
        print(f"{pd.Timestamp(date).date()}: {len(game_rows)} games done, {time.time() - started:.0f}s", flush=True)
    games = pd.DataFrame(game_rows)
    players = pd.DataFrame(player_rows)
    shots = pd.concat(shot_rows, ignore_index=True)
    return games, players, shots, box


def report(games, players, shots, box):
    y = (games.home_score > games.away_score).astype(float)
    margin, total = games.home_score - games.away_score, games.home_score + games.away_score
    summary, per_game = {}, {}
    for label in ("on", "off"):
        p = games[f"p_home_{label}"].clip(.01, .99)
        per_game[label] = dict(brier=(p - y) ** 2, logloss=-(y * np.log(p) + (1 - y) * np.log(1 - p)),
                               margin_err=(games[f"margin_{label}"] - margin).abs(),
                               total_err=(games[f"total_{label}"] - total).abs())
        summary[label] = {"games": len(games), "accuracy": float(((p > .5) == (y == 1)).mean()),
                          "brier": float(per_game[label]["brier"].mean()),
                          "log_loss": float(per_game[label]["logloss"].mean()),
                          "margin_mae": float(per_game[label]["margin_err"].mean()),
                          "total_mae": float(per_game[label]["total_err"].mean()),
                          "team_score_mae": float(pd.concat([
                              ((games[f"total_{label}"] + games[f"margin_{label}"]) / 2 - games.home_score).abs(),
                              ((games[f"total_{label}"] - games[f"margin_{label}"]) / 2 - games.away_score).abs()]).mean()),
                          "first_date": str(games.date.min()), "last_date": str(games.date.max())}
    joined = players.merge(box[["game_id", "player", "PTS", "REB", "AST", "3PM"]], on=["game_id", "player"],
                           suffixes=("", "_real"))
    for label in ("on", "off"):
        part = joined.loc[joined.config.eq(label)]
        for stat in ("PTS", "REB", "AST", "3PM"):
            summary[label][f"player_{stat.lower()}_mae"] = float((part[stat] - part[f"{stat}_real"]).abs().mean())
    summary["on"]["shot_log_loss"] = log_loss(shots.p_on, shots.made)
    summary["off"]["shot_log_loss"] = log_loss(shots.p_off, shots.made)
    summary["shots"] = len(shots)
    diffs = {}
    for metric in ("brier", "logloss", "margin_err"):
        diffs[f"game_{metric}"] = paired_bootstrap(per_game["on"][metric], per_game["off"][metric])
    pts = joined.pivot_table(index=["game_id", "player"], columns="config", values="PTS").join(
        joined.groupby(["game_id", "player"]).PTS_real.first())
    pts_err = pts.assign(on=(pts.on - pts.PTS_real).abs(), off=(pts.off - pts.PTS_real).abs()).groupby("game_id")[
        ["on", "off"]].mean()
    diffs["player_pts_mae"] = paired_bootstrap(pts_err.on, pts_err.off)
    loss = lambda p, y: -(y * np.log(p) + (1 - y) * np.log(1 - p))
    shot_games = shots.assign(on=loss(shots.p_on, shots.made), off=loss(shots.p_off, shots.made)).groupby(
        "game_id")[["on", "off"]].mean()
    diffs["shot_log_loss"] = paired_bootstrap(shot_games.on, shot_games.off)
    summary["on_minus_off"] = {k: {"mean": v[0], "low": v[1], "high": v[2]} for k, v in diffs.items()}
    summary["defender_effect_sd"] = float(shots.effect.std())
    return summary


def quick_report(games, players, box, label):
    """Single-setting summary (no on/off comparison)."""
    y = (games.home_score > games.away_score).astype(float)
    p = games[f"p_home_{label}"].clip(.01, .99)
    joined = players.merge(box[["game_id", "player", "PTS", "MIN"]], on=["game_id", "player"], suffixes=("", "_real"))
    return {"games": len(games), "accuracy": float(((p > .5) == (y == 1)).mean()), "brier": float(((p - y) ** 2).mean()),
            "margin_mae": float((games[f"margin_{label}"] - (games.home_score - games.away_score)).abs().mean()),
            "player_pts_mae": float((joined.PTS - joined.PTS_real).abs().mean()),
            "minutes_mae": float((joined.MIN - joined.MIN_real).abs().mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--configs", nargs="+", default=["on", "off"], help="defender step settings to run")
    parser.add_argument("--minute-iterations", type=int, default=None,
                        help="override the minutes calibration (0 turns it off)")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    if args.minute_iterations is not None:
        sim.MINUTE_ITERATIONS = args.minute_iterations
    games, players, shots, box = run(sims=args.sims, configs=tuple(args.configs))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if len(args.configs) < 2:
        print(json.dumps(quick_report(games, players, box, args.configs[0]), indent=2, default=float))
        games.to_csv(out / "games.csv", index=False)
        players.to_csv(out / "players.csv", index=False)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    games.to_csv(OUT / "games.csv", index=False)
    players.to_csv(OUT / "players.csv", index=False)
    shots.to_csv(OUT / "shots.csv", index=False)
    summary = report(games, players, shots, box)
    (OUT / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
