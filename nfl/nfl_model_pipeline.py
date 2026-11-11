"""Pregame NFL features and time-ordered XGBoost experiments."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import urllib.request

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ALIASES = {"LA": "LAR", "WSH": "WAS", "AZ": "ARI", "OAK": "LV", "SD": "LAC"}
OLD_STATS = [
    "passing_epa", "rushing_epa", "passing_cpoe", "passing_interceptions",
    "fumbles_lost_total", "def_interceptions", "def_sacks", "def_qb_hits",
    "sacks_suffered", "sacks_for",
]
RAW_REQUIRED = [
    "attempts", "sacks_suffered", "carries", "passing_epa", "rushing_epa",
    "passing_cpoe", "passing_yards", "sack_yards_lost", "rushing_yards",
    "passing_first_downs", "rushing_first_downs", "passing_interceptions",
    "fumbles_lost_total", "def_interceptions", "def_sacks", "def_qb_hits",
    "penalty_yards",
]
COLD_TEAMS = {"BUF", "GB", "CHI", "CLE", "PIT", "DEN", "KC", "NE", "NYG", "NYJ", "CIN"}
TEAM_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_week_{season}.csv"
SCHEDULE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PLAYER_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
QB_PRIOR_EPA, QB_PRIOR_PLAYS, QB_HALF_LIFE, QB_REGULAR_WINDOW = -.05, 200., 16., 8
ELO_QB_POINTS = 400
QB_COLUMNS = ["home_qb_rating", "away_qb_rating", "gap_qb_rating", "home_qb_change",
              "away_qb_change", "home_qb_delta", "away_qb_delta", "gap_qb_delta"]


def cached_csv(url, path, refresh=False):
    """Keep a reproducible local snapshot; refresh is an explicit choice."""
    path = Path(path)
    if refresh or not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        from io import BytesIO
        pd.read_csv(BytesIO(payload), nrows=2)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".download")
        temporary.write_bytes(payload)
        temporary.replace(path)
    return pd.read_csv(path)


def load_team_stats(seasons, cache_dir="data/nflverse", refresh=False):
    return pd.concat([
        cached_csv(TEAM_URL.format(season=s), Path(cache_dir) / f"team_stats_{s}.csv", refresh)
        for s in seasons
    ], ignore_index=True)


def load_schedule(cache_dir="data/nflverse", refresh=False):
    return cached_csv(SCHEDULE_URL, Path(cache_dir) / "schedule.csv", refresh)


def load_player_stats(seasons, cache_dir="data/nflverse", refresh=False):
    return pd.concat([
        cached_csv(PLAYER_URL.format(season=s), Path(cache_dir) / f"player_stats_{s}.csv", refresh)
        for s in seasons
    ], ignore_index=True)


def normalize_teams(frame, columns):
    result = frame.copy()
    for column in columns:
        result[column] = result[column].str.strip().str.upper().replace(ALIASES)
    return result


def prepare_schedule(schedule, first_season=2020, last_season=2025):
    games = schedule.loc[
        schedule.season.between(first_season, last_season) & schedule.game_type.eq("REG")
    ].copy()
    games = normalize_teams(games, ["home_team", "away_team"])
    games["gameday"] = pd.to_datetime(games.gameday, errors="raise")
    games["completed"] = games[["home_score", "away_score"]].notna().all(axis=1)
    assert games.game_id.is_unique and games.game_id.notna().all()
    assert games[["home_score", "away_score"]].isna().nunique(axis=1).eq(1).all()
    return games.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)


def reconcile_scores(schedule_games, score_frames):
    """The additional source must agree with the user's cleaned CSV scores."""
    rows = []
    for season, scores in score_frames.items():
        local = normalize_teams(scores, ["HomeTeam", "AwayTeam"]).rename(columns={
            "HomeTeam": "home_team", "AwayTeam": "away_team", "WeekNum": "week",
        })
        public = schedule_games.loc[schedule_games.season.eq(season) & schedule_games.completed]
        merged = local.merge(public, on=["week", "home_team", "away_team"],
                             how="outer", validate="one_to_one", indicator=True)
        assert merged._merge.eq("both").all(), f"Unmatched {season} score rows"
        np.testing.assert_allclose(merged.HomeScore, merged.home_score)
        np.testing.assert_allclose(merged.AwayScore, merged.away_score)
        assert merged.GameDateParsed.eq(merged.gameday).all()
        rows.append({"season": season, "games": len(merged), "scores_and_dates_match": True})
    return pd.DataFrame(rows)


def elo_features(games):
    """Fixed baseline: K=20, 55 home Elo, 25% offseason regression."""
    ratings, records = {}, []
    previous_season = None
    for (season, week), block in games.groupby(["season", "week"], sort=True):
        if previous_season is not None and season != previous_season:
            ratings = {team: 1500 + .75 * (rating - 1500) for team, rating in ratings.items()}
        previous_season = season
        updates = []
        for row in block.itertuples():
            home = ratings.get(row.home_team, 1500.)
            away = ratings.get(row.away_team, 1500.)
            advantage = 0. if row.location == "Neutral" else 55.
            gap = home - away + advantage
            probability = 1 / (1 + 10 ** (-gap / 400))
            records.append({"game_id": row.game_id, "elo_gap": gap,
                            "elo_home_probability": probability})
            if row.completed:
                margin = row.home_score - row.away_score
                outcome = 1. if margin > 0 else 0. if margin < 0 else .5
                winner_gap = gap if margin > 0 else -gap
                multiplier = np.log1p(abs(margin)) * 2.2 / (2.2 + .001 * winner_gap) if margin else 1.
                delta = 20 * multiplier * (outcome - probability)
                updates.append((row.home_team, home + delta, row.away_team, away - delta))
        for ht, hr, at, ar in updates:
            ratings[ht], ratings[at] = hr, ar
    return pd.DataFrame(records)


def build_pregame_features(games, raw_stats):
    """Use only completed, prior-week games, with season-local averages."""
    raw = normalize_teams(raw_stats, ["team", "opponent_team"])
    raw = raw.loc[raw.season_type.eq("REG")].copy()
    assert set(RAW_REQUIRED).issubset(raw), "Required nflverse columns missing"
    assert not raw.duplicated(["game_id", "team"]).any()
    raw[RAW_REQUIRED] = raw[RAW_REQUIRED].apply(pd.to_numeric, errors="raise")
    sides = []
    for side, other in [("home", "away"), ("away", "home")]:
        frame = games[["game_id", "season", "week", "gameday", "completed",
                       f"{side}_team", f"{other}_team", f"{side}_score", f"{other}_score"]].rename(columns={
            f"{side}_team": "team", f"{other}_team": "opponent_team",
            f"{side}_score": "points_for", f"{other}_score": "points_against",
        })
        sides.append(frame)
    team_games = pd.concat(sides, ignore_index=True)
    team_games = team_games.merge(raw[["game_id", "team", "opponent_team"] + RAW_REQUIRED],
                                 on=["game_id", "team", "opponent_team"], how="left",
                                 validate="one_to_one", indicator=True)
    assert team_games.loc[team_games.completed, "_merge"].eq("both").all(), "Completed games lack team stats"
    team_games = team_games.drop(columns="_merge")
    assert team_games.loc[team_games.completed, [c for c in RAW_REQUIRED if c != "passing_cpoe"]].notna().all().all()
    opponent = raw[["game_id", "team"] + RAW_REQUIRED].rename(columns={
        "team": "opponent_team", **{c: f"opp_{c}" for c in RAW_REQUIRED},
    })
    team_games = team_games.merge(opponent, on=["game_id", "opponent_team"],
                                 how="left", validate="many_to_one")
    team_games["sacks_for"] = team_games.opp_sacks_suffered
    team_games["dropbacks"] = team_games.attempts + team_games.sacks_suffered
    team_games["opp_dropbacks"] = team_games.opp_attempts + team_games.opp_sacks_suffered
    team_games["plays"] = team_games.dropbacks + team_games.carries
    team_games["opp_plays"] = team_games.opp_dropbacks + team_games.opp_carries
    team_games["net_passing_yards"] = team_games.passing_yards - team_games.sack_yards_lost
    team_games["opp_net_passing_yards"] = team_games.opp_passing_yards - team_games.opp_sack_yards_lost
    team_games["first_downs"] = team_games.passing_first_downs + team_games.rushing_first_downs
    team_games["opp_first_downs"] = team_games.opp_passing_first_downs + team_games.opp_rushing_first_downs
    team_games["giveaways"] = team_games.passing_interceptions + team_games.fumbles_lost_total
    team_games["takeaways"] = team_games.opp_passing_interceptions + team_games.opp_fumbles_lost_total
    ratios = {
        "pass_epa_rate": ("passing_epa", "dropbacks"),
        "rush_epa_rate": ("rushing_epa", "carries"),
        "def_pass_epa_allowed": ("opp_passing_epa", "opp_dropbacks"),
        "def_rush_epa_allowed": ("opp_rushing_epa", "opp_carries"),
        "net_pass_yards_rate": ("net_passing_yards", "dropbacks"),
        "def_net_pass_yards_allowed": ("opp_net_passing_yards", "opp_dropbacks"),
        "rush_yards_rate": ("rushing_yards", "carries"),
        "def_rush_yards_allowed": ("opp_rushing_yards", "opp_carries"),
        "sack_allowed_rate": ("sacks_suffered", "dropbacks"),
        "sack_made_rate": ("opp_sacks_suffered", "opp_dropbacks"),
        "qb_hit_rate": ("def_qb_hits", "opp_dropbacks"),
        "first_down_rate": ("first_downs", "plays"),
        "def_first_down_allowed": ("opp_first_downs", "opp_plays"),
        "giveaway_rate": ("giveaways", "plays"),
        "takeaway_rate": ("takeaways", "opp_plays"),
        "penalty_yards_rate": ("penalty_yards", "plays"),
    }
    means = list(dict.fromkeys(OLD_STATS + ["points_for", "points_against"]))
    numeric = list(dict.fromkeys(means + [c for pair in ratios.values() for c in pair]))
    positions = {name: i for i, name in enumerate(numeric)}
    records = []
    for (season, team), block in team_games.groupby(["season", "team"], sort=True):
        history = []
        for row in block.sort_values("week").to_dict("records"):
            entry = {"game_id": row["game_id"], "team": team, "season": season,
                     "week": row["week"], "prior_games": len(history)}
            for prefix, window in [("avg", None), ("recent4", 4), ("recent8", 8)]:
                values = np.asarray(history[-window:] if window else history, dtype=float)
                for name in means:
                    v = values[:, positions[name]] if len(history) else np.array([])
                    entry[f"{prefix}_{name}"] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
                for name, (num, den) in ratios.items():
                    denominator = values[:, positions[den]].sum() if len(history) else 0
                    entry[f"{prefix}_{name}"] = values[:, positions[num]].sum() / denominator if denominator > 0 else np.nan
            records.append(entry)
            if row["completed"]:
                history.append([row[name] for name in numeric])
    team_features = pd.DataFrame(records)
    features = games.copy()
    stat_columns = [c for c in team_features if c.startswith(("avg_", "recent4_", "recent8_"))]
    for side in ["home", "away"]:
        side_frame = team_features[["game_id", "team", "prior_games"] + stat_columns].rename(columns={
            "team": f"{side}_team", "prior_games": f"{side}_prior_games",
            **{c: f"{side}_{c}" for c in stat_columns},
        })
        features = features.merge(side_frame, on=["game_id", f"{side}_team"], validate="one_to_one")
    additions = {}
    for column in stat_columns:
        additions[f"gap_{column}"] = features[f"home_{column}"] - features[f"away_{column}"]
    for prefix in ["avg", "recent4", "recent8"]:
        additions[f"net_scoring_gap_{prefix}"] = (
            additions[f"gap_{prefix}_points_for"] - additions[f"gap_{prefix}_points_against"]
        )
        for kind in ["pass", "rush"]:
            additions[f"net_{kind}_epa_gap_{prefix}"] = (
                additions[f"gap_{prefix}_{kind}_epa_rate"] - additions[f"gap_{prefix}_def_{kind}_epa_allowed"]
            )
    features = pd.concat([features, pd.DataFrame(additions, index=features.index)], axis=1)
    features = features.merge(elo_features(games), on="game_id", validate="one_to_one")
    features["rest_gap"] = features.home_rest.clip(upper=21) - features.away_rest.clip(upper=21)
    features["neutral_site"] = features.location.eq("Neutral").astype(int)
    features["cold_game"] = (features.home_team.isin(COLD_TEAMS) & features.gameday.dt.month.isin([11, 12, 1])).astype(int)
    features["home_win"] = np.where(
        features.completed & features.home_score.ne(features.away_score),
        features.home_score.gt(features.away_score).astype(float), np.nan,
    )
    first = team_features.prior_games.eq(0)
    assert team_features.loc[first, stat_columns].isna().all().all(), "Season reset failed"
    return features, team_features


def qb_schedule(schedule, first_season, last_season):
    """All game types, so playoff starts count toward QB history."""
    games = normalize_teams(schedule.loc[schedule.season.between(first_season, last_season)],
                            ["home_team", "away_team"])
    games["gameday"] = pd.to_datetime(games.gameday, errors="raise")
    return games


def correct_qb_ids(schedule, player_stats, overrides=None, season=None, week=None):
    """Fix the schedule's starting-QB fields, which can be stale."""
    schedule = schedule.copy()
    stats = normalize_teams(player_stats.loc[player_stats.position.eq("QB")], ["team"])
    stats["dropbacks"] = pd.to_numeric(stats.attempts, errors="coerce").fillna(0) + \
        pd.to_numeric(stats.sacks_suffered, errors="coerce").fillna(0)
    stats = stats.loc[stats.dropbacks > 0].sort_values(["dropbacks", "player_id"])
    primary = stats.groupby(["game_id", "team"]).tail(1).set_index(["game_id", "team"])
    for side in ["home", "away"]:
        teams = schedule[f"{side}_team"].str.strip().str.upper().replace(ALIASES)
        key = pd.MultiIndex.from_arrays([schedule.game_id, teams])
        found = primary.reindex(key)
        played = schedule.home_score.notna().to_numpy() & found.player_id.notna().to_numpy()
        schedule.loc[played, f"{side}_qb_id"] = found.player_id.to_numpy()[played]
        schedule.loc[played, f"{side}_qb_name"] = found.player_display_name.to_numpy()[played]
    for team, name in (overrides or {}).items():
        matches = stats.loc[stats.player_display_name.eq(name), "player_id"].unique()
        if len(matches) != 1:
            raise ValueError(f"QB override {name!r} matches {len(matches)} players with passing stats")
        for side in ["home", "away"]:
            rows = (schedule.season.eq(season) & schedule.week.eq(week) & schedule.home_score.isna()
                    & schedule[f"{side}_team"].str.strip().str.upper().replace(ALIASES).eq(team))
            schedule.loc[rows, f"{side}_qb_id"] = matches[0]
            schedule.loc[rows, f"{side}_qb_name"] = name
        if not (schedule.season.eq(season) & schedule.week.eq(week)
                & schedule[["home_team", "away_team"]].apply(
                    lambda c: c.str.strip().str.upper().replace(ALIASES)).eq(team).any(axis=1)).any():
            raise ValueError(f"QB override team {team!r} has no Week {week} game")
    return schedule


def qb_history(schedule, player_stats):
    """Each QB's dated games with EPA (pass + rush) and plays (attempts + sacks + carries)."""
    stats = player_stats.loc[player_stats.position.eq("QB")].copy()
    for column in ["passing_epa", "rushing_epa", "attempts", "sacks_suffered", "carries"]:
        stats[column] = pd.to_numeric(stats[column], errors="coerce").fillna(0.)
    stats["epa"] = stats.passing_epa + stats.rushing_epa
    stats["plays"] = stats.attempts + stats.sacks_suffered + stats.carries
    stats = stats.loc[stats.plays > 0].merge(schedule[["game_id", "gameday"]], on="game_id", how="inner")
    return {qb: block.sort_values("gameday")[["gameday", "epa", "plays"]].to_numpy()
            for qb, block in stats.groupby("player_id")}


def qb_rating(history, qb, before):
    """Shrunk, recency-weighted EPA per play from the QB's games before a date."""
    if not isinstance(qb, str) or qb not in history:
        return QB_PRIOR_EPA
    games = history[qb]
    games = games[games[:, 0] < before]
    if not len(games):
        return QB_PRIOR_EPA
    weights = .5 ** (np.arange(len(games))[::-1] / QB_HALF_LIFE)
    epa, plays = games[:, 1].astype(float), games[:, 2].astype(float)
    return (np.sum(weights * epa) + QB_PRIOR_PLAYS * QB_PRIOR_EPA) / (np.sum(weights * plays) + QB_PRIOR_PLAYS)


def add_qb_features(features, schedule, history):
    """Starter rating, starter-vs-regular change and its rating delta per side."""
    from collections import Counter
    starts = []
    for side in ["home", "away"]:
        frame = schedule[["game_id", "gameday", f"{side}_team", f"{side}_qb_id"]]
        starts.append(frame.set_axis(["game_id", "gameday", "team", "qb"], axis=1))
    starts = pd.concat(starts).dropna(subset=["qb"]).sort_values("gameday")
    by_team = {team: block[["gameday", "qb"]].to_numpy() for team, block in starts.groupby("team")}
    rows = []
    for game in features.itertuples():
        row = {"game_id": game.game_id}
        for side in ["home", "away"]:
            starter = getattr(game, f"{side}_qb_id")
            past = by_team.get(getattr(game, f"{side}_team"), np.empty((0, 2)))
            past = past[past[:, 0] < game.gameday][-QB_REGULAR_WINDOW:, 1]
            if len(past):
                counts = Counter(past)
                top = max(counts.values())
                regular = next(qb for qb in past[::-1] if counts[qb] == top)
            else:
                regular = starter
            if not isinstance(starter, str):
                starter = regular
            rating = qb_rating(history, starter, game.gameday)
            row[f"{side}_qb_rating"] = rating
            row[f"{side}_qb_change"] = float(isinstance(starter, str) and starter != regular)
            row[f"{side}_qb_delta"] = rating - qb_rating(history, regular, game.gameday)
        rows.append(row)
    result = features.merge(pd.DataFrame(rows), on="game_id", validate="one_to_one")
    result["gap_qb_rating"] = result.home_qb_rating - result.away_qb_rating
    result["gap_qb_delta"] = result.home_qb_delta - result.away_qb_delta
    return result


def elo_qb_probability(data, points=ELO_QB_POINTS):
    """FiveThirtyEight-style: shift the Elo gap by the starter-vs-regular QB gap."""
    gap = data.elo_gap + points * data.gap_qb_delta
    return (1 / (1 + 10 ** (-gap / 400))).to_numpy()


def feature_sets():
    original = [f"{side}_avg_{stat}" for side in ["home", "away"] for stat in OLD_STATS]
    original += [f"{side}_avg_{stat}" for side in ["home", "away"] for stat in ["points_for", "points_against"]]
    original += ["home_prior_games", "away_prior_games", "cold_game"]
    no_turnovers = [c for c in original if not any(x in c for x in ["interceptions", "fumbles_lost"])]
    metrics = ["points_for", "points_against", "pass_epa_rate", "rush_epa_rate",
               "def_pass_epa_allowed", "def_rush_epa_allowed", "passing_cpoe",
               "sack_allowed_rate", "sack_made_rate", "first_down_rate", "def_first_down_allowed"]
    gaps = [f"gap_avg_{m}" for m in metrics]
    gaps += ["net_scoring_gap_avg", "net_pass_epa_gap_avg", "net_rush_epa_gap_avg"]
    context = gaps + ["elo_gap", "rest_gap", "neutral_site", "home_prior_games", "away_prior_games"]
    context += [f"{side}_avg_{m}" for side in ["home", "away"] for m in metrics[:7]]
    recent_metrics = metrics[:7]
    recent4 = [f"gap_recent4_{m}" for m in recent_metrics] + ["net_scoring_gap_recent4"]
    recent8 = [f"gap_recent8_{m}" for m in recent_metrics] + ["net_scoring_gap_recent8"]
    extras = [f"gap_avg_{m}" for m in ["giveaway_rate", "takeaway_rate", "qb_hit_rate",
                                      "penalty_yards_rate", "net_pass_yards_rate", "def_net_pass_yards_allowed"]]
    return {
        "original": original,
        "original_no_turnovers": no_turnovers,
        "original_no_cold": [c for c in original if c != "cold_game"],
        "original_no_turnovers_cold": [c for c in no_turnovers if c != "cold_game"],
        "efficiency_gaps": gaps,
        "efficiency_context": context,
        "efficiency_recent4": context + recent4,
        "efficiency_recent8": context + recent8,
        "efficiency_turnovers": context + extras[:2],
        "all_enhanced": context + recent4 + recent8 + extras,
    }


def metrics(y, probability):
    probability = np.asarray(probability, dtype=float)
    predicted = probability >= .5
    return {"games": len(y), "correct": int(np.sum(predicted == np.asarray(y))),
            "accuracy": accuracy_score(y, predicted), "log_loss": log_loss(y, probability, labels=[0, 1]),
            "brier": brier_score_loss(y, probability),
            "roc_auc": roc_auc_score(y, probability) if pd.Series(y).nunique() == 2 else np.nan}


def walk_forward_folds(data, first_validation_season=2022):
    folds = []
    for season in sorted(data.season.unique()):
        if season < first_validation_season:
            continue
        for start, end in [(1, 9), (10, 18)]:
            train = data.index[(data.season < season) | ((data.season == season) & (data.week < start))].to_numpy()
            valid = data.index[(data.season == season) & data.week.between(start, end)].to_numpy()
            if len(train) < 200 or not len(valid):
                continue
            assert data.loc[train, "gameday"].max() < data.loc[valid, "gameday"].min()
            folds.append((f"{season} W{start}-{end}", train, valid))
    assert folds, "Insufficient historical data for chronological validation"
    held = np.concatenate([v for _, _, v in folds])
    assert len(held) == len(set(held)), "Validation games counted twice"
    return folds


BASE_PARAMS = dict(max_depth=2, min_child_weight=5, learning_rate=.04,
                   n_estimators=150, reg_lambda=10., reg_alpha=.5, gamma=.25,
                   subsample=.9, colsample_bytree=.9)


def make_candidates(groups, random_state=20260915, samples_per_group=8):
    space = {
        "max_depth": [1, 2, 3], "min_child_weight": [3, 5, 10, 20],
        "learning_rate": [.02, .04, .07, .1], "n_estimators": [50, 100, 200, 350],
        "reg_lambda": [1., 5., 10., 20., 40.], "reg_alpha": [0., .5, 2.],
        "gamma": [0., .25, 1., 2.], "subsample": [.75, .9, 1.],
        "colsample_bytree": [.7, .9, 1.],
    }
    sampled = list(ParameterSampler(space, n_iter=samples_per_group - 1, random_state=random_state))
    candidates = []
    for group in groups:
        for number, params in enumerate([BASE_PARAMS] + sampled):
            candidates.append({"candidate": f"{group}:{number}", "feature_set": group,
                               "params": dict(params), "history_seasons": [None, 3, None, 5][number % 4],
                               "half_life": [None, 2., 4., None][number % 4]})
    return candidates


def fit_xgb(data, columns, spec, seeds=(1,)):
    training = data
    latest = int(training.season.max())
    if spec.get("history_seasons"):
        training = training.loc[training.season > latest - spec["history_seasons"]]
    X = training[columns].astype(float)
    assert not X.isna().all().any(), "All-missing training feature"
    assert not np.isinf(X.to_numpy()).any()
    weights = None
    if spec.get("half_life"):
        weights = 2. ** (-(latest - training.season.to_numpy()) / spec["half_life"])
        weights /= weights.mean()
    models = []
    for seed in seeds:
        model = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                              tree_method="hist", n_jobs=1, random_state=seed, **spec["params"])
        model.fit(X, training.home_win.astype(int), sample_weight=weights)
        models.append(model)
    return models


def predict_models(models, data, columns):
    return np.mean([model.predict_proba(data[columns].astype(float))[:, 1] for model in models], axis=0)


class BlendedPregameModel:
    """Serializable XGBoost/logistic/QB-Elo blend with inspectable components."""

    def __init__(self, xgb_models, logistic_model, columns, weights=(1/3, 1/3, 1/3),
                 elo_qb_points=ELO_QB_POINTS):
        weights = np.asarray(weights, dtype=float)
        if (weights.shape != (3,) or not np.isfinite(weights).all()
                or (weights < 0).any() or not np.isclose(weights.sum(), 1)):
            raise ValueError("Three nonnegative blend weights must sum to one")
        self.xgb_models = xgb_models
        self.logistic_model = logistic_model
        self.columns = list(columns)
        self.weights = weights
        self.elo_qb_points = elo_qb_points

    def component_probabilities(self, data):
        return np.column_stack([
            predict_models(self.xgb_models, data, self.columns),
            self.logistic_model.predict_proba(data[self.columns].astype(float))[:, 1],
            elo_qb_probability(data, self.elo_qb_points),
        ])

    def predict_proba(self, data):
        probability = self.component_probabilities(data) @ self.weights
        return np.column_stack([1 - probability, probability])


def cross_validate_spec(data, folds, groups, spec, seeds=(1,)):
    columns = groups[spec["feature_set"]]
    predictions, fold_rows = [], []
    for label, train, valid in folds:
        models = fit_xgb(data.loc[train], columns, spec, seeds)
        probability = predict_models(models, data.loc[valid], columns)
        fold_rows.append({"fold": label, **metrics(data.loc[valid, "home_win"], probability)})
        predictions.append(pd.DataFrame({"index": valid, "home_win": data.loc[valid, "home_win"].to_numpy(),
                                         "probability": probability, "fold": label}))
    predictions = pd.concat(predictions, ignore_index=True)
    summary = {"candidate": spec["candidate"], "feature_set": spec["feature_set"],
               "features": len(columns), **metrics(predictions.home_win, predictions.probability),
               "worst_fold_accuracy": min(row["accuracy"] for row in fold_rows),
               "fold_accuracy_sd": float(np.std([row["accuracy"] for row in fold_rows])),
               "seeds": len(seeds)}
    return summary, pd.DataFrame(fold_rows), predictions


def search_xgb(data, folds, groups, candidates, workers=4, verbose=True):
    """80 fixed trials, then three-seed probability averaging for the top five."""
    by_id = {spec["candidate"]: spec for spec in candidates}
    rows = []
    def evaluate(spec):
        return cross_validate_spec(data, folds, groups, spec)[0]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for number, row in enumerate(pool.map(evaluate, candidates), 1):
            rows.append(row)
            if verbose and (number % 20 == 0 or number == len(candidates)):
                print(f"Completed {number}/{len(candidates)} chronological search trials", flush=True)
    ranking = pd.DataFrame(rows).sort_values(["accuracy", "log_loss", "features"], ascending=[False, True, True])
    finalists = []
    details = {}
    for candidate in ranking.head(5).candidate:
        result = cross_validate_spec(data, folds, groups, by_id[candidate], seeds=(1, 7, 21))
        finalists.append(result[0])
        details[candidate] = result[1:]
    final_ranking = pd.DataFrame(finalists).sort_values(["accuracy", "log_loss", "features"], ascending=[False, True, True])
    winner = str(final_ranking.iloc[0].candidate)
    return {"selected": by_id[winner], "search_results": ranking.reset_index(drop=True),
            "finalists": final_ranking.reset_index(drop=True), "fold_results": details[winner][0],
            "oof_predictions": details[winner][1]}


def nested_audit(data, groups, candidates, test_seasons=(2023, 2024)):
    """Repeat the entire search using only data before each outer test year."""
    rows = []
    for season in test_seasons:
        past = data.loc[data.season < season].copy()
        held = data.loc[data.season == season].copy()
        inner = walk_forward_folds(past, first_validation_season=2021)
        result = search_xgb(past, inner, groups, candidates, verbose=False)
        selected = result["selected"]
        columns = groups[selected["feature_set"]]
        models = fit_xgb(past, columns, selected, seeds=(1, 7, 21))
        probability = predict_models(models, held, columns)
        rows.append({"season": season, "candidate": selected["candidate"],
                     **metrics(held.home_win, probability),
                     "always_home_accuracy": held.home_win.mean(),
                     "elo_accuracy": metrics(held.home_win, held.elo_home_probability)["accuracy"]})
        print(f"Nested audit {season}: {rows[-1]['accuracy']:.1%} across {len(held)} games", flush=True)
    return pd.DataFrame(rows)


def logistic_baseline(data, folds, columns):
    records = []
    for strength in [.01, .1, 1., 10.]:
        probabilities, outcomes = [], []
        for _, train, valid in folds:
            model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                  StandardScaler(), LogisticRegression(C=strength, max_iter=2000))
            model.fit(data.loc[train, columns], data.loc[train, "home_win"].astype(int))
            probabilities.extend(model.predict_proba(data.loc[valid, columns])[:, 1])
            outcomes.extend(data.loc[valid, "home_win"])
        records.append({"C": strength, **metrics(outcomes, probabilities)})
    ranking = pd.DataFrame(records).sort_values(["accuracy", "log_loss"], ascending=[False, True])
    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                          LogisticRegression(C=float(ranking.iloc[0].C), max_iter=2000))
    model.fit(data[columns], data.home_win.astype(int))
    return model, ranking


def block_bootstrap_difference(y, first, second, weeks, repeats=3000, seed=20260915):
    """Paired week-level bootstrap; a descriptive interval, not a guarantee."""
    delta = ((np.asarray(first) >= .5) == np.asarray(y)).astype(float)
    delta -= ((np.asarray(second) >= .5) == np.asarray(y)).astype(float)
    blocks = [delta[np.asarray(weeks) == week] for week in pd.unique(weeks)]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        sampled = [blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]
        draws.append(np.concatenate(sampled).mean())
    return {"accuracy_difference": delta.mean(), "lower_95": np.quantile(draws, .025),
            "upper_95": np.quantile(draws, .975)}


def save_model_bundle(models, columns, spec, training_data, output_dir="model_artifacts",
                      feature_builder_version="season_local_rates_v1"):
    """Versioned output; the 2025 benchmark model is separate from deployment."""
    import joblib
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    effective_training = training_data
    if spec.get("history_seasons"):
        effective_training = training_data.loc[training_data.season > training_data.season.max() - spec["history_seasons"]]
    import sklearn
    import xgboost
    manifest = {"feature_columns": columns, "selected_spec": spec, "seeds": [1, 7, 21],
                "training_seasons": sorted(map(int, effective_training.season.unique())),
                "training_games": len(effective_training), "last_training_date": str(training_data.gameday.max().date()),
                "threshold": .5, "averages_reset_by_season": True,
                "elo_offseason_carry": .75, "historical_start_season": 2020,
                "feature_builder_version": feature_builder_version,
                "xgboost_version": xgboost.__version__, "sklearn_version": sklearn.__version__,
                "target": "home win; completed regular-season non-ties"}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]
    family = spec.get("model_family", "xgb")
    destination = output / f"nfl_{family}_{digest}.joblib"
    joblib.dump({"models": models, **manifest}, destination)
    destination.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


def predict_upcoming_week(bundle_path, season, week, refresh=True, log=True,
                          game_id=None, return_features=False, qb_overrides=None):
    """Refresh completed history, predict only genuinely future games, log once."""
    import joblib
    bundle = joblib.load(bundle_path)
    raw_schedule = load_schedule(refresh=refresh)
    uses_qb = bool(set(QB_COLUMNS) & set(bundle["feature_columns"]))
    if uses_qb:
        players = pd.concat([load_player_stats(range(2020, season), refresh=False),
                             load_player_stats([season], refresh=refresh)], ignore_index=True)
        raw_schedule = correct_qb_ids(raw_schedule, players, qb_overrides, season, week)
    schedule = prepare_schedule(raw_schedule, 2020, season)
    cutoff = pd.Timestamp.now(tz="America/New_York")
    target = schedule.loc[schedule.season.eq(season) & schedule.week.eq(week)].copy()
    if game_id is not None:
        target = target.loc[target.game_id.eq(game_id)].copy()
    if target.empty:
        raise ValueError("Requested week is absent from the schedule")
    earlier = schedule.loc[(schedule.season < season) | (schedule.season.eq(season) & schedule.week.lt(week))]
    incomplete = earlier.loc[~earlier.completed & ~earlier.game_id.eq("2022_17_BUF_CIN")]
    if not incomplete.empty:
        raise ValueError("Earlier games are still incomplete; choose the current upcoming week")
    if target.gametime.isna().any():
        raise ValueError("Missing kickoff time; cannot verify that predictions are pregame")
    target["kickoff"] = pd.to_datetime(
        target.gameday.dt.strftime("%Y-%m-%d") + " " + target.gametime
    ).dt.tz_localize("America/New_York")
    target = target.loc[~target.completed & target.kickoff.gt(cutoff)].copy()
    if target.empty:
        raise ValueError("No unplayed, future games in the requested week")
    if bundle["last_training_date"] >= str(target.gameday.min().date()):
        raise ValueError("Model training includes games on/after this prediction date")
    earlier = earlier.loc[earlier.completed]
    current_raw = pd.concat([
        load_team_stats(range(2020, season), refresh=False),
        load_team_stats([season], refresh=refresh),
    ], ignore_index=True)
    inputs = pd.concat([earlier, target], ignore_index=True)
    features, _ = build_pregame_features(inputs, current_raw)
    future = features.loc[features.game_id.isin(target.game_id)].copy()
    if uses_qb:
        starts = qb_schedule(raw_schedule, 2020, season)
        future = add_qb_features(future, starts, qb_history(starts, players))
    probability = predict_models(bundle["models"], future, bundle["feature_columns"])
    cutoff = pd.Timestamp.now(tz="America/New_York")
    if not target.kickoff.gt(cutoff).all():
        raise ValueError("A game started while loading inputs; rerun to exclude it")
    shown = ["game_id", "season", "week", "gameday", "home_team", "away_team"]
    result = future[shown + [c for c in ["away_qb_name", "home_qb_name"] if c in future]].copy()
    result["home_win_probability"] = probability
    result["predicted_winner"] = np.where(probability >= .5, result.home_team, result.away_team)
    result["prediction_time"] = str(cutoff)
    result["model_version"] = Path(bundle_path).stem
    if len(bundle["models"]) == 1 and isinstance(bundle["models"][0], BlendedPregameModel):
        components = bundle["models"][0].component_probabilities(future)
        for i, name in enumerate(["xgb", "logistic", "elo_qb"]):
            result[f"{name}_home_probability"] = components[:, i]
        result["component_disagreement"] = (components >= .5).any(axis=1) & (components < .5).any(axis=1)
    if log:
        directory = Path("prediction_logs")
        directory.mkdir(exist_ok=True)
        existing = [game_id for game_id in result.game_id if (directory / f"{game_id}.json").exists()]
        if existing:
            raise FileExistsError(f"Predictions already frozen for {existing}; no files changed")
        for index, row in result.iterrows():
            destination = directory / f"{row.game_id}.json"
            payload = {"prediction": row.to_dict(), "features": future.loc[index, bundle["feature_columns"]].to_dict()}
            try:
                with destination.open("x") as handle:
                    json.dump(payload, handle, indent=2, default=str)
            except FileExistsError:
                raise FileExistsError(f"A prediction for {row.game_id} is already frozen; refusing to overwrite it")
    if return_features:
        return result, future.set_index("game_id")[bundle["feature_columns"]].astype(float)
    return result
