"""Opponent-adjusted, recency-weighted and prior-season-blended team stats."""
from itertools import product
from pathlib import Path
import json
import numpy as np
import pandas as pd
import nfl_model_pipeline as nfl

OUT = Path('model_artifacts/opponent_adjust_experiment')
SEEDS = (1, 7, 21)
HALF_LIFE, PRIOR_GAMES, PRIOR_RETENTION, SHRINK_GAMES, ITERATIONS = 6., 4., .75, 3., 10
STATS = {
    'pass_epa': ('passing_epa', 'dropbacks', 'pass_epa_rate', 'def_pass_epa_allowed'),
    'rush_epa': ('rushing_epa', 'carries', 'rush_epa_rate', 'def_rush_epa_allowed'),
    'first_down': ('first_downs', 'plays', 'first_down_rate', 'def_first_down_allowed'),
    'sack': ('sacks_suffered', 'dropbacks', 'sack_allowed_rate', 'sack_made_rate'),
    'points': ('points_for', 'one', 'points_for', 'points_against'),
    'cpoe': ('cpoe', 'cpoe_games', 'passing_cpoe', None),
}
METRICS = ['points_for', 'points_against', 'pass_epa_rate', 'rush_epa_rate', 'def_pass_epa_allowed',
           'def_rush_epa_allowed', 'passing_cpoe', 'sack_allowed_rate', 'sack_made_rate',
           'first_down_rate', 'def_first_down_allowed']
VARIANTS = {'+'.join(n for n, on in zip(['adjust', 'recency', 'prior'], flags) if on) or 'base': flags
            for flags in product([False, True], repeat=3)}


def team_games(games, raw):
    """One row per team per game with offensive numerators and denominators."""
    raw = nfl.normalize_teams(raw, ['team', 'opponent_team'])
    raw = raw[raw.season_type.eq('REG')].copy()
    raw[nfl.RAW_REQUIRED] = raw[nfl.RAW_REQUIRED].apply(pd.to_numeric, errors='raise')
    sides = []
    for side, other in [('home', 'away'), ('away', 'home')]:
        sides.append(games[['game_id', 'season', 'week', 'completed', f'{side}_team', f'{other}_team',
                            f'{side}_score']].set_axis(['game_id', 'season', 'week', 'completed', 'team',
                                                        'opponent', 'points_for'], axis=1))
    frame = pd.concat(sides, ignore_index=True).merge(
        raw[['game_id', 'team'] + nfl.RAW_REQUIRED], on=['game_id', 'team'], how='left', validate='one_to_one')
    done = frame[frame.completed].copy()
    assert done[[c for c in nfl.RAW_REQUIRED if c != 'passing_cpoe']].notna().all().all()
    done['dropbacks'] = done.attempts + done.sacks_suffered
    done['plays'] = done.dropbacks + done.carries
    done['first_downs'] = done.passing_first_downs + done.rushing_first_downs
    done['one'] = 1.
    done['cpoe_games'] = done.passing_cpoe.notna().astype(float)
    done['cpoe'] = done.passing_cpoe.fillna(0.)
    return frame, done


def strengths(block, teams, num, den):
    """Shrunk offensive and defensive deviations from league average."""
    t, o = block.team_idx.to_numpy(), block.opp_idx.to_numpy()
    x, d = block[num].to_numpy(float), block[den].to_numpy(float)
    size = len(teams)
    league = x.sum() / d.sum()
    games = np.bincount(t, d > 0, size)
    shrink = games / (games + SHRINK_GAMES)
    total = np.bincount(t, d, size)
    off_dev, def_dev = np.zeros(size), np.zeros(size)
    with np.errstate(invalid='ignore', divide='ignore'):
        for _ in range(ITERATIONS):
            offense = np.bincount(t, x - d * def_dev[o], size) / total
            defense = np.bincount(o, x - d * off_dev[t], size) / np.bincount(o, d, size)
            off_dev = np.nan_to_num((offense - league) * shrink)
            def_dev = np.nan_to_num((defense - league) * shrink)
    return league, off_dev, def_dev


def summarize(block, teams, adjust, recency):
    """Weighted ratio-of-sums offense and defense rates per team for one cutoff."""
    block = block.copy()
    order = block.groupby('team_idx').cumcount(ascending=False)
    block['w'] = .5 ** (order / HALF_LIFE) if recency else 1.
    result = {}
    size = len(teams)
    weight_by_team = np.bincount(block.team_idx, block.w, size)
    for key, (num, den, off_name, def_name) in STATS.items():
        x, d, w = block[num].to_numpy(float), block[den].to_numpy(float), block.w.to_numpy()
        t, o = block.team_idx.to_numpy(), block.opp_idx.to_numpy()
        if adjust:
            _, off_dev, def_dev = strengths(block, teams, num, den)
            offense_x, defense_x = x - d * def_dev[o], x - d * off_dev[t]
        else:
            offense_x = defense_x = x
        opp_w = block.set_index(['game_id', 'team_idx']).w
        w_def = opp_w.reindex(pd.MultiIndex.from_arrays([block.game_id, block.opp_idx])).to_numpy()
        with np.errstate(invalid='ignore', divide='ignore'):
            result[off_name] = np.bincount(t, w * offense_x, size) / np.bincount(t, w * d, size)
            if def_name:
                result[def_name] = np.bincount(o, w_def * defense_x, size) / np.bincount(o, w_def * d, size)
    frame = pd.DataFrame(result, index=teams)
    frame['effective_games'] = weight_by_team
    return frame


def rebuild(games, raw, base, adjust, recency, prior):
    """Return base features with the efficiency team-stat columns replaced."""
    schedule, done = team_games(games, raw)
    teams = sorted(set(schedule.team))
    index = {team: i for i, team in enumerate(teams)}
    done['team_idx'], done['opp_idx'] = done.team.map(index), done.opponent.map(index)
    done = done.sort_values(['season', 'week'])
    records = []
    previous = None
    for season in sorted(schedule.season.unique()):
        played = done[done.season == season]
        rows = schedule[schedule.season == season]
        for week in sorted(rows.week.unique()):
            before = played[played.week < week]
            current = summarize(before, teams, adjust, recency) if len(before) else \
                pd.DataFrame(np.nan, index=teams, columns=METRICS).assign(effective_games=0.)
            if prior and previous is not None:
                n = current.effective_games.to_numpy()[:, None]
                old = previous[METRICS].to_numpy()
                now = current[METRICS].to_numpy()
                blended = np.where(np.isfinite(now), (n * np.nan_to_num(now) + PRIOR_GAMES * old) / (n + PRIOR_GAMES), old)
                current[METRICS] = np.where(np.isfinite(old), blended, now)
            for team in rows.loc[rows.week == week, 'team']:
                records.append({'season': season, 'week': week, 'team': team, **current.loc[team, METRICS]})
        full = summarize(played, teams, adjust, False)
        means = full[METRICS].mean()
        previous = means + PRIOR_RETENTION * (full[METRICS] - means)
        previous = previous.where(full.effective_games.gt(0), axis=0)
    stats = pd.DataFrame(records)
    result = base.copy()
    for side in ['home', 'away']:
        side_stats = stats.rename(columns={'team': f'{side}_team', **{m: f'{side}_avg_{m}' for m in METRICS}})
        result = result.drop(columns=[f'{side}_avg_{m}' for m in METRICS]).merge(
            side_stats, on=['season', 'week', f'{side}_team'], how='left', validate='one_to_one')
    for m in METRICS:
        result[f'gap_avg_{m}'] = result[f'home_avg_{m}'] - result[f'away_avg_{m}']
    result['net_scoring_gap_avg'] = result.gap_avg_points_for - result.gap_avg_points_against
    for kind in ['pass', 'rush']:
        result[f'net_{kind}_epa_gap_avg'] = result[f'gap_avg_{kind}_epa_rate'] - result[f'gap_avg_def_{kind}_epa_allowed']
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = nfl.load_team_stats(range(2020, 2027))
    games = nfl.prepare_schedule(nfl.load_schedule(), 2020, 2026)
    base, _ = nfl.build_pregame_features(games, raw)
    columns = nfl.feature_sets()['efficiency_context']
    variants = {name: rebuild(games, raw, base, *flags) for name, flags in VARIANTS.items()}
    check = variants['base'].set_index('game_id').loc[base.game_id, columns]
    pd.testing.assert_frame_equal(check.reset_index(drop=True).astype(float),
                                  base[columns].reset_index(drop=True).astype(float), atol=1e-9)
    variants = {name: data[data.home_win.notna()].reset_index(drop=True) for name, data in variants.items()}
    reference = variants['base']
    for data in variants.values():
        assert data.game_id.equals(reference.game_id)
    print('Built feature variants; base matches the pipeline', flush=True)

    deployment = json.loads(Path('model_artifacts/nfl_xgb_24b9fc23fae6.json').read_text())['selected_spec']
    folds = nfl.walk_forward_folds(reference[reference.season < 2025], 2022)
    index = np.concatenate([valid for _, _, valid in folds])
    y = reference.loc[index, 'home_win'].to_numpy()
    early = reference.loc[index, 'week'].le(4).to_numpy()

    def fit_predict(data, train, held):
        return nfl.predict_models(nfl.fit_xgb(data.loc[train], columns, deployment, SEEDS), data.loc[held], columns)

    rows, tests, predictions, oof = [], [], [], {}
    for name, data in variants.items():
        oof[name] = np.concatenate([fit_predict(data, t, v) for _, t, v in folds])
        rows.append({'name': name, **nfl.metrics(y, oof[name]),
                     'weeks_1_4': nfl.metrics(y[early], oof[name][early]),
                     'vs_base': nfl.block_bootstrap_difference(y, oof[name], oof['base'],
                                                               reference.loc[index, 'season'].astype(str) + '-' +
                                                               reference.loc[index, 'week'].astype(str))})
    print('Validated 2022-24', flush=True)
    for season in [2025, 2026]:
        held_index = reference.index[reference.season == season]
        train_index = reference.index[reference.season < season]
        held = reference.loc[held_index]
        result = {name: fit_predict(data, train_index, held_index) for name, data in variants.items()}
        first = held.week.le(4).to_numpy()
        for name, probability in result.items():
            tests.append({'season': season, 'name': name, **nfl.metrics(held.home_win, probability),
                          'weeks_1_4': nfl.metrics(held.home_win[first], probability[first]),
                          'vs_base': nfl.block_bootstrap_difference(held.home_win, probability,
                                                                    result['base'], held.week)})
            frame = held[['game_id', 'season', 'week', 'home_team', 'away_team', 'home_win']].copy()
            frame['model'], frame['probability'] = name, probability
            predictions.append(frame)
    pooled = pd.concat(predictions)
    pooled_rows = []
    for name, block in pooled.groupby('model', sort=False):
        basis = pooled[pooled.model == 'base']
        pooled_rows.append({'name': name, **nfl.metrics(block.home_win, block.probability),
                            'vs_base': nfl.block_bootstrap_difference(
                                block.home_win, block.probability, basis.probability,
                                block.season.astype(str) + '-' + block.week.astype(str))})

    def flat(records):
        out = []
        for r in records:
            row = {k: v for k, v in r.items() if not isinstance(v, dict)}
            if 'weeks_1_4' in r:
                row.update(w1_4_correct=r['weeks_1_4']['correct'], w1_4_games=r['weeks_1_4']['games'],
                           w1_4_log_loss=r['weeks_1_4']['log_loss'])
            row.update({f'acc_diff_{k}': v for k, v in r['vs_base'].items()})
            out.append(row)
        return pd.DataFrame(out)

    validation, holdout, pooled_table = flat(rows), flat(tests), flat(pooled_rows)
    validation.to_csv(OUT / 'validation.csv', index=False)
    holdout.to_csv(OUT / 'holdout.csv', index=False)
    pooled_table.to_csv(OUT / 'holdout_pooled.csv', index=False)
    pooled.to_csv(OUT / 'predictions.csv', index=False)
    (OUT / 'results.json').write_text(json.dumps({
        'validation_seasons': [2022, 2023, 2024], 'validation_games': len(y),
        'holdout_games': {int(s): int((reference.season == s).sum()) for s in [2025, 2026]},
        'last_2026_week': int(reference[reference.season == 2026].week.max()),
        'settings': {'half_life_games': HALF_LIFE, 'prior_games': PRIOR_GAMES,
                     'prior_retention': PRIOR_RETENTION, 'shrink_games': SHRINK_GAMES,
                     'iterations': ITERATIONS},
        'xgb_spec': deployment, 'feature_columns': columns,
        'validation': rows, 'tests': tests, 'pooled_holdout': pooled_rows}, indent=2, default=float))
    pd.set_option('display.width', 250)
    for title, table in [('validation', validation), ('holdout', holdout), ('pooled', pooled_table)]:
        print(title)
        print(table.drop(columns=['roc_auc']).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
