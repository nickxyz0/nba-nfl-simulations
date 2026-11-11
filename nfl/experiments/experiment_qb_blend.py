"""QB-change features, a QB-adjusted Elo and an equal three-model blend."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import nfl_model_pipeline as nfl

OUT = Path('model_artifacts/qb_blend_experiment')
SEEDS = (1, 7, 21)
ELO_QB_GRID = [0, 100, 200, 300, 400, 500, 600, 800, 1000]
QB_COLUMNS = nfl.QB_COLUMNS
elo_qb_probability = nfl.elo_qb_probability


def fit_predict(train, held, columns, spec):
    if 'C' in spec:
        model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                              LogisticRegression(C=spec['C'], max_iter=2000))
        model.fit(train[columns], train.home_win.astype(int))
        return model.predict_proba(held[columns])[:, 1]
    return nfl.predict_models(nfl.fit_xgb(train, columns, spec, SEEDS), held, columns)


def calibration_table(y, probability):
    """Bucket by the picked side's probability, as the weekly picks are read."""
    y, probability = np.asarray(y), np.asarray(probability)
    picked = np.maximum(probability, 1 - probability)
    correct = (probability >= .5) == y
    rows = []
    for low, high in [(.5, .55), (.55, .6), (.6, .65), (.65, .7), (.7, .75), (.75, 1.01)]:
        mask = (picked >= low) & (picked < high)
        if mask.any():
            rows.append({'bucket': f'{low:.0%}-{min(high, 1):.0%}', 'games': int(mask.sum()),
                         'expected': picked[mask].mean(), 'actual': correct[mask].mean()})
    return pd.DataFrame(rows)


def reliability_plot(table, path, title):
    ink, muted, grid, series = '#0b0b0b', '#52514e', '#e4e3df', '#2a78d6'
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    fig.patch.set_facecolor('#fcfcfb')
    ax.set_facecolor('#fcfcfb')
    ax.plot([.5, .8], [.5, .8], color=muted, lw=1, ls='--', zorder=1)
    ax.text(.775, .79, 'perfectly calibrated', color=muted, fontsize=8, ha='right', rotation=36)
    ax.plot(table.expected, table.actual, color=series, lw=2, zorder=2)
    ax.scatter(table.expected, table.actual, s=36 + table.games / 3, color=series,
               edgecolor='#fcfcfb', linewidth=2, zorder=3)
    for row in table.itertuples():
        ax.annotate(f'{row.games} games', (row.expected, row.actual), textcoords='offset points',
                    xytext=(8, -12), fontsize=8, color=muted)
    ax.set(xlim=(.49, .82), ylim=(.4, .9), xlabel="Model's probability for its pick",
           ylabel='Share of picks that won')
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.grid(color=grid, lw=.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=muted, length=0)
    ax.xaxis.label.set_color(ink)
    ax.yaxis.label.set_color(ink)
    ax.set_title(title, color=ink, fontsize=11, loc='left')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = nfl.load_team_stats(range(2020, 2027))
    players = nfl.load_player_stats(range(2020, 2027))
    corrected = nfl.correct_qb_ids(nfl.load_schedule(), players)
    schedule = nfl.qb_schedule(corrected, 2020, 2026)
    games = nfl.prepare_schedule(corrected, 2020, 2026)
    base, _ = nfl.build_pregame_features(games, raw)
    history = nfl.qb_history(schedule, players)
    data = nfl.add_qb_features(base, schedule, history)
    data = data[data.home_win.notna()].reset_index(drop=True)
    missing = data[['home_qb_id', 'away_qb_id']].isna().any(axis=1).sum()
    assert missing == 0, f'{missing} completed games lack a starting QB id'

    deployment = json.loads(Path('model_artifacts/nfl_xgb_24b9fc23fae6.json').read_text())['selected_spec']
    base_columns = nfl.feature_sets()['efficiency_context']
    column_sets = {'base': base_columns, 'qb': base_columns + QB_COLUMNS}
    development = data[data.season < 2025]
    folds = nfl.walk_forward_folds(development, 2022)
    index = np.concatenate([valid for _, _, valid in folds])
    y = data.loc[index, 'home_win'].to_numpy()

    oof, rows = {}, []
    def validate(name, columns, spec):
        oof[name] = np.concatenate([fit_predict(data.loc[t], data.loc[v], columns, spec) for _, t, v in folds])
        rows.append({'name': name, **nfl.metrics(y, oof[name])})
    specs = {}
    for label, columns in column_sets.items():
        validate(f'xgb_{label}', columns, deployment)
        specs[f'xgb_{label}'] = (columns, deployment)
        best = None
        for c in [.01, .1, 1.]:
            probability = np.concatenate([fit_predict(data.loc[t], data.loc[v], columns, {'C': c})
                                          for _, t, v in folds])
            loss = nfl.metrics(y, probability)['log_loss']
            if best is None or loss < best[0]:
                best = (loss, c, probability)
        oof[f'lr_{label}'] = best[2]
        specs[f'lr_{label}'] = (columns, {'C': best[1]})
        rows.append({'name': f'lr_{label}', 'C': best[1], **nfl.metrics(y, best[2])})
        print(f'Validated {label} features', flush=True)
    held = data.loc[index]
    elo_search = [{'points': p, **nfl.metrics(y, elo_qb_probability(held, p))} for p in ELO_QB_GRID]
    elo_points = min(elo_search, key=lambda r: r['log_loss'])['points']
    oof['elo_base'] = held.elo_home_probability.to_numpy()
    oof['elo_qb'] = elo_qb_probability(held, elo_points)
    rows += [{'name': 'elo_base', **nfl.metrics(y, oof['elo_base'])},
             {'name': 'elo_qb', 'elo_qb_points': elo_points, **nfl.metrics(y, oof['elo_qb'])}]
    blends = {f'blend_{label}': [f'xgb_{label}', f'lr_{label}', f'elo_{label}'] for label in column_sets}
    for name, parts in blends.items():
        oof[name] = np.mean([oof[p] for p in parts], axis=0)
        rows.append({'name': name, **nfl.metrics(y, oof[name])})
    validation = pd.DataFrame(rows)

    tests, predictions, holdout = [], [], {}
    baseline = 'xgb_base'
    for season in [2025, 2026]:
        test = data[data.season == season]
        train = data[data.season < season]
        result = {name: fit_predict(train, test, *spec) for name, spec in specs.items()}
        result['elo_base'] = test.elo_home_probability.to_numpy()
        result['elo_qb'] = elo_qb_probability(test, elo_points)
        for name, parts in blends.items():
            result[name] = np.mean([result[p] for p in parts], axis=0)
        changed = (test.home_qb_change + test.away_qb_change).gt(0).to_numpy()
        for name, probability in result.items():
            row = {'season': season, 'name': name, **nfl.metrics(test.home_win, probability)}
            row['vs_xgb_base'] = nfl.block_bootstrap_difference(test.home_win, probability,
                                                                 result[baseline], test.week)
            row['qb_change_games'] = nfl.metrics(test.home_win[changed], probability[changed])
            tests.append(row)
            frame = test[['game_id', 'season', 'week', 'home_team', 'away_team', 'home_win',
                          'home_qb_change', 'away_qb_change', 'gap_qb_delta']].copy()
            frame['model'], frame['probability'] = name, probability
            predictions.append(frame)
        holdout[season] = result
    changed = (held.home_qb_change + held.away_qb_change).gt(0).to_numpy()
    subset = [{'name': n, **nfl.metrics(y[changed], oof[n][changed])} for n in oof]

    outcomes = np.concatenate([y, data[data.season == 2025].home_win, data[data.season == 2026].home_win])
    calibration = {name: calibration_table(outcomes, np.concatenate([oof[name], holdout[2025][name], holdout[2026][name]]))
                   for name in [baseline, 'blend_qb']}
    reliability_plot(calibration[baseline], OUT / 'reliability_deployed.png',
                     f'Deployed XGBoost settings, 2022–2026 out of sample ({len(outcomes)} games)')

    validation.to_csv(OUT / 'validation.csv', index=False)
    pd.DataFrame(elo_search).to_csv(OUT / 'elo_qb_search.csv', index=False)
    pd.DataFrame(subset).to_csv(OUT / 'validation_qb_change_games.csv', index=False)
    pd.concat(predictions).to_csv(OUT / 'predictions.csv', index=False)
    for name, table in calibration.items():
        table.to_csv(OUT / f'calibration_{name}.csv', index=False)
    (OUT / 'results.json').write_text(json.dumps({
        'validation_seasons': [2022, 2023, 2024], 'validation_games': len(y),
        'validation_qb_change_games': int(changed.sum()), 'elo_qb_points': elo_points,
        'qb_settings': {'prior_epa_per_play': nfl.QB_PRIOR_EPA, 'prior_plays': nfl.QB_PRIOR_PLAYS,
                        'half_life_games': nfl.QB_HALF_LIFE, 'regular_window': nfl.QB_REGULAR_WINDOW},
        'logistic_C': {k: v[1]['C'] for k, v in specs.items() if 'C' in v[1]},
        'tests': tests}, indent=2, default=float))
    pd.set_option('display.width', 200)
    print(validation.drop(columns=['roc_auc']).to_string(index=False))
    print(pd.DataFrame(subset).drop(columns=['roc_auc']).to_string(index=False))
    print(pd.DataFrame(tests).drop(columns=['vs_xgb_base', 'qb_change_games', 'roc_auc']).to_string(index=False))
    for row in tests:
        print(row['season'], row['name'], {k: round(v, 3) for k, v in row['vs_xgb_base'].items()},
              'qb-change games:', row['qb_change_games']['correct'], '/', row['qb_change_games']['games'])
    for name, table in calibration.items():
        print(name); print(table.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
