"""Bounded Week 8 comparison. Choose on 2022-24; later seasons are reports only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import nfl_model_pipeline as nfl

OUT = Path('model_artifacts/week8_experiment')


def load_features():
    players = nfl.load_player_stats(range(2020, 2027))
    schedule = nfl.correct_qb_ids(nfl.load_schedule(), players)
    games = nfl.prepare_schedule(schedule, 2020, 2026)
    features, _ = nfl.build_pregame_features(games, nfl.load_team_stats(range(2020, 2027)))
    starts = nfl.qb_schedule(schedule, 2020, 2026)
    return nfl.add_qb_features(features, starts, nfl.qb_history(starts, players))


def definitions():
    groups = nfl.feature_sets()
    manifest = json.loads(Path('model_artifacts/nfl_xgb_c0f0f2b5583c.json').read_text())
    base = manifest['feature_columns']
    recent = list(dict.fromkeys(groups['efficiency_recent4'] + nfl.QB_COLUMNS))
    turnovers = base + ['gap_avg_giveaway_rate', 'gap_avg_takeaway_rate', 'gap_avg_qb_hit_rate']
    specs = {}
    for label, columns in [('base', base), ('recent4', recent), ('turnovers', turnovers)]:
        specs[f'xgb_{label}'] = dict(kind='xgb', columns=columns, spec=manifest['selected_spec'])
        specs[f'lr_{label}'] = dict(kind='lr', columns=columns, C=.01)
    specs['elo_qb'] = dict(kind='elo', columns=['elo_gap', 'gap_qb_delta'])
    specs['blend_base'] = dict(kind='blend', components=['xgb_base', 'lr_base', 'elo_qb'], weights=[1/3]*3)
    specs['blend_recent4'] = dict(kind='blend', components=['xgb_recent4', 'lr_recent4', 'elo_qb'], weights=[1/3]*3)
    specs['blend_turnovers'] = dict(kind='blend', components=['xgb_turnovers', 'lr_turnovers', 'elo_qb'], weights=[1/3]*3)
    specs['blend_conservative'] = dict(kind='blend', components=['xgb_base', 'lr_base', 'elo_qb'], weights=[.5, .25, .25])
    return specs


def fit_components(train, specs):
    fitted = {}
    for name, definition in specs.items():
        columns = definition.get('columns')
        if definition['kind'] == 'xgb':
            fitted[name] = nfl.fit_xgb(train, columns, definition['spec'], seeds=(1, 7, 21))
        elif definition['kind'] == 'lr':
            model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                                  LogisticRegression(C=definition['C'], max_iter=2000))
            model.fit(train[columns], train.home_win.astype(int))
            fitted[name] = model
    return fitted


def predict(data, specs, fitted):
    result = {}
    for name, definition in specs.items():
        kind = definition['kind']
        if kind == 'xgb':
            result[name] = nfl.predict_models(fitted[name], data, definition['columns'])
        elif kind == 'lr':
            result[name] = fitted[name].predict_proba(data[definition['columns']])[:, 1]
        elif kind == 'elo':
            result[name] = nfl.elo_qb_probability(data)
        else:
            result[name] = sum(w * result[c] for w, c in zip(definition['weights'], definition['components']))
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    specs = definitions()
    data = load_features()
    data = data.loc[data.home_win.notna()].reset_index(drop=True)
    development = data.loc[data.season < 2025]
    folds = nfl.walk_forward_folds(development, 2022)
    frames = []
    for label, train, valid in folds:
        held = data.loc[valid]
        result = predict(held, specs, fit_components(data.loc[train], specs))
        frame = held[['game_id', 'season', 'week', 'home_win']].copy()
        for name, p in result.items():
            frame[name] = p
        frames.append(frame)
        print('Validated', label, flush=True)
    oof = pd.concat(frames)
    validation = pd.DataFrame([dict(name=name, **nfl.metrics(oof.home_win, oof[name])) for name in specs])
    baseline = validation.set_index('name').loc['xgb_base']
    eligible = validation.loc[(validation.accuracy >= baseline.accuracy) &
                              (validation.log_loss < baseline.log_loss)]
    selected = 'xgb_base' if eligible.empty else eligible.sort_values(['log_loss', 'name']).iloc[0]['name']
    tests, predictions = [], []
    for season in [2025, 2026]:
        held = data.loc[data.season.eq(season)]
        if held.empty:
            continue
        result = predict(held, specs, fit_components(data.loc[data.season < season], specs))
        for name, probability in result.items():
            tests.append(dict(season=season, name=name, **nfl.metrics(held.home_win, probability)))
            frame = held[['game_id', 'season', 'week', 'home_team', 'away_team', 'home_win']].copy()
            frame['model'], frame['probability'] = name, probability
            predictions.append(frame)
    validation.to_csv(OUT / 'validation.csv', index=False)
    oof.to_csv(OUT / 'validation_predictions.csv', index=False)
    pd.concat(predictions).to_csv(OUT / 'predictions.csv', index=False)
    report = dict(selection_seasons=[2022, 2023, 2024], selected=selected, definitions=specs,
                  validation=validation.to_dict('records'), tests=tests,
                  last_current_week=int(data.loc[data.season.eq(2026), 'week'].max()),
                  limitations=['2025 is previously inspected, not an untouched holdout.',
                               '2026 is retrospective monitoring, not frozen pregame performance.',
                               'Historical primary QB is a proxy for the announced starter.'])
    (OUT / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Selected:', selected)
    print(validation.to_string(index=False))
    print(pd.DataFrame(tests).to_string(index=False))


if __name__ == '__main__':
    main()
