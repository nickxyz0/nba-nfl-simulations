"""Reproducible offline comparisons; never changes deployment artifacts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import nfl_model_pipeline as nfl

OUT = Path('model_artifacts/improvement_experiment')
SEEDS = (1, 7, 21)


def carryover(games, raw, strength, regression):
    dummy = []
    for season, block in games.groupby('season'):
        teams = sorted(set(block.home_team) | set(block.away_team))
        for i in range(0, len(teams), 2):
            row = block.iloc[-1].copy()
            row['game_id'] = f'prior_{season}_{i}'
            row['week'] = 19
            row['gameday'] = block.gameday.max() + pd.Timedelta(days=7)
            row['home_team'], row['away_team'] = teams[i:i+2]
            row['home_score'] = row['away_score'] = np.nan
            row['completed'] = False
            dummy.append(row)
    augmented, team = nfl.build_pregame_features(pd.concat([games, pd.DataFrame(dummy)], ignore_index=True), raw)
    full = team[team.game_id.str.startswith('prior_')].copy()
    full['season'] += 1
    averages = [c for c in full if c.startswith('avg_')]
    previous = full.set_index(['season', 'team'])[averages]
    result = augmented[~augmented.game_id.str.startswith('prior_')].copy()
    for side in ['home', 'away']:
        key = pd.MultiIndex.from_arrays([result.season, result[f'{side}_team']])
        prior = previous.reindex(key).reset_index(drop=True)
        count = result[f'{side}_prior_games'].to_numpy()
        w = count / (count + strength)
        for stat in averages:
            current = result[f'{side}_{stat}'].to_numpy()
            old = prior[stat].to_numpy()
            league = full.groupby('season')[stat].mean()
            mean = result.season.map(league).to_numpy()
            old = mean + regression * (old - mean)
            blended = np.where(np.isfinite(current), w * np.nan_to_num(current) + (1-w) * old, old)
            result[f'{side}_{stat}'] = np.where(np.isfinite(old), blended, current)
    for stat in averages:
        result[f'gap_{stat}'] = result[f'home_{stat}'] - result[f'away_{stat}']
    result['net_scoring_gap_avg'] = result.gap_avg_points_for - result.gap_avg_points_against
    for kind in ['pass', 'rush']:
        result[f'net_{kind}_epa_gap_avg'] = result[f'gap_avg_{kind}_epa_rate'] - result[f'gap_avg_def_{kind}_epa_allowed']
    return result


def fit_predict(train, valid, columns, spec):
    if 'C' in spec:
        model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                              LogisticRegression(C=spec['C'], max_iter=2000))
        model.fit(train[columns], train.home_win.astype(int))
        return model.predict_proba(valid[columns])[:, 1]
    return nfl.predict_models(nfl.fit_xgb(train, columns, spec, SEEDS), valid, columns)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = nfl.load_team_stats(range(2020, 2027))
    games = nfl.prepare_schedule(nfl.load_schedule(), 2020, 2026)
    base, _ = nfl.build_pregame_features(games, raw)
    variants = {'season_reset': base}
    for strength in [2, 4, 8]:
        for regression in [.5, .75]:
            variants[f'carry_{strength}_reg_{regression}'] = carryover(games, raw, strength, regression)
    variants = {name: data[data.home_win.notna()].reset_index(drop=True) for name, data in variants.items()}
    columns = nfl.feature_sets()['efficiency_context']
    deployment = json.loads(Path('model_artifacts/nfl_xgb_24b9fc23fae6.json').read_text())['selected_spec']
    configs = {'deployed_settings': deployment}
    for depth, trees in [(2, 150), (2, 300), (3, 150)]:
        params = dict(nfl.BASE_PARAMS, max_depth=depth, n_estimators=trees)
        configs[f'depth{depth}_trees{trees}'] = dict(deployment, params=params)
    for c in [.01, .1, 1]:
        configs[f'logistic_C{c}'] = {'C': c}
    reference = variants['season_reset']
    development = reference[reference.season < 2025]
    folds = nfl.walk_forward_folds(development, 2022)
    index = np.concatenate([valid for _, _, valid in folds])
    y = reference.loc[index, 'home_win'].to_numpy()
    oof, definitions, rows = {}, {}, []
    for variant, data in variants.items():
        for config, spec in configs.items():
            name = f'{variant}/{config}'
            probability = np.concatenate([fit_predict(data.loc[train], data.loc[valid], columns, spec)
                                          for _, train, valid in folds])
            oof[name] = probability
            definitions[name] = {'variant': variant, 'config': config}
            rows.append({'name': name, **nfl.metrics(y, probability)})
        print(f'Validated {variant}', flush=True)
    baseline = 'season_reset/deployed_settings'
    oof['elo'] = reference.loc[index, 'elo_home_probability'].to_numpy()
    definitions['elo'] = {'elo': True}
    rows.append({'name': 'elo', **nfl.metrics(y, oof['elo'])})
    ranked = pd.DataFrame(rows).sort_values(['accuracy', 'log_loss'], ascending=[False, True])
    best_xgb = next(n for n in ranked.name if '/depth' in n or '/deployed' in n)
    best_lr = next(n for n in ranked.name if '/logistic' in n)
    for weights in [(.5, .5, 0), (.5, 0, .5), (0, .5, .5), (1/3, 1/3, 1/3), (.6, .2, .2)]:
        components = [best_xgb, best_lr, 'elo']
        name = 'blend_' + '_'.join(f'{w:.3f}' for w in weights)
        oof[name] = sum(w * oof[n] for w, n in zip(weights, components))
        definitions[name] = {'components': components, 'weights': weights}
        rows.append({'name': name, **nfl.metrics(y, oof[name])})
    ranked = pd.DataFrame(rows).sort_values(['accuracy', 'log_loss'], ascending=[False, True])
    winner = ranked.iloc[0]['name']
    best_probability = ranked.sort_values('log_loss').iloc[0]['name']
    selected = list(dict.fromkeys([baseline, winner, best_xgb, best_lr, 'elo',
                                  best_probability, 'blend_0.333_0.333_0.333']))
    def evaluate(name, season):
        definition = definitions[name]
        if 'components' in definition:
            return sum(w * evaluate(n, season) for w, n in zip(definition['weights'], definition['components']))
        data = variants[definition.get('variant', 'season_reset')]
        held = data[data.season == season]
        if definition.get('elo'):
            return held.elo_home_probability.to_numpy()
        return fit_predict(data[data.season < season], held, columns, configs[definition['config']])
    tests, predictions = [], []
    for season in [2025, 2026]:
        held = reference[reference.season == season]
        if held.empty:
            continue
        results = {name: evaluate(name, season) for name in selected}
        for name, probability in results.items():
            row = {'season': season, 'name': name, **nfl.metrics(held.home_win, probability)}
            row['difference_vs_baseline'] = nfl.block_bootstrap_difference(
                held.home_win, probability, results[baseline], held.week)
            early = held.week <= 4
            row['weeks_1_to_4'] = nfl.metrics(held.loc[early, 'home_win'], probability[early])
            tests.append(row)
            frame = held[['game_id', 'season', 'week', 'home_win']].copy()
            frame['model'] = name
            frame['probability'] = probability
            predictions.append(frame)
    ranked.to_csv(OUT / 'validation.csv', index=False)
    pd.concat(predictions).to_csv(OUT / 'predictions.csv', index=False)
    report = {'selection_seasons': [2022, 2023, 2024], 'selected': winner,
              'best_validation_log_loss': best_probability,
              'validation_games': len(y), 'candidate_count': len(ranked),
              'configurations': configs,
              'baseline': baseline, 'definitions': definitions, 'tests': tests,
              'limitations': ['2025 benchmark was previously inspected.',
                              '2026 uses cached available completed games; no refresh.',
                              'No injury/QB experiment in this run.',
                              'Deployment unchanged; statistical carryover is experimental.']}
    (OUT / 'results.json').write_text(json.dumps(report, indent=2))
    print(ranked.head(8).to_string(index=False), flush=True)
    print(pd.DataFrame(tests).drop(columns=['difference_vs_baseline', 'weeks_1_to_4']).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
