"""Fit the validation-selected candidate on pre-2026 games and preview Week 8."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import nfl_model_pipeline as nfl
from experiments.experiment_week8 import OUT, fit_components, load_features, predict


def main():
    report = json.loads((OUT / 'results.json').read_text())
    name, specs = report['selected'], report['definitions']
    definition = specs[name]
    if definition['kind'] != 'blend':
        raise ValueError('No validated blend selected; keep the existing deployment')
    xgb_name, lr_name, elo_name = definition['components']
    assert specs[xgb_name]['kind'] == 'xgb' and specs[lr_name]['kind'] == 'lr'
    assert specs[elo_name]['kind'] == 'elo'
    columns = specs[xgb_name]['columns']
    assert columns == specs[lr_name]['columns']
    data = load_features()
    train = data.loc[(data.season < 2026) & data.home_win.notna()].copy()
    fitted = fit_components(train, specs)
    model = nfl.BlendedPregameModel(fitted[xgb_name], fitted[lr_name], columns, definition['weights'])
    np.testing.assert_allclose(model.predict_proba(train)[:, 1], predict(train, specs, fitted)[name])
    spec = dict(candidate=name, model_family='ensemble', feature_set='efficiency_turnovers_qb',
                history_seasons=None, half_life=None, components={c: specs[c] for c in definition['components']},
                weights=definition['weights'], selection_seasons=report['selection_seasons'])
    bundle = nfl.save_model_bundle([model], columns, spec, train,
                                   feature_builder_version='season_local_rates_v1+qb_v2+blend_v1')
    config = dict(season=2026, week=8, model_path=str(bundle), qb_overrides={})
    Path('nfl_weekly_config.json').write_text(json.dumps(config, indent=2) + '\n')
    preview, inputs = nfl.predict_upcoming_week(bundle, 2026, 8, refresh=False, log=False,
                                               return_features=True, qb_overrides={})
    old = nfl.predict_upcoming_week('model_artifacts/nfl_xgb_c0f0f2b5583c.joblib',
                                   2026, 8, refresh=False, log=False)
    preview = preview.merge(old[['game_id', 'home_win_probability']].rename(
        columns={'home_win_probability': 'previous_home_probability'}), on='game_id', validate='one_to_one')
    preview['winner_probability'] = np.maximum(preview.home_win_probability, 1-preview.home_win_probability)
    preview['previous_pick'] = np.where(preview.previous_home_probability >= .5, preview.home_team, preview.away_team)
    preview['pick_changed'] = preview.predicted_winner.ne(preview.previous_pick)
    preview.to_csv(OUT / 'week8_preview.csv', index=False)
    validation = pd.DataFrame(report['validation']).set_index('name')
    tests = pd.DataFrame(report['tests'])
    lines = ['# Week 8 model comparison', '',
             'Selected on 2022–2024 chronological validation: '+name+'.', '',
             'The selected candidate must match or exceed existing validation accuracy and improve log loss.', '',
             '| Period | Model | Correct / games | Accuracy | Log loss | Brier |',
             '|---|---|---:|---:|---:|---:|']
    for period, frame in [('2022–2024 validation', validation.reset_index()),
                          ('2025 retrospective benchmark', tests.loc[tests.season.eq(2025)]),
                          ('2026 through Week 7 (retrospective)', tests.loc[tests.season.eq(2026)])]:
        for _, row in frame.loc[frame['name'].isin(['xgb_base', name])].iterrows():
            lines.append(f'| {period} | {row["name"]} | {row.correct} / {row.games} | '
                         f'{row.accuracy:.2%} | {row.log_loss:.4f} | {row.brier:.4f} |')
    lines += ['', '## Limits', '',
              'These are modest retrospective improvements, not a guarantee of higher Week 8 accuracy. '
              '2025 has been inspected in prior experiments. 2026 results were not used to select the candidate '
              'and are not frozen live predictions. Training remains fixed through 2025; current-season '
              'pregame features update from completed prior weeks. Historical QB identity uses the player '
              'with most dropbacks, which can differ from the announced starter after an in-game injury.', '',
              'Component disagreement means the three components do not all pick the same winner. '
              'It is a review flag, not a calibrated confidence interval. Starting quarterbacks come from '
              'the current schedule and must be reviewed before kickoff. Weekly overrides live in '
              '`nfl_weekly_config.json`; old Week 7 overrides were cleared.', '',
              '## Reproduce', '', '`.venv/bin/python experiment_week8.py`', '',
              '`.venv/bin/python prepare_week8.py`', '',
              'Preview: `week8_preview.csv`. No picks were logged. Previous model artifacts remain available.', '']
    predictions = pd.read_csv(OUT / 'predictions.csv')
    intervals = {}
    for season in [2025, 2026]:
        baseline = predictions.loc[predictions.season.eq(season) & predictions.model.eq('xgb_base')].set_index('game_id')
        selected = predictions.loc[predictions.season.eq(season) & predictions.model.eq(name)].set_index('game_id').loc[baseline.index]
        intervals[season] = nfl.block_bootstrap_difference(baseline.home_win, selected.probability,
                                                          baseline.probability, baseline.week)
    lines += ['## Accuracy uncertainty', '',
              'Paired week-bootstrap 95% intervals for accuracy difference (blend minus previous model). '
              'Both intervals include zero; a repeatable accuracy gain is not established.', '']
    for season, interval in intervals.items():
        lines.append(f'- {season}: {100*interval["accuracy_difference"]:.2f} percentage points; '
                     f'interval {100*interval["lower_95"]:.2f} to {100*interval["upper_95"]:.2f}.')
    lines.append('')
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    print('Saved', bundle)
    print(preview[['away_team', 'away_qb_name', 'home_team', 'home_qb_name', 'predicted_winner',
                   'winner_probability', 'previous_pick', 'pick_changed', 'component_disagreement']].to_string(index=False))


if __name__ == '__main__':
    main()
