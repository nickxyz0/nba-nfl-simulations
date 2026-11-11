"""Run from the nfl folder: ../.venv/bin/python -m unittest discover -v tests"""
import unittest
from unittest.mock import patch
from pathlib import Path
import os
import tempfile
import numpy as np
import pandas as pd
import nfl_model_pipeline as nfl


def sample_data():
    schedule, raw = [], []
    for season in [2023, 2024]:
        for week, date in [(1, '09-08'), (2, '09-15'), (4, '09-29')]:
            game_id = f'{season}_{week:02d}_NYJ_BUF'
            schedule.append(dict(game_id=game_id, season=season, week=week,
                                 gameday=f'{season}-{date}', gametime='13:00',
                                 home_team='BUF', away_team='NYJ',
                                 home_score=20 + week, away_score=10 + week,
                                 home_rest=14 if week == 4 else 7, away_rest=7,
                                 game_type='REG', location='Home'))
            for team, opponent, factor in [('BUF', 'NYJ', 1.), ('NYJ', 'BUF', 2.)]:
                row = {c: 0. for c in nfl.RAW_REQUIRED}
                row.update(game_id=game_id, season=season, week=week, season_type='REG',
                           team=team, opponent_team=opponent, attempts=20. * week,
                           sacks_suffered=2., carries=20., passing_epa=3. * week * factor,
                           rushing_epa=2. * factor, passing_cpoe=1. * factor,
                           passing_yards=100., sack_yards_lost=10., rushing_yards=70.)
                raw.append(row)
    return nfl.prepare_schedule(pd.DataFrame(schedule), 2023, 2024), pd.DataFrame(raw)


class PregameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.games, cls.raw = sample_data()
        cls.features, cls.teams = nfl.build_pregame_features(cls.games, cls.raw)

    def test_season_reset_and_byes(self):
        first = self.teams.week.eq(1)
        self.assertTrue(self.teams.loc[first, 'prior_games'].eq(0).all())
        self.assertTrue(self.teams.loc[first, 'avg_points_for'].isna().all())
        self.assertTrue(self.teams.loc[self.teams.week.eq(4), 'prior_games'].eq(2).all())

    def test_weighted_rate_and_defense(self):
        row = self.features.loc[self.features.season.eq(2023) & self.features.week.eq(4)].iloc[0]
        self.assertAlmostEqual(row.home_avg_pass_epa_rate, (3 + 6) / (22 + 42))
        self.assertAlmostEqual(row.home_avg_def_pass_epa_allowed, (6 + 12) / (22 + 42))
        self.assertAlmostEqual(row.home_avg_points_for, 21.5)

    def test_current_result_cannot_leak(self):
        altered_games, altered_raw = self.games.copy(), self.raw.copy()
        target = '2023_02_NYJ_BUF'
        altered_games.loc[altered_games.game_id.eq(target), 'home_score'] = 99
        altered_raw.loc[altered_raw.game_id.eq(target), 'passing_epa'] = 999
        altered, _ = nfl.build_pregame_features(altered_games, altered_raw)
        columns = list(dict.fromkeys(sum(nfl.feature_sets().values(), [])))
        before = self.features.loc[self.features.game_id.eq(target), columns]
        after = altered.loc[altered.game_id.eq(target), columns]
        np.testing.assert_allclose(before, after, equal_nan=True)

    def test_future_data_cannot_change_past(self):
        old = self.games.season.eq(2023)
        prefix, _ = nfl.build_pregame_features(self.games.loc[old], self.raw.loc[self.raw.season.eq(2023)])
        columns = list(dict.fromkeys(sum(nfl.feature_sets().values(), [])))
        np.testing.assert_allclose(prefix[columns], self.features.loc[old, columns], equal_nan=True)

    def test_unplayed_games_do_not_update_history(self):
        games = self.games.copy()
        mask = games.season.eq(2024) & games.week.gt(1)
        games.loc[mask, ['home_score', 'away_score']] = np.nan
        games.loc[mask, 'completed'] = False
        features, _ = nfl.build_pregame_features(games, self.raw)
        upcoming = features.loc[mask]
        self.assertTrue(upcoming.home_prior_games.eq(1).all())
        np.testing.assert_allclose(upcoming.home_recent4_points_for, [21., 21.])
        np.testing.assert_allclose(upcoming.home_recent8_pass_epa_rate, [3/22, 3/22])
        self.assertTrue(upcoming.home_win.isna().all())

    def test_explicit_feature_allowlists(self):
        prohibited = {'home_score', 'away_score', 'home_win', 'result', 'spread_line',
                      'home_qb_id', 'away_qb_id', 'temp', 'wind', 'game_id', 'season'}
        for columns in nfl.feature_sets().values():
            self.assertEqual(len(columns), len(set(columns)))
            self.assertFalse(prohibited.intersection(columns))
            self.assertTrue(set(columns).issubset(self.features))

    def test_missing_completed_data_fails(self):
        with self.assertRaises(AssertionError):
            nfl.build_pregame_features(self.games, self.raw.iloc[1:])

    def test_tie_has_no_binary_target_but_counts_in_history(self):
        games = self.games.copy()
        mask = games.game_id.eq('2023_01_NYJ_BUF')
        games.loc[mask, 'home_score'] = games.loc[mask, 'away_score']
        features, _ = nfl.build_pregame_features(games, self.raw)
        self.assertTrue(features.loc[mask, 'home_win'].isna().all())
        self.assertEqual(features.loc[features.game_id.eq('2023_02_NYJ_BUF'), 'home_prior_games'].iloc[0], 1)

    def test_qb_change_uses_prior_games_only(self):
        schedule = self.games.copy()
        schedule['home_qb_id'], schedule['away_qb_id'] = 'starter', 'jets'
        backup = schedule.game_id.eq('2024_04_NYJ_BUF')
        schedule.loc[backup, 'home_qb_id'] = 'backup'
        stats = pd.DataFrame([
            dict(player_id=qb, position='QB', game_id=game_id, passing_epa=epa, rushing_epa=0.,
                 attempts=30., sacks_suffered=2., carries=3.)
            for game_id in schedule.game_id
            for qb, epa in [('starter', 10.), ('jets', 0.)]
        ] + [dict(player_id='backup', position='QB', game_id='2024_04_NYJ_BUF', passing_epa=-50.,
                  rushing_epa=0., attempts=30., sacks_suffered=5., carries=0.)])
        features = nfl.add_qb_features(schedule, schedule, nfl.qb_history(schedule, stats))
        row = features.loc[backup].iloc[0]
        self.assertEqual(row.home_qb_change, 1.)
        self.assertLess(row.home_qb_delta, 0)
        self.assertEqual(row.home_qb_rating, nfl.QB_PRIOR_EPA)
        self.assertEqual(features.loc[~backup, 'home_qb_change'].sum(), 0)

    def test_correct_qb_ids_uses_dropbacks_and_overrides(self):
        schedule = pd.DataFrame([
            dict(game_id='g1', season=2026, week=5, home_team='SEA', away_team='SF',
                 home_score=20., away_score=17., home_qb_id='lock', away_qb_id='purdy',
                 home_qb_name='Drew Lock', away_qb_name='Brock Purdy'),
            dict(game_id='g2', season=2026, week=6, home_team='DEN', away_team='SEA',
                 home_score=np.nan, away_score=np.nan, home_qb_id='nix', away_qb_id='lock',
                 home_qb_name='Bo Nix', away_qb_name='Drew Lock'),
        ])
        stats = pd.DataFrame([
            dict(player_id='darnold', player_display_name='Sam Darnold', position='QB',
                 game_id='g1', team='SEA', attempts=29., sacks_suffered=2.),
            dict(player_id='lock', player_display_name='Drew Lock', position='QB',
                 game_id='g1', team='SEA', attempts=1., sacks_suffered=0.),
        ])
        fixed = nfl.correct_qb_ids(schedule, stats, {'SEA': 'Sam Darnold'}, 2026, 6)
        self.assertEqual(fixed.loc[0, 'home_qb_id'], 'darnold')
        self.assertEqual(fixed.loc[0, 'away_qb_id'], 'purdy')
        self.assertEqual(fixed.loc[1, 'away_qb_id'], 'darnold')
        self.assertEqual(fixed.loc[1, 'home_qb_id'], 'nix')
        with self.assertRaises(ValueError):
            nfl.correct_qb_ids(schedule, stats, {'SEA': 'Nobody'}, 2026, 6)

    def test_candidates_reproducible(self):
        a = nfl.make_candidates(nfl.feature_sets())
        b = nfl.make_candidates(nfl.feature_sets())
        self.assertEqual(a, b)
        self.assertEqual(len(a), 80)

    def test_model_save_reload_preserves_predictions(self):
        import joblib
        data = self.features.copy()
        data['home_win'] = np.arange(len(data)) % 2
        columns = ['elo_gap', 'home_avg_points_for']
        spec = {'candidate': 'test', 'feature_set': 'test',
                'params': {**nfl.BASE_PARAMS, 'n_estimators': 5, 'min_child_weight': 1},
                'history_seasons': 1, 'half_life': None}
        models = nfl.fit_xgb(data, columns, spec, seeds=(1, 7, 21))
        before = nfl.predict_models(models, data, columns)
        with tempfile.TemporaryDirectory(prefix='nfl-bundle-test-') as temporary:
            saved = nfl.save_model_bundle(models, columns, spec, data, temporary)
            bundle = joblib.load(saved)
            after = nfl.predict_models(bundle['models'], data, bundle['feature_columns'])
            np.testing.assert_array_equal(before, after)
            self.assertEqual(bundle['training_seasons'], [2024])
            self.assertEqual(bundle['training_games'], 3)

    def test_week_folds_are_chronological_and_disjoint(self):
        rows = []
        for season in range(2020, 2025):
            for week in range(1, 19):
                for game in range(16):
                    rows.append({'season': season, 'week': week,
                                 'gameday': pd.Timestamp(season, 8, 1) + pd.Timedelta(weeks=week)})
        data = pd.DataFrame(rows)
        folds = nfl.walk_forward_folds(data)
        self.assertEqual(len(folds), 6)
        valid_indices = []
        for _, train, valid in folds:
            self.assertLess(data.loc[train, 'gameday'].max(), data.loc[valid, 'gameday'].min())
            valid_indices.extend(valid)
        self.assertEqual(len(valid_indices), len(set(valid_indices)))

    def test_blend_roundtrip_and_component_weights(self):
        import joblib
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        data = self.features.copy()
        data['home_win'] = np.arange(len(data)) % 2
        data['gap_qb_delta'] = np.linspace(-.2, .2, len(data))
        columns = ['elo_gap', 'home_avg_points_for', 'gap_qb_delta']
        spec = {'candidate': 'test_blend', 'model_family': 'ensemble',
                'params': {**nfl.BASE_PARAMS, 'n_estimators': 5, 'min_child_weight': 1}}
        xgb = nfl.fit_xgb(data, columns, spec, seeds=(1, 7, 21))
        lr = make_pipeline(SimpleImputer(), LogisticRegression()).fit(data[columns], data.home_win)
        model = nfl.BlendedPregameModel(xgb, lr, columns, weights=(.5, .25, .25))
        expected = (.5 * nfl.predict_models(xgb, data, columns)
                    + .25 * lr.predict_proba(data[columns])[:, 1]
                    + .25 * nfl.elo_qb_probability(data))
        np.testing.assert_allclose(model.predict_proba(data)[:, 1], expected)
        with tempfile.TemporaryDirectory(prefix='nfl-blend-test-') as temporary:
            saved = nfl.save_model_bundle([model], columns, spec, data, temporary)
            self.assertTrue(saved.with_suffix('.json').exists())
            loaded = joblib.load(saved)
            np.testing.assert_allclose(nfl.predict_models(loaded['models'], data, columns), expected)
            altered = data.copy()
            altered['gap_qb_delta'] -= .5
            before = loaded['models'][0].component_probabilities(data)[:, 2]
            after = loaded['models'][0].component_probabilities(altered)[:, 2]
            self.assertTrue((after < before).all())
        with self.assertRaises(ValueError):
            nfl.BlendedPregameModel(xgb, lr, columns, weights=(1., 1., -1.))

    def test_future_logging_refuses_overwrite(self):
        class ConstantModel:
            def predict_proba(self, frame):
                return np.tile([.4, .6], (len(frame), 1))

        schedule, raw = sample_data()
        offset = pd.Timestamp.now().year + 1 - 2024
        schedule['season'] += offset
        schedule['gameday'] += pd.DateOffset(years=offset)
        raw['season'] += offset
        target_season = 2024 + offset
        future = schedule.season.eq(target_season) & schedule.week.eq(4)
        schedule.loc[future, ['home_score', 'away_score']] = np.nan
        bundle = {'models': [ConstantModel()], 'feature_columns': ['elo_gap'],
                  'last_training_date': '2024-01-01'}
        previous_directory = os.getcwd()
        try:
            with tempfile.TemporaryDirectory(prefix='nfl-log-test-') as temporary:
                os.chdir(temporary)
                with patch('joblib.load', return_value=bundle), \
                     patch.object(nfl, 'load_schedule', return_value=schedule), \
                     patch.object(nfl, 'load_team_stats', side_effect=lambda seasons, **kwargs: raw.loc[raw.season.isin(seasons)]):
                    result = nfl.predict_upcoming_week('test.joblib', target_season, 4, log=True)
                    self.assertEqual(len(result), 1)
                    log = next(Path('prediction_logs').glob('*.json'))
                    before = log.read_bytes()
                    with self.assertRaises(FileExistsError):
                        nfl.predict_upcoming_week('test.joblib', target_season, 4, log=True)
                    self.assertEqual(log.read_bytes(), before)
        finally:
            os.chdir(previous_directory)


if __name__ == '__main__':
    unittest.main()
