"""Run from the nba folder: ../.venv/bin/python -m unittest discover -v tests"""

import unittest
import warnings

import numpy as np
import pandas as pd

import nba_game_sim as sim


class SimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        events, games = sim.load_events(2025, 2026)
        cls.events = sim.label_free_throws(events)
        cls.league = sim.League(cls.events, games, 2026, as_of=pd.Timestamp("2026-11-01"))
        rows = games.loc[games.season.eq(2026) & games.date.eq(pd.Timestamp("2026-11-01"))]
        rows = rows.loc[rows.game_id.eq(rows.game_id.iloc[0])]
        home, away = rows.loc[rows.home], rows.loc[~rows.home]
        cls.setup = (home.team.iloc[0], away.team.iloc[0], home.player.tolist(), away.player.tolist(),
                     home.player[home.started].tolist(), away.player[away.started].tolist())
        cls.matchup = sim.Matchup(cls.league, *cls.setup)

    def test_same_seed_same_game(self):
        a, b = self.matchup.play([1, 2]), self.matchup.play([1, 2])
        self.assertEqual(a.score, b.score)
        self.assertEqual(a.stats, b.stats)

    def test_box_score_adds_up_to_the_score(self):
        for k in range(20):
            game = self.matchup.play([3, k])
            for side in (0, 1):
                points = sum(row[sim.ST["PTS"]] for row in game.stats[side])
                self.assertEqual(points, game.score[side])
                made = sum(2 * (r[sim.ST["FGM"]] - r[sim.ST["3PM"]]) + 3 * r[sim.ST["3PM"]] + r[sim.ST["FTM"]]
                           for r in game.stats[side])
                self.assertEqual(made, game.score[side])

    def test_five_players_all_game(self):
        for k in range(20):
            game = self.matchup.play([4, k])
            length = sim.period_end(game.period)
            for side in (0, 1):
                seconds = sum(row[0] for row in game.stats[side])
                self.assertAlmostEqual(seconds, 5 * length, delta=1e-6)

    def test_no_ties(self):
        games, _ = self.matchup.simulate(60, seed=5)
        self.assertTrue((games.home_score != games.away_score).all())
        self.assertTrue(((games.periods > 4) == (games.periods >= 5)).all())

    def test_defender_step_only_moves_make_rates(self):
        """Off and on share random numbers, and defender ratings average out per team."""
        off = sim.Matchup(self.league, *self.setup, use_defender=False)
        on = sim.Matchup(self.league, *self.setup, use_defender=True)
        g_off, _ = off.simulate(400, seed=6)
        g_on, _ = on.simulate(400, seed=6)
        self.assertLess(abs((g_on.home_score + g_on.away_score).mean()
                            - (g_off.home_score + g_off.away_score).mean()), 3.)
        for team in on.teams:
            minutes = np.array([sum(o) for o in team.occ])
            centered = np.dot(minutes, np.array(team.dlog) - team.def_avg) / minutes.sum()
            self.assertAlmostEqual(centered, 0., places=9)

    def test_calibration_hits_last_season(self):
        c = self.league.calibration
        self.assertLess(abs(c["sim_pace"] - c["target_pace"]), 1.5)
        self.assertLess(abs(c["sim_points"] - c["target_points"]), 1.5)

    def test_typical_game_matches_the_favorite(self):
        result = sim.simulate_game(self.league, *self.setup, n=100)
        log, box, game = result.replay()
        favorite_home = result.home_win_rate >= .5
        self.assertEqual(game.score[0] > game.score[1], favorite_home)
        self.assertEqual(box.loc[box.team.eq(self.setup[0])].PTS.sum(), game.score[0])
        self.assertEqual(log.iloc[-1].text, "Final")


if __name__ == "__main__":
    unittest.main()
