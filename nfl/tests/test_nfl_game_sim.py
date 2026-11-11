"""Run from the nfl folder: ../.venv/bin/python -m unittest discover -v tests"""
import unittest
import warnings
import numpy as np
import pandas as pd
import nfl_game_sim as sim

SEASON = 2026


class SimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
        cls.league = sim.League.from_cache(SEASON)
        cls.matchup = sim.Matchup(cls.league, "PHI", "NYG")
        cls.games, cls.players = cls.matchup.simulate(300, 0., seed=1)

    def test_same_seed_same_game(self):
        first, second = self.matchup.play(7, record=True), self.matchup.play(7, record=True)
        self.assertEqual(first.score, second.score)
        self.assertEqual([e["text"] for e in first.events], [e["text"] for e in second.events])

    def test_box_score_adds_up_to_the_score(self):
        for seed in range(40):
            game = self.matchup.play(seed, record=True)
            box = game.box()
            for team in (game.home, game.away):
                b = box.loc[box.team.eq(team)]
                self.assertEqual(b.pass_cmp.sum(), b.rec.sum())
                self.assertEqual(b.pass_yds.sum(), b.rec_yds.sum())
                self.assertEqual(b.pass_td.sum(), b.rec_td.sum())
                offense_tds = b.rec_td.sum() + b.rush_td.sum()
                points = game.score[team]
                kicks = 3 * b.fgm.sum() + b.xpm.sum()
                self.assertGreaterEqual(points, 6 * offense_tds + kicks)
            self.assertEqual(game.events[-1]["h"], game.score[game.home])
            self.assertEqual(game.events[-1]["a"], game.score[game.away])

    def test_league_level_realism(self):
        per_team = self.players.groupby(["sim", "team"])[sim.STATS].sum()
        plays = (per_team.pass_att + per_team.sacked + per_team.rush_car).mean()
        points = (self.games.home_score + self.games.away_score).mean() / 2
        self.assertTrue(56 < plays < 68, plays)
        self.assertTrue(17 < points < 28, points)
        self.assertLess((self.games.home_score == self.games.away_score).mean(), .05)

    def test_passing_volume_and_spread_are_realistic(self):
        per_team = self.players.groupby(["sim", "team"])[sim.STATS].sum()
        self.assertTrue(30 < per_team.pass_att.mean() < 36, per_team.pass_att.mean())
        qb = self.matchup.players["PHI"]["qb"]
        lines = self.players.loc[self.players.player_id.eq(qb)].set_index("sim").pass_yds.reindex(range(300)).fillna(0)
        self.assertGreater((lines < 150).mean(), .04)
        self.assertGreater(lines.std(), 55)

    def test_tendencies_exclude_override_situations(self):
        rates = self.league.situation_pass_rate
        self.assertGreater(rates["hurry"], .7)
        self.assertGreater(rates["trail"], .7)
        self.assertLess(rates["milk"], .4)
        situation = sim._game_situation(np.array([3000, 100, 200, 600, 1500]), np.array([0, -3, 7, -10, -10]))
        self.assertEqual(list(situation), ["normal", "hurry", "milk", "trail", "normal"])

    def test_form_levels_shift_passes_only(self):
        samplers = self.matchup.samplers(0.)["PHI"]
        self.assertAlmostEqual(sum(samplers.shifts), 0., places=9)
        low, high = samplers.level(0), samplers.level(sim.FORM_LEVELS - 1)
        epa = self.league.pool["epa"]
        for zone in range(4):
            pass_rows, run_rows = self.league._cells[0, zone], self.league._cells[1, zone]
            self.assertLess(low.weights[0, zone] @ epa[pass_rows], high.weights[0, zone] @ epa[pass_rows])
            np.testing.assert_allclose(low.weights[1, zone], high.weights[1, zone])

    def test_out_players_get_no_snaps(self):
        top_target = self.league.names[self.matchup.players["PHI"]["targets"][0][0]]
        m = sim.Matchup(self.league, "PHI", "NYG", out_players={"PHI": [top_target]})
        out_id = self.league._match_player(top_target, "PHI")
        self.assertNotIn(out_id, set(m.players["PHI"]["targets"][0]))
        _, players = m.simulate(60, 0., seed=2)
        self.assertEqual(players.loc[players.player_id.eq(out_id), ["tgt", "rush_car"]].to_numpy().sum(), 0)
        with self.assertRaises(ValueError):
            sim.Matchup(self.league, "PHI", "NYG", out_players={"PHI": ["Not A Player"]})

    def test_returning_players_get_their_usual_share(self):
        league, cur = self.league, self.league.current
        runs = cur.loc[cur.play_type.eq("run") & cur.rusher_player_id.notna()]
        totals = runs.groupby(["posteam", "rusher_player_id"]).size().sort_values(ascending=False)
        team, price = next((team, pid) for (team, pid), n in totals.items()
                           if n >= 25 and league.positions.get(pid) == "RB"
                           and pid not in set(league.team_players(team)["carries"][0]))
        after = league.team_players(team, returning=[league.names[price]])
        ids, cdf = after["carries"]
        share = dict(zip(ids, np.diff(np.r_[0., cdf])))[price]
        self.assertGreater(share, .1)
        self.assertAlmostEqual(cdf[-1], 1.)
        with self.assertRaises(ValueError):
            self.league.team_players("SEA", out=["Jadarian Price"], returning=["Jadarian Price"])

    def test_bad_run_defense_helps_running_backs(self):
        ratings = self.league.ratings
        worst, best = ratings.run_defense.idxmax(), ratings.run_defense.idxmin()
        offense = next(t for t in ["PHI", "DET", "BAL"] if t not in (worst, best))
        ypc = {}
        for defense in (worst, best):
            _, players = sim.Matchup(self.league, offense, defense).simulate(600, 0., seed=3)
            team = players.loc[players.team.eq(offense)]
            ypc[defense] = team.rush_yds.sum() / team.rush_car.sum()
        self.assertGreater(ypc[worst], ypc[best])

    def test_tilt_solver_hits_target_mean(self):
        rng = np.random.default_rng(0)
        epa, weights = rng.normal(0, 1.3, 4000).clip(-4, 4), rng.random(4000)
        lam = sim._solve_tilt(epa, weights, .15)
        w = weights * np.exp(lam * epa)
        self.assertAlmostEqual(w @ epa / w.sum(), .15, places=4)

    def test_model_weight_moves_toward_model(self):
        light = sim.simulate_game(self.league, "PHI", "NYG", .85, n=400, model_weight=.25, calibration_sims=300)
        heavy = sim.simulate_game(self.league, "PHI", "NYG", .85, n=400, model_weight=1., calibration_sims=300)
        self.assertLess(abs(heavy.home_win_rate - .85), .07)
        self.assertGreater(heavy.home_win_rate, light.home_win_rate)
        self.assertLess(light.target, heavy.target)

    def test_typical_game_is_won_by_the_favorite(self):
        result = sim.simulate_game(self.league, "PHI", "NYG", .6, n=300, calibration_sims=200)
        sim_index = result.typical_sim()
        row = result.games.loc[result.games.sim.eq(sim_index)].iloc[0]
        home_won = row.home_score > row.away_score
        self.assertEqual(home_won, result.home_win_rate >= .5)
        self.assertIn('"events"', sim.replay_html(result, sim_index))
        features, _ = result.typical_features()
        z = ((features.loc[sim_index] - features.median()) / features.std()).abs()
        self.assertLess(z.max(), 1.5, z.sort_values().tail(3))

    def mean_of(self, column, kind, zone, pid, look, team="PHI"):
        rows, cdf = self.matchup.samplers(0.)[team].get(kind, zone, pid, look)
        return np.diff(np.r_[0, cdf]) @ self.league.pool[column][rows]

    def test_ball_carrier_efficiency_follows_his_own_plays(self):
        carriers = self.matchup.players["PHI"]["carries"][0]
        team = self.mean_of("yards", 1, 3, None, (0,))
        own = {pid: self.mean_of("yards", 1, 3, pid, (0,)) for pid in carriers}
        self.assertTrue(any(abs(v - team) > .05 for v in own.values()), own)

    def test_defensive_looks_change_outcomes(self):
        light, stacked = (self.mean_of("yards", 1, 3, None, (box,)) for box in (0, 2))
        self.assertGreater(light, stacked)
        calm, blitz = (self.mean_of("sack", 0, 3, None, (2, b)) for b in (0, 1))
        self.assertGreater(blitz, calm)
        probs = self.league.look_probs("PHI", "NYG")
        for table in probs.values():
            np.testing.assert_allclose(table.sum(axis=1), 1)
        overall, by_situation = self.league.look_league["blitz"]
        self.assertGreater(by_situation[2, 1], by_situation[0, 1])

    def test_starting_qb_override_reaches_passing_and_runs(self):
        league = self.league
        regular = sim.Matchup(league, "BUF", "BAL")
        names = {pid: league.names.get(pid) for pid in league._qb_volume}
        lamar = next(pid for pid, name in names.items() if name == "Lamar Jackson")
        huntley = next(pid for pid, name in names.items() if name == "Tyler Huntley")
        for qb, benched in ((lamar, huntley), (huntley, lamar)):
            m = sim.Matchup(league, "BUF", "BAL", away_qb=league.names[qb])
            self.assertEqual(m.players["BAL"]["qb"], qb)
            carriers = list(m.players["BAL"]["carries"][0])
            self.assertIn(qb, carriers)
            self.assertNotIn(benched, carriers)
        means = []
        for qb in (lamar, huntley):
            m = sim.Matchup(league, "BUF", "BAL", away_qb=league.names[qb])
            rows, cdf = m.samplers(0.)["BAL"].get(0, 3, None, (2, 0))
            means.append(np.diff(np.r_[0, cdf]) @ league.pool["epa"][rows])
        self.assertNotAlmostEqual(means[0], means[1], places=3)
        self.assertIsNotNone(regular.players["BAL"]["qb"])

    def test_prop_probabilities_and_board(self):
        result = sim.simulate_game(self.league, "DAL", "ARI", .6, n=600, calibration_sims=200)
        samples, players = result.player_samples()
        dak = sim.find_player(players, "Dak Prescott")
        prop = sim.prop_probability(samples, players, "dak prescott", "pass_yds", 230)
        self.assertAlmostEqual(prop["over"] + prop["under"] + prop["push"], 1)
        values = samples[("pass_yds", dak)]
        self.assertAlmostEqual(prop["over"], (values > 230).mean())
        self.assertGreaterEqual(prop["over"], sim.prop_probability(samples, players, dak, "pass_yds", 300)["over"])
        board = sim.prop_board(samples, players, .55, ("over", "under"))
        self.assertTrue((board.probability >= .55).all())
        for row in board.itertuples():
            side, line = row.pick.split()
            stat = next(k for k, (label, _) in sim.PROP_STATS.items() if label == row.prop)
            v = samples[(stat, sim.find_player(players, row.player))]
            hit = (v > float(line)).mean() if side == "over" else (v < float(line)).mean()
            self.assertAlmostEqual(hit, row.probability)
        with self.assertRaises(ValueError):
            sim.find_player(players, "Not A Player")

    def test_looks_do_not_move_team_strength(self):
        league, m = self.league, self.matchup
        probs = league.look_probs("PHI", "NYG")
        for kind, label in [(0, "pass"), (1, "run")]:
            target = (league.league_epa[kind] + league.ratings.at["PHI", f"{label}_offense"]
                      + league.ratings.at["NYG", f"{label}_defense"] + (m.qb_shift["PHI"] if kind == 0 else 0))
            means, sizes = [], []
            for zone in range(4):
                cell, rows = (kind, zone), league._cells[kind, zone]
                base = m._base["PHI"][cell]
                w = base * league._look_average(cell, base, probs) * np.exp(m._tilts["PHI"][cell] * league.pool["epa"][rows])
                means.append((w / w.sum()) @ league.pool["epa"][rows])
                sizes.append(len(rows))
            self.assertAlmostEqual(np.average(means, weights=sizes), target, places=3)


if __name__ == "__main__":
    unittest.main()
