# NBA simulator backtest: defender step on vs off

## Player scoring check (2026-11-03)

Question: were the sim's star scores too low and too even? Grouping players by last
season's PPG (known before the games; grouping by points actually scored in these
games exaggerates it, since hot streaks land in the top group), on the same 103 games:

| Last season's PPG | Real | Old sim | All fixes | Final (usage + free throws) |
|---|---|---|---|---|
| under 8 | 4.7 | 5.3 | 5.1 | 5.3 |
| 8–14 | 8.8 | 9.5 | 9.8 | 9.4 |
| 14–20 | 14.1 | 14.6 | 14.7 | 14.5 |
| 20–25 | 20.8 | 19.4 | 19.5 | 19.3 |
| 25+ | 24.9 | 22.3 | 23.6 | 22.5 |

| Version | Winners | Brier | Player pts error | Minutes error |
|---|---|---|---|---|
| Old | 68/103 | 0.2142 | 4.45 | 4.45 |
| All fixes (usage, free throws, minutes calibration) | 66/103 | 0.2175 | 4.54 | 4.80 |
| Final: usage share + foul-drawing free throws, original minutes | 65/103 | 0.2174 | 4.45 | 4.44 |

Changes kept: usage is each player's share of team plays while on the floor (not a
per-minute rate), foul drawing is measured against his own shot mix, and bonus free
throws go to players who draw fouls. Minutes calibration was dropped: it made minutes
and role players' points worse. Winner differences (65–68 of 103) and the Brier change
(+0.003, interval −0.003 to +0.010) are within noise. Stars still run about 2 points
under in the backtest. In the app, Luka's low minutes came from the roster (everyone
from the last 10 games); the app now defaults to each team's 10-man rotation.

Final version over all 107 games: 66/107 winners (61.7%), Brier 0.2195, margin error
11.9, each team's score 10.4, player points 4.44, minutes 4.43 (`results.json`,
`variant_usage_only/`).


**Current run (2026-11-02, simulator with measured late-game fouling):** 103 finished
2026-27 games, Oct 20 – Nov 2. Walk-forward: each date's ratings use last season plus
this season's games before that date only. Each game was simulated 400 times with the
players who actually played and the actual starters, once with the defender step on
and once off, on the same random numbers. Defender ratings come from 2025-26 NBA.com
matchups plus on-floor ratings from this season's ESPN lineups for players without
matchups (mostly rookies). No 2026-27 matchup data was used.

| Metric | Defender off (default) | Defender on |
|---|---|---|
| Winner picked correctly | 68 / 103 (66.0%) | 65 / 103 (63.1%) |
| Brier score (lower is better) | 0.2142 | 0.2191 |
| Log loss | 0.6165 | 0.6270 |
| Margin error (avg points) | 11.91 | 11.99 |
| Each team's score error | 10.48 | 10.53 |
| Total points error | 16.59 | 16.55 |
| Player points error | 4.45 | 4.46 |
| Player rebounds / assists / 3PM error | 1.85 / 1.34 / 0.90 | 1.85 / 1.34 / 0.90 |
| Shot-level log loss (real shots, real defenders on the floor) | 0.6551 | 0.6551 |

On minus off, 95% intervals from resampling games (negative would mean the defender step helps):

- Brier: +0.0050 (−0.0005 to +0.0103)
- Log loss: +0.0106 (−0.0012 to +0.0224)
- Margin error: +0.08 points (−0.07 to +0.24)
- Player points error: +0.010 (−0.001 to +0.020)
- Shot log loss: 0.0000 (−0.0001 to +0.0002)

**Decision (rule set beforehand):** the defender step does not help; if anything it is
slightly worse. Saving NBA.com matchup pages game by game is not worth it. The step
stays off by default (`use_defender=False`).

Calibration (defender off): games the sim gave the favorite 55% → favorite won 55%;
65% → 68%; 75% → 79%; 84% → 86% (7 games).

Other numbers (defender off): no exact final scores (each team's score varies about
10 points game to game); 26% of team scores within 5 points. Player points exact 7.0%,
within 3 points 48.3%; minutes error 4.5.

The first run (92 games, before the late-game fouling fix) reached the same
conclusion: 60/92 winners both ways, no measurable defender effect. Its numbers are in
`results_v1.json`.

Caveats: 103 games is small for game-level metrics; players and starters were the real
ones (known at tip-off); league-wide shot rates come from 2025-26, so nothing leaks from
later games.

Files: `games.csv`, `players.csv`, `shots.csv`, `results.json`. Rerun:
`../.venv/bin/python nba_backtest.py --sims 400` (from the nba folder)
