# NBA Game Simulator

Simulates an NBA game one possession at a time, thousands of times, to estimate who wins, the final score, and each player's stats.

## How it works

### 1. Collect the data
`nba_data.py` downloads and caches play-by-play from three free sources: last season from NBA.com (via [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data)), this season game by game from ESPN, and team stats from NBAstuffer.

### 2. Rate every player and team
From the play-by-play, each player gets a shot profile (where he shoots from and how often he makes it), a usage rate, and turnover, foul and rebounding rates. Players with few games are pulled toward the average for their position. Each team gets defensive ratings adjusted for the opponents it has faced.

### 3. Simulate possessions
`nba_game_sim.py` plays each possession: who ends it, where the shot comes from, whether it goes in, plus fouls, free throws, assists, blocks and rebounds. Rotations come from recent lineups, and the end of the game handles intentional fouls and overtime.

### 4. Turn simulations into projections
Running a game thousands of times gives win probability, the score range, and each player's distribution of points, rebounds and assists. The most typical game is shown as an animated replay.

### 5. Check it against real results
`nba_backtest.py` re-simulates finished games using only data from before each game. `nba_lines_backtest.py` compares the simulator's player props to real sportsbook lines.

What the backtests found:
- The simulator picked the winner in 65 of 103 games.
- Its raw prop percentages run high: picks it rated 70–80% hit about 58%.
- The sportsbook lines were more accurate than the simulator alone, so the prop sheets blend the two.

Full results are in `model_artifacts/*/REPORT.md`.

## Files

| File | What it does |
|---|---|
| `nba_data.py` | Downloads and caches the data |
| `nba_game_sim.py` | Player ratings and the possession simulator |
| `nba_app.py` | Streamlit app: pick a game, set lineups, simulate |
| `nba_backtest.py` | Re-simulates past games to measure accuracy |
| `nba_lines_backtest.py` | Compares simulated props with sportsbook lines |
| `nba_sheet_page.py` | Builds the nightly prop sheet pages |
| `output/simulations/` | Example game replay |
| `tests/` | Unit tests |

## Run it

From this folder:

```bash
../.venv/bin/python -m streamlit run nba_app.py
```

```bash
../.venv/bin/python -m unittest discover tests
```
