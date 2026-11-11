# NBA Game Simulator

Simulates an NBA game one possession at a time, 10,000 times, then replays the most typical of those games.

## How it works

### 1. Collect the data
`nba_data.py` downloads and caches play-by-play from three free sources: last season from NBA.com (via [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data)), this season game by game from ESPN, and team stats from NBAstuffer.

### 2. Rate every player and team
From the play-by-play, each player gets a shot profile (where he shoots from and how often he makes it), a usage rate, and turnover, foul and rebounding rates. Players with few games are pulled toward the average for their position. Each team gets defensive ratings adjusted for the opponents it has faced.

### 3. Simulate possessions
`nba_game_sim.py` plays each possession: who ends it, where the shot comes from, whether it goes in, plus fouls, free throws, assists, blocks and rebounds. Rotations come from recent lineups, and the end of the game handles intentional fouls and overtime.

### 4. Run the game 10,000 times
Each run plays out differently, so together they show how often each team wins, the range of final scores, and what each player usually puts up.

### 5. Show the most typical game
Out of all the runs, the simulator picks the one closest to the middle of everything: both final scores, the margin, and each team's top scorers. That game is replayed on a 3D court with a live scoreboard, win probability, box scores and every shot.

### 6. Check it against real results
`nba_backtest.py` re-simulates finished games using only data from before each game. The simulator picked the winner in 65 of 103 games, and its player point projections were off by about 4.5 points on average. Full results are in `model_artifacts/nba_backtest/REPORT.md`.

## Files

| File | What it does |
|---|---|
| `nba_data.py` | Downloads and caches the data |
| `nba_game_sim.py` | Player ratings and the possession simulator |
| `nba_app.py` | Streamlit app: pick a game, set lineups, simulate and watch the replay |
| `nba_backtest.py` | Re-simulates past games to measure accuracy |
| `nba_game_sim_replay.html` | The animated replay page |
| `output/simulations/` | Example replay (open in a browser) |
| `tests/` | Unit tests |

## Run it

From this folder:

```bash
../.venv/bin/python -m streamlit run nba_app.py
```

```bash
../.venv/bin/python -m unittest discover tests
```
