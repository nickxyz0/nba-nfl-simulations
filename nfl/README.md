# NFL Game Predictor and Simulator

Predicts the winner of each NFL game using only team performance data (no betting lines), then simulates games snap by snap to project scores and player stats.

## How it works

### 1. Collect the data
Game scores for 2023–2025 (in `data/`) and weekly team stats and play-by-play from [nflverse](https://github.com/nflverse), downloaded and cached by `nfl_model_pipeline.py`.

### 2. Build pregame features
For every game, each team's averages from **only the games before it**: passing and rushing EPA, completion percentage over expected, turnovers, sacks, points scored and allowed, plus a cold-weather flag. Using only earlier games keeps the model from seeing the result it is trying to predict.

### 3. Train and tune the model
An XGBoost classifier, trained on 2023 and early 2024 and validated on late 2024, always in time order so it never trains on future games. I compared it against simple baselines: always picking the home team, Elo ratings, and logistic regression.

### 4. Test on a season the model never saw
Results on all 271 games of the 2025 season:

| Model | Accuracy | Log loss |
|---|---:|---:|
| Always pick the home team | 53.9% | 0.690 |
| First XGBoost model | 59.0% | 0.708 |
| Tuned XGBoost | 62.4% | 0.647 |
| Logistic regression | 63.8% | 0.636 |
| Elo ratings | 63.8% | 0.644 |

### 5. Improve the model
Each idea was tested in its own script in `experiments/`, with a written report in `model_artifacts/`:
- adding each starting quarterback's recent form (`qb_blend_experiment`)
- adjusting stats for opponent strength (`opponent_adjust_experiment`)
- blending XGBoost, logistic regression and Elo (`week8_experiment`)

The blend was the best: 66.8% on 2025, and it is the model used for weekly picks now.

### 6. Make weekly predictions
`nfl_weekly_config.json` holds the current week and any quarterback changes. The last cells of `NFL Model.ipynb` and the Streamlit app (`nfl_app.py`) both read it.

### 7. Simulate games
`nfl_game_sim.py` plays a game one snap at a time, thousands of times. Each snap reuses a real play from nflverse, adjusted for how good that offense and defense are. The most typical simulated game is shown as an animated replay, and the full set of games gives projected box scores.

## 2026 season results

I posted a pick for every game each week before kickoff.

| Week | Model | Record | Accuracy |
|---|---|---:|---:|
| 2 | XGBoost | 8-8 | 50.0% |
| 3 | XGBoost | 10-6 | 62.5% |
| 4 | XGBoost | 11-5 | 68.8% |
| 5 | XGBoost | 9-5-1 | 64.3% |
| 6 | XGBoost + QB form | 8-5 | 61.5% |
| 7 | XGBoost + QB form | 8-6 | 57.1% |
| 8 | Blend | 11-3 | 78.6% |
| 9 | Blend | 14-1 | 93.3% |
| **Total** | | **79-39-1** | **66.9%** |

How it got better:
- **More data each week.** The model isn't retrained during the season, but most inputs are a team's averages from its games so far. In Week 2 that is one game per team; by Week 9 it is seven or eight, so the averages are much more reliable.
- **Quarterback form (Week 6).** Added each starting QB's recent efficiency, so injuries and backup QBs change the pick.
- **Blended model (Week 8).** Averaged XGBoost with logistic regression and Elo ratings. On past seasons the blend was more accurate and better calibrated than XGBoost alone. Since the switch it is 25-4 (86%).

Weeks 8 and 9 are only 29 games, so the full-season 66.9% is the fairer number. It matches the 66.8% the blend scored on the 2025 season in testing.

## Files

| File | What it does |
|---|---|
| `NFL Model.ipynb` | The full process: cleaning, features, training, testing and weekly picks |
| `NFL Game Simulator.ipynb` | Runs the simulator for one game and shows the replay |
| `nfl_model_pipeline.py` | Data loading, features, training and prediction |
| `nfl_game_sim.py` | The game simulator |
| `nfl_app.py` | Streamlit app for weekly picks and simulations |
| `experiments/` | Scripts that tested model improvements |
| `model_artifacts/` | Saved models and experiment reports |
| `tests/` | Unit tests |

## Run it

From this folder:

```bash
../.venv/bin/python -m streamlit run nfl_app.py
```

```bash
../.venv/bin/python -m unittest discover tests
```
