# Sports ML Models and Game Simulators

Two sports analytics projects built in Python:

- **[NFL](nfl/)**: a machine learning model that predicts the winner of every NFL game, plus a snap-by-snap game simulator.
- **[NBA](nba/)**: a possession-by-possession game simulator that projects scores, player stats and props.

Each project has its own README that walks through how it works, step by step.

## Folder layout

```
nfl/                 NFL win-probability model and game simulator
  NFL Model.ipynb      the full modeling notebook, from raw scores to weekly picks
  NFL Game Simulator.ipynb
  nfl_model_pipeline.py, nfl_game_sim.py, nfl_app.py
  experiments/         scripts used to test model ideas (results in model_artifacts/)
  tests/
nba/                 NBA game simulator
  nba_data.py, nba_game_sim.py, nba_app.py, nba_backtest.py
  tests/
archive/             old notebook backups and coursework (not part of the projects)
```

## Tools

Python, pandas, NumPy, scikit-learn, XGBoost, Streamlit, and public data from nflverse, ESPN and NBA.com.

## Running it

```bash
python3.9 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then follow the README inside `nfl/` or `nba/`. Commands are run from inside that project's folder.
