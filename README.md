# NFL and NBA ML Models and Game Simulators

Two sports analytics projects built in Python:

- **[NFL](nfl/)**: XGBoost machine learning model that predicts the winner of every NFL game, plus a play-by-play game simulator, running 20,000 simulations of a game to find the most typical game based on the simulation closest to the mean of the 20,0000possibilities.
- **[NBA](nba/)**: a play-by-play game simulator that simulates daily games 10,000 times and displays the most typical game closest to the mean of the 10,000 simulations. 

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
