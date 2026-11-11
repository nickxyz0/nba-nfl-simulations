# NFL and NBA ML Models and Game Simulators

Two sports analytics projects built in Python:

- **[NFL](nfl/)**: XGBoost machine learning model that predicts the winner of every NFL game, plus a play-by-play game simulator, running 20,000 simulations of a game to find the most typical game based on the simulation closest to the mean of the 20,000 possibilities.
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

## Website (both simulators in one place)

`app.py` combines the two Streamlit apps into one site with a Home page, an NFL tab and an NBA tab. Run it locally from the repository root:

```bash
.venv/bin/streamlit run app.py
```

The same site is hosted on Hugging Face Spaces so friends can use it without running anything. The `Dockerfile` builds it, and `.github/workflows/deploy-space.yml` copies the repository to the Space on every push to `main`, so a weekly edit to `nfl/nfl_weekly_config.json` goes live a few minutes after it is pushed. The hosted copy sets `SIM_HOSTED=1`, which runs 5,000 NFL simulations instead of 20,000 (about ±0.7 points of random noise on a win percentage, against ±0.35 at 20,000) and hides the "Save this prediction" button, since saved picks belong on the laptop.

One-time setup:

1. On huggingface.co, create a free account and a **write** access token (Settings → Access Tokens).
2. In this GitHub repository, open Settings → Secrets and variables → Actions. Add a secret `HF_TOKEN` with that token, and a variable `HF_SPACE` set to `<your-hf-username>/nba-nfl-simulations`.
3. Run the "Deploy to Hugging Face Space" workflow from the Actions tab (or push to `main`). It creates the Space if needed. The first build takes about 5 minutes, and the first visit downloads the season's data, which takes a few more.
