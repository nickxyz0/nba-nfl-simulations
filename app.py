"""Combined NBA and NFL simulator site: a home page plus one tab per sport.

Run locally with `streamlit run app.py` from the repository root. The hosted copy (Hugging Face Spaces) runs the
same file through the Dockerfile, with SIM_HOSTED=1 so the pages use lighter defaults.
"""
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
# Each sport's page imports its own modules (nba_data, nfl_model_pipeline, ...) by bare name.
for folder in ("nba", "nfl"):
    path = str(ROOT / folder)
    if path not in sys.path:
        sys.path.insert(0, path)

st.set_page_config(page_title="NBA & NFL Simulators", page_icon="🏟️", layout="wide")


def home():
    st.title("NBA & NFL Simulators")
    st.write("Pick a sport in the tabs at the top. Each one plays a real upcoming matchup thousands of times from "
             "real play-by-play, then replays the most typical game with a live scoreboard and box scores.")
    nba, nfl = st.columns(2)
    with nba:
        with st.container(border=True):
            st.subheader("🏀 NBA")
            st.write("Choose any game in the next few days, set who is available and who starts, and simulate it "
                     "possession by possession. Shows win chances, fair spread and total, and player props.")
            st.page_link(NBA, label="Open the NBA simulator", icon="🏀")
    with nfl:
        with st.container(border=True):
            st.subheader("🏈 NFL")
            st.write("Choose a game this week for the saved model's pick, then simulate it snap by snap. Shows win "
                     "chances, the median score, player projections and props.")
            st.page_link(NFL, label="Open the NFL simulator", icon="🏈")
    st.caption("Simulations take from a few seconds (NBA) to a couple of minutes (NFL), and several people may be "
               "using the site at once, so be patient after clicking Simulate. The first visit after the site has "
               "been asleep also downloads the season's data, which takes a few extra minutes.")
    st.caption("An unofficial personal project for fun. Nothing here is betting advice.")


HOME = st.Page(home, title="Home", icon="🏠", default=True)
NBA = st.Page(str(ROOT / "nba" / "nba_app.py"), title="NBA", icon="🏀", url_path="nba")
NFL = st.Page(str(ROOT / "nfl" / "nfl_app.py"), title="NFL", icon="🏈", url_path="nfl")

st.navigation([HOME, NFL, NBA], position="top").run()
