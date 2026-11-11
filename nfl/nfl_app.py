"""NFL interface. Run with the notebook launcher, streamlit run nfl_app.py, or as the NFL tab of the combined
site (streamlit run app.py from the repository root)."""
from pathlib import Path
import json
import os
import time

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import nfl_model_pipeline as nfl
import nfl_game_sim as sim

ROOT = Path(__file__).resolve().parent
HOSTED = bool(os.environ.get("SIM_HOSTED"))
CONFIG = json.loads((ROOT / "nfl_weekly_config.json").read_text())
MODEL = ROOT / CONFIG["model_path"]
SEASON, WEEK = CONFIG["season"], CONFIG["week"]
QB_OVERRIDES = CONFIG["qb_overrides"]
OUT_PLAYERS = CONFIG.get("out_players", {})
RETURNING_PLAYERS = CONFIG.get("returning_players", {})
FEATURED_GAME = CONFIG.get("featured_game")
SIM_MODEL_WEIGHT = .25
# (games, calibration games). The hosted site has slower CPUs and shared users, so it runs fewer.
FEATURED_SIMS, STANDARD_SIMS = ((5_000, 2_000), (5_000, 2_000)) if HOSTED else ((20_000, 10_000), (20_000, 2_000))
NAMES = dict(
    ARI="Arizona Cardinals", ATL="Atlanta Falcons", BAL="Baltimore Ravens",
    BUF="Buffalo Bills", CAR="Carolina Panthers", CHI="Chicago Bears",
    CIN="Cincinnati Bengals", CLE="Cleveland Browns", DAL="Dallas Cowboys",
    DEN="Denver Broncos", DET="Detroit Lions", GB="Green Bay Packers",
    HOU="Houston Texans", IND="Indianapolis Colts", JAX="Jacksonville Jaguars",
    KC="Kansas City Chiefs", LAC="Los Angeles Chargers", LAR="Los Angeles Rams",
    LV="Las Vegas Raiders", MIA="Miami Dolphins", MIN="Minnesota Vikings",
    NE="New England Patriots", NO="New Orleans Saints", NYG="New York Giants",
    NYJ="New York Jets", PHI="Philadelphia Eagles", PIT="Pittsburgh Steelers",
    SEA="Seattle Seahawks", SF="San Francisco 49ers", TB="Tampa Bay Buccaneers",
    TEN="Tennessee Titans", WAS="Washington Commanders",
)

st.set_page_config(page_title="NFL Matchup Predictor", page_icon="🏈", layout="centered")
st.markdown("""<style>
.block-container {max-width: 1240px; padding-top: 2.5rem;}
[data-testid="stMetricValue"] {font-size: 2.2rem;}
</style>""", unsafe_allow_html=True)
st.caption(f"{'' if HOSTED else 'LOCAL '}MODEL · {SEASON} REGULAR SEASON")
st.title("NFL Matchup Predictor")
st.write(f"Week {WEEK} · Select a game to run your saved model.")


@st.cache_data(ttl=300, show_spinner=False)
def week_schedule(season, week):
    games = nfl.prepare_schedule(nfl.load_schedule(refresh=True), season, season)
    games = games.loc[games.week.eq(week)].copy()
    kickoff_text = games.gameday.dt.strftime("%Y-%m-%d") + " " + games.gametime.fillna("")
    games["kickoff"] = pd.to_datetime(kickoff_text, errors="coerce").dt.tz_localize("America/New_York")
    games.loc[games.gametime.isna(), "kickoff"] = pd.NaT
    return games, pd.Timestamp.now(tz="America/New_York").strftime("%b %d, %Y at %I:%M %p %Z")


def game_label(row):
    date = row.gameday.strftime("%a %b %d")
    return f"{NAMES.get(row.away_team, row.away_team)} at {NAMES.get(row.home_team, row.home_team)} — {date}"


def show_team(team, side):
    slug = {"WAS": "wsh"}.get(team, team.lower())
    st.image(f"https://a.espncdn.com/i/teamlogos/nfl/500/{slug}.png", width=140)
    st.subheader(NAMES.get(team, team))
    st.caption(side)


def save_selected_prediction(prediction, features, kickoff):
    if pd.isna(kickoff) or kickoff <= pd.Timestamp.now(tz="America/New_York"):
        raise ValueError("Kickoff has passed or is unknown; this cannot be saved as a pregame pick.")
    directory = ROOT / "prediction_logs"
    directory.mkdir(exist_ok=True)
    destination = directory / f"{prediction['game_id']}.json"
    clean_features = {name: None if pd.isna(value) else float(value) for name, value in features.items()}
    payload = {"prediction": prediction, "features": clean_features,
               "kickoff": str(kickoff), "saved_at": str(pd.Timestamp.now(tz="America/New_York"))}
    with destination.open("x") as handle:
        json.dump(payload, handle, indent=2, default=str, allow_nan=False)
    return destination


if st.button("Refresh schedule and clear preview"):
    week_schedule.clear()
    st.session_state.pop("nfl_preview", None)

if not MODEL.exists():
    st.error("The saved deployment model is missing. Restore the model_artifacts folder before predicting.")
    st.stop()

try:
    with st.spinner(f"Loading the Week {WEEK} schedule…"):
        games, loaded_at = week_schedule(SEASON, WEEK)
except Exception as exc:
    st.error(f"Could not refresh the schedule: {exc}")
    st.info("Check your internet connection, then use Refresh. No stale predictions have been substituted.")
    st.stop()

now = pd.Timestamp.now(tz="America/New_York")
upcoming = games.loc[~games.completed & games.kickoff.gt(now)].sort_values(["kickoff", "game_id"])
st.caption(f"Schedule refreshed: {loaded_at} · Times are Eastern")
if upcoming.empty:
    st.info(f"No upcoming Week {WEEK} games remain. This page does not generate retrospective picks.")
    st.stop()

by_game = upcoming.set_index("game_id")
game_ids = list(by_game.index)
game_id = st.selectbox(f"Choose a Week {WEEK} game", game_ids,
                       index=game_ids.index(FEATURED_GAME) if FEATURED_GAME in game_ids else 0,
                       format_func=lambda key: game_label(by_game.loc[key]), key=f"game_{SEASON}_{WEEK}")
game = by_game.loc[game_id]
st.caption(f"Kickoff: {game.kickoff.strftime('%A, %B %d at %I:%M %p %Z')} · {game_id}")
away_col, versus, home_col = st.columns([4, 1, 4])
with away_col:
    show_team(game.away_team, "AWAY")
with versus:
    st.markdown("<h2 style='padding-top:55px;text-align:center'>vs</h2>", unsafe_allow_html=True)
with home_col:
    show_team(game.home_team, "HOME")

st.info(f"Uses your saved model and completed Weeks 1–{WEEK - 1} statistics. No retraining. The win pick accounts for starting-QB changes only; the simulation also removes players listed Out in nfl_weekly_config.json and adds back players listed as returning.")
if st.button("Predict selected game", type="primary"):
    st.session_state.pop("nfl_preview", None)
    try:
        with st.spinner("Refreshing statistics and building pregame inputs…"):
            result, features = nfl.predict_upcoming_week(
                MODEL, SEASON, WEEK, refresh=True, log=False,
                game_id=game_id, return_features=True, qb_overrides=QB_OVERRIDES,
            )
        row = result.iloc[0].to_dict()
        st.session_state.nfl_preview = {
            "game_id": game_id, "prediction": row,
            "features": features.loc[game_id].to_dict(), "kickoff": str(game.kickoff),
        }
    except Exception as exc:
        st.error(f"Prediction could not be generated: {exc}")

preview = st.session_state.get("nfl_preview")
if preview and preview["game_id"] == game_id:
    prediction = preview["prediction"]
    home_probability = float(prediction["home_win_probability"])
    winner = NAMES.get(prediction["predicted_winner"], prediction["predicted_winner"])
    winner_probability = max(home_probability, 1 - home_probability)
    st.divider()
    st.caption(f"Scheduled quarterbacks: {prediction.get('away_qb_name', 'Unknown')} ({game.away_team}) · "
               f"{prediction.get('home_qb_name', 'Unknown')} ({game.home_team}). Review before kickoff.")
    st.subheader(f"Predicted winner: {winner}")
    st.metric("Winner probability", f"{winner_probability:.1%}")
    a, h = st.columns(2)
    a.metric(NAMES.get(game.away_team, game.away_team), f"{1 - home_probability:.1%}")
    h.metric(NAMES.get(game.home_team, game.home_team), f"{home_probability:.1%}")
    st.caption(f"Generated: {prediction['prediction_time']} · Model: {prediction['model_version']}")
    st.caption("2025 retrospective benchmark: 66.8% winner accuracy for the blend; 64.2% for the previous model. Future accuracy can differ.")
    if prediction.get("component_disagreement"):
        st.info("The model components disagree on the winner. Treat this as a close or uncertain matchup.")
    if "xgb_home_probability" in prediction:
        with st.expander("Compare model probabilities"):
            st.dataframe(pd.DataFrame([
                {"Model": label, "Home win": prediction[f"{key}_home_probability"],
                 "Away win": 1 - prediction[f"{key}_home_probability"]}
                for key, label in [("xgb", "XGBoost"), ("logistic", "Logistic regression"), ("elo_qb", "QB-adjusted Elo")]
            ]).style.format({"Home win": "{:.1%}", "Away win": "{:.1%}"}), hide_index=True)
    with st.expander("Show the exact model inputs"):
        st.dataframe(pd.Series(preview["features"], name="Input value").rename_axis("Feature").to_frame(), width="stretch")
        st.caption(f"Season averages use {SEASON} Weeks 1–{WEEK - 1} only. Elo carries historical team strength; QB inputs compare this starter with the team's regular one.")
    destination = ROOT / "prediction_logs" / f"{game_id}.json"
    if HOSTED:
        pass  # saved pregame picks are kept on the laptop, not on the shared site
    elif destination.exists():
        st.success("A prediction for this game is already saved. It will not be overwritten.")
    elif st.button("Save this prediction before kickoff"):
        try:
            saved = save_selected_prediction(prediction, preview["features"], pd.Timestamp(preview["kickoff"]))
            st.success(f"Saved {saved.name}. Only this game's prediction was recorded.")
        except Exception as exc:
            st.error(f"Could not save: {exc}")

PBP_RECHECK_HOURS = 6


def pbp_path(season):
    return ROOT / "data" / "nflverse" / f"play_by_play_{season}.parquet"


def pbp_behind(season):
    """True when a completed game is missing from the saved play-by-play."""
    path = pbp_path(season)
    if not path.exists():
        return True
    if time.time() - path.stat().st_mtime < PBP_RECHECK_HOURS * 3600:
        return False
    schedule = nfl.prepare_schedule(nfl.load_schedule(refresh=False), season, season)
    saved = set(pd.read_parquet(path, columns=["game_id"]).game_id)
    return bool(set(schedule.loc[schedule.completed, "game_id"]) - saved)


@st.cache_resource(show_spinner=False)
def _cached_league(season, sim_version, pbp_version):
    return sim.League.from_cache(season)


def load_league(season):
    if pbp_behind(season):
        try:
            with st.spinner("Downloading the latest play-by-play…"):
                sim.load_pbp([season], refresh_seasons=[season])
                sim.load_looks([season - 1, season], refresh_seasons=[season])
        except Exception as exc:
            st.warning(f"Could not update play-by-play ({exc}). Simulating with the saved copy.")
    return _cached_league(season, Path(sim.__file__).stat().st_mtime, pbp_path(season).stat().st_mtime)


def run_simulation(game_id, game, n_games, n_calibration):
    if preview and preview["game_id"] == game_id:
        prediction = preview["prediction"]
    else:
        prediction = nfl.predict_upcoming_week(MODEL, SEASON, WEEK, refresh=False, log=False, game_id=game_id,
                                               qb_overrides=QB_OVERRIDES).iloc[0].to_dict()
    probability = float(prediction.get("xgb_home_probability", prediction["home_win_probability"]))
    started = time.perf_counter()
    result = sim.simulate_game(load_league(SEASON), game.home_team, game.away_team, probability, n=n_games,
                               model_weight=SIM_MODEL_WEIGHT, home_qb=prediction.get("home_qb_name"),
                               away_qb=prediction.get("away_qb_name"), calibration_sims=n_calibration,
                               game_id=game_id, out_players=OUT_PLAYERS,
                               returning_players=RETURNING_PLAYERS)
    typical = result.typical_sim()
    final = result.games.loc[result.games.sim.eq(typical)].iloc[0]
    samples, prop_players = result.player_samples()
    return {"game_id": game_id, "html": sim.replay_html(result, typical, autoplay=True), "typical": typical,
            "home_win": result.home_win_rate, "matchup_only": result.raw_probability, "model": probability,
            "home_median": result.games.home_score.median(), "away_median": result.games.away_score.median(),
            "final": (int(final.away_score), int(final.home_score)), "projections": result.projections(),
            "n": int(result.games.sim.nunique()), "seconds": time.perf_counter() - started,
            "samples": samples, "prop_players": prop_players}


SIM_GAMES, SIM_CALIBRATION = FEATURED_SIMS if game_id == FEATURED_GAME else STANDARD_SIMS
st.divider()
st.subheader(f"Simulate {SIM_GAMES:,} games")
st.write(f"Plays this matchup {SIM_GAMES:,} times, snap by snap, from real nflverse play-by-play, then replays the "
         "most typical game with a live scoreboard, win probability and box scores. Offense and defense matchups "
         f"drive the results; the XGBoost probability counts for {SIM_MODEL_WEIGHT:.0%}.")
if st.button(f"Simulate {SIM_GAMES:,} games", type="primary"):
    try:
        minutes = max(1, round((1.25 * SIM_GAMES + 11 * SIM_CALIBRATION) / 32_000))
        with st.spinner(f"Simulating {SIM_GAMES:,} games… about {minutes} minute{'s' if minutes > 1 else ''}"):
            st.session_state.nfl_simulation = run_simulation(game_id, game, SIM_GAMES, SIM_CALIBRATION)
    except Exception as exc:
        st.error(f"Simulation failed: {exc}")

simulation = st.session_state.get("nfl_simulation")
if simulation and simulation["game_id"] == game_id:
    home, away = game.home_team, game.away_team
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{NAMES.get(away, away)} win", f"{1 - simulation['home_win']:.1%}")
    m2.metric(f"{NAMES.get(home, home)} win", f"{simulation['home_win']:.1%}")
    m3.metric("Median score", f"{away} {simulation['away_median']:.0f} – {simulation['home_median']:.0f} {home}")
    m4.metric(f"Matchups only ({home})", f"{simulation['matchup_only']:.1%}",
              f"model {simulation['model']:.1%}", delta_color="off")
    away_final, home_final = simulation["final"]
    minutes, seconds = divmod(round(simulation["seconds"]), 60)
    st.success(f"Simulated {simulation['n']:,} separate games in {minutes} min {seconds} s.")
    st.caption(f"Replaying simulation {simulation['typical'] + 1:,} of {simulation['n']:,} — the most typical game "
               f"(final {away} {away_final}, {home} {home_final}). Use the speed buttons or slider to scrub.")
    components.html(simulation["html"], height=960, scrolling=True)
    with st.expander("Defensive looks used in the simulation"):
        st.caption("Each snap, the defense picks a look from these tendencies (shown for early downs and 3rd & long); "
                   "outcomes lean on real plays against that look, including each player's own. Coverage is from "
                   "2025 nflverse participation data (2026 not yet published); blitz and box count are from 2025-26 "
                   "FTN charting. Box count stands in for run schemes, which no free source records.")
        looks = sim.look_table(load_league(SEASON), home, away)
        st.dataframe(looks, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(c, format="percent") for c in looks.columns[2:]})
    with st.expander(f"Player averages across all {simulation['n']:,} simulations"):
        projections = simulation["projections"]
        for team in (away, home):
            p = projections.loc[projections.team.eq(team)]
            p = p.loc[p.pass_att.ge(1) | p.rush_car.ge(.5) | p.tgt.ge(.5)]
            st.markdown(f"**{NAMES.get(team, team)}**")
            st.dataframe(p[["player", "pos", "pass_yds", "pass_td", "rush_car", "rush_yds", "rec", "rec_yds",
                            "anytime_td"]].sort_values(["pass_yds", "rush_yds", "rec_yds"], ascending=False).round(1),
                         hide_index=True, width="stretch",
                         column_config={"anytime_td": st.column_config.NumberColumn("Anytime TD", format="percent")})

if simulation and simulation["game_id"] == game_id:
    samples, prop_players = simulation["samples"], simulation["prop_players"]
    st.subheader("Player props")
    st.caption(f"Share of the {simulation['n']:,} simulated games where a player clears a line. These come only from "
               "the simulation (no sportsbook data) and have not been checked against real results yet. "
               "Pass volume and the per-game spread of QB passing yards are calibrated to real 2025-26 games "
               "(about 12% of games under 150 yards); average passing yards run about 3% above real 2026.")
    check, board_col = st.columns([2, 3])
    with check:
        st.markdown("**Check a prop**")
        order = prop_players.sort_values(["team", "player"])
        pid = st.selectbox("Player", list(order.index), key=f"prop_player_{game_id}",
                           format_func=lambda k: f"{order.at[k, 'player']} ({order.at[k, 'team']} {order.at[k, 'pos']})")
        stats = [stat for stat, role in sim.PROP_ROLES.items()
                 if any(prop_players.at[pid, r] for r in role.split("|")) and (stat, pid) in samples]
        stat = st.selectbox("Prop", stats, format_func=lambda k: sim.PROP_STATS[k][0], key=f"prop_stat_{game_id}")
        median = float(samples[(stat, pid)].median())
        line = st.number_input("Line", value=float(median) + .5 if stat != "anytime_td" else .5, step=.5,
                               key=f"prop_line_{game_id}_{pid}_{stat}")
        prop = sim.prop_probability(samples, prop_players, pid, stat, line)
        o, u = st.columns(2)
        o.metric(f"Over {line:g}", f"{prop['over']:.1%}")
        u.metric(f"Under {line:g}", f"{prop['under']:.1%}")
        st.caption(f"Simulated median {prop['median']:g} · average {prop['average']:.1f}"
                   + (f" · push {prop['push']:.1%}" if prop["push"] else ""))
    with board_col:
        st.markdown("**Prop board**")
        threshold = st.slider("Minimum hit rate", .50, .90, .55, .01, format="%.2f", key="nfl_prop_threshold")
        sides = st.multiselect("Sides", ["over", "under"], default=["over"], key="nfl_prop_sides")
        board = sim.prop_board(samples, prop_players, threshold, tuple(sides))
        st.caption("For each player and prop, the most aggressive standard line that hits at least "
                   f"{threshold:.0%} of simulated games.")
        st.dataframe(board, hide_index=True, width="stretch", height=420,
                     column_config={"probability": st.column_config.NumberColumn("Hit rate", format="percent")})

with st.expander("Data and model details"):
    st.write("Team statistics and schedules: nflverse. Logos: ESPN. This is an unofficial personal project.")
    st.write("Predictions refresh current-season stats. The fitted model stays fixed. Past or started games cannot be selected.")
    st.write("Simulations download new play-by-play automatically once a completed game is missing from the saved copy.")
    st.code(str(MODEL.relative_to(ROOT)) if HOSTED else str(MODEL), language=None)
