"""NBA game simulator interface. Run with nba_app_launcher.py, streamlit run nba_app.py, or as the NBA tab of
the combined site (streamlit run app.py from the repository root)."""
from pathlib import Path
import datetime as dt
import json
import os
import time

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import nba_data
import nba_game_sim as sim

ROOT = Path(__file__).resolve().parent
HOSTED = bool(os.environ.get("SIM_HOSTED"))
SEASON = nba_data.current_season()
SIM_CHOICES = [1_000, 5_000, 10_000]
LOOKAHEAD_DAYS = 4
_results = ROOT / "model_artifacts" / "nba_backtest" / "results.json"
_off = json.loads(_results.read_text())["off"] if _results.exists() else {}
BACKTEST = dict(games=_off.get("games", 0), accuracy=_off.get("accuracy", float("nan")),
                margin_mae=round(_off.get("margin_mae", float("nan")), 1),
                team_score_mae=round(_off.get("team_score_mae", float("nan")), 1),
                player_pts_mae=round(_off.get("player_pts_mae", float("nan")), 1),
                note=" (previous simulator version)" if "engine" in (json.loads(_results.read_text())
                                                                     if _results.exists() else {}) else "")

st.set_page_config(page_title="NBA Game Simulator", page_icon="🏀", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1400px; padding-top: 2.2rem;}
[data-testid="stMetricValue"] {font-size: 2rem;}
</style>""", unsafe_allow_html=True)
st.caption(f"{'' if HOSTED else 'LOCAL '}SIMULATOR · {nba_data.season_label(SEASON)} REGULAR SEASON")
st.title("NBA Game Simulator")
st.write("Plays a matchup thousands of times, possession by possession, from real play-by-play: who shoots, "
         "from where, how often it falls, plus fouls, rebounds, rotations and end-of-game logic.")


def _events_stamp():
    path = sim.CACHE_DIR / "sim" / f"events_{SEASON - 1}_{SEASON}.parquet"
    return path.stat().st_mtime if path.exists() else 0.


@st.cache_resource(show_spinner=False)
def _cached_league(season, events_stamp, sim_stamp):
    events, games = sim.load_events(season - 1, season)
    return sim.League(sim.label_free_throws(events), games, season)


def load_league():
    return _cached_league(SEASON, _events_stamp(), Path(sim.__file__).stat().st_mtime)


@st.cache_data(ttl=300, show_spinner=False)
def upcoming_games(days=LOOKAHEAD_DAYS):
    rows = []
    today = dt.datetime.now().date()
    for offset in range(days):
        day = today + dt.timedelta(days=offset)
        board = json.loads(nba_data._get(nba_data.ESPN_SCOREBOARD.format(date=day.strftime("%Y%m%d"))))
        for event in board.get("events", []):
            competition = event["competitions"][0]
            teams = {c["homeAway"]: c["team"] for c in competition["competitors"]}
            rows.append(dict(
                event_id=event["id"], tip=pd.Timestamp(event["date"]).tz_convert("America/New_York"),
                status=event["status"]["type"]["name"], **{
                    f"{side}{suffix}": value for side in ("home", "away") for suffix, value in (
                        ("", sim.ESPN_TEAM.get(teams[side]["abbreviation"], teams[side]["abbreviation"])),
                        ("_espn", teams[side]["abbreviation"]), ("_name", teams[side]["displayName"]),
                        ("_colors", ["#" + teams[side].get("color", "3b4a6b"),
                                     "#" + teams[side].get("alternateColor", "8ea0c8")]))}))
    games = pd.DataFrame(rows)
    if games.empty:
        return games
    return games.loc[games.status.eq("STATUS_SCHEDULED")].sort_values(["tip", "event_id"]).reset_index(drop=True)


@st.cache_data(ttl=600, show_spinner=False)
def injury_report(event_id):
    try:
        return nba_data.espn_injuries(nba_data.espn_summary(event_id))
    except Exception:
        return pd.DataFrame(columns=["team", "athlete_id", "name", "status", "detail", "date"])


def refresh_data():
    nba_data.load_espn_season(SEASON, refresh_schedule=True)
    sim.load_events(SEASON - 1, SEASON, refresh=True)
    upcoming_games.clear()
    injury_report.clear()


top = st.columns([1, 3])
with top[0]:
    if st.button("Refresh data", help="Downloads newly finished games from ESPN and rebuilds the ratings."):
        with st.spinner("Downloading new games and rebuilding events… about a minute"):
            try:
                refresh_data()
                st.session_state.pop("nba_simulation", None)
                st.success("Data refreshed.")
            except Exception as exc:
                st.error(f"Refresh failed: {exc}")
games_file = sim.CACHE_DIR / "espn" / f"games_{SEASON}.parquet"
if games_file.exists():
    finals = pd.read_parquet(games_file)
    finals = finals.loc[finals.status.eq("STATUS_FINAL")]
    with top[1]:
        st.caption(f"Ratings use {len(finals)} finished {nba_data.season_label(SEASON)} games through "
                   f"{pd.to_datetime(finals.date).max():%b %d} plus all of last season. Refresh after each night's games.")

try:
    with st.spinner("Building ratings… about 30 seconds the first time"):
        league = load_league()
except Exception as exc:
    st.error(f"Could not build ratings: {exc}. Run Refresh data, then reload.")
    st.stop()

try:
    games = upcoming_games()
except Exception as exc:
    st.error(f"Could not load the schedule from ESPN: {exc}")
    st.stop()
if games.empty:
    st.info(f"No upcoming games in the next {LOOKAHEAD_DAYS} days.")
    st.stop()

labels = {i: f"{g.away_name} at {g.home_name} — {g.tip:%a %b %d, %I:%M %p} ET" for i, g in games.iterrows()}
choice = st.selectbox("Game", list(labels), format_func=labels.get)
game = games.loc[choice]
home, away = game.home, game.away
game_key = game.event_id
names = {home: game.home_name, away: game.away_name}
colors = {home: list(game.home_colors), away: list(game.away_colors)}

logo = lambda abbr: f"https://a.espncdn.com/i/teamlogos/nba/500/{abbr.lower()}.png"
left, at, right = st.columns([2, 1, 2])
with left:
    st.image(logo(game.away_espn), width=110)
    st.subheader(game.away_name)
    st.caption("Away")
with at:
    st.markdown("<div style='text-align:center;font-size:2.4rem;padding-top:2.4rem;color:#888'>@</div>",
                unsafe_allow_html=True)
with right:
    st.image(logo(game.home_espn), width=110)
    st.subheader(game.home_name)
    st.caption("Home")

injuries = injury_report(game_key)
st.markdown("**Availability and starters**")
st.caption("Each team starts with its 10-man rotation (most minutes over the last five games). Players ruled Out "
           "on ESPN's injury report and deep-bench players start unchecked. Starters default to each team's most "
           "frequent starters over its last three games. Adjust both before simulating.")
setup = {}
team_cols = st.columns(2)
for col, team in zip(team_cols, (away, home)):
    with col:
        everyone, _ = sim.team_roster(league, team, recent_games=10)
        rotation = sim.rotation_players(league, team)
        report = injuries.loc[injuries.team.map(lambda a: sim.ESPN_TEAM.get(a, a)).eq(team)] if len(injuries) else injuries
        ruled_out = {"e" + str(a) for a in report.loc[report.status.eq("Out")].athlete_id} if len(report) else set()
        status = {"e" + str(a): s for a, s in zip(report.athlete_id, report.status)} if len(report) else {}
        name = lambda k: league.names.get(k, k) + (f" ({status[k]})" if k in status else "")
        available = st.multiselect(f"{names[team]} available", everyone,
                                   default=[k for k in rotation if k not in ruled_out], format_func=name,
                                   key=f"avail_{game_key}_{team}")
        out = [k for k in everyone if k not in available]
        players, auto_starters = sim.team_roster(league, team, out=out, recent_games=10)
        players = [k for k in players if k in available]
        starters = st.multiselect(f"{names[team]} starters", players, default=auto_starters[:5], max_selections=5,
                                  format_func=name, key=f"start_{game_key}_{team}_{len(out)}")
        setup[team] = (players, starters)
        if len(starters) != 5:
            st.warning("Pick exactly five starters.")

c1, c2 = st.columns([1, 2])
n_sims = c1.select_slider("Simulations", SIM_CHOICES, value=5_000)
use_defender = c2.checkbox("Use individual defender ratings", value=False,
                           help="In the backtest this changed nothing measurable, so it is off by default. "
                                "The play-by-play names a defender either way.")
ready = all(len(setup[t][1]) == 5 and len(setup[t][0]) >= 8 for t in (home, away))
if not ready:
    st.info("Each team needs exactly five starters and at least eight available players.")
if st.button(f"Simulate {n_sims:,} games", type="primary", disabled=not ready):
    started = time.perf_counter()
    with st.spinner(f"Simulating {n_sims:,} games… about {max(5, round(n_sims / 330))} seconds"):
        try:
            result = sim.simulate_game(league, home, away, setup[home][0], setup[away][0], setup[home][1],
                                       setup[away][1], date=game.tip.tz_localize(None).normalize(), n=n_sims,
                                       use_defender=use_defender)
            typical = result.typical_sim()
            samples, info = sim.prop_samples(result)
            st.session_state.nba_simulation = dict(
                key=game_key, result=result, typical=typical, samples=samples, info=info,
                html=sim.replay_html(result, typical, colors, names, autoplay=True),
                seconds=time.perf_counter() - started)
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")

S = st.session_state.get("nba_simulation")
if not S or S["key"] != game_key:
    st.stop()
result, samples, info = S["result"], S["samples"], S["info"]
g = result.games
margin, total = g.home_score - g.away_score, g.home_score + g.away_score
p_home = result.home_win_rate
st.divider()
m = st.columns(5)
m[0].metric(f"{names[away]} win · {sim.american_odds(1 - p_home)} fair", f"{1 - p_home:.1%}")
m[1].metric(f"{names[home]} win · {sim.american_odds(p_home)} fair", f"{p_home:.1%}")
m[2].metric(f"Median score ({away} – {home})", f"{g.away_score.median():.0f} – {g.home_score.median():.0f}")
fair_line = -(np.floor(margin.median()) + .5)
m[3].metric("Fair spread", f"{home} {fair_line:+.1f}")
m[4].metric("Fair total", f"{np.floor(total.median()) + .5:.1f}")
typical = g.loc[g.sim.eq(S["typical"])].iloc[0]
st.success(f"Simulated {len(g):,} separate games in {S['seconds']:.0f} s · overtime in {(g.periods > 4).mean():.1%}.")
st.caption(f"Replaying simulation {S['typical'] + 1:,} of {len(g):,}, the most typical game (final {away} "
           f"{int(typical.away_score)}, {home} {int(typical.home_score)}). Use the speed buttons or slider to scrub.")
components.html(S["html"], height=1180, scrolling=True)

with st.expander(f"Player averages across all {len(g):,} simulations"):
    averages = result.player_averages().reset_index()
    for team in (away, home):
        part = averages.loc[averages.team.eq(team)].drop(columns="team")
        st.markdown(f"**{names[team]}**")
        st.dataframe(part[["name", "MIN", "PTS", "REB", "AST", "3PM", "STL", "BLK", "TOV", "FGM", "FGA", "FTA", "PF"]],
                     hide_index=True, width="stretch")

st.subheader("Spread and total")
st.caption(f"Share of the {len(g):,} simulated games that cover each line, from the simulation alone (no sportsbook "
           f"data goes in). Backtest on {BACKTEST['games']} games this season{BACKTEST['note']}: favorites won "
           f"{BACKTEST['accuracy']:.0%}, the margin missed by {BACKTEST['margin_mae']} points on average and each "
           f"team's score by {BACKTEST['team_score_mae']}. Treat lines within a few points of the fair line as coin flips. "
           "Odds next to each percentage are fair (no-vig) American odds.")
s1, s2 = st.columns(2)
with s1:
    st.markdown("**Check a spread**")
    line = st.number_input(f"{names[home]} line", value=float(fair_line), step=.5, key=f"spread_{game_key}")
    p = sim.spread_probability(result, line)
    a, b = st.columns(2)
    a.metric(f"{home} {line:+g} covers · {sim.american_odds(p['home'])}", f"{p['home']:.1%}")
    b.metric(f"{away} {-line:+g} covers · {sim.american_odds(p['away'])}", f"{p['away']:.1%}")
    if p["push"]:
        st.caption(f"Push {p['push']:.1%}")
    st.dataframe(sim.spread_ladder(result), hide_index=True, width="stretch", height=300,
                 column_config={c: st.column_config.NumberColumn(c, format="percent")
                                for c in ("home covers", "away covers", "push")})
with s2:
    st.markdown("**Check a total**")
    tline = st.number_input("Total", value=float(np.floor(total.median()) + .5), step=.5, key=f"total_{game_key}")
    p = sim.total_probability(result, tline)
    a, b = st.columns(2)
    a.metric(f"Over {tline:g} · {sim.american_odds(p['over'])}", f"{p['over']:.1%}")
    b.metric(f"Under {tline:g} · {sim.american_odds(p['under'])}", f"{p['under']:.1%}")
    st.dataframe(sim.total_ladder(result), hide_index=True, width="stretch", height=300,
                 column_config={c: st.column_config.NumberColumn(c, format="percent") for c in ("over", "under")})
st.markdown(f"**Simulated margin ({home} minus {away})**")
hist = margin.clip(-40, 40).value_counts().sort_index()
st.bar_chart(pd.DataFrame({"games": hist}), height=220)

st.subheader("Player props")
st.caption(f"Share of simulated games where a player clears a line. Backtest: player points missed by "
           f"{BACKTEST['player_pts_mae']} on average; rotations are the biggest source of error, so check minutes.")
check, board_col = st.columns([2, 3])
with check:
    st.markdown("**Check a prop**")
    pid = st.selectbox("Player", list(info.index), key=f"prop_player_{game_key}",
                       format_func=lambda k: f"{info.at[k, 'name']} ({info.at[k, 'team']}, {info.at[k, 'MIN']:.0f} min)")
    stat = st.selectbox("Prop", list(sim.PROP_STATS), format_func=lambda k: sim.PROP_STATS[k][0],
                        key=f"prop_stat_{game_key}")
    median = float(samples[stat][pid].median())
    pline = st.number_input("Line", value=float(np.floor(median)) + .5, step=.5, key=f"prop_line_{game_key}_{pid}_{stat}")
    prop = sim.prop_probability(samples, pid, stat, pline)
    o, u = st.columns(2)
    o.metric(f"Over {pline:g} · {sim.american_odds(prop['over'])}", f"{prop['over']:.1%}")
    u.metric(f"Under {pline:g} · {sim.american_odds(prop['under'])}", f"{prop['under']:.1%}")
    st.caption(f"Simulated median {prop['median']:g} · average {prop['average']:.1f}"
               + (f" · push {prop['push']:.1%}" if prop["push"] else ""))
    values = samples[stat][pid].clip(upper=samples[stat][pid].quantile(.995)).round().astype(int)
    st.bar_chart(values.value_counts().sort_index().rename("games"), height=180)
with board_col:
    st.markdown("**Prop board**")
    b1, b2 = st.columns(2)
    threshold = b1.slider("Minimum hit rate", .50, .90, .60, .01, format="%.2f", key="nba_prop_threshold")
    sides = b2.multiselect("Sides", ["over", "under"], default=["over"], key="nba_prop_sides")
    stats = st.multiselect("Props", list(sim.PROP_STATS), default=["PTS", "REB", "AST", "3PM", "PRA", "PR", "PA", "RA", "SB"],
                           format_func=lambda k: sim.PROP_STATS[k][0], key="nba_prop_stats")
    board = sim.prop_board(samples, info, threshold, tuple(sides), stats=stats)
    st.caption(f"For each player (12+ simulated minutes) and prop, the most aggressive half-point line that hits at "
               f"least {threshold:.0%} of simulated games.")
    st.dataframe(board, hide_index=True, width="stretch", height=460,
                 column_config={"probability": st.column_config.NumberColumn("Hit rate", format="percent"),
                                "average": st.column_config.NumberColumn("Sim average", format="%.1f")})

with st.expander("Data and model details"):
    st.write("Play-by-play: last season from NBA.com (shufinskiy/nba_data on GitHub), this season from ESPN. "
             "Team colors and logos: ESPN. This is an unofficial personal project; nothing here uses betting lines.")
    st.write("No pregame model is blended in yet; results come from the possession simulation alone. Pace, "
             "scoring and home edge are calibrated to last season's real games.")
    st.write("See model_artifacts/nba_backtest/REPORT.md for the backtest.")
