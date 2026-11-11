"""Possession-by-possession Monte Carlo NBA game simulator."""

from pathlib import Path
import html
import json
import math
import random
import re
import unicodedata

import numpy as np
import pandas as pd

import nba_data


CACHE_DIR = nba_data.CACHE_DIR
ZONES = ["RA", "PAINT", "MID", "C3", "AB3"]
Z = {name: i for i, name in enumerate(ZONES)}
ZONE_VALUE = np.array([2, 2, 2, 3, 3])
ZONE_LABEL = ["at the rim", "in the paint", "a mid-range jumper", "a corner 3", "an above-the-break 3"]
STARTS = ["inbound", "drb", "steal", "oreb"]
S = {name: i for i, name in enumerate(STARTS)}
GROUPS = ["G", "F", "C"]
POSITION_GROUP = {"PG": 0, "SG": 0, "G": 0, "SF": 1, "PF": 1, "F": 1, "GF": 1, "FC": 2, "C": 2, "CF": 2}
ARCHIVE_AREA = {"Restricted Area": "RA", "In The Paint (Non-RA)": "PAINT", "Mid-Range": "MID",
                "Left Corner 3": "C3", "Right Corner 3": "C3", "Above the Break 3": "AB3", "Backcourt": "AB3"}
ESPN_TEAM = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
ESPN_TURNOVER = re.compile(r"Turnover|Traveling|Double Dribble")
NAME_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
PERIOD_SECONDS, OT_SECONDS, REGULATION = 720, 300, 2880
MINUTE_BINS = 48
EVENT_COLUMNS = ["game_id", "season", "date", "home", "away", "order", "period", "t", "team", "kind",
                 "p1", "p2", "p3", "made", "value", "zone", "ft_i", "ft_n", "foul", "stolen"]

PRIOR_WEIGHT, TEAM_PRIOR_WEIGHT = .6, .3
MIX_SHRINK = 40.
FG_SHRINK = {"RA": 60., "PAINT": 80., "MID": 100., "C3": 120., "AB3": 150.}
FT_SHRINK = 60.
DRAW_SHRINK = 150.
USAGE_SHRINK, USAGE_PRIOR_WEIGHT = 250., .4
MINUTE_SIMS, MINUTE_ITERATIONS = 80, 0
RATE_SHRINK_MIN = 300.
TOV_SHRINK = 150.
TEAM_SHRINK = {"tov": 600., "oreb": 400., "nsf": 800., "shot_foul": 1500., "fg": 1500., "mix": 2000.,
               "pace": 1500.}
DEFENDER_SHRINK = 400.
ONFLOOR_RIDGE = 3000.
ASSIGN_SHRINK = 150.
OCC_HALF_LIFE_GAMES = 5.
OCC_PRIOR_GAMES = 3.
SUB_INTERVAL = 110.
GARBAGE_MARGIN, GARBAGE_SECONDS = 20, 330
FOUL_OUT = 6
GAME_FORM_SD = 0.
LEAD_PACE, TRAIL_PACE = .01, .006
REGRESSION = .06
PILOT_SIMS = 150


def _clock_seconds(clock):
    """ESPN '11:03' or '45.3'; NBA.com 'PT11M36.00S'. Seconds left in the period."""
    clock = pd.Series(clock, dtype="string")
    iso = clock.str.extract(r"PT(\d+)M([\d.]+)S")
    plain = clock.str.extract(r"^(?:(\d+):)?([\d.]+)$")
    minutes = iso[0].fillna(plain[0]).astype(float).fillna(0)
    seconds = iso[1].fillna(plain[1]).astype(float)
    return (minutes * 60 + seconds).to_numpy()


def _elapsed(period, left):
    period = np.asarray(period)
    length = np.where(period <= 4, PERIOD_SECONDS, OT_SECONDS)
    start = np.where(period <= 4, (period - 1) * PERIOD_SECONDS, REGULATION + (period - 5) * OT_SECONDS)
    return start + length - left


def period_start(period):
    return (period - 1) * PERIOD_SECONDS if period <= 4 else REGULATION + (period - 5) * OT_SECONDS


def period_end(period):
    return period_start(period) + (PERIOD_SECONDS if period <= 4 else OT_SECONDS)


def normalize_name(name):
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    name = re.sub(r"[^a-z ]", "", name.replace("-", " "))
    return " ".join(NAME_SUFFIX.sub("", name).split())


def _espn_zone(x, y, value):
    distance = np.hypot(x - 25, y - 1)
    zone = np.where(value == 3, np.where(y <= 9, Z["C3"], Z["AB3"]),
                    np.where(distance <= 4, Z["RA"],
                             np.where((np.abs(x - 25) <= 8) & (y <= 14), Z["PAINT"], Z["MID"])))
    return np.where(np.isnan(x) & (value == 2), Z["RA"], np.where(np.isnan(x) & (value == 3), Z["AB3"], zone))


def espn_events(season, cache_dir=CACHE_DIR):
    """This season's ESPN plays as the common event table. Player keys are 'e<ESPN id>'."""
    games = pd.read_parquet(Path(cache_dir) / "espn" / f"games_{season}.parquet")
    games = games.loc[games.status.eq("STATUS_FINAL")]
    plays = pd.read_parquet(Path(cache_dir) / "espn" / f"plays_{season}.parquet")
    plays = plays.loc[plays.event_id.isin(games.event_id)].merge(
        games[["event_id", "date", "home", "away", "home_id", "away_id"]], on="event_id")
    for side in ("home", "away"):
        plays[side] = plays[side].replace(ESPN_TEAM)
    plays = plays.sort_values(["event_id", "sequence"]).reset_index(drop=True)
    team = np.where(plays.team_id.eq(plays.home_id), plays.home,
                    np.where(plays.team_id.eq(plays.away_id), plays.away, None))
    kind = np.full(len(plays), None, dtype=object)
    ptype = plays.type
    kind[plays.field_goal.to_numpy()] = "fga"
    is_ft = plays.free_throw.to_numpy()
    special_ft = ptype.str.contains("Technical|Flagrant", regex=True).to_numpy()
    kind[is_ft & ~special_ft] = "ft"
    kind[is_ft & special_ft] = "ft_special"
    kind[ptype.eq("Defensive Rebound").to_numpy()] = "dreb"
    kind[ptype.eq("Offensive Rebound").to_numpy()] = "oreb"
    kind[ptype.str.contains(ESPN_TURNOVER).to_numpy()] = "tov"
    is_foul = ptype.str.contains("Foul|Charge|Technical", regex=True).to_numpy() & (kind == None)  # noqa: E711
    kind[is_foul] = "foul"
    kind[ptype.eq("End Period").to_numpy()] = "period_end"
    kind[ptype.eq("Substitution").to_numpy()] = "sub"
    foul = np.where(ptype.str.contains("Technical|Double", regex=True), "technical",
                    np.where(ptype.str.contains("Offensive|Charge", regex=True), "offensive",
                             np.where(ptype.str.contains("Shooting"), "shooting", "personal")))
    ft = ptype.str.extract(r"(\d) of (\d)").astype(float)
    value = plays.points_attempted.to_numpy()
    key = lambda column: np.where(plays[column].notna(), "e" + plays[column].astype(str), None).astype(object)
    frame = pd.DataFrame(dict(
        game_id="E" + plays.event_id.astype(str), season=season, date=pd.to_datetime(plays.date),
        home=plays.home, away=plays.away, order=plays.sequence, period=plays.period,
        t=_elapsed(plays.period.to_numpy(), _clock_seconds(plays.clock)), team=team, kind=kind,
        p1=key("player_1"), p2=key("player_2"), p3=None, made=(plays.scoring & plays.shooting).to_numpy(),
        value=np.where(kind == "fga", value, 0), zone=-1, ft_i=ft[0].fillna(0).to_numpy(),
        ft_n=ft[1].fillna(0).to_numpy(), foul=np.where(kind == "foul", foul, None),
        stolen=plays.text.str.contains("steals", na=False).to_numpy() & (kind == "tov")))
    shots = frame.kind.eq("fga")
    frame.loc[shots, "zone"] = _espn_zone(plays.x[shots].to_numpy(), plays.y[shots].to_numpy(), value[shots])
    blocked = shots & plays.blocked
    frame.loc[blocked, "p3"] = frame.loc[blocked, "p2"]
    frame.loc[shots & ~plays.assisted, "p2"] = None
    frame.loc[frame.kind.eq("tov") & ~frame.stolen, "p2"] = None
    frame.loc[frame.kind.isin(["foul", "fga", "ft", "ft_special", "dreb", "oreb", "period_end"]) & ~shots, "p2"] = None
    subs = frame.loc[frame.kind.eq("sub")]
    out_rows = subs.assign(kind="sub_out", p1=subs.p2, p2=None, order=subs.order - .5)
    in_rows = subs.assign(kind="sub_in", p2=None)
    frame = pd.concat([frame.loc[frame.kind.notna() & frame.kind.ne("sub")], out_rows, in_rows])
    return frame[EVENT_COLUMNS].sort_values(["game_id", "order"]).reset_index(drop=True)


def archive_events(season, id_map=None, cache_dir=CACHE_DIR):
    """A past season's NBA.com play-by-play as the common event table."""
    pbp = nba_data.load_archive("cdnnba", season, cache_dir=cache_dir, columns=[
        "gameId", "actionNumber", "orderNumber", "period", "clock", "teamTricode", "personId", "actionType",
        "subType", "qualifiers", "description", "shotResult", "area", "assistPersonId", "blockPersonId",
        "stealPersonId", "foulDrawnPersonId"])
    dates = nba_data.load_archive("shotdetail", season, cache_dir=cache_dir,
                                  columns=["GAME_ID", "GAME_DATE", "HTM", "VTM"]).drop_duplicates("GAME_ID")
    pbp = pbp.merge(dates.rename(columns={"GAME_ID": "gameId"}), on="gameId")
    pbp = pbp.sort_values(["gameId", "orderNumber"]).reset_index(drop=True)
    id_map = id_map or {}

    def key(column):
        ids = pd.to_numeric(pbp[column], errors="coerce")
        keys = ids.map(lambda i: None if pd.isna(i) or i == 0 else id_map.get(int(i), f"n{int(i)}"))
        return keys.astype(object)

    text = lambda column: pbp[column].fillna("").astype(str)
    action, sub = text("actionType"), text("subType")
    description, qualifiers = text("description").str.lower(), text("qualifiers")
    pbp["shotResult"], pbp["area"], pbp["teamTricode"] = text("shotResult"), text("area"), text("teamTricode")
    kind = np.full(len(pbp), None, dtype=object)
    kind[action.isin(["2pt", "3pt", "heave"]).to_numpy()] = "fga"
    special = description.str.contains("technical|flagrant")
    kind[(action.eq("freethrow") & ~special).to_numpy()] = "ft"
    kind[(action.eq("freethrow") & special).to_numpy()] = "ft_special"
    mandatory = qualifiers.str.contains("mandatory")
    kind[(action.eq("rebound") & sub.eq("defensive") & ~mandatory).to_numpy()] = "dreb"
    kind[(action.eq("rebound") & sub.eq("offensive") & ~mandatory).to_numpy()] = "oreb"
    kind[action.eq("turnover").to_numpy()] = "tov"
    kind[action.eq("foul").to_numpy()] = "foul"
    kind[(action.eq("substitution") & sub.eq("out")).to_numpy()] = "sub_out"
    kind[(action.eq("substitution") & sub.eq("in")).to_numpy()] = "sub_in"
    kind[(action.eq("period") & sub.eq("end")).to_numpy()] = "period_end"
    foul = np.where(sub.eq("technical") | description.str.contains("flagrant"), "technical",
                    np.where(sub.eq("offensive"), "offensive",
                             np.where(description.str.contains("shooting"), "shooting", "personal")))
    ft = sub.str.extract(r"(\d) of (\d)").astype(float)
    p1 = key("personId")
    frame = pd.DataFrame(dict(
        game_id="N" + pbp.gameId.astype(str), season=season, date=pd.to_datetime(pbp.GAME_DATE.astype(str)),
        home=pbp.HTM, away=pbp.VTM, order=pbp.orderNumber, period=pbp.period,
        t=_elapsed(pbp.period.to_numpy(), _clock_seconds(pbp.clock)), team=pbp.teamTricode.replace("", None).to_numpy(),
        kind=kind, p1=p1, p2=None, p3=None, made=pbp.shotResult.eq("Made").to_numpy(),
        value=np.where(action.eq("2pt"), 2, np.where(action.isin(["3pt", "heave"]), 3, 0)),
        zone=pbp.area.map(ARCHIVE_AREA).map(Z).fillna(-1).astype(int).to_numpy(),
        ft_i=ft[0].fillna(0).to_numpy(), ft_n=ft[1].fillna(0).to_numpy(),
        foul=np.where(kind == "foul", foul, None), stolen=False))
    shots = frame.kind.eq("fga")
    frame.loc[shots & action.eq("heave"), "zone"] = Z["AB3"]
    frame.loc[shots, "p2"] = key("assistPersonId")[shots]
    frame.loc[shots, "p3"] = key("blockPersonId")[shots]
    turnovers = frame.kind.eq("tov")
    stealer = key("stealPersonId")
    frame.loc[turnovers, "p2"] = stealer[turnovers]
    frame.loc[turnovers, "stolen"] = stealer[turnovers].notna()
    fouls = frame.kind.eq("foul")
    frame.loc[fouls, "p2"] = key("foulDrawnPersonId")[fouls]
    frame = frame.loc[frame.kind.notna()]
    return frame[EVENT_COLUMNS].reset_index(drop=True)


def espn_to_nba_ids(prior_season, current_season, cache_dir=CACHE_DIR):
    """NBA.com person id -> ESPN key, matched on normalized full names."""
    box = pd.read_parquet(Path(cache_dir) / "espn" / f"box_{current_season}.parquet", columns=["athlete_id", "name"])
    espn = box.drop_duplicates("athlete_id").assign(norm=lambda f: f.name.map(normalize_name))
    espn = espn.loc[~espn.norm.duplicated(keep=False)]
    matchups = nba_data.load_archive("matchups", prior_season, cache_dir=cache_dir,
                                     columns=["person_id", "first_name", "family_name",
                                              "matchups_person_id", "matchups_first_name", "matchups_family_name"])
    shots = nba_data.load_archive("shotdetail", prior_season, cache_dir=cache_dir, columns=["PLAYER_ID", "PLAYER_NAME"])
    names = pd.concat([
        pd.DataFrame({"nba": matchups.person_id, "name": matchups.first_name + " " + matchups.family_name}),
        pd.DataFrame({"nba": matchups.matchups_person_id,
                      "name": matchups.matchups_first_name + " " + matchups.matchups_family_name}),
        pd.DataFrame({"nba": shots.PLAYER_ID, "name": shots.PLAYER_NAME})]).dropna().drop_duplicates("nba")
    names["norm"] = names.name.map(normalize_name)
    names = names.loc[~names.norm.duplicated(keep=False)]
    merged = names.merge(espn, on="norm")
    return {int(n): f"e{a}" for n, a in zip(merged.nba, merged.athlete_id)}


def _period_lineups(rows, team, previous_end):
    """The five on the floor at a period's start."""
    first = {}
    for kind, row_team, p1, p2, p3 in rows:
        candidates = []
        if row_team == team and p1 is not None:
            candidates.append((p1, kind))
        if kind == "fga" and row_team == team and p2 is not None:
            candidates.append((p2, kind))
        if row_team is not None and row_team != team:
            if kind == "tov" and p2 is not None:
                candidates.append((p2, kind))
            if kind == "fga" and p3 is not None:
                candidates.append((p3, kind))
            if kind == "foul" and p2 is not None:
                candidates.append((p2, kind))
        for player, how in candidates:
            first.setdefault(player, how)
    starters = [p for p, how in first.items() if how != "sub_in"][:5]
    entered = {p for p, how in first.items() if how == "sub_in"}
    for player in previous_end:
        if len(starters) >= 5:
            break
        if player not in starters and player not in entered:
            starters.append(player)
    return starters


def attach_lineups(events):
    """Rebuild the ten players on the floor for every event, and each possession's start type and elapsed
    time.
    """
    events = events.sort_values(["game_id", "order"]).reset_index(drop=True)
    home_on = np.empty(len(events), dtype=object)
    away_on = np.empty(len(events), dtype=object)
    start = np.full(len(events), -1, dtype=np.int8)
    dur = np.full(len(events), np.nan)
    cols = [events[c].to_numpy() for c in ("kind", "team", "p1", "p2", "p3", "period", "t", "made",
                                           "ft_i", "ft_n", "foul", "stolen", "home", "away")]
    kinds, teams, p1s, p2s, p3s, periods, ts, mades, ft_is, ft_ns, fouls, stolens, homes, aways = cols
    bounds = np.flatnonzero(np.r_[True, events.game_id.to_numpy()[1:] != events.game_id.to_numpy()[:-1], True])
    for g in range(len(bounds) - 1):
        lo, hi = bounds[g], bounds[g + 1]
        home, away = homes[lo], aways[lo]
        other = {home: away, away: home}
        ends = {home: [], away: []}
        i = lo
        while i < hi:
            period = periods[i]
            j = i
            while j < hi and periods[j] == period:
                j += 1
            rows = list(zip(kinds[i:j], teams[i:j], p1s[i:j], p2s[i:j], p3s[i:j]))
            on = {side: _period_lineups(rows, side, ends[side]) for side in (home, away)}
            off, kind_start, start_t = None, S["inbound"], period_start(period)
            shooter_team, pending_ft = None, False
            for r in range(i, j):
                kind, team = kinds[r], teams[r]
                if kind == "sub_out" and p1s[r] in on.get(team, []):
                    on[team].remove(p1s[r])
                elif kind == "sub_in" and team in on and p1s[r] not in on[team]:
                    on[team].append(p1s[r])
                home_on[r] = "|".join(on[home])
                away_on[r] = "|".join(on[away])
                t = ts[r]
                if kind == "fga":
                    start[r] = kind_start if off in (None, team) else S["inbound"]
                    dur[r] = t - start_t
                    shooter_team, pending_ft = team, False
                    if mades[r]:
                        off, kind_start, start_t = other.get(team), S["inbound"], t
                    else:
                        off = team
                elif kind in ("oreb", "dreb") and not pending_ft and team in other:
                    off, start_t = team, t
                    kind_start = S["oreb"] if team == shooter_team else S["drb"]
                elif kind == "tov" and team in other:
                    start[r] = kind_start if off in (None, team) else S["inbound"]
                    dur[r] = t - start_t
                    off, start_t = other[team], t
                    kind_start = S["steal"] if stolens[r] else S["inbound"]
                elif kind == "foul" and team in other and fouls[r] in ("shooting", "personal"):
                    offense = other[team]
                    start[r] = kind_start if off in (None, offense) else S["inbound"]
                    dur[r] = t - start_t
                    off = offense
                    if fouls[r] == "personal":
                        kind_start, start_t = S["inbound"], t
                elif kind == "ft" and team in other:
                    pending_ft = ft_is[r] < ft_ns[r]
                    if not pending_ft:
                        if mades[r]:
                            off, kind_start, start_t = other[team], S["inbound"], t
                        else:
                            off, shooter_team = team, team
            ends = {side: list(on[side]) for side in (home, away)}
            i = j
    events["home_on"], events["away_on"] = home_on, away_on
    events["start"], events["dur"] = start, dur
    return events


def player_games(events):
    """Per player per game: team, minutes, whether he started, and on-floor share in each of the 48
    regulation minutes (from the rebuilt lineups).
    """
    rows = []
    for game_id, game in events.groupby("game_id", sort=False):
        t = game.t.to_numpy()
        nxt = np.r_[t[1:], t[-1]]
        same_period = np.r_[game.period.to_numpy()[1:] == game.period.to_numpy()[:-1], False]
        span = np.where(same_period, np.clip(nxt - t, 0, None), 0)
        first_period = game.period.to_numpy() == 1
        for side, column in (("home", "home_on"), ("away", "away_on")):
            team = game[side].iloc[0]
            seconds, occupancy, starters = {}, {}, set()
            first_row = True
            for on, start_t, length, in_first in zip(game[column].to_numpy(), t, span, first_period):
                players = on.split("|") if on else []
                if first_row and in_first:
                    starters = set(players)
                    first_row = False
                if length <= 0:
                    continue
                for player in players:
                    seconds[player] = seconds.get(player, 0.) + length
                    if start_t < REGULATION:
                        occ = occupancy.setdefault(player, np.zeros(MINUTE_BINS))
                        _spread(occ, start_t, min(start_t + length, REGULATION))
            for player, total in seconds.items():
                rows.append(dict(game_id=game_id, date=game.date.iloc[0], season=game.season.iloc[0], team=team,
                                 home=side == "home", player=player, minutes=total / 60,
                                 started=player in starters,
                                 occ=occupancy.get(player, np.zeros(MINUTE_BINS)) / 60.))
    return pd.DataFrame(rows)


def _spread(occ, a, b):
    """Add seconds on the floor between a and b into minute bins."""
    while a < b:
        minute = int(a // 60)
        edge = min(b, (minute + 1) * 60)
        occ[min(minute, MINUTE_BINS - 1)] += edge - a
        a = edge


def load_events(prior_season=None, current_season=None, refresh=False, cache_dir=CACHE_DIR):
    """Both seasons' events with lineups and player-games, cached as parquet."""
    current_season = current_season or nba_data.current_season()
    prior_season = prior_season or current_season - 1
    out = Path(cache_dir) / "sim"
    out.mkdir(parents=True, exist_ok=True)
    paths = {name: out / f"{name}_{prior_season}_{current_season}.parquet" for name in ("events", "player_games")}
    if refresh or not all(p.exists() for p in paths.values()):
        id_map = espn_to_nba_ids(prior_season, current_season, cache_dir)
        events = pd.concat([archive_events(prior_season, id_map, cache_dir),
                            espn_events(current_season, cache_dir)], ignore_index=True)
        events = attach_lineups(events)
        games = player_games(events)
        events.to_parquet(paths["events"], index=False)
        games.assign(occ=games.occ.map(list)).to_parquet(paths["player_games"], index=False)
        pd.Series(id_map).to_csv(out / f"id_map_{prior_season}.csv")
    events = pd.read_parquet(paths["events"])
    games = pd.read_parquet(paths["player_games"])
    games["occ"] = games.occ.map(np.asarray)
    return events, games


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _expit(x):
    return 1 / (1 + np.exp(-x))


def _ffill_by_game(frame, mask, columns):
    return frame[columns].where(mask).groupby(frame.game_id).ffill()


def label_free_throws(events):
    """Classify each free throw trip (rows with ft_i == 1) by what caused it."""
    events = events.sort_values(["game_id", "order"]).reset_index(drop=True)
    is_foul = events.kind.eq("foul")
    fouls = _ffill_by_game(events.assign(ft_=events.foul, ftt=events.t), is_foul, ["ft_", "ftt"])
    makes = events.kind.eq("fga") & events.made
    shots = _ffill_by_game(events.assign(mt=events.t, mp=events.p1, mz=events.zone), makes, ["mt", "mp", "mz"])
    first = events.kind.eq("ft") & events.ft_i.eq(1)
    same_foul = fouls.ftt.eq(events.t)
    and_one = first & events.ft_n.eq(1) & shots.mt.eq(events.t) & shots.mp.eq(events.p1) & same_foul \
        & fouls.ft_.eq("shooting")
    trip = np.where(~first, None, np.where(and_one, "andone",
                    np.where(same_foul & fouls.ft_.eq("shooting"), "shot",
                             np.where(same_foul & fouls.ft_.eq("personal"), "bonus", "other"))))
    events["trip"] = trip
    and_one_keys = set(zip(events.game_id[and_one], events.t[and_one], events.p1[and_one]))
    events["and_one"] = [(g, t, p) in and_one_keys for g, t, p in zip(events.game_id, events.t, events.p1)] \
        & events.kind.eq("fga")
    events["and_one_zone"] = np.where(and_one, shots.mz, -1)
    points = np.where(events.kind.eq("fga") & events.made, events.value, 0) \
        + np.where(events.kind.isin(["ft", "ft_special"]) & events.made, 1, 0)
    home_points = np.where(events.team.eq(events.home), points, 0)
    events["home_score"] = pd.Series(home_points).groupby(events.game_id).cumsum().to_numpy()
    events["away_score"] = pd.Series(points - home_points).groupby(events.game_id).cumsum().to_numpy()
    return events


def _two_way(frame, y, offense, defense, weight, base, shrink, iterations=6):
    """Opponent-adjusted logit effects for a binary outcome."""
    off, de = {}, {}
    y, weight, base = np.asarray(y, float), np.asarray(weight, float), np.asarray(base, float)
    offense, defense = np.asarray(offense), np.asarray(defense)
    for _ in range(iterations):
        for side, keys, other_keys, other in ((off, offense, defense, de), (de, defense, offense, off)):
            eta = base + pd.Series(other_keys).map(other).fillna(0).to_numpy() \
                + pd.Series(keys).map(side).fillna(0).to_numpy()
            p = _expit(eta)
            grad = pd.Series(weight * (y - p)).groupby(keys).sum()
            hess = pd.Series(weight * p * (1 - p)).groupby(keys).sum()
            current = pd.Series(keys).map(side).fillna(0).groupby(keys).mean()
            pbar = (hess / pd.Series(weight).groupby(keys).sum()).fillna(.25)
            step = (grad - shrink * pbar * current) / (hess + shrink * pbar)
            for key, value in (current + step).items():
                side[key] = value
    return off, de


class League:
    """Everything learned from events before ``as_of``, shared by every matchup."""

    def __init__(self, events, player_games, current_season, as_of=None, use_onfloor=True, cache_dir=CACHE_DIR):
        self.current_season, self.as_of = current_season, as_of
        prior_season = current_season - 1
        keep = (events.season < current_season) | (events.date < as_of if as_of is not None else True)
        ev = events.loc[keep].copy()
        if "trip" not in ev:
            ev = label_free_throws(ev)
        pg = player_games.loc[(player_games.season < current_season)
                              | (player_games.date < as_of if as_of is not None else True)].copy()
        ev["w"] = np.where(ev.season < current_season, PRIOR_WEIGHT, 1.)
        ev["tw"] = np.where(ev.season < current_season, TEAM_PRIOR_WEIGHT, 1.)
        pg["w"] = np.where(pg.season < current_season, PRIOR_WEIGHT, 1.)
        ev["opp"] = np.where(ev.team.eq(ev.home), ev.away, ev.home)
        self.events, self.player_games = ev, pg
        self._positions(cache_dir)
        prior = ev.loc[ev.season.eq(prior_season)]
        self._league_rates(prior)
        self._players(ev, pg)
        self._teams(ev)
        self._home_and_rest(prior, events)
        self._defenders(ev, prior_season, use_onfloor, cache_dir)
        self._rotation_templates(pg)
        self._win_probability(prior)
        self._crews(ev)
        self.duration_scale, self.home_shift, self.fg_shift, self.regression = 1., self.home_fg, 0., REGRESSION
        self.calibrate()

    def calibrate(self, n_games=120, sims=10, iterations=4, seed=7):
        """Match last season's league pace, points per game and home margin by tuning the possession-length
        scale, a league-wide make-rate shift and the home edge.
        """
        prior = self.player_games.loc[self.player_games.season.lt(self.current_season)]
        ids = pd.Series(prior.game_id.unique()).sample(n_games, random_state=seed)
        finals = self.events.loc[self.events.game_id.isin(ids)].groupby("game_id")[
            ["home_score", "away_score"]].last()
        all_finals = self.events.loc[self.events.season.lt(self.current_season)].groupby("game_id")[
            ["home_score", "away_score"]].last()
        target_points = (all_finals.home_score + all_finals.away_score).mean() / 2
        target_margin = (all_finals.home_score - all_finals.away_score).mean()
        target_pace = self.prior_pace
        setups = []
        for game_id in ids:
            rows = prior.loc[prior.game_id.eq(game_id)]
            home, away = rows.loc[rows.home], rows.loc[~rows.home]
            if len(home) < 7 or len(away) < 7:
                continue
            setups.append((home.team.iloc[0], away.team.iloc[0], home.player.tolist(), away.player.tolist(),
                           home.player[home.started].tolist(), away.player[away.started].tolist()))
        matchups = [Matchup(self, *s, pilot=False) for s in setups]
        for m in matchups:
            m.set_expected_margin(n=25)
        for _ in range(iterations):
            points, margins, possessions = [], [], []
            for m in matchups:
                m.duration_scale = self.duration_scale / ((m.teams[0].pace + m.teams[1].pace) / 2)
                m.home_shift = [self.home_shift, -self.home_shift]
                games, players = m.simulate(sims, seed)
                points.append((games.home_score + games.away_score).mean() / 2)
                margins.append((games.home_score - games.away_score).mean())
                minutes = players.groupby("sim").SEC.sum() / 60 / 10
                team = players.groupby(["sim", "side"])[["FGA", "OREB", "TOV", "FTA"]].sum()
                poss = (team.FGA - team.OREB + team.TOV + .44 * team.FTA).groupby("sim").mean()
                possessions.append((poss / minutes * 48).mean())
            pace, ppg, margin = np.mean(possessions), np.mean(points), np.mean(margins)
            self.duration_scale *= pace / target_pace
            self.fg_shift += (target_points - ppg * target_pace / pace) / 45.
            self.home_shift += (target_margin - margin) / 90.
        self.calibration = dict(target_pace=target_pace, sim_pace=pace, target_points=target_points, sim_points=ppg,
                                target_margin=target_margin, sim_margin=margin)

    def is_b2b(self, team, date):
        if date is None:
            return False
        games = self.events.loc[self.events.team.eq(team) & (self.events.date < date)]
        return bool(len(games)) and (pd.Timestamp(date) - games.date.max()).days == 1

    def _crews(self, ev):
        """Each official's fouls per game against the league, from this season's games before as_of; ESPN
        lists the crew.
        """
        self.official_effect = {}
        path = CACHE_DIR / "espn" / f"games_{self.current_season}.parquet"
        if not path.exists():
            return
        games = pd.read_parquet(path, columns=["event_id", "officials"])
        cur = ev.loc[ev.season.eq(self.current_season)]
        fouls = cur.loc[cur.kind.eq("foul") & cur.foul.isin(["shooting", "personal"])].groupby("game_id").size()
        if fouls.empty:
            return
        games = games.assign(game_id="E" + games.event_id.astype(str)).set_index("game_id").reindex(fouls.index)
        league = fouls.mean()
        rows = [(name.strip(), np.log(count / league)) for count, crew in zip(fouls, games.officials.fillna(""))
                for name in crew.split(";") if name.strip()]
        frame = pd.DataFrame(rows, columns=["name", "effect"])
        sums, counts = frame.groupby("name").effect.sum(), frame.groupby("name").size()
        self.official_effect = (sums / (counts + 20.)).to_dict()

    def crew_effect(self, officials):
        names = [n.strip() for n in (officials.split(";") if isinstance(officials, str) else officials) if n.strip()]
        return float(np.mean([self.official_effect.get(n, 0.) for n in names])) if names else 0.

    def _positions(self, cache_dir):
        box = pd.read_parquet(Path(cache_dir) / "espn" / f"box_{self.current_season}.parquet",
                              columns=["athlete_id", "name", "position", "team"])
        box = box.drop_duplicates("athlete_id", keep="last")
        keys = "e" + box.athlete_id.astype(str)
        self.group = dict(zip(keys, box.position.map(POSITION_GROUP).fillna(1).astype(int)))
        self.names = dict(zip(keys, box.name))

    def _league_rates(self, prior):
        fga = prior.loc[prior.kind.eq("fga") & prior.zone.ge(0)]
        self.fg = fga.groupby("zone").made.mean().reindex(range(5)).to_numpy()
        starts = fga.loc[fga.start.ge(0)]
        counts = pd.crosstab(starts.start, starts.zone).reindex(index=range(4), columns=range(5), fill_value=0)
        self.start_mix = (counts.to_numpy() / counts.to_numpy().sum(1, keepdims=True)) \
            / (counts.to_numpy().sum(0) / counts.to_numpy().sum())
        made = starts.pivot_table(index="start", columns="zone", values="made", aggfunc="sum").reindex(
            index=range(4), columns=range(5)).fillna(0).to_numpy()
        self.start_fg = _logit((made + 50 * self.fg) / (counts.to_numpy() + 50)) - _logit(self.fg)
        trips = prior.loc[prior.trip.eq("shot")]
        and_ones = prior.loc[prior.trip.eq("andone")].and_one_zone.value_counts().reindex(range(5), fill_value=0)
        attempts = fga.zone.value_counts().reindex(range(5), fill_value=0).to_numpy()
        trips3 = trips.ft_n.eq(3).sum()
        trips2 = len(trips) - trips3
        share = and_ones.to_numpy().astype(float)
        share2 = share[:3] / share[:3].sum()
        share3 = attempts[3:] / attempts[3:].sum()
        trip_zone = np.r_[trips2 * share2, trips3 * share3]
        self.shot_foul = (trip_zone + share) / (attempts + trip_zone)
        self.and_one = share / (share + trip_zone)
        made_fga = fga.loc[fga.made]
        self.assisted = made_fga.p2.notna().groupby(made_fga.zone).mean().reindex(range(5)).to_numpy()
        missed = fga.loc[~fga.made]
        self.blocked = missed.p3.notna().groupby(missed.zone).mean().reindex(range(5)).to_numpy()
        rebounds = prior.loc[prior.kind.isin(["oreb", "dreb", "fga", "ft"])]
        nxt = rebounds.groupby("game_id").kind.shift(-1)
        miss = rebounds.kind.eq("fga") & ~rebounds.made & nxt.isin(["oreb", "dreb"])
        self.oreb = nxt[miss].eq("oreb").groupby(rebounds.zone[miss]).mean().reindex(range(5)).to_numpy()
        ft_miss = rebounds.kind.eq("ft") & ~rebounds.made & rebounds.ft_i.eq(rebounds.ft_n) & nxt.isin(["oreb", "dreb"])
        self.ft_oreb = nxt[ft_miss].eq("oreb").mean()
        self.ft_pct = prior.loc[prior.kind.eq("ft")].made.mean()
        terminal = prior.loc[prior.start.ge(0) & prior.dur.between(0, 30)]
        outcome = np.where(terminal.kind.eq("tov"), "tov",
                           np.where(terminal.kind.eq("foul") & terminal.foul.eq("personal"), "nsf", "shot"))
        self.durations = [np.clip(terminal.dur[terminal.start.eq(s)].to_numpy(), .5, 24.) for s in range(4)]
        table = pd.crosstab(terminal.start, outcome, normalize="index").reindex(range(4))
        self.nsf_rate = table["nsf"].to_numpy()
        shot_or_tov = table["tov"] + table["shot"]
        self.start_tov = _logit((table["tov"] / shot_or_tov).to_numpy())
        self.tov_rate = (terminal.kind.eq("tov").sum()) / (outcome != "nsf").sum()
        self.start_tov = self.start_tov - _logit(self.tov_rate)
        self.steal_share = prior.loc[prior.kind.eq("tov")].p2.notna().mean()
        self.oreb_logit, self.ft_oreb_logit = [float(v) for v in _logit(self.oreb)], float(_logit(self.ft_oreb))
        self.nsf_logit = [float(v) for v in _logit(self.nsf_rate)]
        self.tov_logit = float(_logit(self.tov_rate))
        self.shot_foul_logit = [float(v) for v in _logit(self.shot_foul)]
        self.offensive_foul_share = .1

    def _players(self, ev, pg):
        fga = ev.loc[ev.kind.eq("fga") & ev.zone.ge(0) & ev.p1.notna()]
        att = fga.pivot_table(index="p1", columns="zone", values="w", aggfunc="sum").reindex(columns=range(5)).fillna(0)
        made = fga.loc[fga.made].pivot_table(index="p1", columns="zone", values="w", aggfunc="sum").reindex(
            index=att.index, columns=range(5)).fillna(0)
        trips = ev.loc[ev.trip.isin(["shot", "andone"])]
        fouled = trips.groupby("p1").w.sum()
        shot_trips = ev.loc[ev.trip.eq("shot")].groupby("p1").w.sum()
        tov = ev.loc[ev.kind.eq("tov") & ev.p1.notna()].groupby("p1").w.sum()
        ft = ev.loc[ev.kind.eq("ft") & ev.p1.notna()]
        fta, ftm = ft.groupby("p1").w.sum(), ft.loc[ft.made].groupby("p1").w.sum()
        count = lambda rows, column: rows.groupby(column).w.sum()
        ast = count(fga.loc[fga.made & fga.p2.notna()], "p2")
        blk = count(fga.loc[fga.p3.notna()], "p3")
        stl = count(ev.loc[ev.kind.eq("tov") & ev.p2.notna()], "p2")
        oreb = count(ev.loc[ev.kind.eq("oreb") & ev.p1.notna()], "p1")
        dreb = count(ev.loc[ev.kind.eq("dreb") & ev.p1.notna()], "p1")
        pf = count(ev.loc[ev.kind.eq("foul") & ev.foul.isin(["shooting", "personal", "offensive"]) & ev.p1.notna()], "p1")
        minutes = (pg.minutes * pg.w).groupby(pg.player).sum()
        keys = minutes.index.union(att.index)
        p = pd.DataFrame(index=keys)
        p["group"] = [self.group.get(k, 1) for k in keys]
        p["minutes"] = minutes.reindex(keys).fillna(0)
        A = att.reindex(keys).fillna(0).to_numpy()
        M = made.reindex(keys).fillna(0).to_numpy()
        g = p.group.to_numpy()
        group_mix = np.vstack([A[g == k].sum(0) / max(A[g == k].sum(), 1) for k in range(3)])
        group_fg = np.vstack([(M[g == k].sum(0) + 1) / (A[g == k].sum(0) + 2) for k in range(3)])
        shots = A.sum(1, keepdims=True)
        mix = (A + MIX_SHRINK * group_mix[g]) / (shots + MIX_SHRINK)
        K = np.array([FG_SHRINK[z] for z in ZONES])
        fg = (M + K * group_fg[g]) / (A + K)
        self.mix = mix
        self.fg_logit = _logit(fg)
        self.mix_cols, self.fg_cols = mix, fg
        p["fta"], p["ftm"] = fta.reindex(keys).fillna(0), ftm.reindex(keys).fillna(0)
        p["ft"] = (p.ftm + FT_SHRINK * self.ft_pct) / (p.fta + FT_SHRINK)
        attempts = shots[:, 0] + shot_trips.reindex(keys).fillna(0).to_numpy()
        fouled_rate = fouled.reindex(keys).fillna(0).to_numpy()
        expected_draw = mix @ self.shot_foul
        draw = (fouled_rate + DRAW_SHRINK * expected_draw) / (attempts + DRAW_SHRINK)
        p["draw"] = _logit(draw) - _logit(expected_draw)
        uses = attempts + tov.reindex(keys).fillna(0).to_numpy()
        tov_k = tov.reindex(keys).fillna(0).to_numpy()
        group_tov = np.array([tov_k[g == k].sum() / max(uses[g == k].sum(), 1) for k in range(3)])
        p["tov"] = _logit((tov_k + TOV_SHRINK * group_tov[g]) / (uses + TOV_SHRINK)) - _logit(self.tov_rate)
        per_minute = {"ast": ast, "blk": blk, "stl": stl, "oreb": oreb, "dreb": dreb, "pf": pf, "ftr": fta}
        mins = p.minutes.to_numpy()
        for name, series in per_minute.items():
            values = series.reindex(keys).fillna(0).to_numpy()
            group_rate = np.array([values[g == k].sum() / max(mins[g == k].sum(), 1) for k in range(3)])
            p[name] = (values + RATE_SHRINK_MIN * group_rate[g]) / (mins + RATE_SHRINK_MIN)
        p["usage"] = self._usage_share(ev, keys, g)
        self.players = p
        self.player_index = {k: i for i, k in enumerate(keys)}

    def _usage_share(self, ev, keys, g):
        """Share of his team's possession-ending plays (shots, shooting-foul trips, turnovers) a player
        uses while he is on the floor, shrunk toward his position.
        """
        uses = ev.loc[(ev.kind.eq("fga") & ev.zone.ge(0)) | ev.trip.eq("shot") | ev.kind.eq("tov")]
        lineup = pd.Series(np.where(uses.team.eq(uses.home), uses.home_on, uses.away_on), index=uses.index)
        lineup = lineup.fillna("").str.split("|")
        ok = lineup.map(len).eq(5)
        uses, lineup = uses.loc[ok], lineup[ok]
        weight = np.where(uses.season < self.current_season, USAGE_PRIOR_WEIGHT, 1.)
        on = pd.DataFrame({"player": lineup.to_numpy(), "w": weight}).explode("player")
        on_floor = on.groupby("player").w.sum().reindex(keys).fillna(0).to_numpy()
        own = pd.Series(weight, index=uses.index)[uses.p1.notna()].groupby(uses.p1[uses.p1.notna()]).sum()
        own = own.reindex(keys).fillna(0).to_numpy()
        group_share = np.array([own[g == k].sum() / max(on_floor[g == k].sum(), 1) for k in range(3)])
        return (own + USAGE_SHRINK * group_share[g]) / (on_floor + USAGE_SHRINK)

    def expected_make(self, shooters, zones, starts=None):
        """Each shooter's own make chance at that zone (logit), with the start effect."""
        idx = pd.Series(shooters).map(self.player_index)
        known = idx.notna().to_numpy()
        out = _logit(self.fg[np.asarray(zones)]).astype(float)
        rows = idx[known].astype(int).to_numpy()
        out[known] = self.fg_logit[rows, np.asarray(zones)[known]]
        if starts is not None:
            starts = np.asarray(starts)
            ok = starts >= 0
            out[ok] += self.start_fg[starts[ok], np.asarray(zones)[ok]]
        return out

    def _teams(self, ev):
        terminal = ev.loc[ev.start.ge(0) & ev.dur.between(0, 30)]
        terminal = terminal.assign(offense=np.where(terminal.kind.eq("foul"), terminal.opp, terminal.team),
                                   defense=np.where(terminal.kind.eq("foul"), terminal.team, terminal.opp))
        non_foul = terminal.loc[~(terminal.kind.eq("foul") & terminal.foul.eq("personal"))]
        base = _logit(self.tov_rate) + self.start_tov[non_foul.start.to_numpy()]
        self.off_tov, self.def_tov = _two_way(non_foul, non_foul.kind.eq("tov"), non_foul.offense, non_foul.defense,
                                              non_foul.tw, base, TEAM_SHRINK["tov"])
        nsf_base = _logit(self.nsf_rate[terminal.start.to_numpy()])
        is_nsf = terminal.kind.eq("foul") & terminal.foul.eq("personal")
        self.off_nsf, self.def_nsf = _two_way(terminal, is_nsf, terminal.offense, terminal.defense, terminal.tw,
                                              nsf_base, TEAM_SHRINK["nsf"])
        reb = ev.loc[ev.kind.isin(["oreb", "dreb", "fga", "ft"])]
        nxt = reb.groupby("game_id").kind.shift(-1)
        nxt_team = reb.groupby("game_id").team.shift(-1)
        miss = reb.kind.eq("fga") & ~reb.made & nxt.isin(["oreb", "dreb"])
        r = reb.loc[miss]
        oreb_base = _logit(self.oreb[r.zone.clip(0).to_numpy()])
        self.off_oreb, self.def_oreb = _two_way(r, nxt[miss].eq("oreb"), r.team, r.opp, r.tw, oreb_base,
                                                TEAM_SHRINK["oreb"])
        attempts = ev.loc[(ev.kind.eq("fga") & ev.zone.ge(0)) | ev.trip.eq("shot")]
        foul_flag = attempts.trip.eq("shot") | attempts.and_one
        defense = attempts.opp
        base = np.full(len(attempts), _logit(np.average(self.shot_foul, weights=np.bincount(
            attempts.zone.clip(0), minlength=5)[:5] + 1)))
        _, self.def_shot_foul = _two_way(attempts, foul_flag, attempts.team, defense, attempts.tw, base,
                                         TEAM_SHRINK["shot_foul"])
        fga = ev.loc[ev.kind.eq("fga") & ev.zone.ge(0)]
        expected = _expit(self.expected_make(fga.p1, fga.zone, fga.start))
        resid = (fga.made - expected) * fga.tw
        var = expected * (1 - expected) * fga.tw
        num, den = resid.groupby(fga.opp).sum(), var.groupby(fga.opp).sum()
        self.def_fg = (num / (den + TEAM_SHRINK["fg"] * .24)).to_dict()
        idx = fga.p1.map(self.player_index)
        exp_mix = np.where(idx.notna().to_numpy()[:, None], self.mix[idx.fillna(0).astype(int).to_numpy()],
                           self.mix.mean(0))
        exp_mix = pd.DataFrame(exp_mix * fga.tw.to_numpy()[:, None]).groupby(fga.opp.to_numpy()).sum()
        obs_mix = pd.crosstab(fga.opp, fga.zone, values=fga.tw, aggfunc="sum").reindex(
            index=exp_mix.index, columns=range(5)).fillna(0)
        league_share = exp_mix.sum(0).to_numpy() / exp_mix.to_numpy().sum()
        k = TEAM_SHRINK["mix"] * league_share
        self.def_mix = {team: (obs_mix.loc[team].to_numpy() + k) / (exp_mix.loc[team].to_numpy() + k)
                        for team in exp_mix.index}
        rows = ev.loc[ev.kind.isin(["fga", "tov", "oreb"]) | ev.kind.eq("ft")]
        poss = (rows.kind.eq("fga").astype(float) + rows.kind.eq("tov") - rows.kind.eq("oreb")
                + .44 * rows.kind.eq("ft"))
        game_poss = poss.groupby([rows.game_id, rows.team]).sum().groupby(level=0).mean()
        game_len = ev.groupby("game_id").t.max().clip(lower=REGULATION) / 60
        per48 = (game_poss / game_len.reindex(game_poss.index) * 48)
        weights = ev.groupby("game_id").tw.first().reindex(per48.index)
        teams = ev.groupby("game_id")[["home", "away"]].first().reindex(per48.index)
        long = pd.DataFrame({"team": np.r_[teams.home, teams.away], "pace": np.r_[per48, per48],
                             "w": np.r_[weights, weights]})
        self.league_pace = np.average(long.pace, weights=long.w)
        prior_games = ev.groupby("game_id").season.first().reindex(per48.index).lt(self.current_season)
        self.prior_pace = per48[prior_games.to_numpy()].mean()
        sums = (long.pace * long.w).groupby(long.team).sum()
        ws = long.w.groupby(long.team).sum()
        shrink_games = 15.
        self.pace = ((sums + shrink_games * self.league_pace) / (ws + shrink_games) / self.league_pace).to_dict()

    def _home_and_rest(self, prior, all_events):
        fga = prior.loc[prior.kind.eq("fga") & prior.zone.ge(0)]
        expected = _expit(self.expected_make(fga.p1, fga.zone, fga.start))
        resid = fga.made - expected
        home = fga.team.eq(fga.home)
        pbar = expected.mean()
        self.home_fg = (resid[home].mean() - resid[~home].mean()) / 2 / (pbar * (1 - pbar))
        tov = prior.loc[prior.kind.eq("tov")]
        self.home_tov = .5 * (_logit(1 - tov.team.eq(tov.home).mean()) - _logit(.5))
        games = all_events.groupby("game_id")[["date", "home", "away"]].first()
        long = pd.concat([games[["date", "home"]].rename(columns={"home": "team"}),
                          games[["date", "away"]].rename(columns={"away": "team"})]).reset_index()
        long = long.sort_values(["team", "date"])
        long["b2b"] = long.groupby("team").date.diff().dt.days.eq(1)
        self.b2b_games = set(zip(long.game_id[long.b2b], long.team[long.b2b]))
        flag = pd.Series(list(zip(fga.game_id, fga.team))).isin(self.b2b_games).to_numpy()
        self.b2b_fg = (resid[flag].mean() - resid[~flag].mean()) / (pbar * (1 - pbar)) if flag.any() else 0.

    def _defenders(self, ev, prior_season, use_onfloor, cache_dir):
        id_map_path = Path(cache_dir) / "sim" / f"id_map_{prior_season}.csv"
        id_map = pd.read_csv(id_map_path, index_col=0).iloc[:, 0].to_dict() if id_map_path.exists() else {}
        key = lambda ids: ids.map(lambda i: id_map.get(int(i), f"n{int(i)}"))
        m = nba_data.load_matchups(prior_season, cache_dir=cache_dir)
        m = m.assign(off=key(m.offense_id), de=key(m.defender_id))
        prior = ev.loc[ev.season.eq(prior_season) & ev.kind.eq("fga")]
        two = prior.loc[prior.value.eq(2)].groupby("p1").made.agg(["sum", "size"])
        three = prior.loc[prior.value.eq(3)].groupby("p1").made.agg(["sum", "size"])
        p2 = ((two["sum"] + 50 * .54) / (two["size"] + 50)).to_dict()
        p3 = ((three["sum"] + 80 * .36) / (three["size"] + 80)).to_dict()
        fga3 = m.matchup_three_pointers_attempted
        fga2 = m.matchup_field_goals_attempted - fga3
        expected = fga2 * m.off.map(p2).fillna(.54) + fga3 * m.off.map(p3).fillna(.36)
        sums = pd.DataFrame({"made": m.matchup_field_goals_made, "expected": expected,
                             "fga": m.matchup_field_goals_attempted}).groupby(m.de).sum()
        delta = (sums.made - sums.expected) / (sums.fga + DEFENDER_SHRINK)
        pbar = .47
        rating = (delta / (pbar * (1 - pbar))).to_dict()
        self.defender_source = {k: "matchups" for k in rating}
        groups = m.off.map(lambda k: self.group.get(k, np.nan))
        guarded = pd.crosstab(m.de, groups, values=m.partial_possessions, aggfunc="sum").reindex(
            columns=[0, 1, 2]).fillna(0)
        self.assign_raw = guarded
        if use_onfloor:
            onfloor = self._onfloor_ratings(ev)
            vets = [k for k in onfloor if k in rating]
            if vets:
                b = np.array([onfloor[k] for k in vets])
                d = np.array([rating[k] for k in vets])
                self.onfloor_scale = float(np.clip((b * d).sum() / max((b * b).sum(), 1e-9), 0, 5))
            else:
                self.onfloor_scale = 0.
            for k, b in onfloor.items():
                if k not in rating:
                    rating[k] = self.onfloor_scale * b
                    self.defender_source[k] = "on-floor"
            self.onfloor = onfloor
        self.defender = rating

    def _onfloor_ratings(self, ev):
        """Ridge regression of this season's make residuals on the five defenders on the floor. Effects are
        in logit units per shot while the player is on.
        """
        cur = ev.loc[ev.season.eq(self.current_season) & ev.kind.eq("fga") & ev.zone.ge(0)]
        if cur.empty:
            return {}
        defense_on = np.where(cur.team.eq(cur.home), cur.away_on, cur.home_on)
        lineups = pd.Series(defense_on).fillna("").str.split("|")
        ok = lineups.map(len).eq(5).to_numpy()
        cur, lineups = cur.loc[ok], lineups[ok]
        expected = _expit(self.expected_make(cur.p1, cur.zone, cur.start))
        resid = (cur.made.to_numpy() - expected) / (expected * (1 - expected)).clip(.15)
        weights = expected * (1 - expected)
        players = sorted(set(p for lineup in lineups for p in lineup))
        index = {p: i for i, p in enumerate(players)}
        X = np.zeros((len(cur), len(players)))
        for r, lineup in enumerate(lineups):
            for p in lineup:
                X[r, index[p]] = 1.
        XtW = X.T * weights
        beta = np.linalg.solve(XtW @ X + ONFLOOR_RIDGE * .24 * np.eye(len(players)), XtW @ resid)
        return dict(zip(players, beta))

    def assignment(self, key):
        """Share of a defender's possessions spent on guards, forwards and centers."""
        own = self.group.get(key, 1)
        default = np.array([.15, .15, .15])
        default[own] = .7
        if own == 1:
            default = np.array([.25, .55, .2])
        counts = self.assign_raw.loc[key].to_numpy() if key in self.assign_raw.index else np.zeros(3)
        return (counts + ASSIGN_SHRINK * default) / (counts.sum() + ASSIGN_SHRINK)

    def _rotation_templates(self, pg):
        prior = pg.loc[pg.season.lt(self.current_season)]
        occ = np.vstack(prior.occ.to_numpy())
        started = prior.started.to_numpy()
        self.starter_occ = occ[started].mean(0)
        self.bench_occ = occ[~started].mean(0)

    def expected_minutes(self, key, started=None):
        """Minutes per game he plays: this season's games weighted by recency, plus last season's average
        counted as a few games. A default if he has neither.
        """
        pg = self.player_games
        rows = pg.loc[pg.player.eq(key)]
        cur = rows.loc[rows.season.eq(self.current_season)].sort_values("date")
        prior = rows.loc[rows.season.lt(self.current_season)]
        total, weight = 0., 0.
        if len(cur):
            w = .5 ** (np.arange(len(cur))[::-1] / OCC_HALF_LIFE_GAMES)
            total += float((cur.minutes.to_numpy() * w).sum())
            weight += w.sum()
        if len(prior):
            total += prior.minutes.mean() * OCC_PRIOR_GAMES
            weight += OCC_PRIOR_GAMES
        if not weight:
            return 28. if started else 10.
        return total / weight

    def occupancy(self, key, started=None):
        """Chance he is on the floor in each regulation minute: recent games this season weighted by
        recency, plus last season's average as a few games' worth.
        """
        pg = self.player_games
        rows = pg.loc[pg.player.eq(key)]
        cur = rows.loc[rows.season.eq(self.current_season)].sort_values("date")
        prior = rows.loc[rows.season.lt(self.current_season)]
        total, weight = np.zeros(MINUTE_BINS), 0.
        if len(cur):
            w = .5 ** (np.arange(len(cur))[::-1] / OCC_HALF_LIFE_GAMES)
            total += (np.vstack(cur.occ.to_numpy()) * w[:, None]).sum(0)
            weight += w.sum()
        if len(prior):
            total += np.vstack(prior.occ.to_numpy()).mean(0) * OCC_PRIOR_GAMES
            weight += OCC_PRIOR_GAMES
        template = self.starter_occ if started else self.bench_occ
        total += template * 1.
        weight += 1.
        return total / weight

    def _win_probability(self, prior):
        from sklearn.linear_model import LogisticRegression
        rows = prior.loc[prior.kind.isin(["fga", "tov", "dreb"])]
        final = prior.groupby("game_id")[["home_score", "away_score"]].last()
        won = (final.home_score > final.away_score).reindex(rows.game_id).to_numpy()
        left = np.clip(REGULATION - rows.t.to_numpy(), 0, None) / 60
        diff = (rows.home_score - rows.away_score).to_numpy()
        X = np.c_[diff / np.sqrt(left + 1), diff]
        self.wp_model = LogisticRegression(C=10.).fit(X, won)

    def live_wp(self, diff, seconds_left, pregame_logit=0.):
        """Home win chance from the score and clock, leaning on the pregame odds early."""
        left = max(seconds_left, 0) / 60
        coef = self.wp_model.coef_[0]
        logit = coef[0] * diff / np.sqrt(left + 1) + coef[1] * diff + self.wp_model.intercept_[0]
        logit += pregame_logit * min(left / 48, 1)
        if seconds_left <= 0 and diff != 0:
            return 1. if diff > 0 else 0.
        return float(_expit(logit))


STATS = ["SEC", "PTS", "FGM", "FGA", "3PM", "3PA", "FTM", "FTA", "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF"]
ST = {name: i for i, name in enumerate(STATS)}
_FOUL_TROUBLE = {1: 2, 2: 3, 3: 4, 4: 5}
INTENTIONAL_FOUL = [[.59, .47, .07, .01], [.72, .57, .49, .10], [.24, .36, .55, .19],
                    [0., 0., .07, .20], [0., .02, .06, .06]]


def minute_targets(expected, total=240.):
    """Fit expected minutes to the 240 a team plays."""
    expected = np.clip(np.asarray(expected, float), 0, 46)
    excess = expected.sum() - total
    if excess <= 0:
        return expected * total / max(expected.sum(), 1e-9)
    targets = expected.copy()
    for _ in range(20):
        weight = targets * np.clip(1 - targets / 42, 0, None) ** 2
        if weight.sum() <= 0 or excess <= 1e-6:
            break
        cut = np.minimum(targets, excess * weight / weight.sum())
        targets -= cut
        excess -= cut.sum()
    return targets * total / targets.sum()


def _intentional_foul_rate(lead, left):
    row = min((lead - 1) // 3, 4)
    column = 0 if left <= 12 else 1 if left <= 24 else 2 if left <= 36 else 3
    return INTENTIONAL_FOUL[row][column]


def _choice(rng, weights):
    total = 0.
    for w in weights:
        total += w
    r = rng.random() * total
    for i, w in enumerate(weights):
        r -= w
        if r <= 0:
            return i
    return len(weights) - 1


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def _lg(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


class _Team:
    """One side's players and team effects, as plain lists for the possession loop."""

    def __init__(self, league, team, players, starters, use_defender, opponent=None):
        L = league
        self.team, self.keys = team, list(players)
        self.starters = [p for p in (starters or []) if p in self.keys]
        frame = L.players.reindex(self.keys)
        weights = L.players.minutes.clip(lower=1)
        group_means = L.players.groupby("group").apply(
            lambda g: g.drop(columns="group").mul(weights[g.index], axis=0).sum() / weights[g.index].sum())
        self.names = [L.names.get(k, k) for k in self.keys]
        self.group = [L.group.get(k, 1) for k in self.keys]
        mix_mean = np.vstack([L.mix[(L.players.group == g).to_numpy()].mean(0) for g in range(3)])
        fg_mean = np.vstack([L.fg_logit[(L.players.group == g).to_numpy()].mean(0) for g in range(3)])
        self.mix, self.fgl = [], []
        for k, g in zip(self.keys, self.group):
            i = L.player_index.get(k)
            self.mix.append(list(L.mix[i] if i is not None else mix_mean[g]))
            self.fgl.append(list(L.fg_logit[i] if i is not None else fg_mean[g]))
        for column in ("ft", "draw", "tov", "usage", "ast", "oreb", "dreb", "stl", "blk", "pf", "ftr"):
            values = []
            for k, g, (_, row) in zip(self.keys, self.group, frame.iterrows()):
                value = row[column]
                values.append(float(group_means.loc[g, column] if pd.isna(value) else value))
            setattr(self, column, values)
        self.occ = [list(L.occupancy(k, started=k in self.starters)) for k in self.keys]
        self.occ_logit = [[_lg(min(max(o, .02), .98)) for o in occ] for occ in self.occ]
        self.offset = [0.] * len(self.keys)
        self.dlog = [float(L.defender.get(k, 0.)) for k in self.keys]
        self.assign = [list(L.assignment(k)) for k in self.keys]
        minutes = np.array([sum(o) for o in self.occ])
        self.def_avg = float(np.dot(minutes, self.dlog) / minutes.sum()) if minutes.sum() else 0.
        self.use_defender = use_defender
        self.tov_off, self.tov_def = L.off_tov.get(team, 0.), L.def_tov.get(team, 0.)
        self.nsf_off, self.nsf_def = L.off_nsf.get(team, 0.), L.def_nsf.get(team, 0.)
        self.oreb_off, self.oreb_def = L.off_oreb.get(team, 0.), L.def_oreb.get(team, 0.)
        self.shot_foul_def = L.def_shot_foul.get(team, 0.)
        self.fg_def = L.def_fg.get(team, 0.)
        self.mix_def = list(L.def_mix.get(team, np.ones(5)))
        self.pace = L.pace.get(team, 1.)


class _Game:
    def __init__(self, matchup, rng, record=False):
        self.m, self.rng, self.record = matchup, rng, record
        self.L = matchup.league
        self.teams = matchup.teams
        self.stats = [[[0.] * len(STATS) for _ in t.keys] for t in self.teams]
        self.score = [0, 0]
        self.t, self.period = 0., 1
        self.lineup = [[], []]
        self.last_sub = [0., 0.]
        self.team_fouls, self.late_fouls = [0, 0], [0, 0]
        self.log = []
        self.form = [rng.gauss(0, GAME_FORM_SD), rng.gauss(0, GAME_FORM_SD)]
        self.off, self.start = 0, S["inbound"]
        self.first_scorer = None

    @property
    def end(self):
        return period_end(self.period)

    def advance(self, dt):
        if dt <= 0:
            return
        for side in (0, 1):
            stats = self.stats[side]
            for i in self.lineup[side]:
                stats[i][0] += dt
        self.t += dt

    def add(self, side, i, stat, value=1):
        self.stats[side][i][ST[stat]] += value

    def note(self, side, text, kind="play", actor=None, zone=None, defender=None):
        if not self.record:
            return
        left = self.end - self.t
        diff = self.score[0] - self.score[1]
        game_left = max(REGULATION - self.t, 0) if self.period <= 4 else left
        row = dict(period=self.period, clock=f"{int(left // 60)}:{int(left % 60):02d}", t=round(self.t, 1),
                   side=side, team=self.teams[side].team if side is not None else "", text=text, kind=kind,
                   home_score=self.score[0], away_score=self.score[1],
                   home_wp=round(self.L.live_wp(diff, game_left, self.m.pregame_logit), 4), actor=actor,
                   zone=ZONES[zone] if zone is not None else None, defender=defender,
                   on=[list(self.lineup[0]), list(self.lineup[1])], fouls=list(self.team_fouls),
                   box=[[[int(round(r[0])), *(int(v) for v in r[1:8]), int(r[8] + r[9]), *(int(v) for v in r[10:])]
                         for r in self.stats[k]] for k in (0, 1)])
        if zone is not None:
            row["spot"] = _shot_spot(zone, random.Random(len(self.log) * 7919 + self.score[0] * 31 + self.score[1]))
        self.log.append(row)

    def name(self, side, i):
        return self.teams[side].names[i]

    def choose_lineup(self, side, use_starters=False):
        team, rng = self.teams[side], self.rng
        stats = self.stats[side]
        alive = [i for i in range(len(team.keys)) if stats[i][ST["PF"]] < FOUL_OUT]
        if len(alive) < 5:
            alive = sorted(range(len(team.keys)), key=lambda i: stats[i][ST["PF"]])[:max(5, len(alive))]
        if use_starters and len(team.starters) == 5:
            picked = [team.keys.index(k) for k in team.starters if team.keys.index(k) in alive]
            if len(picked) == 5:
                self.lineup[side] = picked
                return
        minute = min(int(self.t // 60), MINUTE_BINS - 1)
        left = self.end - self.t
        margin = abs(self.score[0] - self.score[1])
        garbage = self.period >= 4 and margin >= GARBAGE_MARGIN and left <= GARBAGE_SECONDS
        if garbage:
            core = sorted(alive, key=lambda i: -sum(team.occ[i]))[:7]
        scores = {}
        for i in alive:
            score = team.occ_logit[i][minute] + team.offset[i] - math.log(-math.log(rng.random() + 1e-12))
            limit = _FOUL_TROUBLE.get(self.period)
            if limit and stats[i][ST["PF"]] >= limit and not (self.period == 4 and left < 300):
                score -= 2.5
            if garbage and i in core:
                score -= 4.
            scores[i] = score
        order = sorted(alive, key=lambda i: -scores[i])
        lineup = order[:5]
        bench = order[5:]
        if not any(team.group[i] == 0 for i in lineup):
            guards = [i for i in bench if team.group[i] == 0]
            if guards:
                drop = min((i for i in lineup if team.group[i] != 0), key=lambda i: scores[i])
                lineup[lineup.index(drop)] = guards[0]
        centers = [i for i in lineup if team.group[i] == 2]
        if len(centers) > 2:
            others = [i for i in bench if team.group[i] != 2]
            if others:
                drop = min(centers, key=lambda i: scores[i])
                lineup[lineup.index(drop)] = others[0]
        self.lineup[side] = lineup

    def dead_ball(self, chance=1.):
        for side in (0, 1):
            if self.t - self.last_sub[side] >= SUB_INTERVAL and self.rng.random() < chance:
                self.choose_lineup(side)
                self.last_sub[side] = self.t

    def personal_foul(self, side, i, shooting=False):
        """Charge a defensive foul; returns whether it sends the offense to the line."""
        self.add(side, i, "PF")
        in_bonus = self.team_fouls[side] >= 4 or (self.end - self.t <= 120 and self.late_fouls[side] >= 1)
        self.team_fouls[side] += 1
        if self.end - self.t <= 120:
            self.late_fouls[side] += 1
        if self.stats[side][i][ST["PF"]] >= FOUL_OUT:
            self.note(side, f"{self.name(side, i)} fouls out", "foul", actor=i)
            self.choose_lineup(side)
            self.last_sub[side] = self.t
        return in_bonus or shooting

    def free_throws(self, side, i, n):
        made = 0
        for k in range(n):
            self.add(side, i, "FTA")
            if self.rng.random() < self.teams[side].ft[i]:
                self.add(side, i, "FTM")
                self.add(side, i, "PTS")
                if self.first_scorer is None:
                    self.first_scorer = (side, i, "FT")
                self.score[side] += 1
                made += 1
                last_made = True
            else:
                last_made = False
        self.note(side, f"{self.name(side, i)} makes {made} of {n} free throws", "ft", actor=i)
        self.dead_ball()
        if last_made:
            self.off, self.start = 1 - side, S["inbound"]
        else:
            self.rebound(side, None)

    def rebound(self, side, zone):
        O, D = self.teams[side], self.teams[1 - side]
        base = self.L.oreb_logit[zone] if zone is not None else self.L.ft_oreb_logit
        p = _sigmoid(base + O.oreb_off + D.oreb_def)
        if self.rng.random() < p:
            on = self.lineup[side]
            i = on[_choice(self.rng, [O.oreb[j] for j in on])]
            self.add(side, i, "OREB")
            self.off, self.start = side, S["oreb"]
        else:
            on = self.lineup[1 - side]
            i = on[_choice(self.rng, [D.dreb[j] for j in on])]
            self.add(1 - side, i, "DREB")
            self.off, self.start = 1 - side, S["drb"]

    def possession(self):
        rng, L = self.rng, self.L
        side = self.off
        O, D = self.teams[side], self.teams[1 - side]
        on, don = self.lineup[side], self.lineup[1 - side]
        start = self.start
        left = self.end - self.t
        margin = self.score[side] - self.score[1 - side]
        late = self.period >= 4
        if late and 1 <= margin <= 15 and left <= 60 and rng.random() < _intentional_foul_rate(margin, left):
            self.advance(min(left, 1 + 3 * rng.random()))
            fouler = don[int(rng.random() * 5) % len(don)]
            self.note(1 - side, f"{self.name(1 - side, fouler)} fouls on purpose", "foul", actor=fouler)
            if self.personal_foul(1 - side, fouler):
                best = max(on, key=lambda j: O.ft[j])
                shooter = best if rng.random() < .6 else on[_choice(rng, [O.ftr[j] for j in on])]
                self.free_throws(side, shooter, 2)
            else:
                self.start = S["inbound"]
            return
        pool = L.durations[start]
        duration = pool[int(rng.random() * len(pool))] * self.m.duration_scale
        if self.period >= 3:
            capped = min(abs(margin), 10)
            duration *= 1 + LEAD_PACE * capped if margin > 0 else 1 - TRAIL_PACE * capped
        if late and left <= 120:
            if margin > 0:
                duration = max(duration, min(22., left))
            elif margin < 0:
                duration *= .6
        if late and left <= 24 and -3 <= margin <= 0:
            duration = max(left - .5 - 2 * rng.random(), .3)
        rushed = False
        if duration >= left:
            if left < .8:
                self.advance(left)
                return
            duration, rushed = left - .3, left < 3.5
        self.advance(duration)
        left = self.end - self.t
        if not rushed and rng.random() < _sigmoid(L.nsf_logit[start] + D.nsf_def + O.nsf_off + self.m.ref_foul):
            fouler = don[_choice(rng, [D.pf[j] for j in don])]
            self.note(1 - side, f"{self.name(1 - side, fouler)} personal foul", "foul", actor=fouler)
            if self.personal_foul(1 - side, fouler):
                shooter = on[_choice(rng, [O.ftr[j] for j in on])]
                self.free_throws(side, shooter, 2)
            else:
                self.start = S["inbound"]
                self.dead_ball()
            return
        user = on[_choice(rng, [O.usage[j] for j in on])]
        home_sign = 1 if side == 0 else -1
        p_tov = _sigmoid(L.tov_logit + L.start_tov[start] + O.tov[user] + O.tov_off + D.tov_def
                         - home_sign * L.home_tov)
        if not rushed and rng.random() < p_tov:
            self.add(side, user, "TOV")
            if rng.random() < L.steal_share:
                stealer = don[_choice(rng, [D.stl[j] for j in don])]
                self.add(1 - side, stealer, "STL")
                self.note(side, f"{self.name(side, user)} turnover, stolen by {self.name(1 - side, stealer)}", "tov",
                          actor=user)
                self.off, self.start = 1 - side, S["steal"]
            else:
                if rng.random() < L.offensive_foul_share:
                    self.add(side, user, "PF")
                self.note(side, f"{self.name(side, user)} turnover", "tov", actor=user)
                self.off, self.start = 1 - side, S["inbound"]
                self.dead_ball()
            return
        self.shot(side, user, start, rushed, left, margin)

    def shot(self, side, user, start, rushed, left, margin):
        rng, L = self.rng, self.L
        O, D = self.teams[side], self.teams[1 - side]
        on, don = self.lineup[side], self.lineup[1 - side]
        mix = O.mix[user]
        weights = [mix[z] * D.mix_def[z] * L.start_mix[start][z] for z in range(5)]
        need_three = self.period >= 4 and left <= 30 and margin <= -3 and margin >= -8
        if need_three:
            weights = [w * (.15 if z < 3 else 1.) for z, w in enumerate(weights)]
        heave = rushed and left < 2.5 and start in (S["inbound"], S["drb"])
        zone = Z["AB3"] if heave else _choice(rng, weights)
        g = O.group[user]
        defender = don[_choice(rng, [D.assign[j][g] for j in don])]
        logit = O.fgl[user][zone] + L.start_fg[start][zone] + D.fg_def + self.form[side] + self.m.home_shift[side]
        expected = self.m.expected_margin * min(self.t, REGULATION) / REGULATION
        ahead = margin - (expected if side == 0 else -expected)
        logit += L.fg_shift - L.regression * max(-20, min(20, ahead)) / 10
        logit += self.m.b2b[side]
        if O.use_defender:
            logit += D.dlog[defender] - D.def_avg
        if heave:
            logit = -3.18
        elif rushed:
            logit -= .4
        p = _sigmoid(logit)
        value = ZONE_VALUE[zone]
        guard_name = self.name(1 - side, defender)
        if not heave and rng.random() < _sigmoid(L.shot_foul_logit[zone] + O.draw[user] + D.shot_foul_def
                                                 + self.m.ref_foul):
            fouler = defender if rng.random() < .7 else don[_choice(rng, [D.pf[j] for j in don])]
            self.personal_foul(1 - side, fouler, shooting=True)
            and_one = rng.random() < L.and_one[zone] * p / L.fg[zone]
            if and_one:
                self.add(side, user, "FGA")
                if value == 3:
                    self.add(side, user, "3PA")
                self.made_shot(side, user, zone, value, guard_name, and_one=True)
                self.free_throws(side, user, 1)
            else:
                self.note(side, f"{self.name(side, user)} fouled by {self.name(1 - side, fouler)} on "
                                f"{ZONE_LABEL[zone]}", "shooting_foul", actor=user, zone=zone, defender=guard_name)
                self.free_throws(side, user, int(value))
            return
        self.add(side, user, "FGA")
        if value == 3:
            self.add(side, user, "3PA")
        if rng.random() < p:
            self.made_shot(side, user, zone, value, guard_name)
            self.off, self.start = 1 - side, S["inbound"]
            self.dead_ball(.25)
        else:
            text = f"{self.name(side, user)} misses {ZONE_LABEL[zone]} ({guard_name} guarding)"
            if rng.random() < L.blocked[zone]:
                blocker = defender if rng.random() < .5 else don[_choice(rng, [D.blk[j] for j in don])]
                self.add(1 - side, blocker, "BLK")
                text += f", blocked by {self.name(1 - side, blocker)}"
            self.note(side, text, "miss", actor=user, zone=zone, defender=guard_name)
            self.rebound(side, zone)

    def made_shot(self, side, user, zone, value, guard_name, and_one=False):
        O = self.teams[side]
        self.add(side, user, "FGM")
        if value == 3:
            self.add(side, user, "3PM")
        self.add(side, user, "PTS", value)
        if self.first_scorer is None:
            self.first_scorer = (side, user, f"{value}PT")
        self.score[side] += value
        text = f"{self.name(side, user)} makes {ZONE_LABEL[zone]} ({guard_name} guarding)"
        mates = [j for j in self.lineup[side] if j != user]
        if mates and self.rng.random() < self.L.assisted[zone]:
            helper = mates[_choice(self.rng, [O.ast[j] for j in mates])]
            self.add(side, helper, "AST")
            text += f", assist {self.name(side, helper)}"
        self.note(side, text + (" and one" if and_one else ""), "make", actor=user, zone=zone, defender=guard_name)

    def run(self):
        rng = self.rng
        tip = 0 if rng.random() < .5 else 1
        while True:
            self.t = float(period_start(self.period))
            self.team_fouls, self.late_fouls = [0, 0], [0, 0]
            for side in (0, 1):
                self.choose_lineup(side, use_starters=self.period in (1, 3))
                self.last_sub[side] = self.t
            if self.period == 1 or self.period >= 5:
                self.off = tip if self.period == 1 else (0 if rng.random() < .5 else 1)
            else:
                self.off = 1 - tip if self.period in (2, 3) else tip
            self.start = S["inbound"]
            self.note(None, f"Start of {'Q' + str(self.period) if self.period <= 4 else 'OT' + str(self.period - 4)}",
                      "period")
            while self.end - self.t > .05:
                self.possession()
            self.advance(self.end - self.t)
            self.t = float(self.end)
            if self.period >= 4 and self.score[0] != self.score[1]:
                break
            self.period += 1
        self.note(None, "Final", "final")
        return self


class Matchup:
    """One game's two sides, ready to simulate many times."""

    def __init__(self, league, home, away, home_players, away_players, home_starters=None, away_starters=None,
                 date=None, use_defender=False, officials=(), pilot=True):
        self.league, self.home, self.away = league, home, away
        self.use_defender = use_defender
        self.teams = [_Team(league, home, home_players, home_starters, use_defender),
                      _Team(league, away, away_players, away_starters, use_defender)]
        pace = (self.teams[0].pace + self.teams[1].pace) / 2
        self.duration_scale = league.duration_scale / pace
        self.home_shift = [league.home_shift, -league.home_shift]
        self.b2b = [league.b2b_fg if league.is_b2b(team, date) else 0. for team in (home, away)]
        self.ref_foul = league.crew_effect(officials)
        self.pregame_logit = 0.
        self.expected_margin = 0.
        if pilot:
            self.calibrate_minutes()
            self.set_expected_margin()

    def calibrate_minutes(self, n=MINUTE_SIMS, iterations=None, seed=77):
        """Nudge each player's rotation weight so his simulated minutes match his expected minutes."""
        iterations = MINUTE_ITERATIONS if iterations is None else iterations
        if not iterations:
            return
        targets = [minute_targets([self.league.expected_minutes(k, k in t.starters) for k in t.keys])
                   for t in self.teams]
        for it in range(iterations):
            seconds = [np.zeros(len(t.keys)) for t in self.teams]
            for k in range(n):
                game = self.play([seed, it, k])
                for side in (0, 1):
                    seconds[side] += [row[0] for row in game.stats[side]]
            for side, team in enumerate(self.teams):
                simulated = seconds[side] / n / 60
                step = np.log((targets[side] + 1.) / (simulated + 1.)) * (1 + targets[side] / 12)
                team.offset = list(np.clip(np.array(team.offset) + step, -4, 4))

    def set_expected_margin(self, n=PILOT_SIMS, seed=99):
        """Pilot sims without the regression pull give the game's expected margin."""
        regression, self.league.regression = self.league.regression, 0.
        try:
            margins = [g.score[0] - g.score[1] for g in (self.play([seed, k]) for k in range(n))]
        finally:
            self.league.regression = regression
        self.expected_margin = float(np.mean(margins))

    def play(self, seed, record=False):
        state = np.random.SeedSequence(seed).generate_state(2)
        return _Game(self, random.Random(int(state[0]) << 32 | int(state[1])), record).run()

    def simulate(self, n, seed=0):
        games, lines = [], []
        for k in range(n):
            g = self.play([seed, k])
            side, i, how = g.first_scorer
            games.append((k, g.score[0], g.score[1], g.period, side, self.teams[side].keys[i], how))
            for side in (0, 1):
                team = self.teams[side]
                for i, key in enumerate(team.keys):
                    row = g.stats[side][i]
                    if row[0] > 0:
                        lines.append((k, side, key, *row))
        games = pd.DataFrame(games, columns=["sim", "home_score", "away_score", "periods", "first_side",
                                             "first_scorer", "first_how"])
        players = pd.DataFrame(lines, columns=["sim", "side", "player"] + STATS)
        players["MIN"] = players.SEC / 60
        return games, players


class SimulationResult:
    def __init__(self, matchup, games, players, seed):
        self.matchup, self.games, self.players, self.seed = matchup, games, players, seed
        names = {k: n for t in matchup.teams for k, n in zip(t.keys, t.names)}
        self.players["name"] = self.players.player.map(names)
        self.players["team"] = np.where(self.players.side.eq(0), matchup.home, matchup.away)

    @property
    def home_win_rate(self):
        return float((self.games.home_score > self.games.away_score).mean())

    def summary(self):
        g, m = self.games, self.matchup
        return pd.Series({
            f"{m.home} win %": self.home_win_rate, "overtime %": (g.periods > 4).mean(),
            f"{m.away} median points": g.away_score.median(), f"{m.home} median points": g.home_score.median(),
            "median total": (g.home_score + g.away_score).median(),
            f"median margin ({m.home})": (g.home_score - g.away_score).median(), "simulations": len(g)})

    def player_averages(self):
        stats = ["MIN", "PTS", "FGM", "FGA", "3PM", "3PA", "FTM", "FTA", "OREB", "DREB", "AST", "STL", "BLK", "TOV", "PF"]
        n = len(self.games)
        table = self.players.groupby(["team", "name"])[stats].sum() / n
        table["REB"] = table.OREB + table.DREB
        return table.sort_values(["team", "MIN"], ascending=[True, False]).round(1)

    def typical_sim(self):
        """The simulation closest to the median result: both scores plus each side's three biggest scorers'
        points, measured in standard deviations.
        """
        g = self.games.set_index("sim")
        features = g[["home_score", "away_score"]].astype(float)
        features["margin"] = 2. * (g.home_score - g.away_score)
        means = self.players.groupby("player").PTS.sum()
        for side in (0, 1):
            keys = self.players.loc[self.players.side.eq(side)].player.unique()
            for key in means.reindex(keys).nlargest(3).index:
                pts = self.players.loc[self.players.player.eq(key)].set_index("sim").PTS
                features[key] = pts.reindex(g.index).fillna(0)
        z = (features - features.median()) / features.std().replace(0, 1)
        winner_matches = (g.home_score > g.away_score) == (self.home_win_rate >= .5)
        distance = (z ** 2).sum(1) + np.where(winner_matches, 0, 1e6)
        return int(distance.idxmin())

    def replay(self, sim=None):
        """The play-by-play and box score of one simulation (the typical one by default)."""
        sim = self.typical_sim() if sim is None else sim
        game = self.matchup.play([self.seed, sim], record=True)
        box = []
        for side in (0, 1):
            team = self.matchup.teams[side]
            for i, key in enumerate(team.keys):
                row = dict(zip(STATS, game.stats[side][i]))
                if row["SEC"] <= 0:
                    continue
                box.append(dict(team=team.team, player=team.names[i], MIN=round(row["SEC"] / 60),
                                PTS=int(row["PTS"]), FG=f"{int(row['FGM'])}-{int(row['FGA'])}",
                                **{"3P": f"{int(row['3PM'])}-{int(row['3PA'])}"},
                                FT=f"{int(row['FTM'])}-{int(row['FTA'])}", REB=int(row["OREB"] + row["DREB"]),
                                AST=int(row["AST"]), STL=int(row["STL"]), BLK=int(row["BLK"]), TOV=int(row["TOV"]),
                                PF=int(row["PF"])))
        return pd.DataFrame(game.log), pd.DataFrame(box), game


def simulate_game(league, home, away, home_players, away_players, home_starters=None, away_starters=None,
                  date=None, n=1000, seed=2026, use_defender=False, officials=()):
    matchup = Matchup(league, home, away, home_players, away_players, home_starters, away_starters, date,
                      use_defender, officials)
    games, players = matchup.simulate(n, seed)
    matchup.pregame_logit = float(_logit(np.clip((games.home_score > games.away_score).mean(), .01, .99)))
    return SimulationResult(matchup, games, players, seed)


def rotation_players(league, team, size=10, recent_games=5):
    """The players a coach actually uses."""
    pg = league.player_games
    current = pg.loc[pg.season.eq(league.current_season)]
    latest_team = current.sort_values("date").groupby("player").team.last()
    games = current.loc[current.team.eq(team)].sort_values("date")
    recent_ids = games.game_id.drop_duplicates().tail(recent_games)
    minutes = games.loc[games.game_id.isin(recent_ids)].groupby("player").minutes.sum().sort_values(ascending=False)
    return [p for p in minutes.index if latest_team.get(p) == team][:size]


def team_roster(league, team, out=(), recent_games=5):
    """Players for an upcoming game."""
    pg = league.player_games
    current = pg.loc[pg.season.eq(league.current_season)]
    latest_team = current.sort_values("date").groupby("player").team.last()
    games = current.loc[current.team.eq(team)].sort_values("date")
    recent_ids = games.game_id.drop_duplicates().tail(recent_games)
    recent = games.loc[games.game_id.isin(recent_ids) & games.minutes.gt(0)]
    out = set(out)
    players = [p for p in recent.groupby("player").minutes.sum().sort_values(ascending=False).index
               if latest_team.get(p) == team and p not in out]
    last3 = recent.loc[recent.game_id.isin(recent_ids.tail(3)) & recent.started & recent.player.isin(players)]
    order = last3.groupby("player").agg(n=("started", "size"), m=("minutes", "sum")).sort_values(
        ["n", "m"], ascending=False)
    starters = list(order.index[:5])
    for p in players:
        if len(starters) >= 5:
            break
        if p not in starters:
            starters.append(p)
    return players, starters


PROP_STATS = {
    "PTS": ("Points", ["PTS"]), "REB": ("Rebounds", ["OREB", "DREB"]), "AST": ("Assists", ["AST"]),
    "3PM": ("3-pointers made", ["3PM"]), "STL": ("Steals", ["STL"]), "BLK": ("Blocks", ["BLK"]),
    "TOV": ("Turnovers", ["TOV"]), "PRA": ("Pts + Reb + Ast", ["PTS", "OREB", "DREB", "AST"]),
    "PR": ("Pts + Reb", ["PTS", "OREB", "DREB"]), "PA": ("Pts + Ast", ["PTS", "AST"]),
    "RA": ("Reb + Ast", ["OREB", "DREB", "AST"]), "SB": ("Steals + blocks", ["STL", "BLK"]),
    "FTM": ("Free throws made", ["FTM"]),
}


def prop_samples(result):
    """Per simulation and player, every prop stat (zero in sims he did not play)."""
    p = result.players
    sims = result.games.sim.to_numpy()
    out = {}
    for key, (_, columns) in PROP_STATS.items():
        values = p[columns].sum(1)
        table = pd.DataFrame({"sim": p.sim, "player": p.player, "v": values}).pivot_table(
            index="sim", columns="player", values="v", aggfunc="sum").reindex(sims).fillna(0)
        out[key] = table
    info = p.groupby("player").agg(name=("name", "first"), team=("team", "first"), MIN=("MIN", "sum"))
    info["MIN"] = info.MIN / len(sims)
    return out, info.sort_values(["team", "MIN"], ascending=[True, False])


def prop_probability(samples, player, stat, line):
    values = samples[stat][player].to_numpy()
    return dict(over=float((values > line).mean()), under=float((values < line).mean()),
                push=float((values == line).mean()), average=float(values.mean()), median=float(np.median(values)))


def prop_board(samples, info, threshold=.55, sides=("over",), min_minutes=12., stats=None):
    """For each player and prop, the most aggressive half-point line that still hits at least ``threshold``
    of simulated games.
    """
    rows = []
    for stat in stats or PROP_STATS:
        table = samples[stat]
        for player, row in info.loc[info.MIN.ge(min_minutes)].iterrows():
            values = table[player].to_numpy()
            top = max(values.max(), 1)
            lines = np.arange(.5, top + 1, 1.)
            over = np.array([(values > x).mean() for x in lines])
            if "over" in sides and (over >= threshold).any():
                i = np.flatnonzero(over >= threshold).max()
                rows.append(dict(player=row["name"], team=row.team, prop=PROP_STATS[stat][0], side="over",
                                 line=lines[i], probability=over[i], average=values.mean()))
            if "under" in sides and ((1 - over) >= threshold).any():
                i = np.flatnonzero((1 - over) >= threshold).min()
                rows.append(dict(player=row["name"], team=row.team, prop=PROP_STATS[stat][0], side="under",
                                 line=lines[i], probability=1 - over[i], average=values.mean()))
    board = pd.DataFrame(rows, columns=["player", "team", "prop", "side", "line", "probability", "average"])
    return board.sort_values("probability", ascending=False).reset_index(drop=True)


def american_odds(probability):
    """Fair (no-vig) American odds for a probability."""
    probability = min(max(probability, 1e-4), 1 - 1e-4)
    return f"{-round(100 * probability / (1 - probability)):+d}" if probability >= .5 \
        else f"+{round(100 * (1 - probability) / probability)}"


def spread_probability(result, home_line):
    """Home team with ``home_line`` points (sportsbook sign: -5.5 means home must win by 6+)."""
    adjusted = result.games.home_score - result.games.away_score + home_line
    return dict(home=float((adjusted > 0).mean()), away=float((adjusted < 0).mean()), push=float((adjusted == 0).mean()))


def total_probability(result, line):
    total = result.games.home_score + result.games.away_score
    return dict(over=float((total > line).mean()), under=float((total < line).mean()), push=float((total == line).mean()))


def spread_ladder(result, width=8):
    """Cover chances for home lines around the simulated median margin."""
    margin = (result.games.home_score - result.games.away_score).median()
    center = -(np.floor(margin) + .5)
    rows = []
    for line in np.arange(center - width, center + width + .5, 1.):
        p = spread_probability(result, line)
        rows.append({"home line": line, "home covers": p["home"], "away covers": p["away"], "push": p["push"]})
    return pd.DataFrame(rows)


def total_ladder(result, width=10):
    total = (result.games.home_score + result.games.away_score).median()
    center = np.floor(total) + .5
    rows = []
    for line in np.arange(center - width, center + width + .5, 1.):
        p = total_probability(result, line)
        rows.append({"total": line, "over": p["over"], "under": p["under"]})
    return pd.DataFrame(rows)


def _shot_spot(zone, rnd):
    """A plausible spot (feet, rim at 0,0; y toward half court) for a shot in this zone."""
    for _ in range(50):
        if zone == Z["RA"]:
            r, a = rnd.uniform(0, 3.8), rnd.uniform(0, math.pi)
            x, y = r * math.cos(a), r * math.sin(a)
        elif zone == Z["PAINT"]:
            x, y = rnd.uniform(-7.5, 7.5), rnd.uniform(-1, 13.5)
            if math.hypot(x, y) < 4.2:
                continue
        elif zone == Z["MID"]:
            r, a = rnd.uniform(8, 21.5), rnd.uniform(.05 * math.pi, .95 * math.pi)
            x, y = r * math.cos(a), r * math.sin(a)
            if abs(x) <= 8 and y <= 14:
                continue
        elif zone == Z["C3"]:
            x, y = rnd.choice([-1, 1]) * rnd.uniform(22.2, 23.6), rnd.uniform(-3.5, 8)
        else:
            r, a = rnd.uniform(24, 27.5), rnd.uniform(.14 * math.pi, .86 * math.pi)
            x, y = r * math.cos(a), r * math.sin(a)
            if y < 9:
                continue
        return [round(x, 1), round(y, 1)]
    return [0., 0.]


def _short(name):
    parts = str(name).split()
    return f"{parts[0][0]}. {' '.join(parts[1:])}" if len(parts) > 1 else str(name)


def replay_payload(result, sim=None, colors=None, names=None, autoplay=False):
    sim = result.typical_sim() if sim is None else sim
    log, box, game = result.replay(sim)
    m = result.matchup
    teams = {}
    espn_abbr = {v: k for k, v in ESPN_TEAM.items()}
    headshot = lambda k: (f"https://a.espncdn.com/i/headshots/nba/players/full/{k[1:]}.png"
                          if str(k).startswith("e") else None)
    for side, team in enumerate(m.teams):
        teams[team.team] = dict(abbr=team.team, name=(names or {}).get(team.team, team.team),
                                colors=(colors or {}).get(team.team, ["#3b4a6b", "#8ea0c8"]),
                                logo=f"https://a.espncdn.com/i/teamlogos/nba/500/"
                                     f"{espn_abbr.get(team.team, team.team).lower()}.png",
                                players=[dict(name=_short(n), full=n, starter=k in team.starters, group=g,
                                              photo=headshot(k))
                                         for k, n, g in zip(team.keys, team.names, team.group)])
    columns = ["SEC", "PTS", "FGM", "FGA", "3PM", "3PA", "FTM", "FTA", "REB", "AST", "STL", "BLK", "TOV", "PF"]
    events = log.drop(columns=[c for c in ("kind",) if c not in log]).to_dict("records")
    for e in events:
        for key, value in list(e.items()):
            if isinstance(value, float) and math.isnan(value):
                e[key] = None
    return dict(home=m.home, away=m.away, teams=teams, columns=columns, events=events, sim=int(sim) + 1,
                n=len(result.games), simulated=round(result.home_win_rate, 4), autoplay=autoplay,
                final=[int(game.score[0]), int(game.score[1])], periods=int(game.period))


def replay_html(result, sim=None, colors=None, names=None, autoplay=False):
    payload = replay_payload(result, sim, colors, names, autoplay)
    template = Path(__file__).with_name("nba_game_sim_replay.html").read_text()
    title = f"{payload['away']} @ {payload['home']} · Simulated game"
    data = json.dumps(payload, default=lambda v: v.item() if hasattr(v, "item") else str(v))
    return template.replace("__TITLE__", html.escape(title)).replace("__PAYLOAD__", data.replace("</", "<\\/"))


def first_scorers(result, top=None):
    """How often each player scores the game's first points (field goal or free throw)."""
    g = result.games
    table = g.groupby(["first_side", "first_scorer"]).size().rename("games").reset_index()
    names = {k: n for t in result.matchup.teams for k, n in zip(t.keys, t.names)}
    table["player"] = table.first_scorer.map(names)
    table["team"] = np.where(table.first_side.eq(0), result.matchup.home, result.matchup.away)
    table["probability"] = table.games / len(g)
    table["fair odds"] = table.probability.map(american_odds)
    how = g.groupby("first_scorer").first_how.value_counts(normalize=True).unstack(fill_value=0)
    table["by free throw"] = table.first_scorer.map(how.get("FT", pd.Series(dtype=float))).fillna(0)
    table = table.sort_values("probability", ascending=False)[["player", "team", "probability", "fair odds",
                                                              "by free throw", "games"]]
    return table.head(top) if top else table
