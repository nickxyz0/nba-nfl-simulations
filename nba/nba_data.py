"""NBA data loaders for a possession-level game simulator, cached under nba/data."""

from html.parser import HTMLParser
from pathlib import Path
import datetime as dt
import io
import json
import lzma
import re
import tarfile
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd


CACHE_DIR = Path(__file__).resolve().parent / "data"
ARCHIVE_URL = "https://github.com/shufinskiy/nba_data/raw/main/datasets/{kind}_{season}.tar.xz"
ARCHIVE_KINDS = ("cdnnba", "shotdetail", "matchups", "pbpstats", "nbastatsv3")
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date}&limit=50"
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event}"
NBASTUFFER_URL = "https://www.nbastuffer.com/{season}-{next}-nba-{page}/"
NBASTUFFER_PAGES = ("team-stats", "player-stats", "referee-stats")
CDN_COLUMNS = [
    "gameId", "actionNumber", "orderNumber", "period", "clock", "teamId", "teamTricode",
    "personId", "playerNameI", "possession", "actionType", "subType", "descriptor", "qualifiers",
    "x", "y", "area", "areaDetail", "shotDistance", "shotResult", "isFieldGoal", "side",
    "assistPersonId", "blockPersonId", "stealPersonId", "foulDrawnPersonId", "officialId",
    "scoreHome", "scoreAway", "description",
]
ESPN_NO_COORDINATE = -1_000_000
REQUEST_PAUSE = .25
MATCHUP_TEAM_FIELDS = ("teamId", "teamName", "teamCity", "teamTricode", "teamSlug")


def season_label(season):
    return f"{season}-{str(season + 1)[-2:]}"


def current_season(today=None):
    """NBA seasons are named by start year and start in October."""
    today = today or dt.date.today()
    return today.year if today.month >= 9 else today.year - 1


def _get(url, timeout=60, attempts=3):
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404 or attempt == attempts - 1:
                raise
        except urllib.error.URLError:
            if attempt == attempts - 1:
                raise
        time.sleep(2 ** attempt)


def load_archive(kind, season, refresh=False, columns=None, cache_dir=CACHE_DIR):
    """One archived season as a DataFrame, cached as parquet. None if not published yet."""
    if kind not in ARCHIVE_KINDS:
        raise ValueError(f"kind must be one of {ARCHIVE_KINDS}")
    path = Path(cache_dir) / "archive" / f"{kind}_{season}.parquet"
    if refresh or not path.exists():
        try:
            payload = _get(ARCHIVE_URL.format(kind=kind, season=season), timeout=300)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return pd.read_parquet(path, columns=columns) if path.exists() else None
            raise
        with tarfile.open(fileobj=io.BytesIO(lzma.decompress(payload))) as archive:
            member = next(m for m in archive.getmembers() if m.name.endswith(".csv"))
            frame = pd.read_csv(archive.extractfile(member), low_memory=False)
        for column in frame.columns[frame.dtypes.eq(object)]:
            frame[column] = frame[column].astype("string")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".download")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    return pd.read_parquet(path, columns=columns)


def load_shots(season, refresh=False):
    """Every field goal attempt with location, zone, assister and blocker."""
    pbp = load_archive("cdnnba", season, refresh, columns=CDN_COLUMNS)
    chart = load_archive("shotdetail", season, refresh, columns=[
        "GAME_ID", "GAME_EVENT_ID", "ACTION_TYPE", "SHOT_TYPE", "SHOT_ZONE_BASIC",
        "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE", "LOC_X", "LOC_Y", "GAME_DATE", "HTM", "VTM"])
    shots = pbp.loc[pbp.isFieldGoal.eq(1)].copy()
    shots = shots.merge(chart.rename(columns={"GAME_ID": "gameId", "GAME_EVENT_ID": "actionNumber"}),
                        on=["gameId", "actionNumber"], how="left")
    shots["made"] = shots.shotResult.eq("Made").astype(int)
    shots["value"] = np.where(shots.actionType.eq("3pt"), 3, 2)
    return shots.rename(columns={"gameId": "game_id", "personId": "shooter_id", "teamId": "team_id",
                                 "assistPersonId": "assist_id", "blockPersonId": "block_id"})


def load_matchups(season, refresh=False, cache_dir=CACHE_DIR):
    """Who guarded whom each game, with partial possessions and shots against."""
    frame = load_archive("matchups", season, refresh, cache_dir=cache_dir)
    if frame is None:
        frame = _cached_live_matchups(season, cache_dir)
    if frame is None:
        return None
    frame = frame.rename(columns={"person_id": "offense_id", "matchups_person_id": "defender_id",
                                  "team_id": "offense_team_id"})
    totals = frame.groupby(["game_id", "offense_id"]).partial_possessions.transform("sum")
    frame["defender_share"] = frame.partial_possessions / totals.replace(0, np.nan)
    return frame


def _snake(name):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _flatten_matchups(payload):
    """One NBA.com boxscorematchupsv3 response as rows in the archive's column layout."""
    box = payload["boxScoreMatchups"]
    rows = []
    for side in ("homeTeam", "awayTeam"):
        team = box[side]
        team_fields = {_snake(k): team[k] for k in MATCHUP_TEAM_FIELDS}
        for player in team["players"]:
            offense = {_snake(k): v for k, v in player.items() if k != "matchups"}
            for matchup in player.get("matchups", []):
                row = dict(game_id=int(box["gameId"]), away_team_id=box["awayTeamId"],
                           home_team_id=box["homeTeamId"], **team_fields, **offense)
                row.update({f"matchups_{_snake(k)}": v for k, v in matchup.items() if k != "statistics"})
                row.update({_snake(k): v for k, v in matchup["statistics"].items()})
                rows.append(row)
    return pd.DataFrame(rows)


def import_saved_matchups(season=None, cache_dir=CACHE_DIR):
    """Read NBA.com box score pages saved from a browser (Matchups view, Safari "Page Source") in
    data/nba/matchups_manual into the live matchup cache.
    """
    season = season or current_season()
    folder = Path(cache_dir) / "matchups_live" / str(season)
    folder.mkdir(parents=True, exist_ok=True)
    imported = 0
    for page in sorted((Path(cache_dir) / "matchups_manual").glob("*.htm*")):
        found = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                          page.read_text(encoding="utf-8", errors="ignore"), re.S)
        game = json.loads(found.group(1))["props"]["pageProps"].get("game") if found else None
        if not game or "players" not in game.get("homeTeam", {}):
            print(f"  {page.name}: no matchup data (save the Matchups view as Page Source)")
            continue
        if game.get("gameStatus") != 3:
            print(f"  {page.name}: game was not final when saved; save it again after it ends")
            continue
        if not game["gameId"].startswith(f"002{str(season)[-2:]}"):
            continue
        frame = _flatten_matchups({"boxScoreMatchups": game})
        if frame.empty:
            print(f"  {page.name}: page has no matchups")
            continue
        frame.to_parquet(folder / f"{game['gameId']}.parquet", index=False)
        imported += 1
    return imported


def _cached_live_matchups(season, cache_dir=CACHE_DIR):
    games = sorted((Path(cache_dir) / "matchups_live" / str(season)).glob("*.parquet"))
    return pd.concat([pd.read_parquet(g) for g in games], ignore_index=True) if games else None


def fetch_live_matchups(season=None, pause=1.5, cache_dir=CACHE_DIR):
    """Download this season's defensive matchups from stats.nba.com, one game per request."""
    from nba_api.stats.endpoints import boxscorematchupsv3, leaguegamelog

    season = season or current_season()
    folder = Path(cache_dir) / "matchups_live" / str(season)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        log = leaguegamelog.LeagueGameLog(season=season_label(season), player_or_team_abbreviation="T",
                                          season_type_all_star="Regular Season", timeout=60).get_data_frames()[0]
    except Exception as error:
        print(f"stats.nba.com did not answer ({type(error).__name__}). It is blocking this network; "
              "the simulator will use the latest archived season's matchups.")
        return 0
    game_ids = sorted(set(log.GAME_ID))
    todo = [g for g in game_ids if not (folder / f"{g}.parquet").exists()]
    print(f"{len(game_ids)} games played, {len(todo)} to fetch")
    fetched = 0
    for number, game_id in enumerate(todo, 1):
        try:
            payload = boxscorematchupsv3.BoxScoreMatchupsV3(game_id=game_id, timeout=60).get_dict()
            frame = _flatten_matchups(payload)
        except Exception as error:
            print(f"  {game_id}: {error}")
            time.sleep(pause * 4)
            continue
        if not frame.empty:
            frame.to_parquet(folder / f"{game_id}.parquet", index=False)
            fetched += 1
        if number % 25 == 0:
            print(f"  {number}/{len(todo)}")
        time.sleep(pause)
    return fetched


def defender_ratings(matchups, shrink=200.):
    """Points allowed per partial possession by each defender, shrunk to league."""
    league = matchups.player_points.sum() / matchups.partial_possessions.sum()
    totals = matchups.groupby("defender_id")[[
        "partial_possessions", "player_points", "matchup_field_goals_made",
        "matchup_field_goals_attempted", "matchup_three_pointers_made",
        "matchup_three_pointers_attempted", "matchup_turnovers", "shooting_fouls"]].sum()
    names = matchups.groupby("defender_id").matchups_name_i.last()
    possessions = totals.partial_possessions
    totals["pts_per_poss"] = (totals.player_points + shrink * league) / (possessions + shrink)
    totals["pts_per_poss_vs_league"] = totals.pts_per_poss - league
    totals["fg_pct_allowed"] = totals.matchup_field_goals_made / totals.matchup_field_goals_attempted.replace(0, np.nan)
    totals["name"] = names
    return totals.sort_values("pts_per_poss_vs_league")


def espn_schedule(season, through=None, cache_dir=CACHE_DIR):
    """Regular-season and in-season-tournament games from ESPN scoreboards, one row per game."""
    start = dt.date(season, 10, 1)
    end = min(through or dt.date.today(), dt.date(season + 1, 4, 30))
    rows = []
    for offset in range((end - start).days + 1):
        day = start + dt.timedelta(days=offset)
        board = json.loads(_get(ESPN_SCOREBOARD.format(date=day.strftime("%Y%m%d"))))
        for event in board.get("events", []):
            if event["season"]["type"] != 2:
                continue
            competition = event["competitions"][0]
            teams = {c["homeAway"]: c for c in competition["competitors"]}
            rows.append(dict(
                event_id=event["id"], date=day, status=event["status"]["type"]["name"],
                home=teams["home"]["team"]["abbreviation"], away=teams["away"]["team"]["abbreviation"],
                home_id=teams["home"]["team"]["id"], away_id=teams["away"]["team"]["id"],
                home_score=pd.to_numeric(teams["home"].get("score"), errors="coerce"),
                away_score=pd.to_numeric(teams["away"].get("score"), errors="coerce"),
                neutral=competition.get("neutralSite", False)))
        time.sleep(REQUEST_PAUSE)
    schedule = pd.DataFrame(rows).drop_duplicates("event_id")
    path = Path(cache_dir) / "espn" / f"schedule_{season}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(path, index=False)
    return schedule


def espn_summary(event_id, refresh=False, cache_dir=CACHE_DIR):
    """Raw ESPN game summary JSON. Final games are cached for good."""
    path = Path(cache_dir) / "espn" / "games" / f"{event_id}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    summary = json.loads(_get(ESPN_SUMMARY.format(event=event_id)))
    if summary["header"]["competitions"][0]["status"]["type"]["completed"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary))
    return summary


def espn_plays(summary):
    """Flatten one game's plays."""
    event_id = summary["header"]["id"]
    rows = []
    for play in summary.get("plays", []):
        people = [p["athlete"]["id"] for p in play.get("participants", [])] + [None] * 3
        x, y = play.get("coordinate", {}).get("x"), play.get("coordinate", {}).get("y")
        has_spot = x is not None and x > ESPN_NO_COORDINATE
        rows.append(dict(
            event_id=event_id, sequence=int(play["sequenceNumber"]), period=play["period"]["number"],
            clock=play["clock"]["displayValue"], team_id=play.get("team", {}).get("id"),
            type_id=play["type"]["id"], type=play["type"]["text"].replace("\n", " "), text=play.get("text"),
            shooting=play.get("shootingPlay", False), scoring=play.get("scoringPlay", False),
            points=play.get("scoreValue", 0), points_attempted=play.get("pointsAttempted", 0),
            x=x if has_spot else np.nan, y=y if has_spot else np.nan,
            home_score=play.get("homeScore"), away_score=play.get("awayScore"),
            player_1=people[0], player_2=people[1], player_3=people[2]))
    plays = pd.DataFrame(rows)
    if plays.empty:
        return plays
    is_free_throw = plays.type.str.startswith("Free Throw")
    plays["field_goal"] = plays.shooting & ~is_free_throw
    plays["free_throw"] = is_free_throw
    plays["made"] = plays.scoring & plays.shooting
    plays["blocked"] = plays.field_goal & ~plays.made & plays.text.str.contains(" blocks ", na=False)
    plays["assisted"] = plays.field_goal & plays.made & plays.text.str.contains(" assists\\)", na=False)
    return plays


def espn_box(summary):
    """Player box score lines with starter flag and minutes."""
    event_id = summary["header"]["id"]
    rows = []
    for team in summary["boxscore"].get("players", []):
        for block in team["statistics"]:
            labels = block["labels"]
            for line in block["athletes"]:
                row = dict(event_id=event_id, team=team["team"]["abbreviation"],
                           athlete_id=line["athlete"]["id"], name=line["athlete"]["displayName"],
                           position=line["athlete"].get("position", {}).get("abbreviation"),
                           starter=line.get("starter", False), did_not_play=line.get("didNotPlay", False))
                row.update(zip(labels, line.get("stats") or [None] * len(labels)))
                rows.append(row)
    return pd.DataFrame(rows)


def espn_game_info(summary):
    """Officials, venue and attendance (referee crews change foul and free throw rates)."""
    info = summary.get("gameInfo", {})
    officials = [o["displayName"] for o in info.get("officials", [])]
    return dict(event_id=summary["header"]["id"], venue=info.get("venue", {}).get("fullName"),
                attendance=info.get("attendance"), officials="; ".join(officials))


def espn_injuries(summary):
    rows = []
    for team in summary.get("injuries", []):
        for item in team.get("injuries", []):
            rows.append(dict(team=team["team"]["abbreviation"], athlete_id=item["athlete"]["id"],
                             name=item["athlete"]["displayName"], status=item.get("status"),
                             detail=item.get("details", {}).get("type"), date=item.get("date")))
    return pd.DataFrame(rows)


def load_espn_season(season=None, refresh_schedule=True, cache_dir=CACHE_DIR):
    """Plays, box scores and game info for every final game this season, cached as parquet."""
    season = season or current_season()
    schedule_path = Path(cache_dir) / "espn" / f"schedule_{season}.csv"
    if refresh_schedule or not schedule_path.exists():
        schedule = espn_schedule(season, cache_dir=cache_dir)
    else:
        schedule = pd.read_csv(schedule_path, dtype={"event_id": str, "home_id": str, "away_id": str})
    finals = schedule.loc[schedule.status.eq("STATUS_FINAL")]
    plays, boxes, infos = [], [], []
    for event_id in finals.event_id:
        cached = (Path(cache_dir) / "espn" / "games" / f"{event_id}.json").exists()
        summary = espn_summary(event_id, cache_dir=cache_dir)
        plays.append(espn_plays(summary))
        boxes.append(espn_box(summary))
        infos.append(espn_game_info(summary))
        if not cached:
            time.sleep(REQUEST_PAUSE)
    out = Path(cache_dir) / "espn"
    tables = dict(plays=pd.concat(plays, ignore_index=True), box=pd.concat(boxes, ignore_index=True),
                  games=schedule.merge(pd.DataFrame(infos), on="event_id", how="left"))
    for name, frame in tables.items():
        frame.to_parquet(out / f"{name}_{season}.parquet", index=False)
    return tables


class _TableParser(HTMLParser):
    """Collects every <table> as a list of rows of cell text (no lxml needed)."""

    def __init__(self):
        super().__init__()
        self.tables, self._row, self._cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.tables.append([])
        elif tag == "tr" and self.tables:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.tables[-1].append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def load_nbastuffer(page, season=None, refresh=False, cache_dir=CACHE_DIR):
    """The largest table on an nbastuffer stats page, cached per day."""
    if page not in NBASTUFFER_PAGES:
        raise ValueError(f"page must be one of {NBASTUFFER_PAGES}")
    season = season or current_season()
    path = Path(cache_dir) / "nbastuffer" / f"{page}_{season}_{dt.date.today():%Y%m%d}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)
    import requests
    response = requests.get(NBASTUFFER_URL.format(season=season, next=season + 1, page=page), timeout=60)
    response.raise_for_status()
    parser = _TableParser()
    parser.feed(response.text)
    rows = max(parser.tables, key=len)
    header, body = rows[0], [r for r in rows[1:] if len(r) == len(rows[0]) and r != rows[0]]
    frame = pd.DataFrame(body, columns=header)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    for column in frame.columns:
        converted = pd.to_numeric(frame[column].str.rstrip("%"), errors="coerce")
        if converted.notna().mean() > .9:
            frame[column] = converted
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and cache NBA simulator data.")
    parser.add_argument("--archive-seasons", type=int, nargs="*", default=[current_season() - 1],
                        help="past seasons by start year (2025 = 2025-26)")
    parser.add_argument("--skip-espn", action="store_true")
    parser.add_argument("--skip-nbastuffer", action="store_true")
    parser.add_argument("--import-matchups", action="store_true",
                        help="only read box score pages saved in data/nba/matchups_manual")
    parser.add_argument("--live-matchups", action="store_true",
                        help="only fetch this season's matchups from stats.nba.com (needs nba_api)")
    args = parser.parse_args()

    if args.import_matchups:
        imported = import_saved_matchups()
        frame = load_matchups(current_season())
        games = 0 if frame is None else frame.game_id.nunique()
        print(f"imported {imported} saved pages; {games} games of {season_label(current_season())} matchups cached")
        raise SystemExit
    if args.live_matchups:
        fetched = fetch_live_matchups()
        frame = load_matchups(current_season())
        games = 0 if frame is None else frame.game_id.nunique()
        print(f"fetched {fetched} new games; {games} games of {season_label(current_season())} matchups cached")
        raise SystemExit

    for season in args.archive_seasons:
        for kind in ("cdnnba", "shotdetail", "matchups"):
            frame = load_archive(kind, season)
            print(f"{kind} {season_label(season)}:", "not published" if frame is None else f"{len(frame):,} rows")
    if not args.skip_espn:
        tables = load_espn_season()
        print(f"ESPN {season_label(current_season())}: {tables['games'].status.eq('STATUS_FINAL').sum()} final games, "
              f"{len(tables['plays']):,} plays")
    if not args.skip_nbastuffer:
        for page in NBASTUFFER_PAGES:
            print(f"nbastuffer {page}: {len(load_nbastuffer(page)):,} rows")
