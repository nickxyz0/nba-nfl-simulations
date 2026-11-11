"""Play-by-play Monte Carlo NFL game simulator built on nflverse play-by-play."""

from pathlib import Path
import html
import json
import urllib.request

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import nfl_model_pipeline as nfl


PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
PARTICIPATION_URL = ("https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/"
                     "pbp_participation_{season}.csv")
FTN_URL = "https://github.com/nflverse/nflverse-data/releases/download/ftn_charting/ftn_charting_{season}.csv"
PBP_COLUMNS = [
    "game_id", "play_id", "season", "week", "season_type", "home_team", "away_team",
    "posteam", "defteam", "qtr", "down", "ydstogo", "yardline_100", "game_seconds_remaining",
    "play_type", "yards_gained", "qb_dropback", "qb_scramble", "pass_attempt", "complete_pass",
    "sack", "interception", "fumble_lost", "air_yards", "return_yards", "return_touchdown",
    "touchdown", "td_team", "epa", "out_of_bounds", "two_point_attempt", "penalty",
    "penalty_team", "penalty_yards", "penalty_type", "first_down_penalty", "kick_distance",
    "field_goal_result", "extra_point_result", "timeout", "two_point_conv_result", "touchback", "drive",
    "drive_start_transition", "score_differential", "result",
    "passer_player_id", "passer_player_name", "receiver_player_id", "receiver_player_name",
    "rusher_player_id", "rusher_player_name", "kicker_player_id", "kicker_player_name",
    "sack_player_id", "sack_player_name", "half_sack_1_player_id", "half_sack_1_player_name",
    "half_sack_2_player_id", "half_sack_2_player_name", "interception_player_id",
    "interception_player_name", "pass_defense_1_player_id", "pass_defense_1_player_name",
]
POOL_MIX = {"league": .35, "offense": .45, "defense": .20}
RATING_ALPHA = 200.
CARRY_SHRINK, TARGET_SHRINK = 100., 50.
QB_SHRINK = 150.
QB_DESIGNED_RUN_DEFAULT = .05
PASS_RATE_SHRINK = 8.
PRIOR_SEASON_WEIGHT = .3
COVERAGE_GROUPS = {"COVER_0": 0, "COVER_1": 0, "2_MAN": 0, "COVER_2": 1, "COVER_3": 2,
                   "COVER_4": 3, "COVER_6": 3, "COVER_9": 3}
LOOK_LABELS = {"coverage": ["Man coverage", "Cover 2", "Cover 3", "Quarters"],
               "blitz": ["", "Blitz"], "box": ["Light box", "7-man box", "Stacked box (8+)"]}
LOOK_DIMS = {0: ("coverage", "blitz"), 1: ("box",)}
LOOK_STRENGTH = {"coverage": .35, "blitz": .6, "box": .6}
LOOK_SHRINK = 150.
SHARE_HALF_LIFE_GAMES = 3.
RETURNING_GAMES = 4
EPA_CLIP = 4.
GAME_FORM_SD = 0.05
PASS_LEAN_SD = 0.5
FORM_LEVELS = 9
MAX_FG = 57
ZONE_EDGES = [5, 10, 20]
DISTANCE_EDGES = [2, 6, 10]
GO_DISTANCE_EDGES = [1, 2, 5]
YARDLINE_EDGES = [10, 20, 30, 40, 50, 60, 70, 80]
STATS = ["pass_cmp", "pass_att", "pass_yds", "pass_td", "pass_int", "sacked", "sack_yds",
         "rush_car", "rush_yds", "rush_td", "tgt", "rec", "rec_yds", "rec_td",
         "fgm", "fga", "xpm", "xpa", "def_sack", "def_int", "fum_lost"]
S = {name: i for i, name in enumerate(STATS)}
TEAM_COLORS = dict(
    ARI=("#97233F", "#FFB612"), ATL=("#A71930", "#000000"), BAL=("#241773", "#9E7C0C"),
    BUF=("#00338D", "#C60C30"), CAR=("#0085CA", "#101820"), CHI=("#0B162A", "#C83803"),
    CIN=("#FB4F14", "#000000"), CLE=("#311D00", "#FF3C00"), DAL=("#003594", "#869397"),
    DEN=("#FB4F14", "#002244"), DET=("#0076B6", "#B0B7BC"), GB=("#203731", "#FFB612"),
    HOU=("#03202F", "#A71930"), IND=("#002C5F", "#A2AAAD"), JAX=("#006778", "#D7A22A"),
    KC=("#E31837", "#FFB81C"), LAC=("#0080C6", "#FFC20E"), LAR=("#003594", "#FFA300"),
    LV=("#000000", "#A5ACAF"), MIA=("#008E97", "#FC4C02"), MIN=("#4F2683", "#FFC62F"),
    NE=("#002244", "#C60C30"), NO=("#D3BC8D", "#101820"), NYG=("#0B2265", "#A71930"),
    NYJ=("#125740", "#FFFFFF"), PHI=("#004C54", "#A5ACAF"), PIT=("#FFB612", "#101820"),
    SEA=("#002244", "#69BE28"), SF=("#AA0000", "#B3995D"), TB=("#D50A0A", "#34302B"),
    TEN=("#0C2340", "#4B92DB"), WAS=("#5A1414", "#FFB612"),
)
TEAM_NAMES = dict(
    ARI="Cardinals", ATL="Falcons", BAL="Ravens", BUF="Bills", CAR="Panthers", CHI="Bears",
    CIN="Bengals", CLE="Browns", DAL="Cowboys", DEN="Broncos", DET="Lions", GB="Packers",
    HOU="Texans", IND="Colts", JAX="Jaguars", KC="Chiefs", LAC="Chargers", LAR="Rams",
    LV="Raiders", MIA="Dolphins", MIN="Vikings", NE="Patriots", NO="Saints", NYG="Giants",
    NYJ="Jets", PHI="Eagles", PIT="Steelers", SEA="Seahawks", SF="49ers", TB="Buccaneers",
    TEN="Titans", WAS="Commanders",
)


def cached_parquet(url, path, refresh=False, columns=None):
    """Same snapshot policy as nfl.cached_csv: refresh is an explicit choice."""
    path = Path(path)
    if refresh or not path.exists():
        with urllib.request.urlopen(url, timeout=180) as response:
            payload = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".download")
        temporary.write_bytes(payload)
        pd.read_parquet(temporary, columns=["game_id"])
        temporary.replace(path)
    return pd.read_parquet(path, columns=columns)


def load_pbp(seasons, cache_dir="data/nflverse", refresh_seasons=()):
    frames = [cached_parquet(PBP_URL.format(season=s), Path(cache_dir) / f"play_by_play_{s}.parquet",
                             refresh=s in set(refresh_seasons), columns=PBP_COLUMNS) for s in seasons]
    pbp = pd.concat(frames, ignore_index=True)
    pbp = pbp.loc[pbp.season_type.eq("REG")]
    teams = ["home_team", "away_team", "posteam", "defteam", "td_team", "penalty_team"]
    for column in teams:
        pbp[column] = pbp[column].replace(nfl.ALIASES)
    pbp = pbp.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    snaps = pbp.loc[pbp.play_type.notna() & ~(pbp.play_type.eq("no_play") & pbp.penalty.ne(1))]
    following = snaps.groupby("game_id").game_seconds_remaining.shift(-1)
    pbp["elapsed"] = (snaps.game_seconds_remaining - following).clip(2, 45)
    pbp["elapsed"] = pbp.elapsed.fillna(25)
    return pbp


def _cached_table(url, path, refresh=False, usecols=None):
    """Like cached_parquet for CSVs; a season nflverse has not published returns None."""
    import urllib.error
    path = Path(path)
    if refresh or not path.exists():
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                payload = response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return pd.read_csv(path, usecols=usecols, low_memory=False) if path.exists() else None
            raise
        temporary = path.with_suffix(path.suffix + ".download")
        temporary.write_bytes(payload)
        pd.read_csv(temporary, nrows=2)
        temporary.replace(path)
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def load_looks(seasons, cache_dir="data/nflverse", refresh_seasons=()):
    """Per-play defensive look: coverage group, blitz (0/1) and run box (0-2); -1 = unknown."""
    frames = []
    for season in seasons:
        refresh = season in set(refresh_seasons)
        ftn = _cached_table(FTN_URL.format(season=season), Path(cache_dir) / f"ftn_charting_{season}.csv", refresh,
                            ["nflverse_game_id", "nflverse_play_id", "n_defense_box", "n_blitzers"])
        coverage = _cached_table(PARTICIPATION_URL.format(season=season),
                                 Path(cache_dir) / f"pbp_participation_{season}.csv", refresh,
                                 ["nflverse_game_id", "play_id", "defense_coverage_type"])
        if ftn is None and coverage is None:
            continue
        looks = None
        if ftn is not None:
            ftn = ftn.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
            box = np.select([ftn.n_defense_box.le(6), ftn.n_defense_box.eq(7), ftn.n_defense_box.ge(8)], [0, 1, 2], -1)
            looks = pd.DataFrame({"game_id": ftn.game_id, "play_id": ftn.play_id,
                                  "blitz": np.where(ftn.n_blitzers.notna(), ftn.n_blitzers.gt(0).astype(int), -1),
                                  "box": np.where(ftn.n_defense_box.ge(4), box, -1)})
            looks = looks.drop_duplicates(["game_id", "play_id"])
        if coverage is not None:
            coverage = pd.DataFrame({"game_id": coverage.nflverse_game_id, "play_id": coverage.play_id,
                                     "coverage": coverage.defense_coverage_type.map(COVERAGE_GROUPS)})
            coverage = coverage.drop_duplicates(["game_id", "play_id"])
            looks = coverage if looks is None else looks.merge(coverage, on=["game_id", "play_id"], how="outer")
        frames.append(looks)
    looks = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["game_id", "play_id"])
    for dim in ("coverage", "blitz", "box"):
        looks[dim] = looks[dim].fillna(-1).astype(int) if dim in looks else -1
    return looks


def short_name(name):
    parts = str(name).split()
    return name if len(parts) < 2 else f"{parts[0][0]}. {' '.join(parts[1:])}"


def wp_features(diff, seconds, yardline):
    diff, seconds, yardline = (np.asarray(v, dtype=float) for v in (diff, seconds, yardline))
    k = 1 / np.sqrt(seconds / 60 + 1)
    field = (50 - yardline) / 50
    return np.column_stack([diff, diff * k, diff * k * k, field, field * k])


def _bin(edges, value):
    return int(np.searchsorted(edges, value, side="left"))


class League:
    """Everything learned from play-by-play once, shared by every matchup."""

    def __init__(self, pbp, season, player_stats=None):
        self.season = season
        current = pbp.loc[pbp.season.eq(season)]
        recent = pbp.loc[pbp.season.between(season - 1, season)]
        if current.empty:
            raise ValueError(f"No {season} play-by-play loaded")
        self.current = current
        self._usage = recent.loc[recent.play_type.isin(["pass", "run"]) & recent.posteam.notna(),
                                 ["game_id", "season", "week", "posteam", "play_type", "pass_attempt", "sack",
                                  "qb_scramble", "receiver_player_id", "rusher_player_id"]]
        self._build_pool(recent)
        self._build_looks()
        self._build_ratings()
        self._build_tendencies(current, recent)
        self._build_kicking(pbp, current, recent)
        self._build_wp(pbp.loc[pbp.season.lt(season)])
        self._build_names(pbp, player_stats)

    @classmethod
    def from_cache(cls, season, refresh=False, cache_dir="data/nflverse"):
        refresh_seasons = [season] if refresh else ()
        pbp = load_pbp(range(season - 3, season + 1), cache_dir, refresh_seasons)
        looks = load_looks([season - 1, season], cache_dir, refresh_seasons)
        pbp = pbp.merge(looks, on=["game_id", "play_id"], how="left")
        players = nfl.load_player_stats([season - 1, season], cache_dir, refresh=refresh)
        return cls(pbp, season, players)

    def _build_pool(self, recent):
        plays = recent.loc[recent.play_type.isin(["pass", "run"]) & recent.down.notna()
                           & recent.two_point_attempt.ne(1) & recent.yards_gained.notna()
                           & recent.epa.notna() & recent.posteam.notna()].reset_index(drop=True)
        flag = lambda c: plays[c].fillna(0).astype(bool).to_numpy()
        seconds = plays.game_seconds_remaining
        drill = seconds.le(120) | (seconds - 1800).between(0, 120)
        group = [plays.qb_dropback.eq(1), plays.complete_pass.eq(1), plays.out_of_bounds.eq(1)]
        normal = plays.loc[~drill].groupby(group).elapsed.mean()
        plays.loc[drill, "elapsed"] = pd.Series(list(zip(*[g[drill] for g in group])),
                                                index=plays.index[drill]).map(normal)
        off_td = plays.touchdown.eq(1) & plays.td_team.eq(plays.posteam)
        def_td = plays.return_touchdown.eq(1) | (plays.touchdown.eq(1) & plays.td_team.eq(plays.defteam))
        self.pool = dict(
            kind=np.where(plays.qb_dropback.eq(1), 0, 1),
            zone=np.searchsorted(ZONE_EDGES, plays.yardline_100.to_numpy(), side="left"),
            yards=plays.yards_gained.astype(int).to_numpy(), complete=flag("complete_pass"),
            sack=flag("sack"), interception=flag("interception"), fumble_lost=flag("fumble_lost"),
            scramble=flag("qb_scramble"), air_yards=plays.air_yards.fillna(0).astype(int).to_numpy(),
            return_yards=plays.return_yards.fillna(0).astype(int).to_numpy(),
            off_td=off_td.to_numpy(), def_td=def_td.to_numpy(), oob=flag("out_of_bounds"),
            epa=np.clip(plays.epa.to_numpy(), -EPA_CLIP, EPA_CLIP), elapsed=plays.elapsed.to_numpy(),
            posteam=plays.posteam.to_numpy(), defteam=plays.defteam.to_numpy(),
            current=plays.season.eq(self.season).to_numpy(),
            rusher=plays.rusher_player_id.to_numpy(), receiver=plays.receiver_player_id.to_numpy(),
            passer=plays.passer_player_id.fillna(plays.rusher_player_id.where(plays.qb_scramble.eq(1))).to_numpy(),
            attempt=(plays.pass_attempt.eq(1) & plays.sack.ne(1) & plays.qb_scramble.ne(1)).to_numpy(),
            situation=np.select([plays.down.le(2), plays.ydstogo.le(3)], [0, 1], 2),
            **{dim: (plays[dim].fillna(-1).astype(int).to_numpy() if dim in plays else np.full(len(plays), -1))
               for dim in ("coverage", "blitz", "box")},
        )
        self._cells = {(k, z): np.flatnonzero((self.pool["kind"] == k) & (self.pool["zone"] == z))
                       for k in (0, 1) for z in range(4)}
        season_weight = np.where(self.pool["current"], 1., PRIOR_SEASON_WEIGHT)
        self._player_rows, self._player_volume = {}, {}
        for kind, column in [(0, "receiver"), (1, "rusher")]:
            mine = (self.pool["kind"] == kind) & pd.notna(self.pool[column])
            if kind == 0:
                mine &= self.pool["attempt"]
            ids = self.pool[column]
            volume = pd.Series(season_weight[mine]).groupby(ids[mine]).sum()
            self._player_volume[kind] = volume.to_dict()
            for zone in range(4):
                rows = self._cells[kind, zone]
                positions = pd.Series(np.arange(len(rows)))[mine[rows]]
                self._player_rows[kind, zone] = {pid: group.to_numpy()
                                                 for pid, group in positions.groupby(ids[rows][mine[rows]])}
        mine = (self.pool["kind"] == 0) & pd.notna(self.pool["passer"])
        ids = self.pool["passer"]
        self._qb_volume = pd.Series(season_weight[mine]).groupby(ids[mine]).sum().to_dict()
        self._qb_rows = {}
        for zone in range(4):
            rows = self._cells[0, zone]
            positions = pd.Series(np.arange(len(rows)))[mine[rows]]
            self._qb_rows[0, zone] = {pid: group.to_numpy()
                                      for pid, group in positions.groupby(ids[rows][mine[rows]])}
        frame = pd.DataFrame({"game": plays.game_id, "team": plays.posteam, "passer": self.pool["passer"],
                              "rusher": self.pool["rusher"], "kind": self.pool["kind"],
                              "scramble": self.pool["scramble"], "w": season_weight})
        starter = (frame.loc[frame.kind.eq(0)].groupby(["game", "team"]).passer
                   .agg(lambda s: s.value_counts().index[0] if s.notna().any() else None).rename("starter"))
        designed = frame.loc[frame.kind.eq(1)].join(starter, on=["game", "team"])
        designed = designed.loc[designed.starter.notna()]
        own_runs = designed.assign(own=designed.rusher.eq(designed.starter) * designed.w).groupby("starter")
        self._qb_run_share = (own_runs.own.sum() / own_runs.w.sum()).to_dict()

    def _build_looks(self):
        """How often each defense shows each look, and how often each offense faces it, by situation (early
        down, 3rd/4th and short, 3rd/4th and longer).
        """
        p = self.pool
        weight = np.where(p["current"], 1., PRIOR_SEASON_WEIGHT)
        self.look_league, self.look_defense, self.look_offense = {}, {}, {}
        for kind, dims in LOOK_DIMS.items():
            for dim in dims:
                n = len(LOOK_LABELS[dim])
                known = (p["kind"] == kind) & (p[dim] >= 0)
                frame = pd.DataFrame({"label": p[dim][known], "w": weight[known], "situation": p["situation"][known],
                                      "defense": p["defteam"][known], "offense": p["posteam"][known]})
                shares = lambda g: np.bincount(g.label, weights=g.w, minlength=n) / max(g.w.sum(), 1e-9)
                overall = shares(frame) if len(frame) else np.full(n, 1 / n)
                by_situation = np.array([shares(frame.loc[frame.situation.eq(k)]) if frame.situation.eq(k).any()
                                         else overall for k in range(3)])
                self.look_league[dim] = (overall, by_situation)
                for side, store in (("defense", self.look_defense), ("offense", self.look_offense)):
                    store[dim] = {team: (np.bincount(g.label, weights=g.w, minlength=n) + LOOK_SHRINK * overall)
                                  / (g.w.sum() + LOOK_SHRINK) for team, g in frame.groupby(side)}

    def look_probs(self, offense, defense):
        """Per look dimension: probabilities by situation (3 x labels), combining the league rate, this
        defense's tendency and what this offense usually faces.
        """
        probs = {}
        for dims in LOOK_DIMS.values():
            for dim in dims:
                overall, by_situation = self.look_league[dim]
                ratio = np.ones_like(overall)
                for store, team in ((self.look_defense, defense), (self.look_offense, offense)):
                    if team in store[dim]:
                        ratio = ratio * store[dim][team] / np.maximum(overall, 1e-9)
                table = by_situation * ratio
                probs[dim] = table / table.sum(axis=1, keepdims=True)
        return probs

    def _look_average(self, cell, weights, probs):
        """Row multipliers equal to the sampled-look conditioning averaged over this matchup's look mix, so
        team strength can be pinned before looks are applied.
        """
        rows = self._cells[cell]
        multiplier = np.ones(len(rows))
        w = weights / weights.sum()
        for dim in LOOK_DIMS[cell[0]]:
            labels, alpha = self.pool[dim][rows], LOOK_STRENGTH[dim]
            share = probs[dim].mean(axis=0)
            factor = np.full(len(rows), 1 - alpha)
            for label, p_label in enumerate(share):
                match = labels == label
                q = w[match].sum()
                if q > 0:
                    factor[match] += alpha * p_label / q
            multiplier *= factor
        return multiplier

    def _build_ratings(self):
        """Opponent-adjusted EPA per play: offense and defense effects fit jointly."""
        from sklearn.linear_model import Ridge
        p = self.pool
        teams = sorted(set(p["posteam"]) | set(p["defteam"]))
        index = {team: i for i, team in enumerate(teams)}
        weight = np.where(p["current"], 1., PRIOR_SEASON_WEIGHT)
        self.league_epa, columns = {}, {}
        for kind, label in [(0, "pass"), (1, "run")]:
            rows = np.flatnonzero(p["kind"] == kind)
            X = np.zeros((len(rows), 2 * len(teams)))
            X[np.arange(len(rows)), [index[t] for t in p["posteam"][rows]]] = 1
            X[np.arange(len(rows)), [len(teams) + index[t] for t in p["defteam"][rows]]] = 1
            model = Ridge(alpha=RATING_ALPHA).fit(X, p["epa"][rows], sample_weight=weight[rows])
            self.league_epa[kind] = float(np.average(p["epa"][rows], weights=weight[rows]))
            columns[f"{label}_offense"] = model.coef_[:len(teams)]
            columns[f"{label}_defense"] = model.coef_[len(teams):]
        self.ratings = pd.DataFrame(columns, index=pd.Index(teams, name="team"))

    def with_quarterback(self, base, offense, qb):
        """Blend the starting QB's own dropbacks into the offense's pass cells, and return how much better
        (EPA per dropback, shrunk) he is than this offense's recent passing.
        """
        volume = self._qb_volume.get(qb, 0.)
        if volume <= 0:
            return base, 0.
        share = volume / (volume + QB_SHRINK)
        p = self.pool
        season_weight = np.where(p["current"], 1., PRIOR_SEASON_WEIGHT)
        base = dict(base)
        for zone in range(4):
            cell, rows = (0, zone), self._cells[0, zone]
            own = self._qb_rows[cell].get(qb)
            if own is None:
                continue
            w = base[cell] / base[cell].sum() * (1 - share)
            own_weights = season_weight[rows[own]]
            w[own] += share * own_weights / own_weights.sum()
            base[cell] = w
        dropbacks = p["kind"] == 0
        team = dropbacks & (p["posteam"] == offense)
        mine = dropbacks & (p["passer"] == qb)
        if not team.any():
            return base, 0.
        gap = (np.average(p["epa"][mine], weights=season_weight[mine])
               - np.average(p["epa"][team], weights=season_weight[team]))
        return base, share * gap

    def matchup_tilts(self, base, offense, defense, pass_shift=0.):
        """Per-cell tilts so mean EPA = league + offense rating + defense rating, measured after defensive
        looks are mixed in, so looks reshape outcomes without moving the matchup's overall strength (the
        defense rating already includes its looks).
        """
        p, tilts = self.pool, {}
        probs = self.look_probs(offense, defense)
        base = {cell: w * self._look_average(cell, w, probs) for cell, w in base.items()}
        for kind, label in [(0, "pass"), (1, "run")]:
            rating = lambda team, side: (self.ratings.at[team, f"{label}_{side}"]
                                         if team in self.ratings.index else 0.)
            target = self.league_epa[kind] + rating(offense, "offense") + rating(defense, "defense")
            if kind == 0:
                target += pass_shift
            cells = [(kind, z) for z in range(4)]
            sizes = np.array([len(self._cells[c]) for c in cells], dtype=float)
            means = np.array([np.average(p["epa"][self._cells[c]], weights=base[c]) for c in cells])
            shift = target - (sizes / sizes.sum()) @ means
            for cell, mean in zip(cells, means):
                tilts[cell] = _solve_tilt(p["epa"][self._cells[cell]], base[cell], mean + shift)
        return tilts

    def _build_tendencies(self, current, recent):
        plays = current.loc[current.play_type.isin(["pass", "run"]) & current.down.between(1, 3)
                            & current.two_point_attempt.ne(1)].copy()
        plays["d"] = np.searchsorted(DISTANCE_EDGES, plays.ydstogo, side="left")
        plays["is_pass"] = plays.qb_dropback.eq(1)
        situation = _game_situation(plays.game_seconds_remaining.to_numpy(),
                                    plays.score_differential.to_numpy())
        self.situation_pass_rate = {name: float(plays.is_pass[situation == name].mean())
                                    for name in ("hurry", "milk", "trail")}
        plays = plays.loc[situation == "normal"]
        league = plays.groupby(["down", "d"]).is_pass.mean()
        counts = plays.groupby(["posteam", "down", "d"]).is_pass.agg(["sum", "count"])
        self.pass_rate = {}
        for team in plays.posteam.dropna().unique():
            table = np.full((4, 4), .55)
            for down in (1, 2, 3):
                for d in range(4):
                    prior = league.get((down, d), .55)
                    n_pass, n = counts.loc[(team, down, d)] if (team, down, d) in counts.index else (0, 0)
                    table[down - 1, d] = (n_pass + PASS_RATE_SHRINK * prior) / (n + PASS_RATE_SHRINK)
            table[3] = table[2]
            self.pass_rate[team] = table
        fourth = recent.loc[recent.down.eq(4) & recent.play_type.isin(["pass", "run", "punt", "field_goal"])]
        tb = np.searchsorted(GO_DISTANCE_EDGES, fourth.ydstogo, side="left")
        yb = np.searchsorted(YARDLINE_EDGES, fourth.yardline_100, side="left")
        go = fourth.play_type.isin(["pass", "run"]).to_numpy()
        fg = fourth.play_type.eq("field_goal").to_numpy()
        self.go_rate = np.zeros((4, 9))
        self.fg_share = np.zeros(9)
        for y in range(9):
            kicks = (yb == y) & ~go
            self.fg_share[y] = (fg[kicks].sum() + .5) / (kicks.sum() + 1)
            for t in range(4):
                cell = (yb == y) & (tb == t)
                self.go_rate[t, y] = (go[cell].sum() + .5) / (cell.sum() + 1)
        nullified = recent.loc[recent.play_type.eq("no_play") & recent.penalty.eq(1)
                               & recent.penalty_yards.gt(0) & recent.penalty_team.notna()]
        self.penalties = dict(
            on_offense=nullified.penalty_team.eq(nullified.posteam).to_numpy(),
            yards=nullified.penalty_yards.astype(int).to_numpy(),
            first_down=nullified.first_down_penalty.eq(1).to_numpy(),
            kind=nullified.penalty_type.fillna("Penalty").to_numpy(),
            elapsed=nullified.elapsed.to_numpy(),
        )
        snaps = current.play_type.isin(["pass", "run"]).sum()
        flags = (current.play_type.eq("no_play") & current.penalty.eq(1) & current.penalty_yards.gt(0)).sum()
        self.penalty_rate = flags / (snaps + flags)

    def _build_kicking(self, pbp, current, recent):
        kicks = pbp.loc[pbp.play_type.eq("field_goal") & pbp.kick_distance.notna()]
        self.fg_model = LogisticRegression().fit(kicks[["kick_distance"]].to_numpy(),
                                                 kicks.field_goal_result.eq("made").to_numpy())
        tries = recent.loc[recent.play_type.eq("extra_point")]
        self.xp_rate = tries.extra_point_result.eq("good").mean()
        twos = recent.two_point_conv_result.dropna()
        self.two_point_rate = twos.eq("success").mean()
        starts = current.loc[current.down.notna() & current.drive_start_transition.eq("KICKOFF")]
        self.kickoff_starts = starts.groupby(["game_id", "drive"]).yardline_100.first().to_numpy()
        punts = recent.loc[recent.play_type.eq("punt") & recent.kick_distance.notna()].sort_values("yardline_100")
        self.punts = dict(yardline=punts.yardline_100.to_numpy(), distance=punts.kick_distance.to_numpy(),
                          ret=punts.return_yards.fillna(0).to_numpy(), touchback=punts.touchback.eq(1).to_numpy())

    def _build_wp(self, history):
        plays = history.loc[history.play_type.isin(["pass", "run"]) & history.qtr.le(4)
                            & history.result.ne(0) & history.result.notna() & history.posteam.notna()]
        won = (plays.result > 0) == plays.posteam.eq(plays.home_team)
        model = LogisticRegression(max_iter=2000).fit(
            wp_features(plays.score_differential, plays.game_seconds_remaining, plays.yardline_100), won)
        self.wp_coef, self.wp_intercept = model.coef_[0], model.intercept_[0]

    def _build_names(self, pbp, player_stats):
        names = {}
        for column in [c for c in PBP_COLUMNS if c.endswith("_player_id")]:
            pairs = pbp[[column, column.replace("_id", "_name")]].dropna().drop_duplicates(column)
            names.update(zip(pairs[column], pairs[column.replace("_id", "_name")]))
        self.positions, self.headshots = {}, {}
        if player_stats is not None:
            latest = player_stats.sort_values(["season", "week"]).drop_duplicates("player_id", keep="last")
            names.update(zip(latest.player_id, latest.player_display_name))
            self.positions = dict(zip(latest.player_id, latest.position))
            if "headshot_url" in latest:
                self.headshots = {k: v for k, v in zip(latest.player_id, latest.headshot_url) if isinstance(v, str)}
            self.player_teams = latest[["player_id", "player_display_name", "team"]]
        self.names = names

    def fg_probability(self, distance):
        return float(self.fg_model.predict_proba([[distance]])[0, 1])

    def live_wp(self, diff, seconds, yardline):
        """Possession team's win logit from score, clock and field position alone."""
        return self.wp_intercept + wp_features([diff], [seconds], [yardline])[0] @ self.wp_coef

    def base_weights(self, offense, defense):
        """Mix league, offense and defense rows; per (kind, zone) cell, before any tilt."""
        p = self.pool
        season_weight = np.where(p["current"], 1., PRIOR_SEASON_WEIGHT)
        weights = {}
        for cell, rows in self._cells.items():
            total = np.zeros(len(rows))
            for key, raw in [("league", np.ones(len(rows))),
                             ("offense", season_weight[rows] * (p["posteam"][rows] == offense)),
                             ("defense", season_weight[rows] * (p["defteam"][rows] == defense))]:
                if raw.sum() > 0:
                    total += POOL_MIX[key] * raw / raw.sum()
            weights[cell] = total
        return weights

    def sampler(self, base, tilts, theta, players=None, pass_shift=0.):
        return _Sampler(self, base, tilts, theta, players or {}, pass_shift)

    def team_players(self, team, qb=None, out=(), returning=()):
        """Recent, active players with exponentially weighted usage shares."""
        cur = self.current
        games = (cur.loc[cur.posteam.eq(team) | cur.defteam.eq(team), ["game_id", "week"]]
                 .drop_duplicates("game_id").sort_values("week"))
        if games.empty:
            raise ValueError(f"No {self.season} plays for {team}")
        ago = dict(zip(games.game_id, np.arange(len(games))[::-1]))
        recent_games = set(games.game_id.tail(2))
        offense = cur.loc[cur.posteam.eq(team)].copy()
        defense = cur.loc[cur.defteam.eq(team)].copy()
        for frame in (offense, defense):
            frame["w"] = .5 ** (frame.game_id.map(ago) / SHARE_HALF_LIFE_GAMES)
        latest_off = offense.loc[offense.game_id.isin(recent_games)]
        latest_def = defense.loc[defense.game_id.isin(recent_games)]
        active_off = set(pd.concat([latest_off[c] for c in ["passer_player_id", "receiver_player_id",
                                                             "rusher_player_id", "kicker_player_id"]]).dropna())
        active_def = set(pd.concat([latest_def[c] for c in ["sack_player_id", "half_sack_1_player_id",
                                                             "half_sack_2_player_id", "interception_player_id",
                                                             "pass_defense_1_player_id"]]).dropna())
        missing = [name for name in out if self._match_player(name, team) is None]
        if missing:
            raise ValueError(f"{team} out list: no player found for {missing}")
        out_ids = {self._match_player(name, team) for name in out}
        active_off -= out_ids
        active_def -= out_ids

        def shares(rows, credits, active, fallback):
            counts = pd.concat([rows.groupby(column).w.sum() * credit for column, credit in credits])
            counts = counts.groupby(level=0).sum()
            counts = counts.loc[counts.index.isin(active) & counts.gt(0)].sort_values(ascending=False)
            if counts.empty:
                return np.array([fallback]), np.array([1.])
            return counts.index.to_numpy(), np.cumsum(counts.to_numpy()) / counts.sum()

        targets = offense.loc[offense.pass_attempt.eq(1) & offense.sack.ne(1)]
        carries = offense.loc[offense.play_type.eq("run") & offense.qb_scramble.ne(1)]
        team_id = f"{team}_TEAM"
        self.names.setdefault(team_id, f"{team} defense")
        back = {name: self._match_player(name, team) for name in returning}
        missing = [name for name, pid in back.items() if pid is None]
        if missing:
            raise ValueError(f"{team} returning list: no player found for {missing}")
        if set(back.values()) & out_ids:
            raise ValueError(f"{team}: a player is listed as both out and returning")
        returning_shares = {kind: self.returning_shares(team, kind, set(back.values()))
                            for kind in ("targets", "carries")}

        def with_returning(kind, ids, cdf):
            probs = dict(zip(ids, np.diff(np.r_[0., cdf])))
            probs.pop(team_id, None)
            for pid, share in returning_shares[kind].items():
                probs[pid] = probs.get(pid, 0.) + share
            if not probs:
                return ids, cdf
            ordered = sorted(probs.items(), key=lambda item: -item[1])
            return (np.array([pid for pid, _ in ordered]),
                    np.cumsum([p for _, p in ordered]) / sum(p for _, p in ordered))

        players = dict(
            targets=with_returning("targets", *shares(targets, [("receiver_player_id", 1)], active_off, team_id)),
            carries=with_returning("carries", *shares(carries, [("rusher_player_id", 1)], active_off, team_id)),
            sacks=shares(defense, [("sack_player_id", 1), ("half_sack_1_player_id", .5),
                                   ("half_sack_2_player_id", .5)], active_def, team_id),
            picks=shares(defense, [("interception_player_id", 1), ("pass_defense_1_player_id", .25)],
                         active_def, team_id),
        )
        last = offense.loc[offense.game_id.eq(games.game_id.iloc[-1])]
        dropbacks = last.loc[last.qb_dropback.eq(1)].passer_player_id.value_counts()
        recent_starter = dropbacks.index[0] if len(dropbacks) else None
        players["qb"] = self._match_player(qb, team) if qb else None
        if players["qb"] is None:
            players["qb"] = recent_starter or team_id
        ids, cdf = players["carries"]
        probs = dict(zip(ids, np.diff(np.r_[0., cdf])))
        other_qbs = {pid for pid in probs if self.positions.get(pid) == "QB"}
        for benched in ({recent_starter, players["qb"]} | other_qbs) - {None}:
            probs.pop(benched, None)
        share = self._qb_run_share.get(players["qb"], QB_DESIGNED_RUN_DEFAULT)
        total = sum(probs.values())
        probs = {pid: p * (1 - share) / total for pid, p in probs.items()} if total > 0 else {}
        probs[players["qb"]] = probs.get(players["qb"], 0.) + (share if total > 0 else 1.)
        ordered = sorted(probs.items(), key=lambda item: -item[1])
        players["carries"] = (np.array([pid for pid, _ in ordered]),
                              np.cumsum([p for _, p in ordered]) / sum(p for _, p in ordered))
        kicks = offense.loc[offense.kicker_player_id.notna() & offense.play_type.isin(["field_goal", "extra_point"])]
        players["kicker"] = kicks.kicker_player_id.iloc[-1] if len(kicks) else team_id
        return players

    def returning_shares(self, team, kind, player_ids):
        """Each player's mean share of the team's targets or designed carries over his last RETURNING_GAMES
        games played for it (this season or last).
        """
        if not player_ids:
            return {}
        u = self._usage.loc[self._usage.posteam.eq(team)]
        if kind == "targets":
            rows, column = u.loc[u.pass_attempt.eq(1) & u.sack.ne(1)], "receiver_player_id"
        else:
            rows, column = u.loc[u.play_type.eq("run") & u.qb_scramble.ne(1)], "rusher_player_id"
        per_game = rows.groupby("game_id")[column].value_counts(normalize=True).unstack(fill_value=0.)
        order = rows.drop_duplicates("game_id").sort_values(["season", "week"]).game_id
        per_game = per_game.reindex(order[order.isin(per_game.index)])
        u_any = pd.concat([u.receiver_player_id, u.rusher_player_id]).groupby(
            pd.concat([u.game_id, u.game_id])).agg(set)
        result = {}
        for pid in player_ids:
            played = [g for g in per_game.index if pid in u_any.get(g, ())]
            if not played:
                raise ValueError(f"{team}: {self.names.get(pid, pid)} has no {self.season - 1}-{self.season} "
                                 "games to set a returning share")
            share = per_game.loc[played[-RETURNING_GAMES:]].get(pid, pd.Series(0.)).mean()
            if share > 0:
                result[pid] = float(share)
        return result

    def _match_player(self, name, team):
        if not hasattr(self, "player_teams"):
            return None
        table = self.player_teams
        hits = table.loc[table.player_display_name.str.lower().eq(str(name).strip().lower())]
        on_team = hits.loc[hits.team.replace(nfl.ALIASES).eq(team)]
        hits = on_team if len(on_team) else hits
        return hits.player_id.iloc[0] if len(hits) else None


class _Game:
    def __init__(self, matchup, samplers, rng, record=False):
        self.m, self.league, self.samplers, self.rng = matchup, matchup.league, samplers, rng
        self.home, self.away = matchup.home, matchup.away
        self.score = {self.home: 0, self.away: 0}
        self.stats, self.events, self.record = {}, [], record
        self.deltas = []
        self.ot = self.done = False
        self.ot_drive = 0
        self.form = {team: int(rng.integers(FORM_LEVELS)) for team in (self.home, self.away)}
        self.pass_lean = {team: float(rng.normal(0., PASS_LEAN_SD)) for team in (self.home, self.away)}

    def other(self, team):
        return self.away if team == self.home else self.home

    def x(self, team, y):
        return 100 - y if team == self.home else y

    def spot(self, team, y):
        y = int(round(y))
        return "50" if y == 50 else (f"{self.other(team)} {y}" if y < 50 else f"{team} {100 - y}")

    def name(self, pid):
        return short_name(self.league.names.get(pid, pid))

    def stat(self, team, pid, key, value=1):
        row = self.stats.get(pid)
        if row is None:
            row = self.stats[pid] = [team, np.zeros(len(STATS))]
        row[1][S[key]] += value
        if self.record:
            self.deltas.append([team, pid, S[key], value])

    def pick(self, team, role):
        ids, cdf = self.m.players[team][role]
        return ids[min(np.searchsorted(cdf, self.rng.random()), len(ids) - 1)]

    def advance(self, seconds):
        floor = self.half_end if self.ot else max(self.half_end, ((self.g - 1) // 900) * 900)
        self.g = int(max(self.g - int(round(seconds)), floor))

    def clock(self):
        if self.ot:
            return "OT", self.g
        q = min(4, 1 + (3600 - self.g) // 900)
        return f"Q{q}", self.g - (4 - q) * 900

    def situation(self):
        q, left = self.clock()
        return dict(q=q, clock=int(left), pos=self.pos, down=self.down, togo=self.togo, y=self.y,
                    x0=self.x(self.pos, self.y),
                    fd=None if self.togo >= self.y else self.x(self.pos, self.y - self.togo))

    def log(self, before, text, tag="play"):
        if not self.record:
            return
        home_logit = self.home_wp_logit()
        if self.done:
            wp = 1. if self.score[self.home] > self.score[self.away] else (
                0. if self.score[self.home] < self.score[self.away] else .5)
        else:
            wp = float(1 / (1 + np.exp(-home_logit)))
        self.events.append(dict(before, text=text, tag=tag, h=self.score[self.home], a=self.score[self.away],
                                wp=round(wp, 4), x1=self.x(self.pos, self.y), pos1=self.pos,
                                t=round(self.elapsed_minutes(), 3), d=self.deltas))
        self.deltas = []

    def elapsed_minutes(self):
        return (600 - self.g) / 60 + 60 if self.ot else (3600 - self.g) / 60

    def home_wp_logit(self):
        diff = self.score[self.pos] - self.score[self.other(self.pos)]
        remaining = (self.g / 3600) if not self.ot else 0.
        logit = self.league.live_wp(diff, self.g, self.y)
        logit = logit if self.pos == self.home else -logit
        return logit + self.m.pregame_logit * remaining

    def give(self, team, y):
        self.pos, self.y = team, int(np.clip(round(y), 1, 99))
        self.down, self.togo = 1, min(10, self.y)
        if self.ot:
            self.ot_drive += 1
        self.check_ot()

    def check_ot(self):
        if not self.ot or self.done:
            return
        first, second = self.ot_first, self.other(self.ot_first)
        if self.ot_drive >= 3 and self.score[first] != self.score[second]:
            self.done = True
        elif self.score[second] > self.score[first]:
            self.done = True

    def run(self):
        receiver = self.home if self.rng.random() < .5 else self.away
        self.g, self.half_end = 3600, 1800
        self.kickoff(self.other(receiver))
        while self.g > 1800:
            self.snap()
        self.g, self.half_end = 1800, 0
        self.kickoff(receiver)
        while self.g > 0:
            self.snap()
        if self.score[self.home] == self.score[self.away]:
            self.ot, self.g, self.half_end = True, 600, 0
            self.ot_first = self.home if self.rng.random() < .5 else self.away
            self.kickoff(self.other(self.ot_first))
            while self.g > 0 and not self.done:
                self.snap()
        self.done = True
        if self.record and self.events:
            self.events[-1]["wp"] = 1. if self.score[self.home] > self.score[self.away] else (
                0. if self.score[self.home] < self.score[self.away] else .5)
        return self

    def kickoff(self, kicking):
        receiving = self.other(kicking)
        before = dict(q=self.clock()[0], clock=int(self.clock()[1]), pos=kicking, down=0, togo=0,
                      y=65, x0=self.x(kicking, 65), fd=None)
        diff = self.score[kicking] - self.score[receiving]
        late = (self.half_end == 0 and self.g <= 180 and -16 <= diff < 0) or (self.ot and diff < 0)
        kicker = self.name(self.m.players[kicking]["kicker"])
        if late and self.rng.random() < .5:
            if self.rng.random() < .12:
                self.give(kicking, 55)
                return self.log(before, f"{kicker} onside kick — RECOVERED by {kicking}!", "turnover")
            self.give(receiving, 45)
            self.advance(4)
            return self.log(before, f"{kicker} onside kick recovered by {receiving} at the {self.spot(receiving, 45)}")
        start = self.rng.choice(self.league.kickoff_starts)
        self.give(receiving, start)
        self.advance(4 if start == 65 else 6)
        text = f"{kicker} kicks off — touchback" if start == 65 else \
            f"{kicker} kicks off, returned to the {self.spot(receiving, start)}"
        self.log(before, text, "kick")

    def snap(self):
        if self.done:
            return
        L, rng = self.league, self.rng
        off = self.pos
        de = self.other(off)
        y, down, togo = self.y, self.down, self.togo
        left = self.g - self.half_end
        diff = self.score[off] - self.score[de]
        second_half = self.half_end == 0
        late = second_half and self.g <= 300
        hurry = left <= 120 and (not second_half or diff <= 0)
        milk = late and diff > 0 and not self.ot
        before = self.situation()
        qb = self.m.players[off]["qb"]

        if (milk and left <= 40 * (4 - down) + 2) or (not second_half and left <= 30 and y >= 60 and not hurry):
            self.advance(40 if second_half else left)
            self.down, self.togo, self.y = down + 1, togo + 1, min(99, y + 1)
            self.log(before, f"{self.name(qb)} kneels")
            if self.down > 4 and self.g > self.half_end:
                self.give(de, 100 - self.y)
            return
        kick = y + 18
        in_range = kick <= MAX_FG
        if in_range and left <= 20 and (not second_half or self.ot or -3 <= diff <= 0):
            return self.field_goal(before)
        if down == 4:
            choice = self.fourth_down(y, togo, diff, second_half, late, milk)
            if choice == "punt":
                return self.punt(before)
            if choice == "fg":
                return self.field_goal(before)
        if rng.random() < L.penalty_rate:
            return self.penalty(before)

        rate = L.pass_rate.get(off, np.full((4, 4), .55))[down - 1, _bin(DISTANCE_EDGES, togo)]
        if hurry:
            rate = max(rate, L.situation_pass_rate["hurry"])
        elif milk:
            rate = min(rate, L.situation_pass_rate["milk"])
        elif second_half and self.g <= 900 and diff <= -9:
            rate = max(rate, L.situation_pass_rate["trail"])
        rate = _expit(_logit(rate) + self.pass_lean[off])
        kind = 0 if rng.random() < rate else 1
        zone = _bin(ZONE_EDGES, y)
        chosen = self.pick(off, "targets" if kind == 0 else "carries")
        situation = 0 if down <= 2 else (1 if togo <= 3 else 2)
        look = tuple(min(np.searchsorted(cdf[situation], rng.random()), cdf.shape[1] - 1)
                     for cdf in self.m.look_cdfs[off][kind])
        before["look"] = " · ".join(filter(None, (LOOK_LABELS[dim][label] for dim, label
                                                  in zip(LOOK_DIMS[kind], look))))
        rows, cdf = self.samplers[off].get(kind, zone, chosen, look, self.form[off] if kind == 0 else None)
        i = rows[min(np.searchsorted(cdf, rng.random()), len(rows) - 1)]
        P = L.pool
        yards = int(P["yards"][i])
        td = bool(P["off_td"][i]) if zone < 3 else yards >= y
        yards = y if td else min(yards, y - 1)
        fumble = bool(P["fumble_lost"][i])
        if fumble:
            td, yards = False, min(yards, y - 1)
        stopped = False
        carrier = qb

        if P["sack"][i]:
            yards = min(yards, 0)
            td = False
            sacker = self.pick(de, "sacks")
            self.stat(off, qb, "sacked")
            self.stat(off, qb, "sack_yds", -yards)
            self.stat(de, sacker, "def_sack")
            text = f"{self.name(qb)} sacked by {self.name(sacker)} for {yards} yards"
        elif P["interception"][i]:
            target, picker = chosen, self.pick(de, "picks")
            self.stat(off, qb, "pass_att")
            self.stat(off, qb, "pass_int")
            self.stat(off, target, "tgt")
            self.stat(de, picker, "def_int")
            catch = y - int(P["air_yards"][i])
            text = f"{self.name(qb)} pass intended for {self.name(target)} INTERCEPTED by {self.name(picker)}"
            self.advance(P["elapsed"][i])
            if P["def_td"][i]:
                return self.defensive_touchdown(de, before, text + " — returned for a TOUCHDOWN!")
            new_y = 80 if catch <= 0 else 100 - catch - int(P["return_yards"][i])
            if new_y <= 0:
                return self.defensive_touchdown(de, before, text + " — returned for a TOUCHDOWN!")
            self.give(de, min(new_y, 99))
            return self.log(before, text, "turnover")
        elif P["scramble"][i]:
            self.stat(off, qb, "rush_car")
            self.stat(off, qb, "rush_yds", yards)
            text = f"{self.name(qb)} scrambles for {yards} yards"
        elif kind == 0:
            target = chosen
            self.stat(off, qb, "pass_att")
            self.stat(off, target, "tgt")
            if P["complete"][i]:
                carrier = target
                self.stat(off, qb, "pass_cmp")
                self.stat(off, qb, "pass_yds", yards)
                self.stat(off, target, "rec")
                self.stat(off, target, "rec_yds", yards)
                text = f"{self.name(qb)} pass complete to {self.name(target)} for {yards} yards"
            else:
                yards, td, fumble, stopped = 0, False, False, True
                text = f"{self.name(qb)} pass incomplete intended for {self.name(target)}"
        else:
            carrier = chosen
            self.stat(off, carrier, "rush_car")
            self.stat(off, carrier, "rush_yds", yards)
            text = f"{self.name(carrier)} rushes for {yards} yards"

        elapsed = P["elapsed"][i]
        stopped = stopped or bool(P["oob"][i])
        if hurry and not stopped:
            elapsed = min(elapsed, rng.integers(6, 15))
        elif late and diff < 0 and not stopped:
            elapsed = min(elapsed, 22)
        elif milk and not stopped:
            elapsed = max(elapsed, 38)
        self.advance(elapsed)

        if td:
            if P["sack"][i] or P["scramble"][i] or kind == 1:
                self.stat(off, carrier, "rush_td")
            else:
                self.stat(off, qb, "pass_td")
                self.stat(off, carrier, "rec_td")
            self.y = 0
            return self.touchdown(off, before, text + " — TOUCHDOWN!")
        if fumble:
            self.stat(off, carrier, "fum_lost")
            if P["def_td"][i]:
                return self.defensive_touchdown(de, before, text + ", FUMBLES — returned for a TOUCHDOWN!")
            self.give(de, 100 - (y - yards))
            return self.log(before, text + f", FUMBLES — recovered by {de}", "turnover")
        if y - yards >= 100:
            self.score[de] += 2
            self.y = 99
            self.log(before, text + " — SAFETY!", "score")
            self.check_ot()
            return self.free_kick(off)
        self.y = y - yards
        self.togo = togo - yards
        if self.togo <= 0:
            self.down, self.togo = 1, min(10, self.y)
        elif down == 4:
            self.give(de, 100 - self.y)
            return self.log(before, text + " — turnover on downs", "turnover")
        else:
            self.down = down + 1
        self.log(before, text)

    def fourth_down(self, y, togo, diff, second_half, late, milk):
        L, rng = self.league, self.rng
        in_range = y + 18 <= MAX_FG
        desperate = (late and diff < 0) or (self.ot and diff < 0)
        if desperate:
            return "fg" if in_range and diff >= -3 else "go"
        go = L.go_rate[_bin(GO_DISTANCE_EDGES, togo), _bin(YARDLINE_EDGES, y)]
        if milk and y > 40:
            go *= .5
        if rng.random() < go:
            return "go"
        if in_range and rng.random() < L.fg_share[_bin(YARDLINE_EDGES, y)]:
            return "fg"
        return "punt" if y > 35 or not in_range else "fg"

    def field_goal(self, before):
        off, de = self.pos, self.other(self.pos)
        kicker = self.m.players[off]["kicker"]
        distance = self.y + 18
        self.stat(off, kicker, "fga")
        self.advance(5)
        if self.rng.random() < self.league.fg_probability(distance):
            self.stat(off, kicker, "fgm")
            self.score[off] += 3
            self.log(before, f"{self.name(kicker)} {distance}-yard field goal is GOOD", "score")
            self.check_ot()
            return self.kick_after_score(off)
        self.give(de, min(80, 100 - (self.y + 7)))
        self.log(before, f"{self.name(kicker)} {distance}-yard field goal is NO GOOD", "miss")

    def punt(self, before):
        off, de = self.pos, self.other(self.pos)
        P, y = self.league.punts, self.y
        lo, hi = np.searchsorted(P["yardline"], [y - 6, y + 6])
        if hi - lo < 20:
            lo, hi = np.searchsorted(P["yardline"], [y - 15, y + 15])
        i = self.rng.integers(lo, max(hi, lo + 1))
        land = y - P["distance"][i]
        self.advance(7)
        if land <= 0 or P["touchback"][i]:
            self.give(de, 80)
            return self.log(before, f"{off} punts into the end zone — touchback", "kick")
        self.give(de, 100 - (land + P["ret"][i]))
        self.log(before, f"{off} punts {int(P['distance'][i])} yards, {de} takes over at the {self.spot(de, self.y)}",
                 "kick")

    def penalty(self, before):
        P, rng = self.league.penalties, self.rng
        i = rng.integers(len(P["yards"]))
        off, y = self.pos, self.y
        yards = int(P["yards"][i])
        self.advance(P["elapsed"][i])
        if P["on_offense"][i]:
            yards = min(yards, (100 - y) // 2)
            self.y, self.togo = y + yards, self.togo + yards
            who = off
        else:
            yards = min(yards, max(1, y // 2))
            self.y = max(1, y - yards)
            self.togo -= yards
            if P["first_down"][i] or self.togo <= 0:
                self.down, self.togo = 1, min(10, self.y)
            who = self.other(off)
        self.log(before, f"PENALTY on {who}: {P['kind'][i]}, {yards} yards", "penalty")

    def try_after(self, team):
        diff = self.score[team] - self.score[self.other(team)]
        kicker = self.m.players[team]["kicker"]
        if self.half_end == 0 and self.g <= 900 and diff in (-10, -5, -2, 1, 5):
            good = self.rng.random() < self.league.two_point_rate
            self.score[team] += 2 * good
            return " Two-point try " + ("GOOD." if good else "fails.")
        self.stat(team, kicker, "xpa")
        good = self.rng.random() < self.league.xp_rate
        if good:
            self.stat(team, kicker, "xpm")
        self.score[team] += good
        return " Extra point good." if good else " Extra point MISSED."

    def touchdown(self, team, before, text):
        self.score[team] += 6
        self.check_ot()
        if not self.done:
            text += self.try_after(team)
        self.log(before, text, "score")
        self.kick_after_score(team)

    def kick_after_score(self, team):
        if not self.done and self.g > self.half_end:
            self.kickoff(team)

    def defensive_touchdown(self, team, before, text):
        self.pos, self.y = team, 0
        return self.touchdown(team, before, text)

    def free_kick(self, kicking):
        if self.done or self.g <= self.half_end:
            return
        start = int(np.clip(self.rng.choice(self.league.kickoff_starts) - 12, 35, 80))
        before = self.situation()
        self.give(self.other(kicking), start)
        self.log(before, f"{kicking} free kick after the safety", "kick")

    def box(self):
        rows = [dict(team=team, player_id=pid, **dict(zip(STATS, values)))
                for pid, (team, values) in self.stats.items()]
        return pd.DataFrame(rows, columns=["team", "player_id"] + STATS)


class Matchup:
    def __init__(self, league, home, away, home_qb=None, away_qb=None, out_players=None,
                 returning_players=None):
        self.league, self.home, self.away = league, home, away
        out_players, returning_players = out_players or {}, returning_players or {}
        self.out = {team: list(out_players.get(team, ())) for team in (home, away)}
        self.returning = {team: list(returning_players.get(team, ())) for team in (home, away)}
        self.players = {home: league.team_players(home, home_qb, self.out[home], self.returning[home]),
                        away: league.team_players(away, away_qb, self.out[away], self.returning[away])}
        self._base, self._tilts, self.qb_shift = {}, {}, {}
        for offense, defense in ((home, away), (away, home)):
            base, shift = league.with_quarterback(league.base_weights(offense, defense), offense,
                                                  self.players[offense]["qb"])
            self._base[offense], self.qb_shift[offense] = base, shift
            self._tilts[offense] = league.matchup_tilts(base, offense, defense, shift)
        self._samplers = {}
        self.pregame_logit = 0.
        self.look_cdfs = {}
        for offense, defense in ((home, away), (away, home)):
            probs = league.look_probs(offense, defense)
            self.look_cdfs[offense] = {kind: [np.cumsum(probs[dim], axis=1) for dim in dims]
                                       for kind, dims in LOOK_DIMS.items()}

    def samplers(self, theta):
        key = round(float(theta), 6)
        if key not in self._samplers:
            if len(self._samplers) >= 2:
                self._samplers.pop(next(iter(self._samplers)))
            players = {team: {0: self.players[team]["targets"][0], 1: self.players[team]["carries"][0],
                              "qb": self.players[team]["qb"]} for team in (self.home, self.away)}
            self._samplers[key] = {
                team: _FormSamplers(self.league, self._base[team], self._tilts[team], sign * theta, players[team])
                for team, sign in ((self.home, 1), (self.away, -1))}
        return self._samplers[key]

    def play(self, seed, theta=0., record=False):
        return _Game(self, self.samplers(theta), np.random.default_rng(seed), record).run()

    def simulate(self, n, theta=0., seed=0, keep_players=True):
        games, players = [], []
        for sim in range(n):
            game = self.play(seed + sim, theta)
            games.append((sim, seed + sim, game.score[self.home], game.score[self.away], game.ot))
            if keep_players:
                box = game.box()
                box.insert(0, "sim", sim)
                players.append(box)
        games = pd.DataFrame(games, columns=["sim", "seed", "home_score", "away_score", "overtime"])
        return games, (pd.concat(players, ignore_index=True) if keep_players else None)

    def home_win_rate(self, theta, n, seed):
        games, _ = self.simulate(n, theta, seed, keep_players=False)
        return _win_rate(games)

    def calibrate(self, target, n=300, seed=10_000, iterations=10):
        """Bisection on the EPA tilt; common random numbers keep it monotone enough."""
        lo, hi = -1.5, 1.5
        for _ in range(iterations):
            mid = (lo + hi) / 2
            if self.home_win_rate(mid, n, seed) < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2


class _FormSamplers:
    """One team's samplers for every passing-form level, built on first use. Runs and the default (level
    None) use the middle level, which has no form shift.
    """

    def __init__(self, league, base, tilts, theta, players):
        from statistics import NormalDist
        self.args = (league, base, tilts, theta, players)
        self.middle = FORM_LEVELS // 2
        quantiles = [NormalDist().inv_cdf((i + .5) / FORM_LEVELS) for i in range(FORM_LEVELS)]
        self.shifts = [GAME_FORM_SD * q for q in quantiles]
        self.levels = {}

    def level(self, index):
        hit = self.levels.get(index)
        if hit is None:
            hit = self.levels[index] = _Sampler(*self.args, pass_shift=self.shifts[index])
        return hit

    def get(self, kind, zone, pid, look, level=None):
        return self.level(self.middle if level is None else level).get(kind, zone, pid, look)


class _Sampler:
    """Cumulative outcome distributions, built on first use, per (play kind, field zone, ball carrier or
    target, defensive look).
    """

    def __init__(self, league, base, tilts, theta, players, pass_shift=0.):
        self.league, self.cache = league, {}
        self.weights = {}
        self.shift = {cell: theta + (pass_shift if cell[0] == 0 else 0.) for cell in league._cells}
        for cell, rows in league._cells.items():
            w = base[cell] * np.exp((tilts[cell] + self.shift[cell]) * league.pool["epa"][rows])
            self.weights[cell] = w / w.sum()
        self.tilts, self.theta = tilts, theta
        self.mixable = {kind: {pid for pid in players.get(kind, ()) if league._player_volume[kind].get(pid, 0.) > 0}
                        for kind in (0, 1)}


    def get(self, kind, zone, pid, look):
        pid = pid if pid in self.mixable.get(kind, ()) else None
        key = (kind, zone, pid, look)
        hit = self.cache.get(key)
        if hit is None:
            hit = self.cache[key] = self._build(kind, zone, pid, look)
        return hit

    def _build(self, kind, zone, pid, look):
        L, cell = self.league, (kind, zone)
        rows, w = L._cells[cell], self.weights[cell]
        own = L._player_rows[cell].get(pid) if pid is not None else None
        if own is not None:
            attempt = L.pool["attempt"][rows] if kind == 0 else np.ones(len(rows), bool)
            volume = L._player_volume[kind][pid]
            share = volume / (volume + (TARGET_SHRINK if kind == 0 else CARRY_SHRINK))
            own_weights = (np.where(L.pool["current"][rows[own]], 1., PRIOR_SEASON_WEIGHT)
                           * np.exp((self.tilts[cell] + self.shift[cell]) * L.pool["epa"][rows[own]]))
            mass = w[attempt].sum()
            w = w.copy()
            w[attempt] *= 1 - share
            w[own] += share * mass * own_weights / own_weights.sum()
        for dim, label in zip(LOOK_DIMS[kind], look):
            match = L.pool[dim][rows] == label
            q = w[match].sum() / w.sum()
            if q > 0:
                alpha = LOOK_STRENGTH[dim]
                w = w * ((1 - alpha) + alpha * match / q)
        cumulative = np.cumsum(w)
        return rows, cumulative / cumulative[-1]


def look_table(league, home, away):
    """What each defense is expected to show this offense, overall and by situation."""
    rows = []
    for offense, defense in ((away, home), (home, away)):
        probs = league.look_probs(offense, defense)
        overall = {dim: probs[dim].mean(axis=0) for dim in probs}
        row = {"defense": defense, "vs offense": offense}
        for label, share in zip(LOOK_LABELS["coverage"], overall["coverage"]):
            row[label] = share
        row["Blitz (early downs)"] = probs["blitz"][0, 1]
        row["Blitz (3rd & long)"] = probs["blitz"][2, 1]
        for label, share in zip(LOOK_LABELS["box"], probs["box"][0]):
            row[f"{label} (early-down runs)"] = share
        rows.append(row)
    return pd.DataFrame(rows)


def _game_situation(game_seconds, score_diff):
    """Label each snap as the snap loop sees it: hurry, milk, trail or normal."""
    second = game_seconds <= 1800
    left = np.where(second, game_seconds, game_seconds - 1800)
    hurry = (left <= 120) & (~second | (score_diff <= 0))
    milk = second & (game_seconds <= 300) & (score_diff > 0)
    trail = second & (game_seconds <= 900) & (score_diff <= -9)
    return np.select([hurry, milk, trail], ["hurry", "milk", "trail"], "normal")


def _solve_tilt(epa, weights, target, iterations=25):
    """Newton steps for lambda with weighted mean of epa under exp(lambda*epa) = target."""
    lam = 0.
    for _ in range(iterations):
        w = weights * np.exp(lam * epa)
        w = w / w.sum()
        mean = w @ epa
        variance = w @ (epa - mean) ** 2
        step = (target - mean) / max(variance, 1e-6)
        lam = float(np.clip(lam + step, -3, 3))
        if abs(step) < 1e-6:
            break
    return lam


def matchup_table(league, home, away):
    """Each side's offense rating next to the opposing defense rating, in EPA/play and rank."""
    r = league.ratings
    rank = lambda column, ascending: r[column].rank(ascending=ascending).astype(int)
    rows = []
    for offense, defense in [(away, home), (home, away)]:
        for label in ("pass", "run"):
            rows.append({
                "offense": offense, "play": label,
                "offense EPA/play (+ better)": r.at[offense, f"{label}_offense"],
                "offense rank": rank(f"{label}_offense", False)[offense],
                "defense": defense,
                "defense EPA allowed (+ worse)": r.at[defense, f"{label}_defense"],
                "defense rank (32 = worst)": rank(f"{label}_defense", True)[defense],
                "expected edge": r.at[offense, f"{label}_offense"] + r.at[defense, f"{label}_defense"],
            })
    return pd.DataFrame(rows)


def _win_rate(games):
    return float(((games.home_score > games.away_score) + .5 * (games.home_score == games.away_score)).mean())


def _expit(x):
    return float(1 / (1 + np.exp(-x)))


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(np.log(p / (1 - p)))


class SimulationResult:
    def __init__(self, matchup, games, players, theta, model_probability, target, raw_probability, game_id=None):
        self.matchup, self.games, self.players, self.theta = matchup, games, players, theta
        self.model_probability, self.target, self.raw_probability = model_probability, target, raw_probability
        self.game_id = game_id
        self.home, self.away = matchup.home, matchup.away

    @property
    def home_win_rate(self):
        return _win_rate(self.games)

    def summary(self):
        g = self.games
        return pd.Series({
            "model home win %": self.model_probability, "play-by-play only home win %": self.raw_probability,
            "simulated home win %": self.home_win_rate, "ties %": (g.home_score == g.away_score).mean(),
            "overtime %": g.overtime.mean(), f"{self.away} median points": g.away_score.median(),
            f"{self.home} median points": g.home_score.median(),
            "median total": (g.home_score + g.away_score).median(),
            f"median margin ({self.home})": (g.home_score - g.away_score).median(), "EPA tilt": self.theta,
            "simulations": len(g),
        })

    def typical_features(self):
        """Per-simulation score, team yards and turnovers, and the key players' lines."""
        g, p = self.games.set_index("sim"), self.players
        features = pd.DataFrame({"home_score": g.home_score, "away_score": g.away_score,
                                 "margin": g.home_score - g.away_score})
        weights = dict.fromkeys(features, 3.)
        team = (p.assign(yards=p.pass_yds - p.sack_yds + p.rush_yds, turnovers=p.pass_int + p.fum_lost)
                .groupby(["sim", "team"])[["yards", "turnovers"]].sum().unstack("team"))
        for stat in ("yards", "turnovers"):
            for t in (self.home, self.away):
                features[f"{t} {stat}"] = team[(stat, t)]
                weights[f"{t} {stat}"] = 1.
        means = p.groupby(["team", "player_id"])[["pass_yds", "rush_yds", "rec_yds"]].sum() / len(g)
        names = self.matchup.league.names
        for t in (self.home, self.away):
            m = means.loc[t]
            key = [(m.pass_yds.idxmax(), "pass_yds"), (m.rush_yds.idxmax(), "rush_yds")]
            key += [(pid, "rec_yds") for pid in m.rec_yds.nlargest(2).index]
            for pid, stat in key:
                rows = p.loc[p.team.eq(t) & p.player_id.eq(pid)].set_index("sim")[stat]
                column = f"{names.get(pid, pid)} {stat}"
                features[column] = rows
                weights[column] = 1.
        return features.fillna(0), pd.Series(weights)

    def typical_sim(self):
        """Medoid-style pick: among games won by the more frequent winner, the real simulated game closest
        to the median on score, team yards and turnovers, and each team's main passer, rusher and two
        receivers.
        """
        features, weights = self.typical_features()
        spread = features.std().replace(0, 1)
        distance = ((((features - features.median()) / spread) ** 2) * weights).sum(axis=1)
        home_favored = self.home_win_rate >= .5
        won = features.margin > 0 if home_favored else features.margin < 0
        return int(distance[won].idxmin())

    def replay(self, sim):
        seed = int(self.games.loc[self.games.sim.eq(sim), "seed"].iloc[0])
        return self.matchup.play(seed, self.theta, record=True)

    def projections(self):
        names, positions = self.matchup.league.names, self.matchup.league.positions
        n = len(self.games)
        totals = self.players.groupby(["team", "player_id"])[STATS].sum() / n
        any_td = (self.players.assign(scored=(self.players.rush_td + self.players.rec_td) > 0)
                  .groupby(["team", "player_id"]).scored.sum() / n)
        table = totals.assign(anytime_td=any_td).reset_index()
        table.insert(2, "player", table.player_id.map(lambda pid: names.get(pid, pid)))
        table.insert(3, "pos", table.player_id.map(positions).fillna(""))
        return table

    def player_samples(self):
        """Every simulated game's line for each player with a real role."""
        p = self.players.assign(rush_rec_yds=self.players.rush_yds + self.players.rec_yds,
                                anytime_td=self.players.rush_td + self.players.rec_td)
        n = len(self.games)
        usage = p.groupby(["team", "player_id"])[["pass_att", "rush_car", "tgt"]].sum() / n
        roles = usage.assign(passer=usage.pass_att.ge(10), rusher=usage.rush_car.ge(2), receiver=usage.tgt.ge(1.5))
        roles = roles.loc[roles.passer | roles.rusher | roles.receiver].reset_index()
        names, positions = self.matchup.league.names, self.matchup.league.positions
        roles["player"] = roles.player_id.map(lambda pid: names.get(pid, pid))
        roles["pos"] = roles.player_id.map(positions).fillna("")
        keep = p.loc[p.player_id.isin(roles.player_id)]
        samples = (keep.set_index(["sim", "player_id"])[list(PROP_STATS)].unstack("player_id")
                   .reindex(range(n)).fillna(0).astype(np.int16))
        return samples, roles.set_index("player_id")


PROP_STATS = {
    "pass_yds": ("Passing yards", np.arange(149.5, 400, 25)), "pass_td": ("Passing TDs", [.5, 1.5, 2.5, 3.5]),
    "pass_cmp": ("Completions", np.arange(13.5, 34, 2)), "pass_int": ("Interceptions", [.5, 1.5]),
    "rush_yds": ("Rushing yards", np.arange(9.5, 160, 10)), "rush_car": ("Rush attempts", np.arange(5.5, 28, 2)),
    "rec": ("Receptions", np.arange(1.5, 12, 1)), "rec_yds": ("Receiving yards", np.arange(9.5, 160, 10)),
    "rush_rec_yds": ("Rush + rec yards", np.arange(19.5, 200, 10)), "anytime_td": ("Anytime TD", [.5]),
}
PROP_ROLES = {"pass_yds": "passer", "pass_td": "passer", "pass_cmp": "passer", "pass_int": "passer",
              "rush_yds": "rusher", "rush_car": "rusher", "rec": "receiver", "rec_yds": "receiver",
              "rush_rec_yds": "rusher|receiver", "anytime_td": "rusher|receiver"}


def find_player(players, name):
    """Player id from a full or partial name (case-insensitive), within this game's players."""
    name = str(name).strip().lower()
    exact = players.index[players.player.str.lower().eq(name)]
    partial = players.index[players.player.str.lower().str.contains(name, regex=False)]
    hits = exact if len(exact) else partial
    if len(hits) != 1:
        raise ValueError(f"{name!r} matches {len(hits)} players in this game: "
                         f"{', '.join(players.loc[hits, 'player']) if len(hits) else 'none'}")
    return hits[0]


def prop_probability(samples, players, player, stat, line):
    """Share of simulated games over, under and exactly on a line, plus the median."""
    pid = player if player in players.index else find_player(players, player)
    if (stat, pid) not in samples:
        raise ValueError(f"No {PROP_STATS[stat][0].lower()} recorded for {players.at[pid, 'player']}")
    values = samples[(stat, pid)].to_numpy()
    return {"player": players.at[pid, "player"], "team": players.at[pid, "team"], "prop": PROP_STATS[stat][0],
            "line": line, "over": float((values > line).mean()), "under": float((values < line).mean()),
            "push": float((values == line).mean()), "median": float(np.median(values)),
            "average": float(values.mean()), "games": len(values)}


def prop_board(samples, players, threshold=.55, sides=("over",)):
    """For each player and prop, the most aggressive standard line that hits at least `threshold` of
    simulated games.
    """
    rows = []
    for (stat, pid), column in samples.items():
        if pid not in players.index or not any(players.at[pid, r] for r in PROP_ROLES[stat].split("|")):
            continue
        values = column.to_numpy()
        label, ladder = PROP_STATS[stat]
        for side in sides:
            hits = [(line, (values > line).mean() if side == "over" else (values < line).mean()) for line in ladder]
            hits = [(line, prob) for line, prob in hits if prob >= threshold]
            if not hits:
                continue
            line, prob = max(hits) if side == "over" else min(hits)
            rows.append({"team": players.at[pid, "team"], "player": players.at[pid, "player"],
                         "pos": players.at[pid, "pos"], "prop": label, "pick": f"{side} {line:g}",
                         "probability": prob, "sim median": float(np.median(values))})
    board = pd.DataFrame(rows, columns=["team", "player", "pos", "prop", "pick", "probability", "sim median"])
    return board.sort_values(["team", "player", "probability"], ascending=[True, True, False]).reset_index(drop=True)


def simulate_game(league, home, away, model_probability, n=1000, model_weight=.25, seed=2026,
                  home_qb=None, away_qb=None, calibration_sims=500, game_id=None, out_players=None,
                  returning_players=None):
    """model_weight blends win probabilities in logit space: 0 uses the play-by-play matchup alone, 1
    forces the simulated win rate onto the model.
    """
    matchup = Matchup(league, home, away, home_qb, away_qb, out_players, returning_players)
    raw = matchup.home_win_rate(0., max(calibration_sims, n // 4), seed)
    target = _expit(model_weight * _logit(model_probability) + (1 - model_weight) * _logit(raw))
    theta = 0.
    if model_weight > 0:
        block = 10_000
        block_raw = matchup.home_win_rate(0., calibration_sims, block)
        theta = matchup.calibrate(_expit(_logit(block_raw) + _logit(target) - _logit(raw)), calibration_sims, block)
    games, players = matchup.simulate(n, theta, seed)
    matchup.pregame_logit = _logit(_win_rate(games))
    return SimulationResult(matchup, games, players, theta, model_probability, float(target), raw, game_id)


def box_tables(game, league):
    """Readable passing / rushing / receiving / kicking tables for one simulated game."""
    box = game.box()
    box.insert(1, "player", box.player_id.map(lambda pid: league.names.get(pid, pid)))
    tables = {}
    for team in (game.away, game.home):
        b = box.loc[box.team.eq(team)]
        tables[team] = {
            "Passing": b.loc[b.pass_att.gt(0) | b.sacked.gt(0)].assign(
                **{"C/ATT": lambda d: d.pass_cmp.astype(int).astype(str) + "/" + d.pass_att.astype(int).astype(str)})
            [["player", "C/ATT", "pass_yds", "pass_td", "pass_int", "sacked"]],
            "Rushing": b.loc[b.rush_car.gt(0)].sort_values("rush_yds", ascending=False)
            [["player", "rush_car", "rush_yds", "rush_td"]],
            "Receiving": b.loc[b.tgt.gt(0)].sort_values("rec_yds", ascending=False)
            [["player", "rec", "tgt", "rec_yds", "rec_td"]],
            "Kicking": b.loc[b.fga.gt(0) | b.xpa.gt(0)][["player", "fgm", "fga", "xpm", "xpa"]],
            "Defense": b.loc[b.def_sack.gt(0) | b.def_int.gt(0)][["player", "def_sack", "def_int"]],
        }
        for key, frame in tables[team].items():
            numeric = frame.select_dtypes("number").columns
            tables[team][key] = frame.astype({c: int for c in numeric}).reset_index(drop=True)
    return tables


def replay_payload(result, sim, autoplay=False):
    game = result.replay(sim)
    league, m = result.matchup.league, result.matchup
    ids = set(game.stats)
    for team in (m.home, m.away):
        p = m.players[team]
        ids |= {p["qb"], p["kicker"]} | set(p["carries"][0][:3]) | set(p["targets"][0][:5])
    shots = getattr(league, "headshots", {})
    players = {pid: [short_name(league.names.get(pid, pid)), league.positions.get(pid, ""), shots.get(pid)]
               for pid in ids}
    starters = {team: dict(qb=m.players[team]["qb"], kicker=m.players[team]["kicker"],
                           carries=list(m.players[team]["carries"][0][:3]),
                           targets=list(m.players[team]["targets"][0][:5])) for team in (m.home, m.away)}
    teams = {team: dict(abbr=team, name=TEAM_NAMES.get(team, team), colors=TEAM_COLORS.get(team, ("#444", "#999")),
                        logo=f"https://a.espncdn.com/i/teamlogos/nfl/500/{ {'WAS': 'wsh'}.get(team, team.lower()) }.png")
             for team in (m.home, m.away)}
    return dict(home=m.home, away=m.away, teams=teams, players=players, starters=starters, stats=STATS,
                events=game.events, sim=int(sim) + 1, n=len(result.games), game_id=result.game_id,
                model=round(result.model_probability, 4), simulated=round(result.home_win_rate, 4),
                autoplay=autoplay)


def replay_html(result, sim=None, autoplay=False):
    sim = result.typical_sim() if sim is None else sim
    payload = json.dumps(replay_payload(result, sim, autoplay), default=lambda v: v.item() if hasattr(v, "item") else str(v))
    template = (Path(__file__).with_name("nfl_game_sim_replay.html")).read_text()
    title = f"{result.away} @ {result.home} · Simulated game"
    return template.replace("__TITLE__", html.escape(title)).replace("__PAYLOAD__", payload.replace("</", "<\\/"))


def show_replay(result, sim=None, output_dir="output/simulations", height=900):
    """Write a standalone HTML replay and embed it in the notebook."""
    import warnings
    from IPython.display import HTML
    sim = result.typical_sim() if sim is None else sim
    document = replay_html(result, sim)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{result.game_id or result.away + '_' + result.home}_sim{sim + 1}.html"
    path.write_text(document)
    frame = (f'<iframe srcdoc="{html.escape(document, quote=True)}" style="width:100%;height:{height}px;'
             f'border:0;border-radius:12px;background:#e8e5de"></iframe>')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return HTML(frame), path
