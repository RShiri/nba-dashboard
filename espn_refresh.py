"""
Fill the current season from ESPN's public feeds when stats.nba.com gives us nothing.

stats.nba.com stopped answering automated requests in October 2026 (every call times out,
from GitHub's runners and from a home connection alike), so fetch_data.py keeps the old pickle
and the site silently freezes. This step runs after fetch_data.py and before build_data.py.

It only ever writes the current season, and only for a player whose ESPN game count is higher
than what the pickle already holds, so official data wins again as soon as it returns.

Not covered by ESPN, and therefore left as they are: advanced career stats (USG%, net rating)
and the league-wide All-Star / League Trends datasets.

Usage: python espn_refresh.py [--dry-run]
"""
from __future__ import annotations

import math
import pickle
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

import fetch_data

ESPN_IDS = {
    "deni_avdija": "4683021",
    "ben_saraf": "5242502",
    "danny_wolf": "5107173",
    "emanuel_sharp": "5106058",
}

GAMELOG_URL = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{id}/gamelog"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
HEADERS = {"User-Agent": "nba-dashboard-refresh"}
EASTERN = ZoneInfo("America/New_York")
ESPN_TO_NBA_ABBR = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
NBA_ABBRS = {
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM",
    "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
}

# ESPN's court is in whole feet, 0-50 sideline to sideline, measured from the baseline.
HOOP_X, HOOP_Y = 25.0, 0.3
# Calibrated against 406 official 2025-26 shots: same left/right sign as LOC_X, hoop 0.3 ft in.
X_SIGN = 1


def _get(url: str, params: dict | None = None) -> dict:
    last = None
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"ESPN request failed: {url} ({last})")


def _nba_abbr(abbr: str) -> str:
    return ESPN_TO_NBA_ABBR.get(abbr, abbr)


def _espn_season(season: str) -> int:
    """'2026-27' -> 2027."""
    return int(season[:4]) + 1


def _pair(value: str) -> tuple[int, int]:
    made, att = value.split("-")
    return int(made), int(att)


def _int(value) -> int:
    try:
        return int(str(value).replace("+", ""))
    except ValueError:
        return 0


def _game_date(iso: str) -> pd.Timestamp:
    """ESPN gives UTC tip-off; the NBA dates games in US Eastern."""
    utc = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return pd.Timestamp(utc.astimezone(EASTERN).date())


def fetch_game_log(espn_id: str, nba_id: int, season: str) -> pd.DataFrame:
    """Regular-season game log in the same columns as nba_api's PlayerGameLog."""
    data = _get(GAMELOG_URL.format(id=espn_id), {"season": _espn_season(season)})
    names = data.get("names", [])
    idx = {name: i for i, name in enumerate(names)}
    events = data.get("events", {})
    season_id = f"2{season[:4]}"

    rows = []
    for stype in data.get("seasonTypes", []):
        label = stype.get("displayName", "")
        if "Regular Season" not in label or "Play In" in label:
            continue
        for cat in stype.get("categories", []):
            if cat.get("type") != "event":
                continue
            for ev in cat.get("events", []):
                info, stats = events.get(ev["eventId"]), ev["stats"]
                if not info or info.get("gameResult") not in ("W", "L"):
                    continue
                fgm, fga = _pair(stats[idx["fieldGoalsMade-fieldGoalsAttempted"]])
                fg3m, fg3a = _pair(stats[idx["threePointFieldGoalsMade-threePointFieldGoalsAttempted"]])
                ftm, fta = _pair(stats[idx["freeThrowsMade-freeThrowsAttempted"]])
                team = _nba_abbr(info["team"]["abbreviation"])
                opponent = _nba_abbr(info["opponent"]["abbreviation"])
                if team not in NBA_ABBRS or opponent not in NBA_ABBRS:  # All-Star exhibition games
                    continue
                joiner = "vs." if info.get("atVs") == "vs" else "@"
                rows.append({
                    "SEASON_ID": season_id,
                    "Player_ID": nba_id,
                    "Game_ID": ev["eventId"],
                    "GAME_DATE": _game_date(info["gameDate"]),
                    "MATCHUP": f"{team} {joiner} {opponent}",
                    "WL": info["gameResult"],
                    "MIN": _int(stats[idx["minutes"]]),
                    "FGM": fgm, "FGA": fga, "FG_PCT": round(fgm / fga, 3) if fga else 0.0,
                    "FG3M": fg3m, "FG3A": fg3a, "FG3_PCT": round(fg3m / fg3a, 3) if fg3a else 0.0,
                    "FTM": ftm, "FTA": fta, "FT_PCT": round(ftm / fta, 3) if fta else 0.0,
                    "REB": _int(stats[idx["totalRebounds"]]),
                    "AST": _int(stats[idx["assists"]]),
                    "STL": _int(stats[idx["steals"]]),
                    "BLK": _int(stats[idx["blocks"]]),
                    "TOV": _int(stats[idx["turnovers"]]),
                    "PF": _int(stats[idx["fouls"]]),
                    "PTS": _int(stats[idx["points"]]),
                })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("Game_ID")
    return df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)


def classify_zone(loc_x: float, loc_y: float, is_three: bool) -> tuple[str, str, str]:
    """(SHOT_ZONE_BASIC, SHOT_ZONE_AREA, SHOT_ZONE_RANGE), from tenth-of-a-foot coordinates."""
    dist = math.hypot(loc_x, loc_y) / 10
    if is_three:
        if loc_y <= 92.5 and abs(loc_x) >= 200:
            return ("Left Corner 3", "Left Side(L)", "24+ ft.") if loc_x < 0 \
                else ("Right Corner 3", "Right Side(R)", "24+ ft.")
        basic = "Above the Break 3"
    elif dist <= 4:
        return "Restricted Area", "Center(C)", "Less Than 8 ft."
    elif abs(loc_x) <= 80 and loc_y <= 137.5:
        basic = "In The Paint (Non-RA)"
    else:
        basic = "Mid-Range"

    # Angle boundaries read off the official 2025-26 zone labels.
    angle = math.degrees(math.atan2(loc_x, max(loc_y, 0.1)))
    if basic == "In The Paint (Non-RA)":
        area = "Center(C)" if dist < 8 or abs(angle) <= 30 else ("Left Side(L)" if angle < 0 else "Right Side(R)")
    elif abs(angle) <= 18:
        area = "Center(C)"
    elif abs(angle) <= (68 if is_three else 45):
        area = "Left Side Center(LC)" if angle < 0 else "Right Side Center(RC)"
    else:
        area = "Left Side(L)" if angle < 0 else "Right Side(R)"

    if is_three or dist >= 24:
        rng = "24+ ft."
    elif dist >= 16:
        rng = "16-24 ft."
    elif dist >= 8:
        rng = "8-16 ft."
    else:
        rng = "Less Than 8 ft."
    return basic, area, rng


def shots_from_summary(summary: dict, espn_id: str, nba_id: int, name: str, game_id: str, game_date: pd.Timestamp) -> list[dict]:
    rows = []
    for play in summary.get("plays", []):
        if not play.get("shootingPlay") or "Free Throw" in play.get("type", {}).get("text", ""):
            continue
        people = play.get("participants") or []
        if not people or str(people[0]["athlete"]["id"]) != espn_id:
            continue
        coord = play.get("coordinate") or {}
        x, y = coord.get("x"), coord.get("y")
        if x is None or y is None or not (0 <= x <= 50 and 0 <= y <= 94):
            continue
        loc_x = (x - HOOP_X) * 10 * X_SIGN
        loc_y = (y - HOOP_Y) * 10
        is_three = play.get("pointsAttempted") == 3
        made = bool(play.get("scoringPlay"))
        basic, area, rng = classify_zone(loc_x, loc_y, is_three)
        clock = play.get("clock", {}).get("displayValue", "0:0")
        minutes, seconds = clock.split(":") if ":" in clock else ("0", clock.split(".")[0])
        rows.append({
            "GRID_TYPE": "Shot Chart Detail", "GAME_ID": game_id, "GAME_EVENT_ID": play.get("sequenceNumber"),
            "PLAYER_ID": nba_id, "PLAYER_NAME": name, "PERIOD": play.get("period", {}).get("number"),
            "MINUTES_REMAINING": int(minutes), "SECONDS_REMAINING": int(seconds),
            "EVENT_TYPE": "Made Shot" if made else "Missed Shot",
            "ACTION_TYPE": play.get("type", {}).get("text", ""),
            "SHOT_TYPE": "3PT Field Goal" if is_three else "2PT Field Goal",
            "SHOT_ZONE_BASIC": basic, "SHOT_ZONE_AREA": area, "SHOT_ZONE_RANGE": rng,
            "SHOT_DISTANCE": int(math.hypot(loc_x, loc_y) / 10),
            "LOC_X": loc_x, "LOC_Y": loc_y, "SHOT_ATTEMPTED_FLAG": 1, "SHOT_MADE_FLAG": int(made),
            "GAME_DATE": game_date.strftime("%Y%m%d"),
        })
    return rows


def box_line(summary: dict, espn_id: str) -> dict:
    """OREB / DREB / +/- / started, which ESPN's game log leaves out."""
    for team in summary.get("boxscore", {}).get("players", []):
        for group in team.get("statistics", []):
            keys = group.get("keys") or []
            for ath in group.get("athletes", []):
                if str(ath.get("athlete", {}).get("id")) != espn_id or not ath.get("stats"):
                    continue
                stat = dict(zip(keys, ath["stats"]))
                return {
                    "OREB": _int(stat.get("offensiveRebounds", 0)),
                    "DREB": _int(stat.get("defensiveRebounds", 0)),
                    "PLUS_MINUS": _int(stat.get("plusMinus", 0)),
                    "STARTED": bool(ath.get("starter")),
                }
    return {"OREB": 0, "DREB": 0, "PLUS_MINUS": 0, "STARTED": False}


def fetch_summary(event_id: str, cache: dict) -> dict:
    if event_id not in cache:
        cache[event_id] = _get(SUMMARY_URL, {"event": event_id})
        time.sleep(0.25)
    return cache[event_id]


def enrich(logs: pd.DataFrame, espn_id: str, nba_id: int, name: str, cache: dict) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Add box-score-only columns to the log. Returns (logs, shots, games started)."""
    extras, shots = [], []
    for row in logs.itertuples():
        summary = fetch_summary(row.Game_ID, cache)
        extras.append(box_line(summary, espn_id))
        shots += shots_from_summary(summary, espn_id, nba_id, name, row.Game_ID, row.GAME_DATE)
    extra = pd.DataFrame(extras)
    started = int(extra.pop("STARTED").sum())
    logs = pd.concat([logs.reset_index(drop=True), extra], axis=1)
    return logs, pd.DataFrame(shots), started


def add_season_row(career: pd.DataFrame, logs: pd.DataFrame, season: str, nba_id: int, started: int) -> pd.DataFrame:
    career = fetch_data.local_patch_career_stats(career.copy(), logs.copy(), season)
    idx = career.index[career["SEASON_ID"] == season][0]
    prev = career[career["SEASON_ID"] != season].sort_values("SEASON_ID").tail(1)
    team = logs.iloc[0]["MATCHUP"].split(" ")[0]
    fill = {
        "PLAYER_ID": nba_id, "LEAGUE_ID": "00", "TEAM_ABBREVIATION": team, "GS": started,
        "OREB": logs["OREB"].mean(), "DREB": logs["DREB"].mean(),
    }
    if not prev.empty:
        fill["PLAYER_AGE"] = prev.iloc[0]["PLAYER_AGE"] + 1
        if prev.iloc[0]["TEAM_ABBREVIATION"] == team:
            fill["TEAM_ID"] = prev.iloc[0]["TEAM_ID"]
    for col, val in fill.items():
        if col in career.columns:
            career.at[idx, col] = val
    return career.sort_values("SEASON_ID").reset_index(drop=True)


def refresh(dry_run: bool = False) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    season = fetch_data.CURRENT_SEASON
    key_col = f"game_logs_{fetch_data.season_key(season)}"
    with open(fetch_data.OUTPUT_FILE, "rb") as f:
        data = pickle.load(f)

    changed, cache = 0, {}
    for key, espn_id in ESPN_IDS.items():
        meta = fetch_data.PLAYERS[key]
        player = data["players"][key]
        have = player.get(key_col, pd.DataFrame())
        have_n = 0 if have is None else len(have)
        try:
            logs = fetch_game_log(espn_id, meta["id"], season)
        except RuntimeError as exc:
            print(f"{meta['name']}: ESPN failed, keeping current data ({exc})")
            continue
        if len(logs) <= have_n:
            print(f"{meta['name']}: {have_n} games stored, ESPN has {len(logs)}, nothing to do")
            continue
        logs, shots, started = enrich(logs, espn_id, meta["id"], meta["name"], cache)
        print(f"{meta['name']}: {have_n} -> {len(logs)} games, {len(shots)} shots, latest {logs.GAME_DATE.max().date()}")
        player[key_col] = logs
        player.setdefault("shot_charts", {})[season] = shots
        career = player.get("career_basic")
        if not fetch_data.is_valid_df(career):  # rookie with no official career row yet
            template = data["players"]["deni_avdija"]["career_basic"]
            career = pd.DataFrame([{"SEASON_ID": season}], columns=template.columns)
            career = career.astype({c: "float64" for c in career.columns if c in template.select_dtypes("number").columns})
        player["career_basic"] = add_season_row(career, logs, season, meta["id"], started)
        changed += 1

    if changed and not dry_run:
        data["fetched_at"] = datetime.now().isoformat()
        with open(fetch_data.OUTPUT_FILE, "wb") as f:
            pickle.dump(data, f)
    print(f"Updated {changed} player(s){' (dry run, nothing written)' if dry_run else ''}.")
    return changed


if __name__ == "__main__":
    refresh(dry_run="--dry-run" in sys.argv)
