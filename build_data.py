"""
Static-site data exporter.

Reads nba_data.pkl (produced by fetch_data.py) and writes the JSON files consumed
by the static GitHub Pages dashboard under nba_dashboard/data/. Run this after
fetch_data.py, or via the GitHub Actions workflow.

Usage:
    python build_data.py
"""

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.static import players as nba_players

import fetch_data

DATA_FILE = Path(__file__).parent / "nba_data.pkl"
OUT_DIR = Path(__file__).parent / "nba_dashboard" / "data"


def headshot_url(player_id) -> str:
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"


def clean_records(df: pd.DataFrame, cols=None) -> list:
    """DataFrame -> list of plain-JSON-safe dicts (NaN/NaT -> None, numpy -> python)."""
    if df is None or df.empty:
        return []
    d = df[cols].copy() if cols else df.copy()
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].dt.strftime("%Y-%m-%d")
    d = d.replace({np.nan: None})
    return json.loads(d.to_json(orient="records"))


# -----------------------------
# Court zone geometry (ported from app.py's get_court_zones/create_zone_efficiency_map)
# -----------------------------
def get_court_zones() -> list:
    zones = []
    R_HOOP, R_3PT, R_FAR = 40.0, 237.5, 500.0
    X_CORNER, Y_PAINT_TOP = 220.0, 142.5
    y_break = np.sqrt(R_3PT**2 - X_CORNER**2)
    theta_break_r = np.arctan2(y_break, X_CORNER)
    theta_cut_r, theta_cut_l = np.radians(72), np.radians(108)
    theta_break_l = np.pi - theta_break_r

    def arc(r, t1, t2, steps=30):
        t = np.linspace(t1, t2, steps)
        return r * np.cos(t), r * np.sin(t)

    ar_x, ar_y = arc(R_3PT, theta_break_r, theta_cut_r)
    ac_x, ac_y = arc(R_3PT, theta_cut_r, theta_cut_l)
    al_x, al_y = arc(R_3PT, theta_cut_l, theta_break_l)
    ra_x, ra_y = arc(R_HOOP, 0, np.pi)

    zones.append({"name": "Restricted Area", "keys": ["Restricted Area_Center(C)"],
        "x": np.concatenate(([40, 40, -40, -40], ra_x[::-1])), "y": np.concatenate(([-47.5, 0, 0, -47.5], ra_y[::-1]))})
    zones.append({"name": "Paint", "keys": ["In The Paint (Non-RA)_Center(C)", "In The Paint (Non-RA)_Right Side(R)", "In The Paint (Non-RA)_Left Side(L)"],
        "x": np.concatenate(([80, 80, -80, -80, -40], ra_x, [40])), "y": np.concatenate(([-47.5, Y_PAINT_TOP, Y_PAINT_TOP, -47.5, -47.5], ra_y, [-47.5]))})
    zones.append({"name": "MR Right", "keys": ["Mid-Range_Right Side(R)"], "x": [80, X_CORNER, X_CORNER, 80, 80], "y": [-47.5, -47.5, y_break, y_break, -47.5]})
    zones.append({"name": "MR RC", "keys": ["Mid-Range_Right Side Center(RC)"],
        "x": np.concatenate(([80], ar_x[::-1], [X_CORNER, 80, 80])), "y": np.concatenate(([Y_PAINT_TOP], ar_y[::-1], [y_break, y_break, Y_PAINT_TOP]))})
    zones.append({"name": "MR Center", "keys": ["Mid-Range_Center(C)"],
        "x": np.concatenate(([80], ac_x[::-1], [-80, 80])), "y": np.concatenate(([Y_PAINT_TOP], ac_y[::-1], [Y_PAINT_TOP, Y_PAINT_TOP]))})
    zones.append({"name": "MR LC", "keys": ["Mid-Range_Left Side Center(LC)"],
        "x": np.concatenate(([-80, -X_CORNER], al_x[::-1], [-80, -80])), "y": np.concatenate(([y_break, y_break], al_y[::-1], [Y_PAINT_TOP, y_break]))})
    zones.append({"name": "MR Left", "keys": ["Mid-Range_Left Side(L)"], "x": [-80, -X_CORNER, -X_CORNER, -80, -80], "y": [y_break, y_break, -47.5, -47.5, y_break]})
    zones.append({"name": "Right Corner 3", "keys": ["Right Corner 3_Right Side(R)"], "x": [X_CORNER, 250, 250, X_CORNER, X_CORNER], "y": [-47.5, -47.5, y_break, y_break, -47.5]})
    zones.append({"name": "Left Corner 3", "keys": ["Left Corner 3_Left Side(L)"], "x": [-X_CORNER, -250, -250, -X_CORNER, -X_CORNER], "y": [-47.5, -47.5, y_break, y_break, -47.5]})

    far_r_x, far_r_y = arc(R_FAR, theta_break_r, theta_cut_r, steps=10)
    zones.append({"name": "AB3 RC", "keys": ["Above the Break 3_Right Side Center(RC)"],
        "x": np.concatenate((ar_x, far_r_x[::-1], [ar_x[0]])), "y": np.concatenate((ar_y, far_r_y[::-1], [ar_y[0]]))})
    far_c_x, far_c_y = arc(R_FAR, theta_cut_r, theta_cut_l, steps=10)
    zones.append({"name": "AB3 Center", "keys": ["Above the Break 3_Center(C)"],
        "x": np.concatenate((ac_x, far_c_x[::-1], [ac_x[0]])), "y": np.concatenate((ac_y, far_c_y[::-1], [ac_y[0]]))})
    far_l_x, far_l_y = arc(R_FAR, theta_cut_l, theta_break_l, steps=10)
    zones.append({"name": "AB3 LC", "keys": ["Above the Break 3_Left Side Center(LC)"],
        "x": np.concatenate((al_x, far_l_x[::-1], [al_x[0]])), "y": np.concatenate((al_y, far_l_y[::-1], [al_y[0]]))})

    for z in zones:
        z["x"] = [round(float(v), 2) for v in z["x"]]
        z["y"] = [round(float(v), 2) for v in z["y"]]
    return zones


ZONES = get_court_zones()


def build_shot_chart_json(shot_df: pd.DataFrame) -> dict:
    if shot_df is None or shot_df.empty:
        return {"points": [], "zones": []}

    clean = shot_df.dropna(subset=["LOC_X", "LOC_Y"])
    points = [
        {"x": round(float(r.LOC_X), 1), "y": round(float(r.LOC_Y), 1), "made": r.EVENT_TYPE == "Made Shot"}
        for r in clean.itertuples()
    ]

    df = shot_df.copy()
    df["ZONE_GROUP"] = df["SHOT_ZONE_BASIC"] + "_" + df["SHOT_ZONE_AREA"]
    agg = df.groupby("ZONE_GROUP").agg(FGM=("SHOT_MADE_FLAG", "sum"), FGA=("SHOT_ATTEMPTED_FLAG", "count")).reset_index()

    zone_stats = []
    for z in ZONES:
        rows = agg[agg["ZONE_GROUP"].isin(z["keys"])]
        fgm, fga = int(rows["FGM"].sum()), int(rows["FGA"].sum())
        zone_stats.append({"name": z["name"], "fgm": fgm, "fga": fga, "pct": (fgm / fga) if fga else None})

    return {"points": points, "zones": zone_stats}


# -----------------------------
# Main export
# -----------------------------
def resolve_headshots(names: list) -> dict:
    out = {}
    for name in names:
        try:
            hits = nba_players.find_players_by_full_name(name)
            if hits:
                out[name] = headshot_url(hits[0]["id"])
        except Exception:
            pass
    return out


def build():
    with open(DATA_FILE, "rb") as f:
        data = pickle.load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    current_season = fetch_data.CURRENT_SEASON
    prev_season = fetch_data.PREV_SEASON
    prev2_season = fetch_data.PREV2_SEASON
    benchmark_season = fetch_data.BENCHMARK_SEASON

    meta_players = []
    for key, meta in fetch_data.PLAYERS.items():
        meta_players.append({
            "key": key, "name": meta["name"], "team_full": meta["team_full"],
            "position": meta["position"], "headshot": headshot_url(meta["id"]),
        })

    for key, meta in fetch_data.PLAYERS.items():
        pdata = data.get("players", {}).get(key, {})
        career_basic = pdata.get("career_basic", pd.DataFrame())
        career_adv = pdata.get("career_advanced", pd.DataFrame())

        career = pd.DataFrame()
        if career_basic is not None and not career_basic.empty:
            career = career_basic.copy()
            if career_adv is not None and not career_adv.empty and "SEASON_ID" in career_adv.columns:
                adv_cols = [c for c in ["SEASON_ID", "NET_RATING", "AST_TO", "TS_PCT", "USG_PCT"] if c in career_adv.columns]
                career = career.merge(career_adv[adv_cols], on="SEASON_ID", how="left")
            career = career.sort_values("SEASON_ID")

        game_logs = {}
        for season in (current_season, prev_season, prev2_season):
            gl = pdata.get(f"game_logs_{fetch_data.season_key(season)}", pd.DataFrame())
            if gl is not None and not gl.empty:
                game_logs[season] = clean_records(gl.sort_values("GAME_DATE"), [
                    "GAME_DATE", "WL", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV",
                    "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                ])

        shot_charts = {}
        for season, df in (pdata.get("shot_charts", {}) or {}).items():
            sc = build_shot_chart_json(df)
            if sc["points"]:
                shot_charts[season] = sc

        out = {
            "player_id": meta["id"],
            "player_name": meta["name"],
            "team_full": meta["team_full"],
            "position": meta["position"],
            "headshot": headshot_url(meta["id"]),
            "career": clean_records(career) if not career.empty else [],
            "game_logs": game_logs,
            "shot_charts": shot_charts,
        }
        with open(OUT_DIR / f"{key}.json", "w") as f:
            json.dump(out, f)
        print(f"wrote {key}.json ({len(out['career'])} career rows, {sum(len(v) for v in game_logs.values())} games, {sum(len(v['points']) for v in shot_charts.values())} shots)")

    # Shared / league-wide data
    allstar_names = list(dict.fromkeys(
        data.get("allstar_stats", pd.DataFrame()).get("PLAYER_NAME", pd.Series(dtype=str)).tolist()
        + [m["name"] for m in meta_players]
    ))
    headshots = resolve_headshots(allstar_names)

    shared = {
        "allstar_bench": {
            "season": benchmark_season,
            "stats": clean_records(data.get("allstar_stats", pd.DataFrame())),
            "detailed": clean_records(data.get("allstar_detailed_stats", pd.DataFrame())),
        },
        "allstar_race": {
            "season": current_season,
            "stats": clean_records(data.get("allstar_stats_26", pd.DataFrame())),
            "detailed": clean_records(data.get("allstar_detailed_26", pd.DataFrame())),
        },
        "league_ft": {
            "season": current_season,
            "leaders": clean_records(data.get("league_ft_stats", pd.DataFrame()).sort_values("FTM", ascending=False).head(20)) if not data.get("league_ft_stats", pd.DataFrame()).empty else [],
        },
        "league_trends": {
            "heliocentric": clean_records(data.get("heliocentric_data", pd.DataFrame()).sort_values("TEAM_PPG", ascending=False)) if not data.get("heliocentric_data", pd.DataFrame()).empty else [],
            "drives": clean_records(data.get("drives_data", pd.DataFrame()).sort_values("DRIVES", ascending=False).head(10)) if not data.get("drives_data", pd.DataFrame()).empty else [],
            "misc": clean_records(data.get("misc_stats", pd.DataFrame()).sort_values("PFD", ascending=False).head(10)) if not data.get("misc_stats", pd.DataFrame()).empty and "PFD" in data.get("misc_stats", pd.DataFrame()).columns else [],
            "passing": clean_records(data.get("passing_data", pd.DataFrame()).sort_values("AST_3P", ascending=False).head(10)) if not data.get("passing_data", pd.DataFrame()).empty else [],
        },
        "headshots": headshots,
    }
    with open(OUT_DIR / "shared.json", "w") as f:
        json.dump(shared, f)
    print(f"wrote shared.json ({len(shared['allstar_bench']['stats'])} bench all-stars, {len(headshots)} headshots)")

    meta = {
        "current_season": current_season,
        "prev_season": prev_season,
        "prev2_season": prev2_season,
        "benchmark_season": benchmark_season,
        "fetched_at": data.get("fetched_at"),
        "players": meta_players,
    }
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f)
    print("wrote meta.json")

    with open(OUT_DIR / "zones.json", "w") as f:
        json.dump(ZONES, f)
    print("wrote zones.json")


if __name__ == "__main__":
    build()
