import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from nba_api.stats.endpoints import (
    PlayerCareerStats,
    PlayerDashboardByYearOverYear,
    PlayerGameLog,
    shotchartdetail,
)
from nba_api.stats.static import players


# -----------------------------
# Config and constants
# -----------------------------
PLAYER_NAME = "Deni Avdija"
CURRENT_SEASON = "2025-26"

st.set_page_config(
    page_title="Deni Avdija: 360° Performance Analytics",
    layout="wide",
)


# -----------------------------
# Data helpers with caching
# -----------------------------
@st.cache_data(show_spinner=False)
def get_player_id(full_name: str = PLAYER_NAME) -> int:
    """Resolve NBA player ID, falling back to known value if lookup fails."""
    try:
        hits = players.find_players_by_full_name(full_name)
        if hits:
            return hits[0]["id"]
    except Exception:
        pass
    return 1630166  # Known ID for Deni Avdija


@st.cache_data(show_spinner=False)
def fetch_career_basic(player_id: int) -> pd.DataFrame:
    """Fetch career basic per-season stats."""
    try:
        df = PlayerCareerStats(player_id=player_id, per_mode36="PerGame").get_data_frames()[0]
        df = df.copy()
        # Convert SEASON_ID to string immediately to avoid warnings
        df["SEASON_ID"] = df["SEASON_ID"].astype(str)
        return df
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not load career basic stats: {exc}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def fetch_career_advanced(player_id: int) -> pd.DataFrame:
    """Fetch career advanced per-season stats."""
    try:
        adv = PlayerDashboardByYearOverYear(
            player_id=player_id,
            per_mode_detailed="PerGame",
            measure_type_detailed="Advanced",
        ).get_data_frames()[1]
        adv = adv.copy()
        adv["SEASON_ID"] = adv["GROUP_VALUE"].astype(str)
        return adv
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not load career advanced stats: {exc}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def fetch_game_logs(player_id: int, season: str = CURRENT_SEASON) -> pd.DataFrame:
    """Fetch game logs for a target season."""
    try:
        logs = PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
        logs = logs.copy()
        
        # Convert SEASON_ID to string immediately to avoid API format warnings
        if "SEASON_ID" in logs.columns:
            logs["SEASON_ID"] = logs["SEASON_ID"].astype(str)
        
        if "GAME_DATE" in logs.columns:
            logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
        for col in ("MIN", "PTS", "REB", "AST"):
            if col in logs.columns:
                logs[col] = pd.to_numeric(logs[col], errors="coerce")
        return logs
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not load {season} game logs: {exc}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def fetch_shot_data(player_id: int, season: str) -> pd.DataFrame:
    """Fetch shot chart data for a player and season."""
    try:
        shot_data = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=player_id,
            context_measure_simple="FGA",
            season_nullable=season,
        ).get_data_frames()[0]
        return shot_data.copy()
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not load shot data for {season}: {exc}")
        return pd.DataFrame()


# -----------------------------
# Transform helpers
# -----------------------------
def merge_career_frames(basic_df: pd.DataFrame, adv_df: pd.DataFrame) -> pd.DataFrame:
    """Merge basic and advanced frames on SEASON_ID."""
    if basic_df.empty or adv_df.empty:
        return pd.DataFrame()
    adv_cols = ["SEASON_ID", "NET_RATING", "AST_TO", "TS_PCT", "USG_PCT"]
    missing_cols = [c for c in adv_cols if c not in adv_df.columns]
    if missing_cols:
        st.warning(f"Advanced data missing columns: {', '.join(missing_cols)}")
        return pd.DataFrame()
    merged = basic_df.merge(adv_df[adv_cols], on="SEASON_ID", how="inner")
    merged = merged.sort_values("SEASON_ID")
    return merged


def normalize_growth(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Normalize selected columns to rookie season = 100."""
    if df.empty:
        return pd.DataFrame()
    norm_df = df[["SEASON_ID"] + columns].copy()
    for col in columns:
        first_valid = norm_df[col].dropna()
        if first_valid.empty or first_valid.iloc[0] == 0:
            norm_df[col] = pd.NA
        else:
            base = first_valid.iloc[0]
            norm_df[col] = (norm_df[col] / base) * 100
    return norm_df


# -----------------------------
# Plot builders
# -----------------------------
def plot_per_game_stats(df: pd.DataFrame) -> go.Figure:
    """Per Game Stats - Grouped Bar Chart (PTS, REB, AST)."""
    fig = go.Figure()
    
    metrics = ["PTS", "REB", "AST"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]  # Blue, Green, Orange
    
    for metric, color in zip(metrics, colors):
        fig.add_trace(
            go.Bar(
                name=metric,
                x=df["SEASON_ID"],
                y=df[metric],
                marker_color=color,
                hovertemplate=f"{metric}: %{{y:.1f}}<extra></extra>",
            )
        )
    
    fig.update_layout(
        title="ממוצעים למשחק (Per Game)",
        xaxis_title="Season",
        yaxis_title="Value",
        barmode="group",
        hovermode="x unified",
        showlegend=True,
    )
    return fig


def plot_per_36_stats(df: pd.DataFrame) -> go.Figure:
    """Per 36 Minutes Stats - Grouped Bar Chart (PTS, REB, AST, STL, TOV)."""
    fig = go.Figure()
    
    # All possible metrics we want to show
    all_metrics = ["PTS", "REB", "AST", "STL", "TOV"]
    all_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]  # Blue, Green, Orange, Purple, Red
    
    # Filter to only metrics that exist in the dataframe
    available_metrics = [m for m in all_metrics if m in df.columns]
    available_colors = [all_colors[all_metrics.index(m)] for m in available_metrics]
    
    df_work = df.copy()
    
    # Check if we have MIN column for calculation
    if "MIN" in df.columns:
        # Calculate Per36 for each available metric
        for metric in available_metrics:
            # Avoid division by zero
            df_work[f"{metric}_PER36"] = df_work.apply(
                lambda row: (row[metric] / row["MIN"]) * 36 if row["MIN"] > 0 else 0,
                axis=1
            )
    
    # Add traces for each available metric
    for metric, color in zip(available_metrics, available_colors):
        col_name = f"{metric}_PER36" if "MIN" in df.columns else metric
        if col_name in df_work.columns:
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=df_work["SEASON_ID"],
                    y=df_work[col_name],
                    marker_color=color,
                    hovertemplate=f"{metric} (Per 36): %{{y:.1f}}<extra></extra>",
                )
            )
    
    fig.update_layout(
        title="יעילות לדקה (מנורמל ל-36 דקות)",
        xaxis_title="Season",
        yaxis_title="Value (Per 36 Minutes)",
        barmode="group",
        hovermode="x unified",
        showlegend=True,
    )
    return fig


def plot_usage_growth(df: pd.DataFrame) -> go.Figure:
    """Usage Rate Growth - Line Chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["SEASON_ID"],
            y=df["USG_PCT"],
            mode="lines+markers",
            name="Usage %",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=8),
            hovertemplate="Season: %{x}<br>Usage %: %{y:.2%}<extra></extra>",
            )
        )
    fig.update_layout(
        title="עלייה בנפח השימוש (Usage %)",
        xaxis_title="Season",
        yaxis_title="Usage Percentage",
        hovermode="x unified",
        yaxis=dict(tickformat=".0%"),
    )
    return fig


def plot_ts_growth(df: pd.DataFrame) -> go.Figure:
    """True Shooting Growth - Line Chart with League Average."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["SEASON_ID"],
            y=df["TS_PCT"],
            mode="lines+markers",
            name="True Shooting %",
            line=dict(color="#2ca02c", width=3),
            marker=dict(size=8),
            hovertemplate="Season: %{x}<br>TS %: %{y:.2%}<extra></extra>",
        )
    )
    # Add league average line
    fig.add_hline(
        y=0.58,
        line_dash="dash",
        line_color="gray",
        annotation_text="League Average (0.58)",
        annotation_position="right",
    )
    fig.update_layout(
        title="שיפור ביעילות קליעה (True Shooting %)",
        xaxis_title="Season",
        yaxis_title="True Shooting Percentage",
        hovermode="x unified",
        yaxis=dict(tickformat=".0%"),
    )
    return fig




@st.cache_data(show_spinner=False, ttl=3600)
def fetch_allstar_stats() -> pd.DataFrame:
    """Fetch 2024-25 stats for complete list of 2024 All-Stars and calculate weighted averages."""
    # Exact list of 2024 NBA All-Stars (26 players)
    allstar_names = [
        "Giannis Antetokounmpo",
        "Jayson Tatum",
        "Joel Embiid",
        "Tyrese Haliburton",
        "Damian Lillard",
        "Jalen Brunson",
        "Donovan Mitchell",
        "Tyrese Maxey",
        "Paolo Banchero",
        "Jaylen Brown",
        "Julius Randle",
        "Bam Adebayo",
        "Trae Young",
        "Scottie Barnes",
        "LeBron James",
        "Kevin Durant",
        "Nikola Jokic",
        "Luka Doncic",
        "Shai Gilgeous-Alexander",
        "Stephen Curry",
        "Anthony Edwards",
        "Kawhi Leonard",
        "Paul George",
        "Devin Booker",
        "Anthony Davis",
        "Karl-Anthony Towns",
    ]
    
    all_stats = []
    
    for player_name in allstar_names:
        try:
            hits = players.find_players_by_full_name(player_name)
            if hits and len(hits) > 0:
                player_id = hits[0]["id"]
                stats = PlayerCareerStats(player_id=player_id, per_mode36="PerGame").get_data_frames()[0]
                # Convert SEASON_ID to string immediately to avoid warnings
                stats["SEASON_ID"] = stats["SEASON_ID"].astype(str)
                # Get 2024-25 season - handle multiple format variations
                season_mask = (
                    (stats["SEASON_ID"] == "2024-25") | 
                    (stats["SEASON_ID"] == "22025") |
                    (stats["SEASON_ID"].str.startswith("2024", na=False)) |
                    (stats["SEASON_ID"].str.startswith("2202", na=False))
                )
                season_24_25 = stats[season_mask]
                if not season_24_25.empty:
                    row = season_24_25.iloc[0]
                    # Ensure we have valid numeric values and round to 1 decimal
                    gp_val = pd.to_numeric(row.get("GP", 0), errors="coerce") or 0
                    pts_val = round(pd.to_numeric(row.get("PTS", 0), errors="coerce") or 0, 1)
                    reb_val = round(pd.to_numeric(row.get("REB", 0), errors="coerce") or 0, 1)
                    ast_val = round(pd.to_numeric(row.get("AST", 0), errors="coerce") or 0, 1)
                    
                    if gp_val > 0:  # Only add if player has games played
                        all_stats.append({
                            "PLAYER_NAME": player_name,
                            "GP": float(gp_val),
                            "PTS": float(pts_val),
                            "REB": float(reb_val),
                            "AST": float(ast_val),
                        })
        except Exception:
            continue  # Skip if player not found or error
    
    if not all_stats:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_stats)
    
    # Calculate weighted averages based on Games Played (round to 1 decimal)
    total_gp = df["GP"].sum()
    if total_gp > 0:
        df["WEIGHTED_PTS"] = round((df["PTS"] * df["GP"]).sum() / total_gp, 1)
        df["WEIGHTED_REB"] = round((df["REB"] * df["GP"]).sum() / total_gp, 1)
        df["WEIGHTED_AST"] = round((df["AST"] * df["GP"]).sum() / total_gp, 1)
    
    return df


def plot_allstar_comparison(deni_stats: dict, allstar_df: pd.DataFrame) -> tuple[go.Figure, dict, pd.DataFrame]:
    """Grouped bar chart comparing Deni to All-Star average and bottom 4 entry level. Returns figure, stats dict, and bottom 4 df."""
    if allstar_df.empty or not deni_stats:
        fig = go.Figure()
        fig.update_layout(title="All-Star Comparison")
        return fig, {}, pd.DataFrame()
    
    # Get weighted averages and round to 1 decimal
    total_gp = allstar_df["GP"].sum()
    if total_gp == 0:
        fig = go.Figure()
        fig.update_layout(title="All-Star Comparison")
        return fig, {}, pd.DataFrame()
    
    allstar_pts = round((allstar_df["PTS"] * allstar_df["GP"]).sum() / total_gp, 1)
    allstar_reb = round((allstar_df["REB"] * allstar_df["GP"]).sum() / total_gp, 1)
    allstar_ast = round((allstar_df["AST"] * allstar_df["GP"]).sum() / total_gp, 1)
    
    # Get Deni's stats (use current season averages) and round to 1 decimal
    deni_pts = round(deni_stats.get("PTS", 0), 1)
    deni_reb = round(deni_stats.get("REB", 0), 1)
    deni_ast = round(deni_stats.get("AST", 0), 1)
    
    # Get bottom 4 All-Stars (sorted by PTS ascending)
    bottom_4_df = allstar_df.nsmallest(4, "PTS").copy()
    bottom_4_df = bottom_4_df.sort_values("PTS", ascending=True)
    
    # Create grouped bar chart
    categories = ["PTS", "REB", "AST"]
    
    fig = go.Figure()
    
    # Deni trace (Green)
    fig.add_trace(
        go.Bar(
            name="Deni Avdija (2025-26)",
            x=categories,
            y=[deni_pts, deni_reb, deni_ast],
            marker_color="Green",
            text=[deni_pts, deni_reb, deni_ast],
            textposition="outside",
        )
    )
    
    # All-Star Average trace (Blue)
    fig.add_trace(
            go.Bar(
                name="All-Star Average (2024-25)",
                x=categories,
                y=[allstar_pts, allstar_reb, allstar_ast],
                marker_color="#1f77b4",  # Blue
                text=[allstar_pts, allstar_reb, allstar_ast],
                textposition="outside",
            )
        )
    
    # Bottom 4 Entry Level All-Stars (Gray/LightBlue)
    for idx, row in bottom_4_df.iterrows():
        player_name = row["PLAYER_NAME"]
        fig.add_trace(
            go.Bar(
                name=f"{player_name} (Entry Level)",
                x=categories,
                y=[round(row["PTS"], 1), round(row["REB"], 1), round(row["AST"], 1)],
                marker_color="LightGray",
                text=[round(row["PTS"], 1), round(row["REB"], 1), round(row["AST"], 1)],
                textposition="outside",
                showlegend=True,
            )
        )
    
    fig.update_layout(
        title="The All-Star Threshold: Deni vs All-Star Average vs Entry Level",
        xaxis_title="Metric",
        yaxis_title="Value (Per Game)",
        barmode="group",
        hovermode="x unified",
        showlegend=True,
    )
    
    stats_dict = {
        "allstar_pts": allstar_pts,
        "allstar_reb": allstar_reb,
        "allstar_ast": allstar_ast,
        "deni_pts": deni_pts,
        "deni_reb": deni_reb,
        "deni_ast": deni_ast,
    }
    
    return fig, stats_dict, bottom_4_df


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_allstar_detailed_stats() -> pd.DataFrame:
    """Fetch detailed stats for All-Stars including USG_PCT and TS_PCT for scatter plots."""
    allstar_names = [
        "Giannis Antetokounmpo",
        "Jayson Tatum",
        "Joel Embiid",
        "Tyrese Haliburton",
        "Damian Lillard",
        "Jalen Brunson",
        "Donovan Mitchell",
        "Tyrese Maxey",
        "Paolo Banchero",
        "Jaylen Brown",
        "Julius Randle",
        "Bam Adebayo",
        "Trae Young",
        "Scottie Barnes",
        "LeBron James",
        "Kevin Durant",
        "Nikola Jokic",
        "Luka Doncic",
        "Shai Gilgeous-Alexander",
        "Stephen Curry",
        "Anthony Edwards",
        "Kawhi Leonard",
        "Paul George",
        "Devin Booker",
        "Anthony Davis",
        "Karl-Anthony Towns",
    ]
    
    all_stats = []
    
    for player_name in allstar_names:
        try:
            hits = players.find_players_by_full_name(player_name)
            if hits and len(hits) > 0:
                player_id = hits[0]["id"]
                # Get advanced stats for USG_PCT and TS_PCT
                adv_stats = PlayerDashboardByYearOverYear(
                    player_id=player_id,
                    per_mode_detailed="PerGame",
                    measure_type_detailed="Advanced",
                ).get_data_frames()[1]
                adv_stats["SEASON_ID"] = adv_stats["GROUP_VALUE"].astype(str)
                
                # Get basic stats for PTS, REB, AST
                basic_stats = PlayerCareerStats(player_id=player_id, per_mode36="PerGame").get_data_frames()[0]
                basic_stats["SEASON_ID"] = basic_stats["SEASON_ID"].astype(str)
                
                # Get 2024-25 season
                season_mask = (
                    (adv_stats["SEASON_ID"] == "2024-25") | 
                    (adv_stats["SEASON_ID"] == "22025") |
                    (adv_stats["SEASON_ID"].str.startswith("2024", na=False)) |
                    (adv_stats["SEASON_ID"].str.startswith("2202", na=False))
                )
                adv_24_25 = adv_stats[season_mask]
                basic_24_25 = basic_stats[
                    (basic_stats["SEASON_ID"] == "2024-25") | 
                    (basic_stats["SEASON_ID"] == "22025") |
                    (basic_stats["SEASON_ID"].str.startswith("2024", na=False)) |
                    (basic_stats["SEASON_ID"].str.startswith("2202", na=False))
                ]
                
                if not adv_24_25.empty and not basic_24_25.empty:
                    adv_row = adv_24_25.iloc[0]
                    basic_row = basic_24_25.iloc[0]
                    
                    # Round all values to 1 decimal (or 3 for percentages)
                    all_stats.append({
                        "PLAYER_NAME": player_name,
                        "PTS": round(pd.to_numeric(basic_row.get("PTS", 0), errors="coerce") or 0, 1),
                        "REB": round(pd.to_numeric(basic_row.get("REB", 0), errors="coerce") or 0, 1),
                        "AST": round(pd.to_numeric(basic_row.get("AST", 0), errors="coerce") or 0, 1),
                        "USG_PCT": round(pd.to_numeric(adv_row.get("USG_PCT", 0), errors="coerce") or 0, 3),
                        "TS_PCT": round(pd.to_numeric(adv_row.get("TS_PCT", 0), errors="coerce") or 0, 3),
                    })
        except Exception:
            continue
    
    if not all_stats:
        return pd.DataFrame()
    
    return pd.DataFrame(all_stats)


def plot_separation_chart(allstar_df: pd.DataFrame, deni_stats: dict, career_df_ref: pd.DataFrame = None) -> go.Figure:
    """Scatter plot showing Usage vs True Shooting (The Separation Chart)."""
    if allstar_df.empty or not deni_stats:
        fig = go.Figure()
        fig.update_layout(title="The Separation Chart")
        return fig
    
    fig = go.Figure()
    
    # Plot All-Stars
    fig.add_trace(
        go.Scatter(
            x=allstar_df["USG_PCT"] * 100,  # Convert to percentage
            y=allstar_df["TS_PCT"] * 100,
            mode="markers",
            name="All-Stars",
            marker=dict(color="LightGray", size=10),
            text=allstar_df["PLAYER_NAME"],
            hovertemplate="%{text}<br>Usage: %{x:.1f}%<br>TS%: %{y:.1f}%<extra></extra>",
        )
    )
    
    # Plot Deni
    deni_usg = deni_stats.get("USG_PCT", 0)
    deni_ts = deni_stats.get("TS_PCT", 0)
    
    # Convert to percentage if needed
    if isinstance(deni_usg, (int, float)) and deni_usg <= 1:
        deni_usg = deni_usg * 100
    if isinstance(deni_ts, (int, float)) and deni_ts <= 1:
        deni_ts = deni_ts * 100
    
    # Get from career_df_ref if available
    if (deni_usg == 0 or deni_ts == 0) and career_df_ref is not None and not career_df_ref.empty:
        latest = career_df_ref.iloc[-1]
        deni_usg = round(latest.get("USG_PCT", 0) * 100, 1)
        deni_ts = round(latest.get("TS_PCT", 0) * 100, 1)
    
    if deni_usg > 0 or deni_ts > 0:
        fig.add_trace(
            go.Scatter(
                x=[round(deni_usg, 1)],
                y=[round(deni_ts, 1)],
                mode="markers+text",
                name="Deni Avdija",
                marker=dict(color="Red", size=18, symbol="star"),
                text=["Deni"],
                textposition="top center",
                hovertemplate=f"Deni Avdija<br>Usage: {deni_usg:.1f}%<br>TS%: {deni_ts:.1f}%<extra></extra>",
            )
        )
        
        # Label top outliers
        if len(allstar_df) > 0:
            top_usage = allstar_df.nlargest(2, "USG_PCT")
            for _, row in top_usage.iterrows():
                fig.add_annotation(
                    x=row["USG_PCT"] * 100,
                    y=row["TS_PCT"] * 100,
                    text=row["PLAYER_NAME"].split()[-1],
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-30,
                    font=dict(size=10),
                )
    
    fig.update_layout(
        title="The Separation Chart: Usage vs True Shooting Efficiency",
        xaxis_title="Usage Percentage (%)",
        yaxis_title="True Shooting Percentage (%)",
        hovermode="closest",
        showlegend=True,
    )
    
    return fig


def plot_triple_threat(allstar_df: pd.DataFrame, deni_stats: dict) -> go.Figure:
    """3D scatter plot showing Points, Rebounds, Assists (The Triple Threat)."""
    if allstar_df.empty or not deni_stats:
        fig = go.Figure()
        fig.update_layout(title="The Triple Threat")
        return fig
    
    fig = go.Figure()
    
    # Plot All-Stars
    fig.add_trace(
        go.Scatter3d(
            x=allstar_df["PTS"],
            y=allstar_df["REB"],
            z=allstar_df["AST"],
            mode="markers",
            name="All-Stars",
            marker=dict(
                color="#1f77b4",
                size=8,
                opacity=0.7,
            ),
            text=allstar_df["PLAYER_NAME"],
            hovertemplate="%{text}<br>PTS: %{x:.1f}<br>REB: %{y:.1f}<br>AST: %{z:.1f}<extra></extra>",
        )
    )
    
    # Plot Deni
    deni_pts = round(deni_stats.get("PTS", 0), 1)
    deni_reb = round(deni_stats.get("REB", 0), 1)
    deni_ast = round(deni_stats.get("AST", 0), 1)
    
    fig.add_trace(
        go.Scatter3d(
            x=[deni_pts],
            y=[deni_reb],
            z=[deni_ast],
            mode="markers+text",
            name="Deni Avdija",
            marker=dict(
                color="Green",
                size=15,
                symbol="diamond",
            ),
            text=["Deni"],
            textposition="middle center",
            hovertemplate=f"Deni Avdija<br>PTS: {deni_pts:.1f}<br>REB: {deni_reb:.1f}<br>AST: {deni_ast:.1f}<extra></extra>",
        )
    )
    
    fig.update_layout(
        title="The Triple Threat: Points, Rebounds, Assists (3D View)",
        scene=dict(
            xaxis_title="Points",
            yaxis_title="Rebounds",
            zaxis_title="Assists",
        ),
        hovermode="closest",
    )
    
    return fig


def plot_current_season(logs: pd.DataFrame, season: str) -> go.Figure:
    logs_sorted = logs.sort_values("GAME_DATE")
    bar_colors = ["#2ca02c" if res == "W" else "#d62728" for res in logs_sorted.get("WL", [])]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=logs_sorted["GAME_DATE"],
            y=logs_sorted["MIN"],
            marker_color=bar_colors,
            name="Minutes Played",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>MIN: %{y}<br>Result: %{customdata}",
            customdata=logs_sorted.get("WL", []),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=logs_sorted["GAME_DATE"],
            y=logs_sorted["PTS"],
            mode="lines+markers",
            yaxis="y2",
            name="Points",
            line=dict(color="#1f77b4", width=3),
        )
    )
    fig.update_layout(
        title=f"{season} Game Impact: Minutes vs Points",
        xaxis_title="Game Date",
        yaxis=dict(title="Minutes Played"),
        yaxis2=dict(title="Points", overlaying="y", side="right"),
        hovermode="x unified",
    )
    return fig


def draw_nba_court() -> go.Figure:
    """Create a Plotly figure with NBA half-court lines drawn as shapes."""
    fig = go.Figure()
    
    # Court dimensions in NBA API coordinate system (Half Court View)
    # X: -250 to 250 (50ft wide)
    # Y: -47.5 to 422.5 (47ft = half court depth, with basket at bottom)
    court_width = 500  # -250 to 250
    half_court_depth = 470  # -47.5 to 422.5
    
    # Baseline (bottom)
    fig.add_shape(
        type="rect",
        x0=-250, y0=-47.5, x1=250, y1=-47.5,
        line=dict(color="black", width=2),
        layer="below",
    )
    # Sidelines (only half court)
    fig.add_shape(
        type="line",
        x0=-250, y0=-47.5, x1=-250, y1=422.5,
        line=dict(color="black", width=2),
        layer="below",
    )
    fig.add_shape(
        type="line",
        x0=250, y0=-47.5, x1=250, y1=422.5,
        line=dict(color="black", width=2),
        layer="below",
    )
    # Half-court line (top of offensive half)
    fig.add_shape(
        type="line",
        x0=-250, y0=422.5, x1=250, y1=422.5,
        line=dict(color="black", width=2, dash="dash"),
        layer="below",
    )
    
    # Free throw line (15ft from baseline = 180 units from baseline at y=-47.5, so y=132.5)
    free_throw_y = -47.5 + 180
    fig.add_shape(
        type="line",
        x0=-80, y0=free_throw_y, x1=80, y1=free_throw_y,
        line=dict(color="black", width=2),
        layer="below",
    )
    
    # Paint/Lane (16ft wide = 192 units, extends 15ft = 180 units from baseline)
    # Left side
    fig.add_shape(
        type="line",
        x0=-80, y0=-47.5, x1=-80, y1=free_throw_y,
        line=dict(color="black", width=2),
        layer="below",
    )
    # Right side
    fig.add_shape(
        type="line",
        x0=80, y0=-47.5, x1=80, y1=free_throw_y,
        line=dict(color="black", width=2),
        layer="below",
    )
    
    # Restricted area arc (4ft radius = 48 units from basket at (0, 0))
    # Draw semicircle using scatter plot
    restricted_arc_x = []
    restricted_arc_y = []
    for i in range(101):
        x = -48 + (i / 100) * 96  # x from -48 to 48
        y = (48**2 - x**2)**0.5  # Calculate y from circle equation
        restricted_arc_x.append(x)
        restricted_arc_y.append(y)
    fig.add_trace(
        go.Scatter(
            x=restricted_arc_x,
            y=restricted_arc_y,
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    
    # Three-point line
    # Top arc: 23.75ft = 285 units radius, center at (0, 0)
    # The arc extends to 14ft from baseline in corners = 168 units from baseline
    three_pt_radius = 285  # 23.75ft in NBA coordinate units
    corner_y = -47.5 + 168  # 14ft from baseline
    
    # Calculate three-point arc points
    # Arc goes from x=-220 to x=220, with radius 285 from center (0,0)
    three_pt_arc_x = []
    three_pt_arc_y = []
    for i in range(101):
        x = -220 + (i / 100) * 440  # x from -220 to 220
        if abs(x) < three_pt_radius:
            y = (three_pt_radius**2 - x**2)**0.5
            if y >= corner_y:  # Only show part above corner distance
                three_pt_arc_x.append(x)
                three_pt_arc_y.append(y)
    fig.add_trace(
        go.Scatter(
            x=three_pt_arc_x,
            y=three_pt_arc_y,
            mode="lines",
            line=dict(color="black", width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    # Left corner line
    fig.add_shape(
        type="line",
        x0=-250, y0=-47.5, x1=-250, y1=corner_y,
        line=dict(color="black", width=2),
        layer="below",
    )
    # Right corner line
    fig.add_shape(
        type="line",
        x0=250, y0=-47.5, x1=250, y1=corner_y,
        line=dict(color="black", width=2),
        layer="below",
    )
    
    # Hoop (circle at basket location (0, 0))
    fig.add_shape(
        type="circle",
        xref="x", yref="y",
        x0=-7.5, y0=-7.5, x1=7.5, y1=7.5,
        line=dict(color="orange", width=2),
        fillcolor="orange",
        layer="below",
    )
    
    # Backboard (6ft wide = 72 units, 4ft from baseline = 48 units from baseline)
    backboard_y = -47.5 + 48
    fig.add_shape(
        type="rect",
        x0=-36, y0=backboard_y, x1=36, y1=backboard_y,
        line=dict(color="black", width=2),
        layer="below",
    )
    
    # Set axis properties (Half Court View) - Fixed aspect ratio with hardwood floor
    fig.update_layout(
        xaxis=dict(
            range=[-260, 260],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            range=[-60, 440],
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        plot_bgcolor="#d2b48c",  # Hardwood tan color
        paper_bgcolor="white",
        width=600,
        height=564,  # Proper ratio for 50ft x 47ft
    )
    
    return fig


# -----------------------------
# Sidebar & About Me
# -----------------------------
def sidebar_header() -> None:
    st.sidebar.markdown(
        """
        <div style="display:flex; flex-direction:column; align-items:center; gap:12px;">
            <div style="
                width:120px; height:120px; border-radius:60px;
                background: linear-gradient(135deg, #1f77b4, #2ca02c);
                display:flex; align-items:center; justify-content:center;
                color:white; font-weight:700; font-size:26px;
            ">
                DA
            </div>
            <div style="text-align:center;">
                <h3 style="margin:0;">Deni Avdija</h3>
                <p style="margin:0; color:#6c757d;">360° Performance Analytics</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def about_me_page() -> None:
    st.subheader("About Me")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            """
            <div style="
                width:160px; height:160px; border-radius:80px;
                background: linear-gradient(135deg, #1f77b4, #2ca02c);
                display:flex; align-items:center; justify-content:center;
                color:white; font-weight:800; font-size:32px;
                box-shadow:0 4px 12px rgba(0,0,0,0.15);
            ">
                DA
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown("### Bio")
        st.write(
            "Data professional with a focus on sports analytics, building dashboards "
            "that translate on-court performance into actionable insights for coaches, "
            "scouts, and fans."
        )
        st.markdown("### Skills")
        st.write("- Python  •  SQL  •  Data Visualization (Plotly, Streamlit)")
        st.markdown("### Contact")
        st.write("LinkedIn: https://www.linkedin.com/in/your-profile")
        st.write("GitHub: https://github.com/your-profile")


# -----------------------------
# Layout & page selection
# -----------------------------
sidebar_header()
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Detailed Analysis", "Raw Data", "Shot Maps", "Research: Deep Dive", "About Me"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Data via nba_api • Plots via Plotly • Cached for speed")


# -----------------------------
# Data fetch
# -----------------------------
player_id = get_player_id()
career_basic_df = fetch_career_basic(player_id)
career_adv_df = fetch_career_advanced(player_id)
logs_df = fetch_game_logs(player_id, "2025-26")  # Explicitly fetch 2025-26
logs_df_24_25 = fetch_game_logs(player_id, "2024-25")  # 2024-25
career_df = merge_career_frames(career_basic_df, career_adv_df)


# -----------------------------
# Pages
# -----------------------------
st.title("Deni Avdija: 360° Performance Analytics")
st.write(
    "Tracking scoring, rebounding, playmaking, efficiency, and current-season impact "
    "to give a complete scouting read on Deni Avdija."
)

if page == "Dashboard":
    st.markdown("### Current Season Game Logs")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### {CURRENT_SEASON}")
    if logs_df.empty:
        st.info(f"No {CURRENT_SEASON} game logs yet.")
    else:
            st.plotly_chart(plot_current_season(logs_df, CURRENT_SEASON), use_container_width=True)
    with col2:
        st.markdown("#### 2024-25")
        if logs_df_24_25.empty:
            st.info("No 2024-25 game logs yet.")
        else:
            st.plotly_chart(plot_current_season(logs_df_24_25, "2024-25"), use_container_width=True)

elif page == "Detailed Analysis":
    st.subheader("Career Trends")
    if career_df.empty:
        st.warning("No career data returned yet. Try again shortly.")
    else:
        st.plotly_chart(plot_per_game_stats(career_df), use_container_width=True)
        st.plotly_chart(plot_per_36_stats(career_df), use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_usage_growth(career_df), use_container_width=True)
        with c2:
            st.plotly_chart(plot_ts_growth(career_df), use_container_width=True)

    st.subheader("Season Deep Dive")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### {CURRENT_SEASON}")
    if logs_df.empty:
        st.info(f"No {CURRENT_SEASON} game logs yet.")
    else:
            st.plotly_chart(plot_current_season(logs_df, CURRENT_SEASON), use_container_width=True)
    with col2:
        st.markdown("#### 2024-25")
        if logs_df_24_25.empty:
            st.info("No 2024-25 game logs yet.")
        else:
            st.plotly_chart(plot_current_season(logs_df_24_25, "2024-25"), use_container_width=True)

elif page == "Raw Data":
    st.subheader("Master Career Data")
    
    if career_df.empty:
        st.warning("No career data available.")
    else:
        # Column renaming mapping
        column_mapping = {
            "SEASON_ID": "Season",
            "PTS": "Points",
            "REB": "Rebounds",
            "AST": "Assists",
            "GP": "Games",
            "MIN": "Minutes",
            "FG_PCT": "FG%",
            "FG3_PCT": "3P%",
            "FT_PCT": "FT%",
            "USG_PCT": "Usage",
            "TS_PCT": "True Shooting",
            "NET_RATING": "Net Rating",
            "AST_TO": "AST/TO",
        }
        
        # Create display dataframe with renamed columns
        display_df = career_df.copy()
        
        # Drop unwanted columns
        columns_to_drop = ["PLAYER_ID", "LEAGUE_ID", "TEAM_ID", "Team_ID"]
        for col in columns_to_drop:
            if col in display_df.columns:
                display_df = display_df.drop(columns=[col])
        
        display_df = display_df.rename(columns=column_mapping)
        
        # Multi-Trend Viewer (before formatting percentages)
        st.markdown("### 📊 Multi-Trend Viewer: Select metrics to compare over seasons")
        
        # Get available numeric metrics for selection (use original column names)
        numeric_metrics = []
        metric_mapping = {}  # Maps display name to original column name in career_df
        
        # Build list of available numeric metrics
        if "PTS" in career_df.columns:
            numeric_metrics.append("Points")
            metric_mapping["Points"] = "PTS"
        if "REB" in career_df.columns:
            numeric_metrics.append("Rebounds")
            metric_mapping["Rebounds"] = "REB"
        if "AST" in career_df.columns:
            numeric_metrics.append("Assists")
            metric_mapping["Assists"] = "AST"
        if "MIN" in career_df.columns:
            numeric_metrics.append("Minutes")
            metric_mapping["Minutes"] = "MIN"
        if "NET_RATING" in career_df.columns:
            numeric_metrics.append("Net Rating")
            metric_mapping["Net Rating"] = "NET_RATING"
        if "AST_TO" in career_df.columns:
            numeric_metrics.append("AST/TO")
            metric_mapping["AST/TO"] = "AST_TO"
        if "USG_PCT" in career_df.columns:
            numeric_metrics.append("Usage")
            metric_mapping["Usage"] = "USG_PCT"
        if "TS_PCT" in career_df.columns:
            numeric_metrics.append("True Shooting")
            metric_mapping["True Shooting"] = "TS_PCT"
        
        if numeric_metrics and "SEASON_ID" in career_df.columns:
            selected_metrics = st.multiselect(
                "Select Metrics",
                numeric_metrics,
                default=["Points"] if "Points" in numeric_metrics else [numeric_metrics[0]] if numeric_metrics else [],
            )
            
            if selected_metrics:
                # Create trend dataframe from original career_df
                trend_df = career_df[["SEASON_ID"]].copy()
                trend_df.columns = ["Season"]
                
                # Add selected metrics (convert percentages to percentage points)
                for metric_display in selected_metrics:
                    orig_col = metric_mapping[metric_display]
                    if orig_col in career_df.columns:
                        values = career_df[orig_col].copy()
                        # Convert percentage columns to percentage points (0.45 -> 45)
                        if "PCT" in orig_col or "USG" in orig_col:
                            values = values * 100
                        trend_df[metric_display] = values
                
                # Create multi-line chart
                fig = px.line(
                    trend_df,
                    x="Season",
                    y=selected_metrics,
                    title="Career Trends: Multiple Metrics Comparison",
                    markers=True,
                )
                fig.update_layout(
                    xaxis_title="Season",
                    yaxis_title="Value",
                    hovermode="x unified",
                    legend=dict(title="Metrics"),
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Now format percentage columns for display
        percentage_cols = ["FG%", "3P%", "FT%"]
        for col in percentage_cols:
            if col in display_df.columns:
                display_df[col] = (display_df[col] * 100).round(1).astype(str) + "%"
        
        # Format Usage and True Shooting as percentages
        if "Usage" in display_df.columns:
            display_df["Usage"] = (display_df["Usage"] * 100).round(1).astype(str) + "%"
        if "True Shooting" in display_df.columns:
            display_df["True Shooting"] = (display_df["True Shooting"] * 100).round(1).astype(str) + "%"
        
        # Display master table
        st.markdown("### Career Statistics Table")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

elif page == "Shot Maps":
    st.subheader("Shot Maps")
    season = st.selectbox("Select Season", ["2022-23", "2023-24", "2024-25", "2025-26"], index=3)
    
    shot_df = fetch_shot_data(player_id, season)
    
    if shot_df.empty:
        st.warning(f"No shot data available for {season}.")
    else:
        # Create court figure
        fig = draw_nba_court()
        
        # Filter out shots with missing location data
        shot_df_clean = shot_df.dropna(subset=["LOC_X", "LOC_Y"])
        
        if shot_df_clean.empty:
            st.warning("No shot location data available.")
        else:
            # Separate made and missed shots
            made_shots = shot_df_clean[shot_df_clean["EVENT_TYPE"] == "Made Shot"]
            missed_shots = shot_df_clean[shot_df_clean["EVENT_TYPE"] == "Missed Shot"]
            
            # Add made shots (green)
            if not made_shots.empty:
                fig.add_trace(
                    go.Scatter(
                        x=made_shots["LOC_X"],
                        y=made_shots["LOC_Y"],
                        mode="markers",
                        name="Made Shot",
                        marker=dict(color="green", size=6, opacity=0.7),
                        hovertemplate="X: %{x}<br>Y: %{y}<extra></extra>",
                    )
                )
            
            # Add missed shots (red)
            if not missed_shots.empty:
                fig.add_trace(
                    go.Scatter(
                        x=missed_shots["LOC_X"],
                        y=missed_shots["LOC_Y"],
                        mode="markers",
                        name="Missed Shot",
                        marker=dict(color="red", size=6, opacity=0.7),
                        hovertemplate="X: %{x}<br>Y: %{y}<extra></extra>",
                    )
                )
            
            # Update title
            fig.update_layout(
                title=f"Shot Chart - {season}",
                showlegend=True,
            )
            
            st.plotly_chart(fig, use_container_width=True)

elif page == "Research: Deep Dive":
    st.subheader("Research: Deep Dive Analysis")
    
    # All-Star Threshold Section
    st.markdown("### The All-Star Threshold")
    st.write("Comparing Deni's 2025-26 performance against the weighted average of 2024 NBA All-Stars (2024-25 season stats) and the 'Entry Level' All-Stars (bottom 4 scorers).")
    
    # Fetch All-Star data
    with st.spinner("Loading All-Star comparison data..."):
        allstar_df = fetch_allstar_stats()
        allstar_detailed_df = fetch_allstar_detailed_stats()
    
    # Get Deni's current season stats - improved fallback logic
    deni_stats = {}
    deni_advanced_stats = {}
    
    # Try to get Deni's 2025-26 stats from career data first
    if not career_df.empty:
        deni_current = career_df[career_df["SEASON_ID"] == "2025-26"]
        if not deni_current.empty:
            deni_stats = {
                "PTS": round(deni_current.iloc[0].get("PTS", 0), 1),
                "REB": round(deni_current.iloc[0].get("REB", 0), 1),
                "AST": round(deni_current.iloc[0].get("AST", 0), 1),
            }
            deni_advanced_stats = {
                "USG_PCT": round(deni_current.iloc[0].get("USG_PCT", 0), 3),
                "TS_PCT": round(deni_current.iloc[0].get("TS_PCT", 0), 3),
            }
    
    # If not found, calculate from game logs
    if not deni_stats and not logs_df.empty:
        if "PTS" in logs_df.columns and len(logs_df) > 0:
            deni_stats = {
                "PTS": round(logs_df["PTS"].mean(), 1),
                "REB": round(logs_df["REB"].mean() if "REB" in logs_df.columns else 0, 1),
                "AST": round(logs_df["AST"].mean() if "AST" in logs_df.columns else 0, 1),
            }
    
    # Final fallback to most recent season from career data
    if not deni_stats and not career_df.empty:
        latest_season = career_df.iloc[-1]
        deni_stats = {
            "PTS": round(latest_season.get("PTS", 0), 1),
            "REB": round(latest_season.get("REB", 0), 1),
            "AST": round(latest_season.get("AST", 0), 1),
        }
        deni_advanced_stats = {
            "USG_PCT": round(latest_season.get("USG_PCT", 0), 3),
            "TS_PCT": round(latest_season.get("TS_PCT", 0), 3),
        }
    
    # Merge advanced stats into deni_stats
    deni_stats.update(deni_advanced_stats)
    
    # Show status messages and ensure we always have data
    if allstar_df.empty:
        st.error("❌ Could not fetch All-Star comparison data.")
        st.info("💡 **Troubleshooting:**")
        st.write("- The All-Star data is being fetched from the NBA API")
        st.write("- This may take a moment on first load - please refresh the page")
        st.write("- API rate limits may cause delays - wait a few seconds and try again")
        if st.button("🔄 Retry Loading All-Star Data"):
            st.cache_data.clear()
            st.rerun()
    else:
        # Ensure we have Deni stats
        if not deni_stats:
            st.warning("⚠️ Could not fetch Deni's current season stats for comparison.")
            st.info("💡 Using most recent available season data as fallback.")
            deni_stats = {"PTS": 0.0, "REB": 0.0, "AST": 0.0}
        
        # Display All-Star Comparison Chart
        try:
            comparison_fig, stats, bottom_4_df = plot_allstar_comparison(deni_stats, allstar_df)
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            # Full Comparison Table with Deni
            st.markdown("### Full Comparison Table")
            
            # Prepare comparison dataframe with all All-Stars + Deni
            comparison_df = allstar_df[["PLAYER_NAME", "GP", "PTS", "REB", "AST"]].copy()
            # Round All-Star stats to 1 decimal (ensure proper float format)
            comparison_df["PTS"] = comparison_df["PTS"].round(1)
            comparison_df["REB"] = comparison_df["REB"].round(1)
            comparison_df["AST"] = comparison_df["AST"].round(1)
            # Rename columns to clean names
            comparison_df.columns = ["Player", "GP", "Points", "Rebounds", "Assists"]
            
            # Add Deni's row (round to 1 decimal)
            deni_row = pd.DataFrame({
                "Player": ["Deni Avdija (2025-26)"],
                "GP": [logs_df.shape[0] if not logs_df.empty else 0],
                "Points": [round(deni_stats["PTS"], 1)],
                "Rebounds": [round(deni_stats["REB"], 1)],
                "Assists": [round(deni_stats["AST"], 1)],
            })
            comparison_df = pd.concat([comparison_df, deni_row], ignore_index=True)
            
            # Sort by Points descending
            comparison_df = comparison_df.sort_values("Points", ascending=False)
            
            # Reset index for styling
            comparison_df = comparison_df.reset_index(drop=True)
            
            # Apply yellow highlighting to Deni's row
            def highlight_deni(row):
                if "Deni Avdija" in str(row["Player"]):
                    return ["background-color: yellow"] * len(row)
                return [""] * len(row)
            
            styled_df = comparison_df.style.apply(highlight_deni, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Data Source Expander (All-Stars only, sorted)
            with st.expander("📋 All-Star Data Source (2024-25 Stats)", expanded=False):
                source_df = allstar_df[["PLAYER_NAME", "GP", "PTS", "REB", "AST"]].copy()
                # Round to 1 decimal before renaming
                source_df["PTS"] = source_df["PTS"].round(1)
                source_df["REB"] = source_df["REB"].round(1)
                source_df["AST"] = source_df["AST"].round(1)
                # Rename to clean column names
                source_df.columns = ["Player", "GP", "Points", "Rebounds", "Assists"]
                source_df = source_df.sort_values("Points", ascending=False)
                # Add weighted average row
                total_gp = allstar_df["GP"].sum()
                weighted_row = pd.DataFrame({
                    "Player": ["**Weighted Average**"],
                    "GP": [int(total_gp)],
                    "Points": [round(stats["allstar_pts"], 1)],
                    "Rebounds": [round(stats["allstar_reb"], 1)],
                    "Assists": [round(stats["allstar_ast"], 1)],
                })
                source_df = pd.concat([source_df, weighted_row], ignore_index=True)
                st.dataframe(source_df, use_container_width=True, hide_index=True)
            
            # Analysis conclusion
            if stats:
                pts_delta = round(stats["deni_pts"] - stats["allstar_pts"], 1)
                reb_delta = round(stats["deni_reb"] - stats["allstar_reb"], 1)
                ast_delta = round(stats["deni_ast"] - stats["allstar_ast"], 1)
                
                st.markdown("#### Analysis")
                st.write(f"**Points**: Deni {stats['deni_pts']:.1f} vs All-Star Avg {stats['allstar_pts']:.1f} ({pts_delta:+.1f})")
                st.write(f"**Rebounds**: Deni {stats['deni_reb']:.1f} vs All-Star Avg {stats['allstar_reb']:.1f} ({reb_delta:+.1f})")
                st.write(f"**Assists**: Deni {stats['deni_ast']:.1f} vs All-Star Avg {stats['allstar_ast']:.1f} ({ast_delta:+.1f})")
                
                # Simple conclusion logic based on rounded numbers
                metrics_above = sum([pts_delta > 0, reb_delta > 0, ast_delta > 0])
                if metrics_above >= 2:
                    conclusion = "✅ **Deni is performing at an All-Star level** - matching or exceeding All-Star averages in multiple categories."
                elif metrics_above == 1:
                    conclusion = "⚠️ **Deni is approaching All-Star level** - showing strength in some categories but needs improvement in others."
                else:
                    conclusion = "📊 **Deni is below All-Star level** - needs improvement across key metrics to reach All-Star standards."
                
                st.markdown(f"**Conclusion**: {conclusion}")
            
            # New Scouting Visualizations
            st.markdown("---")
            st.markdown("### Advanced Scouting Visualizations")
            
            # The Separation Chart
            st.markdown("---")
            st.markdown("### Advanced Scouting Visualizations")
            st.markdown("#### The Separation Chart: Usage vs Efficiency")
            st.write("Compare Deni's efficiency (True Shooting %) against his usage rate, relative to All-Stars. Elite players typically occupy the high-usage, high-efficiency quadrant.")
            
            if not allstar_detailed_df.empty and len(allstar_detailed_df) > 0:
                separation_fig = plot_separation_chart(allstar_detailed_df, deni_stats, career_df)
                st.plotly_chart(separation_fig, use_container_width=True)
            else:
                st.warning("⚠️ Detailed All-Star stats (Usage/TS%) not available for separation chart.")
            
            # The Triple Threat
            st.markdown("#### The Triple Threat: 3D All-Around Game")
            st.write("Visualize Deni's all-around production (Points, Rebounds, Assists) in 3D space compared to All-Stars. This shows if he's a volume contributor or specialist.")
            
            triple_threat_fig = plot_triple_threat(allstar_df, deni_stats)
            st.plotly_chart(triple_threat_fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error displaying comparison: {str(e)}")
            st.info("💡 Please try refreshing the page or contact support if the issue persists.")

elif page == "About Me":
    about_me_page()

