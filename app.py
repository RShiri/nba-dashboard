"""
Deni Avdija: 360° Performance Analytics Dashboard

Features:
- Smart Data Patching: Updates career stats from game logs if stale.
- Hexbin Shot Maps: Advanced density visualization with fixed aspect ratios.
- Multi-Trend Viewer: Interactive career analysis.
- Triple Threat & Threshold: Deep dive all-star comparisons restored.
- Dynamic Ranking: Customizable player ranking with correct 1-decimal formatting.
- Dual-Season Dashboard: Side-by-side impact analysis.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import os
import time
import fetch_data
import pickle
import numpy as np
from nba_api.stats.static import players
from pathlib import Path
from datetime import datetime, timedelta

# -----------------------------
# Config and constants
# -----------------------------
PLAYER_NAME = "Deni Avdija"
CURRENT_SEASON = "2025-26"
DATA_FILE = "nba_data.pkl"

# Colorblind-Safe Color Palette
COLOR_POSITIVE = "#2c7bb6"  # Strong Blue (PTS)
COLOR_NEGATIVE = "#d7191c"  # Strong Red-Orange
COLOR_DENI = "#00CC96"  # Teal
COLOR_HIGHLIGHT = "#AB63FA"  # Purple
COLOR_AST = "#E69F00" # Orange/Gold (Colorblind Safe for Assists)
COLOR_GRAY = "#636EFA"  # Blue-gray

# Plotly Config for High-Res Downloads
PLOT_CONFIG = {
    'toImageButtonOptions': {
        'format': 'png', # one of png, svg, jpeg, webp
        'filename': 'custom_image',
        'height': 1000,
        'width': 1400,
        'scale': 2 # Multiply title/legend/axis/canvas sizes by this factor
    },
    'displayModeBar': True
}

st.set_page_config(
    page_title="Deni Avdija Analytics",
    layout="wide",
)


# -----------------------------
# Data Loading & Patching
# -----------------------------
# -----------------------------
# Auto-Update Logic
# -----------------------------
def check_and_update_data():
    """
    Smart update checker that detects new games and updates data automatically.
    Uses session state caching to prevent excessive API calls.
    """
    try:
        # Initialize session state for caching
        if "last_game_check_time" not in st.session_state:
            st.session_state.last_game_check_time = None
            st.session_state.last_known_game_count = 0
        
        # CASE 1: No data file exists - fetch initial data
        if not Path(DATA_FILE).exists():
            status = st.empty()
            status.info("⏳ No data found. Fetching initial data...")
            fetch_data.smart_update()
            status.success("✅ Initial data loaded!")
            time.sleep(1)
            status.empty()
            st.rerun()
            return
        
        # CASE 2: Smart schedule-based checking (only after Portland games)
        now = datetime.now()
        
        if st.session_state.last_game_check_time:
            # Check if we should look for new games based on schedule
            should_check = fetch_data.should_check_for_new_game(
                last_check=st.session_state.last_game_check_time,
                existing_logs=logs_25_26
            )
        else:
            # First run - check once
            should_check = True
        
        if should_check:
            # Load existing data to check game count
            try:
                with open(DATA_FILE, "rb") as f:
                    existing_data = pickle.load(f)
                
                logs_25_26 = existing_data.get("game_logs_2025_26", pd.DataFrame())
                
                # Check if there's a new completed game
                status = st.empty()
                status.info("🔍 Checking for new games...")
                
                has_new_game = fetch_data.check_new_games(logs_25_26)
                st.session_state.last_game_check_time = now
                
                if has_new_game:
                    status.success("🎮 New game detected! Updating stats...")
                    fetch_data.smart_update()
                    status.success("✅ Stats updated with latest game!")
                    time.sleep(2)
                    status.empty()
                    st.rerun()
                else:
                    # Update game count for tracking
                    st.session_state.last_known_game_count = len(logs_25_26)
                    status.empty()
                    
            except Exception as e:
                print(f"Error checking for new games: {e}")
                # Don't crash the app, just log the error
                
    except Exception as e:
        print(f"Update check failed: {e}")

# Run check immediately on app load
check_and_update_data()


@st.cache_data(show_spinner=False)
def load_nba_data(mtime: float) -> dict:
    if not Path(DATA_FILE).exists():
        return {}
    try:
        with open(DATA_FILE, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Error loading data file: {e}")
        return {}


def patch_career_stats(career_df: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    """Overwrite 2025-26 career stats with fresh aggregates from game logs if logs have more games."""
    if logs.empty or career_df.empty:
        return career_df
    
    # Ensure correct types
    if "SEASON_ID" in career_df.columns:
        career_df["SEASON_ID"] = career_df["SEASON_ID"].astype(str)
        
    true_gp = len(logs)
    if true_gp == 0: return career_df
    
    # Calculate Fresh Stats
    updated_row = {
        "GP": true_gp,
        "GS": true_gp, 
        "MIN": logs["MIN"].mean() if "MIN" in logs.columns else 0,
        "PTS": logs["PTS"].mean(),
        "REB": logs["REB"].mean(),
        "AST": logs["AST"].mean(),
        "STL": logs["STL"].mean() if "STL" in logs.columns else 0,
        "BLK": logs["BLK"].mean() if "BLK" in logs.columns else 0,
        "TOV": logs["TOV"].mean() if "TOV" in logs.columns else 0,
        "FG_PCT": (logs["FGM"].sum() / logs["FGA"].sum()) if logs["FGA"].sum() > 0 else 0,
        "FG3_PCT": (logs["FG3M"].sum() / logs["FG3A"].sum()) if logs["FG3A"].sum() > 0 else 0,
        "FT_PCT": (logs["FTM"].sum() / logs["FTA"].sum()) if logs["FTA"].sum() > 0 else 0,
    }
    
    mask = career_df["SEASON_ID"] == "2025-26"
    if mask.any():
        idx = career_df.index[mask][0]
        current_gp = career_df.at[idx, "GP"]
        if true_gp >= current_gp: 
             for col, val in updated_row.items():
                 if col in career_df.columns:
                     career_df.at[idx, col] = val
    return career_df


def merge_career_frames(basic_df: pd.DataFrame, adv_df: pd.DataFrame) -> pd.DataFrame:
    if basic_df.empty: return pd.DataFrame()
    if adv_df.empty: return basic_df
    adv_cols = ["SEASON_ID", "NET_RATING", "AST_TO", "TS_PCT", "USG_PCT"]
    adv_cols = [c for c in adv_cols if c in adv_df.columns]
    merged = basic_df.merge(adv_df[adv_cols], on="SEASON_ID", how="left")
    return merged.sort_values("SEASON_ID")


# -----------------------------
# Visualization Helpers
# -----------------------------
def create_hexbin_heatmap(shot_df: pd.DataFrame, hex_size: float = 25.0) -> list:
    """Generate hexbin data for heatmap (Smaller hex size for better detail)."""
    if shot_df.empty or "LOC_X" not in shot_df.columns or "LOC_Y" not in shot_df.columns:
        return []
    valid = shot_df.dropna(subset=["LOC_X", "LOC_Y"])
    if valid.empty: return []
    
    x, y = valid["LOC_X"].values, valid["LOC_Y"].values
    hex_bins = []
    
    # NBA Court width is 500 units (-250 to 250), height 470
    h_spacing = hex_size * np.sqrt(3)
    v_spacing = hex_size * 1.5
    
    x_centers = np.arange(-250, 250 + h_spacing, h_spacing)
    y_centers = np.arange(-47.5, 422.5 + v_spacing, v_spacing)
    
    for i, yc in enumerate(y_centers):
        for j, xc in enumerate(x_centers):
            xo = (h_spacing / 2) if (i % 2 == 1) else 0
            xc_actual = xc + xo
            
            dists = np.sqrt((x - xc_actual)**2 + (y - yc)**2)
            count = np.sum(dists <= hex_size)
            
            if count > 0:
                vx, vy = [], []
                for k in range(6):
                    angle = k * np.pi / 3
                    vx.append(xc_actual + hex_size * np.cos(angle))
                    vy.append(yc + hex_size * np.sin(angle))
                vx.append(vx[0])
                vy.append(vy[0])
                hex_bins.append({"x": vx, "y": vy, "count": int(count)})
    return hex_bins


# =========================================================
#  VERTEX-BASED ZONE MAP (STRICT USER SPEC)
# =========================================================

# =========================================================
#  SMART INTEGRATION ZONE MAP (NBA STANDARD GEOMETRY)
# =========================================================

def get_court_zones() -> list:
    """
    Returns a list of 14 distinct dictionary polygons representing the NBA court zones.
    Geometry: 
    - 3-Point Arc Radius: 23.75 ft (237.5 units)
    - Corner 3 Width: 22 ft (220 units)
    - Key Width: 16 ft (160 units)
    - Key Height: 19 ft (190 units) - but typically visualised to the circle top (142.5)
    """
    zones = []
    
    # --- Dimensions (x10 scale) ---
    R_HOOP = 40.0
    R_3PT = 237.5
    R_FAR = 500.0
    X_CORNER = 220.0
    X_PAINT = 80.0
    Y_PAINT_TOP = 142.5
    
    # --- 1. Shared Boundaries Calculation ---
    # Intersection of 3PT line and Corner (Break Point)
    y_break = np.sqrt(R_3PT**2 - X_CORNER**2) # ~89.47
    
    # Angles for the 'Fan' cuts (Radians)
    # NEW STANDARD ANGLES: 72 deg and 108 deg for Center Zone
    # This creates a 36-degree wide Center channel (72-108), matching standard analytics.
    theta_break_r = np.arctan2(y_break, X_CORNER)
    theta_cut_r = np.radians(72)
    theta_cut_l = np.radians(108)
    theta_break_l = np.pi - theta_break_r
    
    # Helper to generate smooth arcs
    def get_arc(r, theta1, theta2, steps=30):
        t = np.linspace(theta1, theta2, steps)
        return r * np.cos(t), r * np.sin(t)

    # Generate the 3PT Arc Segments (Used for both Inner and Outer zones)
    # Right Side Arc (Break -> 72deg)
    arc_3pt_r_x, arc_3pt_r_y = get_arc(R_3PT, theta_break_r, theta_cut_r)
    # Center Arc (72deg -> 108deg)
    arc_3pt_c_x, arc_3pt_c_y = get_arc(R_3PT, theta_cut_r, theta_cut_l)
    # Left Side Arc (108deg -> Break)
    arc_3pt_l_x, arc_3pt_l_y = get_arc(R_3PT, theta_cut_l, theta_break_l)

    # =============================================
    # ZONE 1: Restricted Area & Paint
    # =============================================
    # Restricted Area (Circle)
    ra_x, ra_y = get_arc(R_HOOP, 0, np.pi)
    zones.append({
        "name": "Restricted Area", "key": "Restricted Area_Center(C)",
        "x": np.concatenate(([40, 40, -40, -40], ra_x[::-1])),
        "y": np.concatenate(([-47.5, 0, 0, -47.5], ra_y[::-1]))
    })
    
    # Paint (Non-RA) - Box minus RA
    zones.append({
        "name": "Paint", "key": "In The Paint (Non-RA)_Center(C)",
        "x": np.concatenate(([80, 80, -80, -80, -40], ra_x, [40])),
        "y": np.concatenate(([-47.5, Y_PAINT_TOP, Y_PAINT_TOP, -47.5, -47.5], ra_y, [-47.5]))
    })

    # =============================================
    # ZONE 2: Mid-Range (Inside 3PT)
    # =============================================
    # MR Right Side (Rectangular-ish strip)
    zones.append({
        "name": "MR Right", "key": "Mid-Range_Right Side(R)",
        "x": [80, X_CORNER, X_CORNER, 80, 80], # Close loop
        "y": [-47.5, -47.5, y_break, y_break, -47.5]
    })
    
    # MR Right Center (Wedge 22-72 deg)
    zones.append({
        "name": "MR RC", "key": "Mid-Range_Right Side Center(RC)",
        "x": np.concatenate(([80], arc_3pt_r_x[::-1], [X_CORNER, 80, 80])),
        "y": np.concatenate(([Y_PAINT_TOP], arc_3pt_r_y[::-1], [y_break, y_break, Y_PAINT_TOP]))
    })
    
    # MR Center (Wedge 72-108 deg)
    zones.append({
        "name": "MR Center", "key": "Mid-Range_Center(C)",
        "x": np.concatenate(([80], arc_3pt_c_x[::-1], [-80, 80])),
        "y": np.concatenate(([Y_PAINT_TOP], arc_3pt_c_y[::-1], [Y_PAINT_TOP, Y_PAINT_TOP]))
    })
    
    # MR Left Center (Wedge 108-158 deg)
    zones.append({
        "name": "MR LC", "key": "Mid-Range_Left Side Center(LC)",
        "x": np.concatenate(([-80, -X_CORNER], arc_3pt_l_x[::-1], [-80, -80])),
        "y": np.concatenate(([y_break, y_break], arc_3pt_l_y[::-1], [Y_PAINT_TOP, y_break]))
    })
    
    # MR Left Side
    zones.append({
        "name": "MR Left", "key": "Mid-Range_Left Side(L)",
        "x": [-80, -X_CORNER, -X_CORNER, -80, -80],
        "y": [y_break, y_break, -47.5, -47.5, y_break]
    })

    # =============================================
    # ZONE 3: 3-Point Zones (Outside)
    # =============================================
    # Corners
    zones.append({"name": "Right Corner 3", "key": "Right Corner 3_Right Side(R)", 
                  "x": [X_CORNER, 250, 250, X_CORNER, X_CORNER], "y": [-47.5, -47.5, y_break, y_break, -47.5]})
    zones.append({"name": "Left Corner 3", "key": "Left Corner 3_Left Side(L)", 
                  "x": [-X_CORNER, -250, -250, -X_CORNER, -X_CORNER], "y": [-47.5, -47.5, y_break, y_break, -47.5]})

    # Above Break 3 - Right Center
    # Inner Boundary is the Exact 3PT Arc (Right Segment)
    # Outer Boundary is ARC at R_FAR
    far_r_x, far_r_y = get_arc(R_FAR, theta_break_r, theta_cut_r, steps=10)
    zones.append({
        "name": "AB3 RC", "key": "Above the Break 3_Right Side Center(RC)",
        "x": np.concatenate((arc_3pt_r_x, far_r_x[::-1], [arc_3pt_r_x[0]])),
        "y": np.concatenate((arc_3pt_r_y, far_r_y[::-1], [arc_3pt_r_y[0]]))
    })
    
    # Above Break 3 - Center
    far_c_x, far_c_y = get_arc(R_FAR, theta_cut_r, theta_cut_l, steps=10)
    zones.append({
        "name": "AB3 Center", "key": "Above the Break 3_Center(C)",
        "x": np.concatenate((arc_3pt_c_x, far_c_x[::-1], [arc_3pt_c_x[0]])),
        "y": np.concatenate((arc_3pt_c_y, far_c_y[::-1], [arc_3pt_c_y[0]]))
    })
    
    # Above Break 3 - Left Center
    far_l_x, far_l_y = get_arc(R_FAR, theta_cut_l, theta_break_l, steps=10)
    zones.append({
        "name": "AB3 LC", "key": "Above the Break 3_Left Side Center(LC)",
        "x": np.concatenate((arc_3pt_l_x, far_l_x[::-1], [arc_3pt_l_x[0]])),
        "y": np.concatenate((arc_3pt_l_y, far_l_y[::-1], [arc_3pt_l_y[0]]))
    })
    
    # AB3 Strips (The area above the corners)
    zones.append({"name": "AB3 Right Strip", "key": "Above the Break 3_Right Side(R)", 
                  "x": [X_CORNER, 250, 250, X_CORNER, X_CORNER], "y": [y_break, y_break, 422.5, 422.5, y_break]})
    zones.append({"name": "AB3 Left Strip", "key": "Above the Break 3_Left Side(L)", 
                  "x": [-X_CORNER, -250, -250, -X_CORNER, -X_CORNER], "y": [y_break, y_break, 422.5, 422.5, y_break]})

    # Backcourt (Optional, but usually not included in 14-zone maps)
    
    return zones


def create_clean_shot_chart(shot_df: pd.DataFrame, season: str) -> go.Figure:
    """Mode A: Clean Scatter Chart (Made=Green Circle, Missed=Red X)."""
    fig = draw_nba_court()
    
    clean_df = shot_df.dropna(subset=["LOC_X", "LOC_Y"])
    if clean_df.empty:
        fig.update_layout(title=f"{season} - No Data")
        return fig
    
    made = clean_df[clean_df["EVENT_TYPE"] == "Made Shot"]
    missed = clean_df[clean_df["EVENT_TYPE"] == "Missed Shot"]
    
    # Made
    fig.add_trace(go.Scatter(
        x=made["LOC_X"], y=made["LOC_Y"], 
        mode="markers",
        marker=dict(color="#2ca02c", size=6, opacity=0.7, line=dict(width=0)),
        name="Made", showlegend=True
    ))
    
    # Missed
    fig.add_trace(go.Scatter(
        x=missed["LOC_X"], y=missed["LOC_Y"], 
        mode="markers",
        marker=dict(color="#d62728", size=6, opacity=0.7, symbol="x"),
        name="Missed", showlegend=True
    ))
    
    fig.update_layout(
        title=dict(text=f"{season} Shot Chart", x=0.5, xanchor='center'),
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        height=650,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def create_zone_efficiency_map(shot_df: pd.DataFrame, season: str) -> go.Figure:
    """Mode B: 14-Zone Efficiency Map (Polygons colored by FG%)."""
    fig = draw_nba_court()
    
    if shot_df.empty:
        fig.update_layout(title=f"{season} - No Data")
        return fig

    # 1. Aggregate
    if "SHOT_ZONE_BASIC" in shot_df.columns and "SHOT_ZONE_AREA" in shot_df.columns:
        shot_df["ZONE_GROUP"] = shot_df["SHOT_ZONE_BASIC"] + "_" + shot_df["SHOT_ZONE_AREA"]
    else:
        # Fallback if columns missing? Should assume they exist per fetching.
        pass
        
    stats = shot_df.groupby("ZONE_GROUP").agg(
        FGM=("SHOT_MADE_FLAG", "sum"),
        FGA=("SHOT_ATTEMPTED_FLAG", "count")
    ).reset_index()
    stats["PCT"] = (stats["FGM"] / stats["FGA"]).fillna(0)
    
    zones = get_court_zones()
    
    # 2. Draw Zones
    # Use 'toself' fill which is gap-free if coordinates match perfectly.
    # Add a small stroke to force overlap if needed, but strict math should hold.
    
    for z in zones:
        row = stats[stats["ZONE_GROUP"] == z["key"]]
        
        val_text = ""
        pct_text = ""
        hover_text = z["name"]
        
        if not row.empty:
            fgm = int(row.iloc[0]["FGM"])
            fga = int(row.iloc[0]["FGA"])
            pct = row.iloc[0]["PCT"]
            
            # Colors
            if pct < 0.35: fill_color = "#4575b4" # Blue
            elif pct < 0.45: fill_color = "#ffffbf" # Yellow
            else: fill_color = "#d73027"            # Red
            
            # Opacity
            fill_color = f"rgba{tuple(int(fill_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.85,)}"
                
            val_text = f"<b>{fgm}/{fga}</b>"
            pct_text = f"<b>{pct:.1%}</b>"
            hover_text += f"<br>{fgm}/{fga} ({pct:.1%})"
        else:
            fill_color = "rgba(220, 220, 220, 0.4)" # Gray
            # Don't draw text for empty
        
        # Plot Polygon
        fig.add_trace(go.Scatter(
            x=z["x"], y=z["y"],
            fill="toself", mode="lines",
            line=dict(color=fill_color, width=1), # Match stroke to fill to hide anti-alias gaps
            fillcolor=fill_color,
            hoverinfo="text",
            text=hover_text,
            showlegend=False
        ))
        
        # Centroid Text
        if val_text:
            cx, cy = np.mean(z["x"]), np.mean(z["y"])
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy], mode="text", 
                text=[f"{val_text}<br>{pct_text}"],
                textfont=dict(family="Arial Black", size=10, color="black"),
                showlegend=False, hoverinfo="skip"
            ))

    fig.update_layout(
        title=dict(text=f"{season} Zone Efficiency", x=0.5, xanchor='center'),
        height=650,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def draw_nba_court(fig=None):
    if fig is None: fig = go.Figure()
    shapes = []
    # Outer
    shapes.append(dict(type="rect", x0=-250, y0=-47.5, x1=250, y1=422.5, line=dict(color="black", width=2)))
    # Paint
    shapes.append(dict(type="rect", x0=-80, y0=-47.5, x1=80, y1=142.5, line=dict(color="black", width=2)))
    shapes.append(dict(type="rect", x0=-60, y0=-47.5, x1=60, y1=142.5, line=dict(color="black", width=2)))
    # Hoop
    shapes.append(dict(type="circle", x0=-7.5, y0=-7.5, x1=7.5, y1=7.5, line=dict(color="#ec7607", width=2)))
    shapes.append(dict(type="line", x0=-30, y0=-40, x1=30, y1=-40, line=dict(color="black", width=2)))
    shapes.append(dict(type="line", x0=0, y0=-40, x1=0, y1=-7.5, line=dict(color="#ec7607", width=2)))
    # 3PT
    shapes.append(dict(type="line", x0=-220, y0=-47.5, x1=-220, y1=92.5, line=dict(color="black", width=2)))
    shapes.append(dict(type="line", x0=220, y0=-47.5, x1=220, y1=92.5, line=dict(color="black", width=2)))
    # Arcs
    arc_x = []
    arc_y = []
    for i in range(500):
        x = -220 + (i/499)*440
        if abs(x) <= 237.5:
            y = np.sqrt(237.5**2 - x**2)
            if y > 92.5:
                arc_x.append(x)
                arc_y.append(y)
    fig.add_trace(go.Scatter(x=arc_x, y=arc_y, mode="lines", line=dict(color="black", width=2), showlegend=False, hoverinfo="skip"))
    
    # Center Circle (Half)
    cc_x = [60 * np.cos(t) for t in np.linspace(0, np.pi, 50)]
    cc_y = [422.5 + 60 * np.sin(t) for t in np.linspace(0, np.pi, 50)]
    fig.add_trace(go.Scatter(x=cc_x, y=cc_y, mode="lines", line=dict(color="black", width=2), showlegend=False, hoverinfo="skip"))

    # STRICT LAYOUT FOR IDENTICAL SIZING
    fig.update_layout(
        shapes=shapes,
        xaxis=dict(range=[-250, 250], showgrid=False, zeroline=False, visible=False, fixedrange=True),
        yaxis=dict(range=[-47.5, 422.5], scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False, visible=False, fixedrange=True),
        plot_bgcolor="#f8f8f8",
        margin=dict(l=0,r=0,t=25,b=0),
        width=650, height=600, # Fixed dimensions
        autosize=False
    )
    return fig


    if not made.empty:
        fig.add_trace(go.Scatter(x=made["LOC_X"], y=made["LOC_Y"], mode="markers", 
                                marker=dict(color=COLOR_POSITIVE, size=5, line=dict(width=1, color="white")),
                                name="Made", opacity=0.85))
    if not missed.empty:
        fig.add_trace(go.Scatter(x=missed["LOC_X"], y=missed["LOC_Y"], mode="markers", 
                                marker=dict(color=COLOR_NEGATIVE, size=5, symbol="x"), 
                                name="Missed", opacity=0.7))
        
    fig.update_layout(title=dict(text=f"{season}", x=0.5, xanchor='center'))
    return fig


# -----------------------------
# Restored Deep Dive Charts
# -----------------------------
def plot_allstar_thresh(deni_stats: dict, allstar_df: pd.DataFrame) -> go.Figure:
    """Restored Grouped Bar Chart: Deni vs All-Star Avg vs Entry Level"""
    if allstar_df.empty: return go.Figure()
    
    # Averages
    avg_pts = (allstar_df["PTS"] * allstar_df["GP"]).sum() / allstar_df["GP"].sum()
    avg_reb = (allstar_df["REB"] * allstar_df["GP"]).sum() / allstar_df["GP"].sum()
    avg_ast = (allstar_df["AST"] * allstar_df["GP"]).sum() / allstar_df["GP"].sum()
    
    # Bottom 4
    bottom = allstar_df.nsmallest(4, "PTS").sort_values("PTS")
    
    fig = go.Figure()
    cats = ["PTS", "REB", "AST"]
    
    # Deni
    d_vals = [deni_stats.get("PTS",0), deni_stats.get("REB",0), deni_stats.get("AST",0)]
    fig.add_trace(go.Bar(name="Deni Avdija", x=cats, y=d_vals, marker_color=COLOR_DENI, text=[f"{v:.1f}" for v in d_vals], textposition="outside"))
    
    # Avg
    a_vals = [avg_pts, avg_reb, avg_ast]
    fig.add_trace(go.Bar(name="All-Star Avg", x=cats, y=a_vals, marker_color=COLOR_POSITIVE, text=[f"{v:.1f}" for v in a_vals], textposition="outside"))
    
    # Entry Level
    for _, row in bottom.iterrows():
        p_name = row["PLAYER_NAME"]
        vals = [row["PTS"], row["REB"], row["AST"]]
        fig.add_trace(go.Bar(name=f"{p_name} (Entry)", x=cats, y=vals, opacity=0.4, marker_color="gray"))
        
    fig.update_layout(title="The All-Star Threshold", barmode="group", yaxis_title="Per Game")
    return fig


def plot_triple_threat(allstar_df: pd.DataFrame, deni_stats: dict, is_2d: bool = True) -> go.Figure:
    """
    Restored Triple Threat Chart (2D Only - Faces).
    X=PTS, Y=AST, Size=REB (Face Size)
    """
    if allstar_df.empty: return go.Figure()
    
    # Helper for URL (Local scope to avoid global clutter, cached by Streamlit usually but here plain python)
    def get_face_url_local(name):
        try:
            hits = players.find_players_by_full_name(name)
            if hits:
                pid = hits[0]['id']
                return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png"
        except: pass
        return None

    fig = go.Figure()
    d_pts, d_reb, d_ast = deni_stats["PTS"], deni_stats["REB"], deni_stats["AST"]
    
    # We use a base scale factor for sizing images based on REB
    # Rebounds usually range 3 to 12.
    # Image size in plot coordinates. 
    # Let's say max REB is 13-15. We want max size ~2.0 units?
    # Scale = REB * 0.15 ? 10 rebs -> 1.5 units. 4 rebs -> 0.6 units.
    SIZE_FACTOR = 0.16
    
    # 1. Add Invisible Markers (for Hover + Auto-scale)
    # We add Deni to this list for scaling
    pts_list = allstar_df["PTS"].tolist() + [d_pts]
    ast_list = allstar_df["AST"].tolist() + [d_ast]
    reb_list = allstar_df["REB"].tolist() + [d_reb]
    names_list = allstar_df["PLAYER_NAME"].tolist() + ["Deni Avdija"]
    
    fig.add_trace(go.Scatter(
        x=pts_list, y=ast_list,
        mode="markers",
        name="Players",
        text=names_list,
        # Size for markers (just for hover area)
        marker=dict(size=[r*4 for r in reb_list], color="rgba(0,0,0,0)"),
        hovertemplate="<b>%{text}</b><br>PTS: %{x:.1f}<br>AST: %{y:.1f}<br>REB: %{customdata:.1f}<extra></extra>",
        customdata=reb_list
    ))
    
    # 2. Add Images
    images = []
    
    # All-Stars
    for _, row in allstar_df.iterrows():
        url = get_face_url_local(row["PLAYER_NAME"])
        if url:
            size_val = max(row["REB"] * SIZE_FACTOR, 0.5) # Min size
            images.append(dict(
                source=url,
                xref="x", yref="y",
                x=row["PTS"], y=row["AST"],
                sizex=size_val, sizey=size_val,
                xanchor="center", yanchor="middle",
                layer="above"
            ))
            
    # Deni
    d_url = get_face_url_local("Deni Avdija")
    if d_url:
        d_size = max(d_reb * SIZE_FACTOR, 0.5)
        images.append(dict(
            source=d_url,
            xref="x", yref="y",
            x=d_pts, y=d_ast,
            sizex=d_size, sizey=d_size,
            xanchor="center", yanchor="middle",
            layer="above"
        ))
        
    # Add Deni Text Label
    fig.add_trace(go.Scatter(
        x=[d_pts], y=[d_ast - (d_reb * SIZE_FACTOR * 0.6)], # Shift text below face
        mode="text",
        text=["Deni"],
        textposition="bottom center",
        textfont=dict(size=14, color="black", family="Arial Black")
    ))

    fig.update_layout(
        title="Triple Threat (2D): PTS vs AST (Face Size = REB)",
        xaxis_title="Points Per Game",
        yaxis_title="Assists Per Game",
        images=images,
        height=700,
        # Add some padding to ranges so faces don't get cut off
        xaxis=dict(range=[min(pts_list)-2, max(pts_list)+2]),
        yaxis=dict(range=[min(ast_list)-1, max(ast_list)+1]),
        showlegend=False
    )
    
    return fig


def analytical_verdict(deni_stats: dict, allstar_df: pd.DataFrame):
    """
    Displays the percentile ranking of Deni vs All-Stars with correct grammar.
    """
    if allstar_df.empty: return
    
    # 1. Helper for correct suffixes (1st, 2nd, 3rd, 4th...)
    def get_ordinal(n):
        if 11 <= (n % 100) <= 13: suffix = 'th'
        else: suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"

    # 2. Calculate Percentiles (Among All-Stars)
    pts_p = int((allstar_df["PTS"] < deni_stats["PTS"]).mean() * 100)
    reb_p = int((allstar_df["REB"] < deni_stats["REB"]).mean() * 100)
    ast_p = int((allstar_df["AST"] < deni_stats["AST"]).mean() * 100)
    
    st.markdown("### 🎯 The Analytical Verdict")
    
    # 3. Display Metrics with Fix
    c1, c2, c3 = st.columns(3)
    c1.metric("Scoring Percentile", get_ordinal(pts_p), help="Rank among All-Star roster")
    c2.metric("Rebounding Percentile", get_ordinal(reb_p), help="Rank among All-Star roster")
    c3.metric("Playmaking Percentile", get_ordinal(ast_p), help="Rank among All-Star roster")
    
    # 4. Context Explanation
    st.caption(f"""
    ℹ️ **Context:** These percentages compare Deni specifically against the **2024-25 All-Star Roster**.
    For example, being in the **{get_ordinal(pts_p)} percentile** means he outscores {pts_p}% of the NBA's elite.
    """)
    
    # 5. Summary Logic (Existing)
    avg_p = (pts_p + reb_p + ast_p) / 3
    if avg_p > 50:
        st.success(f"🏆 **All-Star Caliber**: Deni ranks in the top half of All-Stars ({get_ordinal(int(avg_p))} percentile avg).")
    elif avg_p > 30:
        st.warning(f"⚡ **Borderline**: Deni is competitive with lower-tier All-Stars ({get_ordinal(int(avg_p))} percentile avg).")
    else:
        st.info(f"📈 **Developing**: Deni shows flashes but trails the All-Star pack ({get_ordinal(int(avg_p))} percentile avg).")


def plot_what_if_analysis() -> go.Figure:
    """
    Creates a grouped bar chart comparing Deni's actual stats, 
    Deni's projected stats at Luka's usage, and Luka's actual stats.
    """
    # 1. Hardcoded Stats (Projections 25-26)
    # Deni Avdija
    deni_ppg, deni_rpg, deni_apg = 25.6, 7.2, 7.0
    deni_usage = 28.0

    # Luka Doncic
    luka_ppg, luka_rpg, luka_apg = 33.6, 8.1, 8.7
    luka_usage = 37.9

    # 2. Calculate Projections
    # Formula: Stat * (Luka_Usage / Deni_Usage)
    usage_ratio = luka_usage / deni_usage
    
    proj_ppg = deni_ppg * usage_ratio
    proj_rpg = deni_rpg * usage_ratio
    proj_apg = deni_apg * usage_ratio

    # 3. Prepare Data for Grouped Bar Chart
    categories = ['Points', 'Rebounds', 'Assists']
    
    # Trace 1: Deni Actual
    trace1 = go.Bar(
        name='Deni (Actual - 28% USG)', 
        x=categories, 
        y=[deni_ppg, deni_rpg, deni_apg],
        marker_color='rgb(160, 160, 160)', # Grey
        text=[deni_ppg, deni_rpg, deni_apg],
        textposition='auto'
    )

    # Trace 2: Deni Projected (Highlight)
    trace2 = go.Bar(
        name=f'Deni (Projected @ {luka_usage}% USG)', 
        x=categories, 
        y=[proj_ppg, proj_rpg, proj_apg],
        marker_color='rgb(0, 204, 150)', # Green
        text=[f"{p:.1f}" for p in [proj_ppg, proj_rpg, proj_apg]],
        textposition='auto'
    )

    # Trace 3: Luka Actual
    trace3 = go.Bar(
        name='Luka (Actual)', 
        x=categories, 
        y=[luka_ppg, luka_rpg, luka_apg],
        marker_color='rgb(55, 83, 109)', # Blue/Navy
        text=[luka_ppg, luka_rpg, luka_apg],
        textposition='auto'
    )

    # 4. Construct Figure
    fig = go.Figure(data=[trace1, trace2, trace3])

    fig.update_layout(
        title="<b>Usage-Adjusted Efficiency: Deni vs. Luka</b><br><i>What if Deni had Luka's Usage Rate?</i>",
        barmode='group',
        yaxis_title="Per Game Stats",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
        margin=dict(t=100) # Space for title/legend
    )
    
    return fig


def plot_versatility_radar(deni_stats: dict, allstar_df: pd.DataFrame) -> go.Figure:
    """Radar Chart: Deni vs All-Star Average (Normalized)."""
    if allstar_df.empty: return go.Figure()

    metrics = ["PTS", "REB", "AST", "STL", "BLK"]
    
    # 1. Prepare Data
    d_vals = [deni_stats.get(m, 0) for m in metrics]
    avg_stats = allstar_df[metrics].mean().tolist()
    
    # Max values for normalization
    max_vals = []
    for i, m in enumerate(metrics):
        max_val = max(allstar_df[m].max(), d_vals[i])
        max_vals.append(max_val if max_val > 0 else 1) 
        
    # Normalize
    d_norm = [d / m for d, m in zip(d_vals, max_vals)]
    a_norm = [a / m for a, m in zip(avg_stats, max_vals)]
    
    # Close the loop
    metrics += [metrics[0]]
    d_norm += [d_norm[0]]
    a_norm += [a_norm[0]]
    
    # 2. Plot
    fig = go.Figure()
    
    # All-Star Avg
    fig.add_trace(go.Scatterpolar(
        r=a_norm, theta=metrics,
        fill='toself', name='All-Star Avg',
        line=dict(color=COLOR_GRAY, width=2),
        fillcolor="rgba(99, 110, 250, 0.2)"
    ))
    
    # Deni
    fig.add_trace(go.Scatterpolar(
        r=d_norm, theta=metrics,
        fill='toself', name='Deni Avdija',
        line=dict(color=COLOR_DENI, width=3),
        fillcolor="rgba(0, 204, 150, 0.3)"
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1]) 
        ),
        showlegend=True,
        title="Type A: The Multidimensional Wing (Normalized)",
        height=500
    )
    return fig


def plot_offensive_engine(deni_stats: dict, allstar_df: pd.DataFrame) -> go.Figure:
    """Stacked Bar: Points Scored + Points Created."""
    if allstar_df.empty: return go.Figure()
    
    # 1. Prepare Data
    df = allstar_df.copy()
    
    # Add Deni if not in list
    if "Deni Avdija" not in df["PLAYER_NAME"].values:
        d_row = {"PLAYER_NAME": "Deni Avdija"}
        for k, v in deni_stats.items():
            if k in df.columns: d_row[k] = v
        df = pd.concat([df, pd.DataFrame([d_row])], ignore_index=True)
        
    # Calculate Engine Stats
    df["PTS_CREATED"] = df["AST"] * 2.3
    df["TOTAL_OUTPUT"] = df["PTS"] + df["PTS_CREATED"]
    
    # Sort
    df = df.sort_values("TOTAL_OUTPUT", ascending=False)
    
    # Top 15 + Deni check
    top_15 = df.head(15)
    if "Deni Avdija" not in top_15["PLAYER_NAME"].values:
        deni_row = df[df["PLAYER_NAME"] == "Deni Avdija"]
        plot_df = pd.concat([top_15, deni_row])
        plot_df = plot_df.sort_values("TOTAL_OUTPUT", ascending=False)
    else:
        plot_df = top_15
        
    # 2. Plot
    fig = go.Figure()
    
    names = plot_df["PLAYER_NAME"].tolist()
    
    # PTS Bar
    fig.add_trace(go.Bar(
        name="Points Scored", x=names, y=plot_df["PTS"],
        marker_color=[COLOR_DENI if x == "Deni Avdija" else "#7f7f7f" for x in names]
    ))
    
    # Created Bar
    fig.add_trace(go.Bar(
        name="Points Created (Est)", x=names, y=plot_df["PTS_CREATED"],
        marker_color=[COLOR_HIGHLIGHT if x == "Deni Avdija" else "#1f77b4" for x in names]
    ))
    
    fig.update_layout(
        barmode='stack',
        title="Type B: The Offensive Engine (Scoring + Playmaking)",
        xaxis_tickangle=-45,
        yaxis_title="Total Points Production",
        height=500
    )
    return fig


def render_scouting_report():
    """Renders the static scouting report text in a clean format."""
    with st.expander("📋 READ: Scouting Report & Analysis (2025)", expanded=False):
        st.markdown("""
        ### 🧐 Analysis: The Expanded Role
        **Early returns (11/26):** The most impressive aspect of Avdija’s star-making season has been his capacity for scaling up his production to fit his expanded role. 
        * **Pick-and-Roll Volume:** Avdija has already logged more possessions as a P&R initiator than in his full years 2 or 3.
        * **Elite Driving:** He is fully tapping into his physicality. No one in the league drives more often, and few pass out of drives more frequently.
        * **Free Throw Rate:** His downhill speed and "incessant drives" have him getting to the line at a rate on par with **Shai Gilgeous-Alexander**.
        
        ---
        ### ⚡ Defining Trait: The One-Man Fast Break
        Avdija has become a reliable one-man fast break. He is equally adept at finishing at full tilt or shifting gears (Euro-step or shoulder bumps) to dislodge defenders.
        
        > *"The most bullish sign of Avdija’s ascent might be his ability to draw contact. Avdija had one of the highest free throw attempt rates in the league... the only other non-bigs in his cohort were **Jimmy Butler** and **James Harden**."*
        
        ---
        ### 🧬 Modern NBA Archetype: The Multidimensional Wing
        Multidimensional wings are the lifeblood of the modern game. Avdija represents a high-reward venture that is paying off:
        * **Growth:** Incremental growth throughout his career, with breakthroughs in years 4 and 5.
        * **Foundation:** His vision and ballhandling got him noticed; his defense and rebounding instincts kept him on the floor long enough to see the fruits of his labor come to the fore.
        """)


# -----------------------------
# Main App
# -----------------------------
def main():
    st.sidebar.title("Deni Avdija Analytics")
    
    if st.sidebar.button("🔄 Refresh Data"):
        with st.spinner("Forcing update..."):
            fetch_data.smart_update(force_refresh=True)
        st.success("Data updated!")
        time.sleep(1)
        st.rerun()
    
    # Data Status Display
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Status")
    
    if Path(DATA_FILE).exists():
        # Show last update time
        fetched_at = None
        game_count = 0
        
        try:
            with open(DATA_FILE, "rb") as f:
                temp_data = pickle.load(f)
            fetched_at = temp_data.get("fetched_at")
            logs = temp_data.get("game_logs_2025_26", pd.DataFrame())
            game_count = len(logs) if not logs.empty else 0
        except:
            pass
        
        if fetched_at:
            try:
                update_time = datetime.fromisoformat(fetched_at)
                st.sidebar.caption(f"📅 Last updated: {update_time.strftime('%b %d, %H:%M')}")
            except:
                st.sidebar.caption("📅 Last updated: Recently")
        
        if game_count > 0:
            st.sidebar.caption(f"🏀 Games tracked: **{game_count}**")
        
        # Show next game info instead of countdown
        try:
            next_game = fetch_data.get_next_portland_game()
            if next_game:
                location = "vs" if next_game['is_home'] else "@"
                st.sidebar.caption(f"🗓️ Next game: {location} {next_game['opponent']} ({next_game['date_str']})")
            else:
                st.sidebar.caption("🗓️ No upcoming games in next 7 days")
        except:
            pass
    else:
        st.sidebar.warning("⚠️ No data file found")
    
    st.sidebar.markdown("---")

    page = st.sidebar.radio("Navigate", ["Dashboard", "Career Analysis", "League Trends", "Raw Data", "Shot Maps", "Research: Deep Dive", "About Me"])
    
    # Get mtime to force cache invalidation on update
    mtime = 0
    if Path(DATA_FILE).exists():
        mtime = Path(DATA_FILE).stat().st_mtime
    
    data = load_nba_data(mtime)
    if not data:
        st.error(f"Missing data file. Run `python fetch_data.py`.")
        st.stop()
        
    # Unpack
    career_basic = data.get("career_basic", pd.DataFrame())
    career_adv = data.get("career_advanced", pd.DataFrame())
    logs_26 = data.get("game_logs_2025_26", pd.DataFrame())
    logs_25 = data.get("game_logs_2024_25", pd.DataFrame())
    logs_23 = data.get("game_logs_2023_24", pd.DataFrame())
    shot_charts = data.get("shot_charts", {})
    allstar = data.get("allstar_stats", pd.DataFrame())
    allstar_detailed = data.get("allstar_detailed_stats", pd.DataFrame())
    
    # NEW: 25/26 RACE
    allstar_26 = data.get("allstar_stats_26", pd.DataFrame())
    allstar_detailed_26 = data.get("allstar_detailed_26", pd.DataFrame())
    
    league_ft = data.get("league_ft_stats", pd.DataFrame())
    
    # NEW: League Trends Data
    drives_df = data.get("drives_data", pd.DataFrame())
    misc_df = data.get("misc_stats", pd.DataFrame())
    passing_df = data.get("passing_data", pd.DataFrame())
    heliocentric_df = data.get("heliocentric_data", pd.DataFrame())
    
    # 1. Patch Career Stats 
    career_basic = patch_career_stats(career_basic, logs_26)
    career_df = merge_career_frames(career_basic, career_adv)
    
    # -----------------------------
    # PAGE: Dashboard
    # -----------------------------
    if page == "Dashboard":
        st.title("Performance Dashboard")
        st.info("🔥 **#1 in NBA Free Throw Attempts** (250+ FTA)")
        
        # 2025-26 Season (Full Width)
        st.subheader(f"25/26 Impact")
        if not logs_26.empty:
            df = logs_26.sort_values("GAME_DATE")
            avg_pts = df["PTS"].mean()
            avg_min = df["MIN"].mean()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df["GAME_DATE"], y=df["PTS"], marker_color=["#2ca02c" if w=="W" else "#d7191c" for w in df["WL"]], name="PTS"))
            fig.add_trace(go.Scatter(x=df["GAME_DATE"], y=df["MIN"], mode="lines", name="MIN", yaxis="y2", line=dict(color="gold", width=2)))
            
            # Averages
            fig.add_hline(y=avg_pts, line_dash="dash", line_color="gray", annotation_text=f"Avg PTS: {avg_pts:.1f}", annotation_position="top left")
            fig.add_trace(go.Scatter(x=df["GAME_DATE"], y=[avg_min]*len(df), mode="lines", name=f"Avg MIN ({avg_min:.1f})", yaxis="y2", line=dict(color="gold", width=1, dash="dot")))
            
            fig.update_layout(yaxis=dict(title="Points"), yaxis2=dict(title="Minutes", overlaying="y", side="right", range=[0, 48]), legend=dict(orientation="h", y=1.1), height=400)
            st.plotly_chart(fig, width="stretch")

        st.divider()

        # 2024-25 Season (Full Width)
        st.subheader("24/25 Impact")
        if not logs_25.empty:
            df = logs_25.sort_values("GAME_DATE")
            avg_pts = df["PTS"].mean()
            avg_min = df["MIN"].mean()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df["GAME_DATE"], y=df["PTS"], marker_color=["#2ca02c" if w=="W" else "#d7191c" for w in df["WL"]], name="PTS"))
            fig.add_trace(go.Scatter(x=df["GAME_DATE"], y=df["MIN"], mode="lines", name="MIN", yaxis="y2", line=dict(color="gold", width=2)))
            
            # Averages
            fig.add_hline(y=avg_pts, line_dash="dash", line_color="gray", annotation_text=f"Avg PTS: {avg_pts:.1f}", annotation_position="top left")
            fig.add_trace(go.Scatter(x=df["GAME_DATE"], y=[avg_min]*len(df), mode="lines", name=f"Avg MIN ({avg_min:.1f})", yaxis="y2", line=dict(color="gold", width=1, dash="dot")))

            fig.update_layout(yaxis=dict(title="Points"), yaxis2=dict(title="Minutes", overlaying="y", side="right", range=[0, 48]), legend=dict(orientation="h", y=1.1), height=400)
            st.plotly_chart(fig, width="stretch")

        st.divider()

        # 2023-24 Season (Full Width)
        st.subheader("23/24 Impact")
        if not logs_23.empty:
            df = logs_23.sort_values("GAME_DATE")
            avg_pts = df["PTS"].mean()
            avg_min = df["MIN"].mean()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df["GAME_DATE"], y=df["PTS"], marker_color=["#2ca02c" if w=="W" else "#d7191c" for w in df["WL"]], name="PTS"))
            fig.add_trace(go.Scatter(x=df["GAME_DATE"], y=df["MIN"], mode="lines", name="MIN", yaxis="y2", line=dict(color="gold", width=2)))
            
            # Averages
            fig.add_hline(y=avg_pts, line_dash="dash", line_color="gray", annotation_text=f"Avg PTS: {avg_pts:.1f}", annotation_position="top left")
            fig.add_trace(go.Scatter(x=df["GAME_DATE"], y=[avg_min]*len(df), mode="lines", name=f"Avg MIN ({avg_min:.1f})", yaxis="y2", line=dict(color="gold", width=1, dash="dot")))

            fig.update_layout(yaxis=dict(title="Points"), yaxis2=dict(title="Minutes", overlaying="y", side="right", range=[0, 48]), legend=dict(orientation="h", y=1.1), height=400)
            st.plotly_chart(fig, width="stretch")

    # -----------------------------
    # PAGE: Career Analysis
    # -----------------------------
    elif page == "Career Analysis":
        st.title("Career Trajectory Analysis")
        if not career_df.empty:
            # Per Game Stats
            st.subheader("Per Game Stats")
            fig = px.bar(career_df, x="SEASON_ID", y=["PTS", "REB", "AST"], barmode="group", title="PTS / REB / AST")
            fig.update_layout(
                xaxis=dict(showgrid=False, title="Season"),
                yaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=1, zerolinecolor='black', dtick=5),
                plot_bgcolor='white',
                margin=dict(t=40, l=40, r=40, b=40)
            )
            st.plotly_chart(fig, width="stretch")

            st.divider()

            # Per 36 Minutes
            st.subheader("Per 36 Minutes")
            df_36 = career_df.copy()
            for c in ["PTS", "REB", "AST", "STL", "TOV"]:
                if c in df_36.columns and "MIN" in df_36.columns:
                        df_36[f"{c}_36"] = df_36.apply(lambda r: (r[c]/r["MIN"]*36) if r["MIN"]>0 else 0, axis=1)
            fig = px.bar(df_36, x="SEASON_ID", y=[c+"_36" for c in ["PTS", "REB", "AST", "STL", "TOV"]], barmode="group", title="Per 36 Min")
            fig.update_layout(
                xaxis=dict(showgrid=False, title="Season"),
                yaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=1, zerolinecolor='black', dtick=5),
                plot_bgcolor='white',
                margin=dict(t=40, l=40, r=40, b=40)
            )
            st.plotly_chart(fig, width="stretch")
            
            st.divider()

            # Prepare plotting data with formatted Season
            plot_df = career_df.copy()
            if "SEASON_ID" in plot_df.columns:
                plot_df["Season"] = plot_df["SEASON_ID"].apply(
                    lambda x: f"{x[2:4]}/{x[5:]}" if isinstance(x, str) and len(x) >= 7 else x
                )
            else:
                plot_df["Season"] = plot_df.index
            
            # Usage Rate
            st.subheader("Usage Rate")
            st.caption("""
            **Definition:** Percentage of team plays used by the player while on floor.
            * **High (>30%):** Primary Scorers (Luka, Giannis) | **Low (<15%):** Role Players
            """)
            fig = px.line(plot_df, x="Season", y="USG_PCT", markers=True, title="Usage %", 
                          labels={"USG_PCT": "Usage Percentage", "Season": "Season"})
            fig.add_hline(y=0.20, line_dash="dash", annotation_text="League Avg (20%)")
            st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)

            st.divider()

            # True Shooting %
            st.subheader("True Shooting %")
            st.caption("""
            **Definition:** Shooting efficiency adjusting for 3-pointers (1.5x) and Free Throws.
            * **Elite (>60%):** Curry/Jokic | **Avg (~58%)** | **Poor (<52%)**
            """)
            fig = px.line(plot_df, x="Season", y="TS_PCT", markers=True, title="TS %",
                          labels={"TS_PCT": "True Shooting Percentage", "Season": "Season"})
            fig.add_hline(y=0.58, line_dash="dash", annotation_text="League Avg (58%)")
            st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)

    # -----------------------------
    # PAGE: League Trends
    # -----------------------------
    elif page == "League Trends":
        st.title("League Trends & Advanced Metrics")
        
        # 1. Heliocentric Graph
        st.subheader("1. Heliocentric Offenses: Team vs. Star Output")
        st.caption("How much of a team's total offense comes from their biggest star? (Points + Created Points)")
        
        if not heliocentric_df.empty:
            # Sort by Team PPG
            h_df = heliocentric_df.sort_values("TEAM_PPG", ascending=False)
            
            fig = go.Figure()
            
            # Layer 1: Team PPG
            fig.add_trace(go.Bar(
                x=h_df["TEAM_NAME"],
                y=h_df["TEAM_PPG"],
                name="Team Avg PPG",
                marker_color="#e0e0e0", # Light Grey
                textposition="none"
            ))
            
            # Layer 2: Star Output
            # We want this 'in front' or overlay. 
            # Ideally standard group bar with 'barmode=overlay' so they sit on top of each other?
            # Or just bar chart with custom width?
            
            # Let's use 'overlay' mode
            fig.add_trace(go.Bar(
                x=h_df["TEAM_NAME"],
                y=h_df["STAR_OUTPUT"],
                name="Star Player Total Output",
                marker_color="#00CC96", # Teal
                text=h_df["STAR_NAME"],
                hovertemplate="<b>%{text}</b><br>Output: %{y:.1f}<extra></extra>"
            ))
            
            fig.update_layout(
                barmode='overlay',
                xaxis_tickangle=-45,
                title="Offensive Dependence (Sorted by Team Offense)",
                yaxis_title="Points Production",
                height=500,
                legend=dict(orientation="h", y=1.05)
            )
            st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
        else:
            st.info("No data available yet. Please run update.")

        st.divider()
        
        # 2. Top 10 Tables
        st.subheader("2. Specialized Leaderboards")
        
        c1, c2, c3 = st.columns(3)
        
        # Table A: Sniper Finders (AST_3P)
        with c1:
            st.markdown("#### 🎯 Sniper Finders")
            st.caption("Best at creating 3-point looks.")
            if not passing_df.empty:
                # Filter Top 10
                top_snipers = passing_df.sort_values("AST_3P", ascending=False).head(10).reset_index(drop=True)
                top_snipers.index += 1
                
                st.dataframe(
                    top_snipers[["PLAYER_NAME", "TEAM_ABBREVIATION", "AST_3P", "AST"]],
                    width="stretch",
                    column_config={
                        "PLAYER_NAME": "Player",
                        "TEAM_ABBREVIATION": "Team",
                        "AST_3P": st.column_config.NumberColumn("Assists to 3P", format="%.1f"),
                        "AST": st.column_config.NumberColumn("Total AST", format="%d")
                    }
                )
            else: st.write("No Data")

        # Table B: Foul Magnets (PFD)
        with c2:
            st.markdown("#### 💪 Foul Magnets")
            st.caption("Most fouls drawn (PFD).")
            
            if not misc_df.empty and "PFD" in misc_df.columns:
                top_pfd = misc_df.sort_values("PFD", ascending=False).head(10).reset_index(drop=True)
                top_pfd.index += 1
                
                st.dataframe(
                    top_pfd,
                    width="stretch",
                    column_config={
                        "PLAYER_NAME": "Player",
                        "TEAM_ABBREVIATION": "Team",
                        "PFD": st.column_config.NumberColumn("Fouls Drawn", format="%d")
                    }
                )
            else:
                st.warning("PFD Data unavailable.")

        # Table C: Rim Pressure (Drives)
        with c3:
            st.markdown("#### 🚂 Rim Pressure")
            st.caption("Most drives per game.")
            if not drives_df.empty:
                top_drives = drives_df.sort_values("DRIVES", ascending=False).head(10).reset_index(drop=True)
                top_drives.index += 1
                
                st.dataframe(
                    top_drives,
                    width="stretch",
                    column_config={
                        "PLAYER_NAME": "Player",
                        "TEAM_ABBREVIATION": "Team",
                        "DRIVES": st.column_config.NumberColumn("Drives/G", format="%.1f"),
                        "DRIVE_PTS": st.column_config.NumberColumn("Drive PTS", format="%.1f"),
                    }
                )
            else: st.write("No Data")

    # -----------------------------
    # PAGE: Raw Data
    # -----------------------------
    elif page == "Raw Data":
        st.title("Raw Data & Custom Trends")
        
        st.subheader("Interactive Trend Viewer")
        all_cols = career_df.columns.tolist()
        numeric_cols = [c for c in all_cols if career_df[c].dtype in ['float64', 'int64']]
        defaults = ["PTS", "REB"] if "PTS" in numeric_cols else []
        sel_metrics = st.multiselect("Select Metrics", numeric_cols, default=defaults)
        if sel_metrics:
            fig = px.line(career_df, x="SEASON_ID", y=sel_metrics, markers=True, title="Custom Trends")
            st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
          # 2. Table
        st.divider()
        st.subheader("Career Data Table")
        
        # Format tweaks
        df_display = career_df.copy()
        
        # 1. Remove Technical Columns
        drop_cols = ["PLAYER_ID", "LEAGUE_ID", "TEAM_ID"]
        df_display = df_display.drop(columns=[c for c in drop_cols if c in df_display.columns])
        
        # 2. Rename Columns
        # Format SEASON_ID to xx/xx (e.g., 2024-25 -> 24/25)
        if "SEASON_ID" in df_display.columns:
            df_display["SEASON_ID"] = df_display["SEASON_ID"].apply(
                lambda x: f"{x[2:4]}/{x[5:]}" if isinstance(x, str) and len(x) >= 7 else x
            )
            
        rename_map = {"SEASON_ID": "Season", "TEAM_ABBREVIATION": "TEAM", "PLAYER_AGE": "AGE"}
        df_display = df_display.rename(columns=rename_map)
        
        # Valid columns to cast to int
        int_cols = ["GP", "GS", "age"] # 'age' sometimes in career stats
        for c in int_cols:
            if c in df_display.columns:
                df_display[c] = pd.to_numeric(df_display[c], errors='coerce').fillna(0).astype(int)

        # Configure formatting for Averages columns to 1 decimal
        avg_cols = ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "PF", "USG_PCT", "TS_PCT"]
        col_config = {}
        for c in avg_cols:
            if c in df_display.columns:
                # Pct columns handle differently?
                if "PCT" in c:
                    col_config[c] = st.column_config.NumberColumn(format="%.1%")
                else:
                    col_config[c] = st.column_config.NumberColumn(format="%.1f")
            
        st.dataframe(df_display, width="stretch", hide_index=True, column_config=col_config)

    # -----------------------------
    # PAGE: Shot Maps 
    # -----------------------------
    elif page == "Shot Maps":
        st.title("Shot Analysis")
        
        c_ctrl, c_view = st.columns([1, 4])
        with c_ctrl:
            compare = st.checkbox("Compare Mode", value=False)
            s_a = st.selectbox("Season A", ["2022-23", "2023-24", "2024-25", "2025-26"], index=3)
            s_b = st.selectbox("Season B", ["2022-23", "2023-24", "2024-25", "2025-26"], index=2) if compare else None
            
            view_type = st.radio("Map Style", ["Shot Chart", "14-Zone Efficiency"], index=0, horizontal=True)
            
        with c_view:
            df_a = shot_charts.get(s_a, pd.DataFrame())
            
            # STRICT SIZING: Use cols to force layout
            # If Compare: 2 cols, equal width. If Single: Just display one.
            if not compare:
                if view_type == "14-Zone Efficiency":
                    fig = create_zone_efficiency_map(df_a, s_a)
                    st.plotly_chart(fig, use_container_width=False, config=PLOT_CONFIG)
                else:
                    fig = create_clean_shot_chart(df_a, s_a)
                    st.plotly_chart(fig, use_container_width=False, config=PLOT_CONFIG)
                
                # Download HTML Button
                import io
                buffer = io.StringIO()
                fig.write_html(buffer, include_plotlyjs="cdn")
                html_bytes = buffer.getvalue().encode()
                st.download_button(
                    label="💾 Download Interactive HTML",
                    data=html_bytes,
                    file_name=f"deni_avdija_{s_a}_{view_type.replace(' ', '_')}.html",
                    mime="text/html"
                )
            else:
                df_b = shot_charts.get(s_b, pd.DataFrame())
                c1, c2 = st.columns(2)
                with c1:
                    if view_type == "14-Zone Efficiency":
                        st.plotly_chart(create_zone_efficiency_map(df_a, s_a), use_container_width=False, config=PLOT_CONFIG)
                    else:
                        st.plotly_chart(create_clean_shot_chart(df_a, s_a), use_container_width=False, config=PLOT_CONFIG)
                with c2:
                    if view_type == "14-Zone Efficiency":
                        st.plotly_chart(create_zone_efficiency_map(df_b, s_b), use_container_width=False, config=PLOT_CONFIG)
                    else:
                        st.plotly_chart(create_clean_shot_chart(df_b, s_b), use_container_width=False, config=PLOT_CONFIG)

    # -----------------------------
    # PAGE: Deep Dive (RESTORED GRAPHS)
    # -----------------------------
    elif page == "Research: Deep Dive":
        st.title("All-Star Comparison")
        
        render_scouting_report()
        
        # --- NEW SECTION: Free Throw Leaders ---
        if not league_ft.empty:
            st.divider()
            st.subheader("🔥 2025-26 Season: Free Throw Leaders (Top 10)")

            # 0. Base Data: Top 10 by FTM (The "Leaders")
            # We filter first, then let user RE-SORT this specific group.
            ft_df = league_ft.sort_values("FTM", ascending=False).head(10).reset_index(drop=True)

            # 1. Dynamic Sort Widget
            sort_metric = st.selectbox(
                "Sort Leaderboard By:",
                ["Total Attempts", "Total Made", "FT%"],
                index=0,  # Default to Attempts (User preference or FTM?) User snippet said index=0 (Attempts).
                key="ft_sort_box"
            )

            # 2. Map Selection to Column Names
            col_map = {
                "Total Attempts": "FTA",
                "Total Made": "FTM",
                "FT%": "FT_PCT"
            }
            target_col = col_map[sort_metric]

            # 3. Sort and Re-Rank
            # We sort the Top 10 subset by the chosen metric
            df_display = ft_df.sort_values(target_col, ascending=False).reset_index(drop=True)
            df_display.index += 1  # Start rank at 1
            df_display.index.name = "Rank"

            # 4. Highlight Logic
            def highlight_deni(row):
                if row.get("PLAYER_NAME") == "Deni Avdija":
                    return ['background-color: #00CC96; color: black'] * len(row)
                return [''] * len(row)
            
            # 5. Render
            st.dataframe(
                df_display.style.apply(highlight_deni, axis=1),
                width="stretch",
                column_config={
                    "FT_PCT": st.column_config.NumberColumn("FT%", format="%.1%"),
                    "FTM": st.column_config.NumberColumn("Total Made", format="%d"),
                    "FTA": st.column_config.NumberColumn("Total Attempts", format="%d"),
                    "GP": st.column_config.NumberColumn("Games", format="%d"),
                },
                hide_index=False,
                key=f"ft_leaderboard_{sort_metric}"
            )
        # ---------------------------------------
        

        
        
        # TABS FOR COMPARISON
        tab_bench, tab_race = st.tabs(["Benchmark (24/25 All-Stars)", "The Race (25/26 All-Star Stats)"])
        
        # --- TAB 1: BENCHMARK (Frozen 24/25) ---
        with tab_bench:
            st.caption("Comparing Deni's **current** stats against the **final** stats of 24/25 All-Stars.")
            
            if not allstar.empty and not logs_26.empty:
                # Prepare data
                deni_stats = {
                    "PTS": logs_26["PTS"].mean(), "REB": logs_26["REB"].mean(), "AST": logs_26["AST"].mean(),
                    "STL": logs_26["STL"].mean(), "BLK": logs_26["BLK"].mean(), "TOV": logs_26["TOV"].mean(),
                }
                if not career_df.empty:
                    cur = career_df[career_df["SEASON_ID"] == "2025-26"]
                    if not cur.empty:
                        deni_stats["USG_PCT"] = cur.iloc[0]["USG_PCT"]
                        deni_stats["TS_PCT"] = cur.iloc[0]["TS_PCT"]

                # 1. VERDICT & THRESHOLD
                st.subheader("1. The All-Star Threshold")
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.plotly_chart(plot_allstar_thresh(deni_stats, allstar), width="stretch", config=PLOT_CONFIG)
                with c2:
                    analytical_verdict(deni_stats, allstar)

                # 2. TRIPLE THREAT
                st.divider()
                st.subheader("2. The Triple Threat")
                show_2d = st.toggle("Switch to 2D Bubble View", value=False, key="toggle_2d_bench")
                st.plotly_chart(plot_triple_threat(allstar, deni_stats, show_2d), width="stretch", config=PLOT_CONFIG)

                # 3. SEPARATION CHART
                if not allstar_detailed.empty:
                    st.divider()
                    st.subheader("3. Separation Chart (Usage vs Efficiency)")
                    
                    @st.cache_data
                    def get_face_url(name):
                        try:
                            hits = players.find_players_by_full_name(name)
                            if hits:
                                pid = hits[0]['id']
                                return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png"
                        except: pass
                        return None

                    fig = go.Figure()

                    # Add Deni Text Label Only (No Marker)
                    if "USG_PCT" in deni_stats:
                        dx, dy = deni_stats["USG_PCT"]*100, deni_stats["TS_PCT"]*100
                        fig.add_trace(go.Scatter(
                            x=[dx], y=[dy], mode="text", 
                            name="Deni", text=["Deni"], textposition="top center",
                            textfont=dict(size=14, color="black", family="Arial Black")
                        ))
                        
                        # Add Deni Face
                        d_url = get_face_url("Deni Avdija")
                        if d_url:
                            fig.add_layout_image(dict(
                                source=d_url, xref="x", yref="y", x=dx, y=dy,
                                sizex=1.5, sizey=1.5, xanchor="center", yanchor="middle", layer="above"
                            ))

                    # Add All-Stars Images
                    for _, row in allstar_detailed.iterrows():
                        if row["PLAYER_NAME"] == "Deni Avdija": continue # Skip Deni
                        
                        url = get_face_url(row["PLAYER_NAME"])
                        if url:
                            fig.add_layout_image(dict(
                                source=url, xref="x", yref="y",
                                x=row["USG_PCT"]*100, y=row["TS_PCT"]*100,
                                sizex=1.5, sizey=1.5, xanchor="center", yanchor="middle", layer="above"
                            ))
                            
                    # Invisible markers for hover
                    fig.add_trace(go.Scatter(
                        x=allstar_detailed[allstar_detailed["PLAYER_NAME"] != "Deni Avdija"]["USG_PCT"]*100, 
                        y=allstar_detailed[allstar_detailed["PLAYER_NAME"] != "Deni Avdija"]["TS_PCT"]*100, 
                        mode="markers", name="All-Stars", 
                        text=allstar_detailed[allstar_detailed["PLAYER_NAME"] != "Deni Avdija"]["PLAYER_NAME"],
                        marker=dict(color="rgba(0,0,0,0)", size=30), hoverinfo="text+x+y"
                    ))

                    fig.update_layout(
                        xaxis_title="Usage %", yaxis_title="True Shooting %",
                        width=800, height=800,
                        xaxis=dict(range=[18, 40]), yaxis=dict(range=[48, 70])
                    )
                    st.plotly_chart(fig, config=PLOT_CONFIG)

                # 4. TABLE
                st.divider()
                st.subheader("4. Full League Comparison Table")
                rank_metric = st.selectbox("🏆 Rank Players By:", ["PTS", "REB", "AST", "STL", "BLK", "TOV"], index=0, key="rank_bench")
                
                t_df = allstar[["PLAYER_NAME", "PTS", "REB", "AST", "STL", "BLK", "TOV", "GP"]].copy()
                
                # Remove Deni if he exists in the fetched data (to avoid duplication with our manual row)
                t_df = t_df[t_df["PLAYER_NAME"] != "Deni Avdija"]
                
                d_row = {"PLAYER_NAME": "Deni Avdija", "GP": len(logs_26)}
                for k in ["PTS", "REB", "AST", "STL", "BLK", "TOV"]: d_row[k] = deni_stats[k]
                
                t_df = pd.concat([t_df, pd.DataFrame([d_row])], ignore_index=True)
                t_df = t_df.sort_values(rank_metric, ascending=False).reset_index(drop=True)
                t_df.insert(0, "Rank", range(1, len(t_df) + 1))
                
                cfg = {c: st.column_config.NumberColumn(format="%.1f") for c in ["PTS", "REB", "AST", "STL", "BLK", "TOV"]}
                def hl_deni(x):
                    return ['background-color: #00CC96; color: black' if x["PLAYER_NAME"] == "Deni Avdija" else '' for _ in x]
                st.dataframe(t_df.style.apply(hl_deni, axis=1), width="stretch", hide_index=True, column_config=cfg)

                # 5. Advanced Case Studies
                st.divider()
                st.subheader("5. Advanced Case Studies")
                c_adv1, c_adv2 = st.columns(2)
                with c_adv1:
                    st.plotly_chart(plot_versatility_radar(deni_stats, allstar), width="stretch", config=PLOT_CONFIG)
                with c_adv2:
                    st.plotly_chart(plot_offensive_engine(deni_stats, allstar), width="stretch", config=PLOT_CONFIG)

        # --- TAB 2: THE RACE (25/26 Live) ---
        with tab_race:
            st.caption("Comparing Deni's **current** stats against the **current (25/26)** stats of the same All-Star cohort.")
            
            if allstar_26.empty:
                st.warning("⚠️ No 2025-26 All-Star data found. Please click 'Refresh Data' in the sidebar to fetch the latest comparison stats.")
            else:
                # Use same Deni stats
                deni_stats_race = deni_stats.copy()

                # 1. VERDICT & THRESHOLD - RACE
                st.subheader("1. The Race Threshold")
                c1r, c2r = st.columns([2, 1])
                with c1r:
                    st.plotly_chart(plot_allstar_thresh(deni_stats_race, allstar_26), width="stretch", config=PLOT_CONFIG)
                with c2r:
                    analytical_verdict(deni_stats_race, allstar_26)

                # 2. TRIPLE THREAT - RACE
                st.divider()
                st.subheader("2. The Triple Threat (25/26)")
                show_2d_race = st.toggle("Switch to 2D Bubble View", value=False, key="toggle_2d_race")
                st.plotly_chart(plot_triple_threat(allstar_26, deni_stats_race, show_2d_race), width="stretch", config=PLOT_CONFIG)

                # 3. SEPARATION CHART - RACE
                if not allstar_detailed_26.empty:
                    st.divider()
                    st.subheader("3. Separation Chart (25/26 Performance)")
                    
                    fig_race = go.Figure()

                    # Add Deni Text Label Only (No Marker)
                    if "USG_PCT" in deni_stats_race:
                        dx, dy = deni_stats_race["USG_PCT"]*100, deni_stats_race["TS_PCT"]*100
                        fig_race.add_trace(go.Scatter(
                            x=[dx], y=[dy], mode="text", 
                            name="Deni", text=["Deni"], textposition="top center",
                            textfont=dict(size=14, color="black", family="Arial Black")
                        ))
                        # Add Deni Face
                        d_url = get_face_url("Deni Avdija")
                        if d_url:
                            fig_race.add_layout_image(dict(
                                source=d_url, xref="x", yref="y", x=dx, y=dy,
                                sizex=1.5, sizey=1.5, xanchor="center", yanchor="middle", layer="above"
                            ))

                    # Add All-Stars Images
                    for _, row in allstar_detailed_26.iterrows():
                        if row["PLAYER_NAME"] == "Deni Avdija": continue # Skip Deni (Added manually above)
                        
                        url = get_face_url(row["PLAYER_NAME"])
                        if url:
                            fig_race.add_layout_image(dict(
                                source=url, xref="x", yref="y",
                                x=row["USG_PCT"]*100, y=row["TS_PCT"]*100,
                                sizex=1.5, sizey=1.5, xanchor="center", yanchor="middle", layer="above"
                            ))
                            
                    # Invisible markers for hover
                    fig_race.add_trace(go.Scatter(
                        x=allstar_detailed_26[allstar_detailed_26["PLAYER_NAME"] != "Deni Avdija"]["USG_PCT"]*100, 
                        y=allstar_detailed_26[allstar_detailed_26["PLAYER_NAME"] != "Deni Avdija"]["TS_PCT"]*100, 
                        mode="markers", name="All-Stars", 
                        text=allstar_detailed_26[allstar_detailed_26["PLAYER_NAME"] != "Deni Avdija"]["PLAYER_NAME"],
                        marker=dict(color="rgba(0,0,0,0)", size=30), hoverinfo="text+x+y"
                    ))

                    fig_race.update_layout(
                        xaxis_title="Usage %", yaxis_title="True Shooting %",
                        width=800, height=800,
                        xaxis=dict(range=[18, 40]), yaxis=dict(range=[48, 70])
                    )
                    st.plotly_chart(fig_race, config=PLOT_CONFIG)

                # 4. TABLE - RACE
                st.divider()
                st.subheader("4. 25/26 Leaderboard")
                rank_metric_race = st.selectbox("🏆 Rank Players By:", ["PTS", "REB", "AST", "STL", "BLK", "TOV"], index=0, key="rank_race")
                
                t_df_race = allstar_26[["PLAYER_NAME", "PTS", "REB", "AST", "STL", "BLK", "TOV", "GP"]].copy()
                
                # Remove Deni if he exists in the fetched data (to avoid duplication with our manual row)
                t_df_race = t_df_race[t_df_race["PLAYER_NAME"] != "Deni Avdija"]
                
                d_row = {"PLAYER_NAME": "Deni Avdija", "GP": len(logs_26)}
                for k in ["PTS", "REB", "AST", "STL", "BLK", "TOV"]: d_row[k] = deni_stats_race[k]
                
                t_df_race = pd.concat([t_df_race, pd.DataFrame([d_row])], ignore_index=True)
                t_df_race = t_df_race.sort_values(rank_metric_race, ascending=False).reset_index(drop=True)
                t_df_race.insert(0, "Rank", range(1, len(t_df_race) + 1))
                
                st.dataframe(t_df_race.style.apply(hl_deni, axis=1), width="stretch", hide_index=True, column_config=cfg)

                # 5. Advanced Case Studies - RACE
                st.divider()
                st.subheader("5. Advanced Case Studies (25/26)")
                c_adv1r, c_adv2r = st.columns(2)
                with c_adv1r:
                    st.plotly_chart(plot_versatility_radar(deni_stats_race, allstar_26), width="stretch", config=PLOT_CONFIG)
                with c_adv2r:
                    st.plotly_chart(plot_offensive_engine(deni_stats_race, allstar_26), width="stretch", config=PLOT_CONFIG)

                # 6. What-If Analysis - RACE
                st.divider()
                st.subheader("6. What-If: Deni vs Luka Efficiency")
                st.caption("How would Deni compare if he had Luka Dončić's usage rate (37.9%)?")
                st.plotly_chart(plot_what_if_analysis(), width="stretch", config=PLOT_CONFIG)

    elif page == "About Me":
        st.title("About the Creator")
        
        c1, c2 = st.columns([1, 2.5])
        
        with c1:
            st.markdown("# 👨‍💻") # Simple avatar emoji
            st.markdown("### Ram Shiri")
            st.markdown("**Data Engineering Student**")
            st.link_button("Connect on LinkedIn", "https://www.linkedin.com/in/ram-shiri-1a1056304/?originalSubdomain=il")
        
        with c2:
            st.subheader("👋 Hello!")
            st.write("""
            I'm a **3rd year B.Sc. Data Engineering student** specializing in data science with a passion for building smart, practical solutions. 
            I love combining creativity with technical skills to drive real-world impact—especially in the world of sports analytics.
            """)
            
            st.subheader("🛠️ Skills & Approach")
            st.write("""
            - **Tech Stack:** Python, Java, SQL, Pandas, Streamlit, Plotly
            - **Soft Skills:** Creative thinking, fast learning, hands-on problem solving
            - **Philosophy:** Comfortable working with AI tools to accelerate development (like this dashboard!) while maintaining deep understanding of the core logic.
            """)
            
            st.subheader("❤️ Passions")
            st.write("🏀 Basketball | ⚽ Football | 🏎️ F1 Racing | 🧱 LEGO")
            
            st.divider()
            st.info("🚀 **Open to Work:** Actively seeking a student or full-time position in software or data engineering to grow, contribute, and thrive in a dynamic environment.")

if __name__ == "__main__":
    main()
