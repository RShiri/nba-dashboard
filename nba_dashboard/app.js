/* Israeli NBA Players Analytics — static dashboard (Broadcast Kinetic theme) */

const COLOR = {
  accent: "#d7ff3a",
  accentInk: "#0c0d10",
  negative: "#ff2a4d",
  info: "#9fd0ff",
  muted: "#737b88",
  mutedLine: "rgba(255,255,255,0.18)",
  text: "#f5f7fa",
};

const PAGES = ["dashboard", "career", "league", "raw", "shots", "deepdive", "about"];

const state = {
  meta: null,
  shared: null,
  playerKey: null,
  playerData: null,
  page: "dashboard",
  shotSeason: null,
  shotSeasonB: null,
  shotCompare: false,
  shotView: "chart",
  deepTab: "bench",
};

// -----------------------------
// Utilities
// -----------------------------
function seasonLabel(s) { return s ? `${s.slice(2, 4)}/${s.slice(5)}` : ""; }
function fmt1(x) { return (x === null || x === undefined || isNaN(x)) ? "—" : Number(x).toFixed(1); }
function fmtPct1(x) { return (x === null || x === undefined || isNaN(x)) ? "—" : (Number(x) * 100).toFixed(1) + "%"; }
function fmtInt(x) { return (x === null || x === undefined || isNaN(x)) ? "—" : Math.round(x).toString(); }
function mean(arr, key) {
  const v = arr.map(r => r[key]).filter(x => x !== null && x !== undefined && !isNaN(x));
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0;
}
function sum(arr, key) {
  return arr.reduce((a, r) => a + (Number(r[key]) || 0), 0);
}
function ordinal(n) {
  n = Math.round(n);
  const s = ["th", "st", "nd", "rd"], v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}
function firstName(name) { return name.split(" ")[0]; }
function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}
function headshotUrl(playerId) { return `https://cdn.nba.com/headshots/nba/latest/1040x760/${playerId}.png`; }
function allstarHeadshot(name) { return (state.shared.headshots || {})[name]; }

// -----------------------------
// Plotly theme
// -----------------------------
function baseLayout(overrides) {
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Barlow, sans-serif", color: COLOR.text, size: 13 },
    colorway: [COLOR.accent, COLOR.info, COLOR.negative, COLOR.muted, "#ffb020"],
    margin: { t: 50, l: 50, r: 30, b: 50 },
    xaxis: { gridcolor: COLOR.mutedLine, zerolinecolor: COLOR.mutedLine, linecolor: COLOR.mutedLine, color: "#b9bfc9" },
    yaxis: { gridcolor: COLOR.mutedLine, zerolinecolor: COLOR.mutedLine, linecolor: COLOR.mutedLine, color: "#b9bfc9" },
    legend: { font: { color: "#b9bfc9" } },
  }, overrides || {});
}
const PLOTLY_CONFIG = { displayModeBar: true, responsive: true };

function plot(containerId, traces, layout, config) {
  Plotly.newPlot(containerId, traces, baseLayout(layout), Object.assign({}, PLOTLY_CONFIG, config || {}));
}

// -----------------------------
// Data loading
// -----------------------------
async function loadJSON(path) {
  const res = await fetch(path, { cache: "no-cache" });
  if (!res.ok) throw new Error(`Failed to load ${path}`);
  return res.json();
}

async function loadPlayer(key) {
  state.playerData = await loadJSON(`data/${key}.json`);
  state.playerKey = key;
}

async function boot() {
  [state.meta, state.shared, state.zones] = await Promise.all([
    loadJSON("data/meta.json"),
    loadJSON("data/shared.json"),
    loadJSON("data/zones.json"),
  ]);

  const select = document.getElementById("player-select");
  select.innerHTML = state.meta.players.map(p => `<option value="${p.key}">${p.name}</option>`).join("");
  select.addEventListener("change", async (e) => {
    await loadPlayer(e.target.value);
    state.shotSeason = null;
    renderSidebarPlayer();
    renderPage();
  });

  document.getElementById("nav-list").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-page]");
    if (!btn) return;
    state.page = btn.dataset.page;
    renderNav();
    renderPage();
  });

  await loadPlayer(state.meta.players[0].key);
  renderSidebarPlayer();
  renderNav();
  renderPage();
}

function renderNav() {
  document.querySelectorAll("#nav-list button").forEach(b => b.classList.toggle("active", b.dataset.page === state.page));
}

function renderSidebarPlayer() {
  const p = state.playerData;
  document.getElementById("topbar-title").innerHTML = `${p.player_name} <span>Analytics</span>`;
  document.getElementById("player-card").innerHTML = `
    <img src="${p.headshot}" alt="${p.player_name}" onerror="this.style.visibility='hidden'" />
    <div class="pc-name">${p.player_name}</div>
    <div class="pc-role">${p.team_full}</div>
  `;

  const seasonKey = state.meta.current_season;
  const logs = (p.game_logs && p.game_logs[seasonKey]) || [];
  const fetched = state.meta.fetched_at ? new Date(state.meta.fetched_at).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—";
  document.getElementById("status-block").innerHTML = `
    <div>&#128197; Last updated: <b>${fetched}</b></div>
    <div>&#127936; ${p.player_name} games tracked: <b>${logs.length}</b></div>
  `;
}

function renderPage() {
  renderNav();
  const main = document.getElementById("page-content");
  main.innerHTML = "";
  const renderers = {
    dashboard: renderDashboard, career: renderCareer, league: renderLeague,
    raw: renderRaw, shots: renderShots, deepdive: renderDeepDive, about: renderAbout,
  };
  renderers[state.page]();
}

// -----------------------------
// Shared bits
// -----------------------------
function heroHtml() {
  const p = state.playerData;
  return `
    <div class="hero">
      <img src="${p.headshot}" alt="${p.player_name}" onerror="this.style.visibility='hidden'" />
      <div>
        <div class="kicker">${p.team_full} &middot; ${p.position}</div>
        <h1 class="name">${p.player_name}</h1>
        <div class="sub">360&deg; Performance Analytics &middot; ${state.meta.current_season} Season</div>
      </div>
    </div>`;
}

function kpiCard(label, value, delta) {
  let deltaHtml = "";
  if (delta !== undefined && delta !== null) {
    const cls = delta.startsWith("-") ? "down" : "up";
    deltaHtml = `<div class="kpi-delta ${cls}">${delta}</div>`;
  }
  return `<div class="kpi-card"><div class="kpi-label">${label}</div><div class="kpi-value">${value}</div>${deltaHtml}</div>`;
}

function seasonAggregates(logs) {
  if (!logs || !logs.length) return null;
  const fgm = sum(logs, "FGM"), fga = sum(logs, "FGA");
  const fg3m = sum(logs, "FG3M"), fg3a = sum(logs, "FG3A");
  const ftm = sum(logs, "FTM"), fta = sum(logs, "FTA");
  return {
    pts: mean(logs, "PTS"), reb: mean(logs, "REB"), ast: mean(logs, "AST"), min: mean(logs, "MIN"),
    fg: fga ? fgm / fga : 0, fg3: fg3a ? fg3m / fg3a : 0, ft: fta ? ftm / fta : 0, gp: logs.length,
  };
}

function renderKpiStrip(container) {
  const p = state.playerData;
  const cur = (p.game_logs || {})[state.meta.current_season] || [];
  const prev = (p.game_logs || {})[state.meta.prev_season] || [];
  const c = seasonAggregates(cur);
  if (!c) return;
  const pr = seasonAggregates(prev);
  const d = (cv, pv, isPct) => {
    if (!pr) return null;
    const diff = cv - pv;
    const scaled = isPct ? diff * 100 : diff;
    const sign = scaled >= 0 ? "+" : "";
    return `${sign}${scaled.toFixed(1)}${isPct ? " pp" : ""}`;
  };
  container.appendChild(el(`<div class="kpi-grid">
    ${kpiCard("Points / G", fmt1(c.pts), d(c.pts, pr && pr.pts))}
    ${kpiCard("Rebounds / G", fmt1(c.reb), d(c.reb, pr && pr.reb))}
    ${kpiCard("Assists / G", fmt1(c.ast), d(c.ast, pr && pr.ast))}
    ${kpiCard("Minutes / G", fmt1(c.min), d(c.min, pr && pr.min))}
  </div>
  <div class="kpi-grid">
    ${kpiCard("Field Goal %", fmtPct1(c.fg), d(c.fg, pr && pr.fg, true))}
    ${kpiCard("3-Point %", fmtPct1(c.fg3), d(c.fg3, pr && pr.fg3, true))}
    ${kpiCard("Free Throw %", fmtPct1(c.ft), d(c.ft, pr && pr.ft, true))}
    ${kpiCard("Games Played", c.gp, pr ? `${c.gp - pr.gp >= 0 ? "+" : ""}${c.gp - pr.gp}` : null)}
  </div>`));
}

// -----------------------------
// Dashboard
// -----------------------------
function renderDashboard() {
  const main = document.getElementById("page-content");
  const p = state.playerData;
  main.appendChild(el(heroHtml()));

  const anyLogs = p.game_logs && Object.keys(p.game_logs).length > 0;
  if (!anyLogs) {
    main.appendChild(el(`<div class="banner info">&#127936; ${p.player_name} hasn't logged an NBA game yet — check back once the season tips off.</div>`));
  }
  if (state.playerKey === "deni_avdija") {
    main.appendChild(el(`<div class="banner">&#128293; <b>#1 in NBA Free Throw Attempts</b> (250+ FTA)</div>`));
  }

  renderKpiStrip(main);
  main.appendChild(el("<hr/>"));

  const seasons = [state.meta.current_season, state.meta.prev_season, state.meta.prev2_season];
  seasons.forEach((season, i) => {
    const logs = ((p.game_logs || {})[season] || []).slice().sort((a, b) => a.GAME_DATE.localeCompare(b.GAME_DATE));
    if (!logs.length) return;
    const block = el(`<div class="chart-block"><h3>${seasonLabel(season)} Impact</h3><div class="plot" id="dash-plot-${i}" style="height:400px"></div></div>`);
    main.appendChild(block);
    const avgPts = mean(logs, "PTS"), avgMin = mean(logs, "MIN");
    const dates = logs.map(r => r.GAME_DATE);
    plot(`dash-plot-${i}`, [
      { type: "bar", x: dates, y: logs.map(r => r.PTS), name: "PTS", marker: { color: logs.map(r => r.WL === "W" ? COLOR.accent : COLOR.negative) } },
      { type: "scatter", mode: "lines", x: dates, y: logs.map(r => r.MIN), name: "MIN", yaxis: "y2", line: { color: "gold", width: 2 } },
      { type: "scatter", mode: "lines", x: dates, y: dates.map(() => avgMin), name: `Avg MIN (${avgMin.toFixed(1)})`, yaxis: "y2", line: { color: "gold", width: 1, dash: "dot" } },
    ], {
      yaxis: { title: "Points", gridcolor: COLOR.mutedLine },
      yaxis2: { title: "Minutes", overlaying: "y", side: "right", range: [0, 48] },
      legend: { orientation: "h", y: 1.15 },
      shapes: [{ type: "line", x0: 0, x1: 1, xref: "paper", y0: avgPts, y1: avgPts, yref: "y", line: { color: "gray", dash: "dash" } }],
      annotations: [{ x: 0.02, xref: "paper", y: avgPts, yref: "y", text: `Avg PTS: ${avgPts.toFixed(1)}`, showarrow: false, font: { color: COLOR.muted } }],
    });
  });
}

// -----------------------------
// Career Analysis
// -----------------------------
function renderCareer() {
  const main = document.getElementById("page-content");
  const p = state.playerData;
  main.appendChild(el(`<h1>${p.player_name} &mdash; Career Trajectory Analysis</h1>`));

  const career = p.career || [];
  if (!career.length) {
    main.appendChild(el(`<p style="color:var(--muted)">Not enough career data yet to chart a trajectory.</p>`));
    return;
  }

  main.appendChild(el(`<div class="chart-block"><h3>Per Game Stats</h3><div class="plot" id="career-pergame" style="height:420px"></div></div>`));
  const seasons = career.map(r => r.SEASON_ID);
  plot("career-pergame", ["PTS", "REB", "AST"].map((k, i) => ({
    type: "bar", name: k, x: seasons, y: career.map(r => r[k]),
  })), { barmode: "group", yaxis: { gridcolor: COLOR.mutedLine, dtick: 5 } });

  main.appendChild(el(`<div class="chart-block"><h3>Per 36 Minutes</h3><div class="plot" id="career-per36" style="height:420px"></div></div>`));
  plot("career-per36", ["PTS", "REB", "AST", "STL", "TOV"].map(k => ({
    type: "bar", name: `${k}_36`,
    x: seasons,
    y: career.map(r => (r.MIN > 0 ? (r[k] / r.MIN) * 36 : 0)),
  })), { barmode: "group", yaxis: { gridcolor: COLOR.mutedLine, dtick: 5 } });

  const hasUsg = career.some(r => r.USG_PCT !== null && r.USG_PCT !== undefined);
  if (hasUsg) {
    main.appendChild(el(`<div class="chart-block"><h3>Usage Rate</h3>
      <p style="color:var(--muted);font-size:13px">Percentage of team plays used by the player while on floor. High (&gt;30%): primary scorers | Low (&lt;15%): role players</p>
      <div class="plot" id="career-usg" style="height:380px"></div></div>`));
    plot("career-usg", [{ type: "scatter", mode: "lines+markers", x: seasons, y: career.map(r => r.USG_PCT), line: { color: COLOR.accent } }], {
      yaxis: { title: "Usage %", tickformat: ".0%", gridcolor: COLOR.mutedLine },
      shapes: [{ type: "line", x0: 0, x1: 1, xref: "paper", y0: 0.20, y1: 0.20, line: { dash: "dash", color: COLOR.muted } }],
      annotations: [{ x: 0.98, xref: "paper", y: 0.20, text: "League Avg (20%)", showarrow: false, font: { color: COLOR.muted } }],
    });
  }

  const hasTs = career.some(r => r.TS_PCT !== null && r.TS_PCT !== undefined);
  if (hasTs) {
    main.appendChild(el(`<div class="chart-block"><h3>True Shooting %</h3>
      <p style="color:var(--muted);font-size:13px">Shooting efficiency adjusting for 3-pointers (1.5x) and free throws. Elite (&gt;60%) | Avg (~58%) | Poor (&lt;52%)</p>
      <div class="plot" id="career-ts" style="height:380px"></div></div>`));
    plot("career-ts", [{ type: "scatter", mode: "lines+markers", x: seasons, y: career.map(r => r.TS_PCT), line: { color: COLOR.info } }], {
      yaxis: { title: "TS %", tickformat: ".0%", gridcolor: COLOR.mutedLine },
      shapes: [{ type: "line", x0: 0, x1: 1, xref: "paper", y0: 0.58, y1: 0.58, line: { dash: "dash", color: COLOR.muted } }],
      annotations: [{ x: 0.98, xref: "paper", y: 0.58, text: "League Avg (58%)", showarrow: false, font: { color: COLOR.muted } }],
    });
  }
}

// -----------------------------
// Raw Data
// -----------------------------
function renderRaw() {
  const main = document.getElementById("page-content");
  const p = state.playerData;
  main.appendChild(el(`<h1>${p.player_name} &mdash; Raw Data &amp; Custom Trends</h1>`));

  const career = p.career || [];
  if (!career.length) {
    main.appendChild(el(`<p style="color:var(--muted)">No career data yet.</p>`));
    return;
  }

  const numericCols = Object.keys(career[0]).filter(k => typeof career[0][k] === "number" && k !== "PLAYER_ID" && k !== "LEAGUE_ID" && k !== "TEAM_ID");
  const defaults = numericCols.includes("PTS") ? ["PTS", "REB"] : [];

  main.appendChild(el(`<h3>Interactive Trend Viewer</h3>`));
  const controls = el(`<div class="controls-row"><div class="control">
      <label class="field-label">Select Metrics</label>
      <select id="raw-metric-select" multiple size="6" style="min-width:220px"></select>
    </div></div>`);
  main.appendChild(controls);
  const sel = controls.querySelector("#raw-metric-select");
  sel.innerHTML = numericCols.map(c => `<option value="${c}" ${defaults.includes(c) ? "selected" : ""}>${c}</option>`).join("");

  main.appendChild(el(`<div class="chart-block" id="raw-chart-wrap"><div class="plot" id="raw-trend" style="height:420px"></div></div>`));

  const seasons = career.map(r => r.SEASON_ID);
  const drawTrend = () => {
    const chosen = Array.from(sel.selectedOptions).map(o => o.value);
    plot("raw-trend", chosen.map(c => ({ type: "scatter", mode: "lines+markers", name: c, x: seasons, y: career.map(r => r[c]) })), {});
  };
  sel.addEventListener("change", drawTrend);
  drawTrend();

  main.appendChild(el("<hr/>"));
  main.appendChild(el(`<h3>Career Data Table</h3>`));
  const dropCols = new Set(["PLAYER_ID", "LEAGUE_ID", "TEAM_ID"]);
  const cols = Object.keys(career[0]).filter(c => !dropCols.has(c));
  const rename = { SEASON_ID: "Season", TEAM_ABBREVIATION: "TEAM", PLAYER_AGE: "AGE" };
  const pctCols = new Set(["FG_PCT", "FG3_PCT", "FT_PCT", "USG_PCT", "TS_PCT"]);
  const wrap = el(`<div class="table-wrap"><table class="data-table"><thead><tr>${cols.map(c => `<th>${rename[c] || c}</th>`).join("")}</tr></thead><tbody>
    ${career.map(row => `<tr>${cols.map(c => {
      let v = row[c];
      if (c === "SEASON_ID") v = seasonLabel(v).replace("/", "/");
      else if (typeof v === "number") v = pctCols.has(c) ? fmtPct1(v) : (Number.isInteger(v) ? v : v.toFixed(1));
      return `<td>${v === null || v === undefined ? "—" : v}</td>`;
    }).join("")}</tr>`).join("")}
  </tbody></table></div>`);
  main.appendChild(wrap);
}

// -----------------------------
// League Trends (shared, not player-specific)
// -----------------------------
function renderLeague() {
  const main = document.getElementById("page-content");
  main.appendChild(el(`<h1>League Trends &amp; Advanced Metrics</h1>`));

  const lt = state.shared.league_trends;
  main.appendChild(el(`<h3>1. Heliocentric Offenses: Team vs. Star Output</h3><p style="color:var(--muted);font-size:13px">How much of a team's total offense comes from their biggest star? (Points + Created Points)</p>`));
  if (lt.heliocentric && lt.heliocentric.length) {
    main.appendChild(el(`<div class="chart-block"><div class="plot" id="helio-chart" style="height:500px"></div></div>`));
    const h = lt.heliocentric;
    plot("helio-chart", [
      { type: "bar", x: h.map(r => r.TEAM_NAME), y: h.map(r => r.TEAM_PPG), name: "Team Avg PPG", marker: { color: "#5a5f68" } },
      {
        type: "bar", x: h.map(r => r.TEAM_NAME), y: h.map(r => r.STAR_OUTPUT), name: "Star Player Total Output", marker: { color: COLOR.accent },
        text: h.map(r => r.STAR_NAME), hovertemplate: "<b>%{text}</b><br>Output: %{y:.1f}<extra></extra>",
      },
    ], { barmode: "overlay", xaxis: { tickangle: -45 }, yaxis: { title: "Points Production" }, legend: { orientation: "h", y: 1.08 } });
  } else {
    main.appendChild(el(`<div class="banner info">No data available yet. Please run update.</div>`));
  }

  main.appendChild(el("<hr/>"));
  main.appendChild(el(`<h3>2. Specialized Leaderboards</h3>`));
  const cols = el(`<div class="three-col"></div>`);
  main.appendChild(cols);

  function tableCol(title, caption, rows, columns) {
    if (!rows || !rows.length) return `<div><h4>${title}</h4><p style="color:var(--muted)">No Data</p></div>`;
    return `<div><h4>${title}</h4><p style="color:var(--muted);font-size:12px">${caption}</p>
      <div class="table-wrap"><table class="data-table"><thead><tr>${columns.map(c => `<th>${c[1]}</th>`).join("")}</tr></thead><tbody>
      ${rows.map(r => `<tr>${columns.map(c => `<td>${typeof r[c[0]] === "number" ? (c[2] ? c[2](r[c[0]]) : fmt1(r[c[0]])) : (r[c[0]] ?? "—")}</td>`).join("")}</tr>`).join("")}
      </tbody></table></div></div>`;
  }

  cols.innerHTML =
    tableCol("&#127919; Sniper Finders", "Best at creating 3-point looks.", lt.passing, [["PLAYER_NAME", "Player"], ["TEAM_ABBREVIATION", "Team"], ["AST_3P", "Assists to 3P"], ["AST", "Total AST", v => fmtInt(v)]]) +
    tableCol("&#128170; Foul Magnets", "Most fouls drawn (PFD).", lt.misc, [["PLAYER_NAME", "Player"], ["TEAM_ABBREVIATION", "Team"], ["PFD", "Fouls Drawn", v => fmtInt(v)]]) +
    tableCol("&#128646; Rim Pressure", "Most drives per game.", lt.drives, [["PLAYER_NAME", "Player"], ["TEAM_ABBREVIATION", "Team"], ["DRIVES", "Drives/G"], ["DRIVE_PTS", "Drive PTS"]]);
}

// -----------------------------
// Shot Maps
// -----------------------------
function courtShapes() {
  const c = COLOR.muted;
  return [
    { type: "rect", x0: -250, y0: -47.5, x1: 250, y1: 422.5, line: { color: c, width: 2 } },
    { type: "rect", x0: -80, y0: -47.5, x1: 80, y1: 142.5, line: { color: c, width: 2 } },
    { type: "rect", x0: -60, y0: -47.5, x1: 60, y1: 142.5, line: { color: c, width: 2 } },
    { type: "circle", x0: -7.5, y0: -7.5, x1: 7.5, y1: 7.5, line: { color: COLOR.accent, width: 2 } },
    { type: "line", x0: -30, y0: -40, x1: 30, y1: -40, line: { color: c, width: 2 } },
    { type: "line", x0: 0, y0: -40, x1: 0, y1: -7.5, line: { color: COLOR.accent, width: 2 } },
    { type: "line", x0: -220, y0: -47.5, x1: -220, y1: 92.5, line: { color: c, width: 2 } },
    { type: "line", x0: 220, y0: -47.5, x1: 220, y1: 92.5, line: { color: c, width: 2 } },
  ];
}
function courtArcTraces() {
  const c = COLOR.muted;
  const arcX = [], arcY = [];
  for (let i = 0; i < 500; i++) {
    const x = -220 + (i / 499) * 440;
    if (Math.abs(x) <= 237.5) {
      const y = Math.sqrt(237.5 ** 2 - x ** 2);
      if (y > 92.5) { arcX.push(x); arcY.push(y); }
    }
  }
  const ccX = [], ccY = [];
  for (let i = 0; i <= 49; i++) {
    const t = Math.PI * (i / 49);
    ccX.push(60 * Math.cos(t)); ccY.push(422.5 + 60 * Math.sin(t));
  }
  return [
    { type: "scatter", mode: "lines", x: arcX, y: arcY, line: { color: c, width: 2 }, showlegend: false, hoverinfo: "skip" },
    { type: "scatter", mode: "lines", x: ccX, y: ccY, line: { color: c, width: 2 }, showlegend: false, hoverinfo: "skip" },
  ];
}
function courtLayout(extra) {
  return Object.assign({
    xaxis: { range: [-250, 250], showgrid: false, zeroline: false, visible: false, fixedrange: true },
    yaxis: { range: [-47.5, 422.5], scaleanchor: "x", scaleratio: 1, showgrid: false, zeroline: false, visible: false, fixedrange: true },
    shapes: courtShapes(),
    margin: { l: 0, r: 0, t: 40, b: 0 },
    height: 620,
    plot_bgcolor: "#090a0d",
  }, extra || {});
}

function drawShotChart(divId, shotData, season) {
  const traces = courtArcTraces();
  if (shotData && shotData.points.length) {
    const made = shotData.points.filter(p => p.made);
    const missed = shotData.points.filter(p => !p.made);
    traces.push(
      { type: "scatter", mode: "markers", x: made.map(p => p.x), y: made.map(p => p.y), name: "Made", marker: { color: COLOR.accent, size: 6, opacity: 0.85 } },
      { type: "scatter", mode: "markers", x: missed.map(p => p.x), y: missed.map(p => p.y), name: "Missed", marker: { color: COLOR.negative, size: 6, opacity: 0.75, symbol: "x" } },
    );
  }
  const title = (shotData && shotData.points.length) ? `${season} Shot Chart` : `${season} - No Data`;
  plot(divId, traces, courtLayout({ title: { text: title, x: 0.5 }, legend: { orientation: "h", y: 1.02, x: 0.5, xanchor: "center" } }), { staticPlot: false });
}

function drawZoneMap(divId, shotData, season) {
  const byName = {};
  ((shotData && shotData.zones) || []).forEach(z => byName[z.name] = z);
  const traces = courtArcTraces();
  (state.zones || []).forEach(z => {
    const st = byName[z.name];
    let fill = "rgba(28,31,38,0.6)", hover = z.name, centroidText = null;
    if (st && st.fga > 0) {
      const pct = st.pct;
      fill = pct < 0.35 ? "rgba(69,117,180,0.85)" : pct < 0.45 ? "rgba(255,255,191,0.85)" : "rgba(215,48,39,0.85)";
      hover += `<br>${st.fgm}/${st.fga} (${(pct * 100).toFixed(1)}%)`;
      centroidText = `${st.fgm}/${st.fga}<br>${(pct * 100).toFixed(1)}%`;
    }
    traces.push({ type: "scatter", x: z.x, y: z.y, fill: "toself", mode: "lines", line: { color: "rgba(255,255,255,0.9)", width: 1.6 }, fillcolor: fill, hoverinfo: "text", text: hover, showlegend: false });
    if (centroidText) {
      const cx = z.x.reduce((a, b) => a + b, 0) / z.x.length;
      const cy = z.y.reduce((a, b) => a + b, 0) / z.y.length;
      traces.push({ type: "scatter", x: [cx], y: [cy], mode: "text", text: [centroidText], textfont: { size: 10, color: "black", family: "Arial Black" }, showlegend: false, hoverinfo: "skip" });
    }
  });
  plot(divId, traces, courtLayout({ title: { text: `${season} Zone Efficiency`, x: 0.5 } }));
}

function renderShots() {
  const main = document.getElementById("page-content");
  const p = state.playerData;
  main.appendChild(el(`<h1>${p.player_name} &mdash; Shot Analysis</h1>`));

  const seasons = Object.keys(p.shot_charts || {}).sort().reverse();
  if (!seasons.length) {
    main.appendChild(el(`<p style="color:var(--muted)">No shot chart data available yet for ${p.player_name}.</p>`));
    return;
  }
  if (!state.shotSeason || !seasons.includes(state.shotSeason)) state.shotSeason = seasons[0];
  if (!state.shotSeasonB || !seasons.includes(state.shotSeasonB)) state.shotSeasonB = seasons[Math.min(1, seasons.length - 1)];

  const controls = el(`<div class="controls-row">
    <div class="control"><label class="checkbox-row"><input type="checkbox" id="shot-compare" ${state.shotCompare ? "checked" : ""}/> Compare Mode</label></div>
    <div class="control"><label class="field-label">Season A</label><select id="shot-season-a">${seasons.map(s => `<option value="${s}" ${s === state.shotSeason ? "selected" : ""}>${s}</option>`).join("")}</select></div>
    <div class="control" id="shot-season-b-wrap" style="display:${state.shotCompare ? "flex" : "none"}"><label class="field-label">Season B</label><select id="shot-season-b">${seasons.map(s => `<option value="${s}" ${s === state.shotSeasonB ? "selected" : ""}>${s}</option>`).join("")}</select></div>
    <div class="control"><label class="field-label">Map Style</label>
      <div class="tabs" style="border-bottom:none;margin:0">
        <button data-view="chart" class="${state.shotView === "chart" ? "active" : ""}">Shot Chart</button>
        <button data-view="zone" class="${state.shotView === "zone" ? "active" : ""}">14-Zone Efficiency</button>
      </div>
    </div>
  </div>`);
  main.appendChild(controls);

  const viewWrap = el(`<div id="shot-view-wrap" style="display:flex;gap:20px;flex-wrap:wrap"></div>`);
  main.appendChild(viewWrap);

  function redraw() {
    viewWrap.innerHTML = "";
    if (!state.shotCompare) {
      viewWrap.appendChild(el(`<div id="shot-plot-a" style="height:640px;width:650px"></div>`));
      const sc = p.shot_charts[state.shotSeason];
      if (state.shotView === "chart") drawShotChart("shot-plot-a", sc, state.shotSeason); else drawZoneMap("shot-plot-a", sc, state.shotSeason);
    } else {
      viewWrap.appendChild(el(`<div id="shot-plot-a" style="height:640px;width:650px"></div>`));
      viewWrap.appendChild(el(`<div id="shot-plot-b" style="height:640px;width:650px"></div>`));
      const scA = p.shot_charts[state.shotSeason], scB = p.shot_charts[state.shotSeasonB];
      if (state.shotView === "chart") { drawShotChart("shot-plot-a", scA, state.shotSeason); drawShotChart("shot-plot-b", scB, state.shotSeasonB); }
      else { drawZoneMap("shot-plot-a", scA, state.shotSeason); drawZoneMap("shot-plot-b", scB, state.shotSeasonB); }
    }
  }

  controls.querySelector("#shot-compare").addEventListener("change", (e) => {
    state.shotCompare = e.target.checked;
    controls.querySelector("#shot-season-b-wrap").style.display = state.shotCompare ? "flex" : "none";
    redraw();
  });
  controls.querySelector("#shot-season-a").addEventListener("change", (e) => { state.shotSeason = e.target.value; redraw(); });
  controls.querySelector("#shot-season-b").addEventListener("change", (e) => { state.shotSeasonB = e.target.value; redraw(); });
  controls.querySelectorAll(".tabs button").forEach(b => b.addEventListener("click", () => {
    state.shotView = b.dataset.view;
    controls.querySelectorAll(".tabs button").forEach(x => x.classList.toggle("active", x === b));
    redraw();
  }));

  redraw();
}

// -----------------------------
// Deep Dive
// -----------------------------
const SCOUTING_REPORTS = {
  deni_avdija: `
    <h4>&#129300; Analysis: The Expanded Role</h4>
    <p><b>Early returns (11/26):</b> The most impressive aspect of Avdija's star-making season has been his capacity for scaling up his production to fit his expanded role.</p>
    <ul>
      <li><b>Pick-and-Roll Volume:</b> Avdija has already logged more possessions as a P&amp;R initiator than in his full years 2 or 3.</li>
      <li><b>Elite Driving:</b> He is fully tapping into his physicality. No one in the league drives more often, and few pass out of drives more frequently.</li>
      <li><b>Free Throw Rate:</b> His downhill speed and "incessant drives" have him getting to the line at a rate on par with <b>Shai Gilgeous-Alexander</b>.</li>
    </ul>
    <h4>&#9889; Defining Trait: The One-Man Fast Break</h4>
    <p>Avdija has become a reliable one-man fast break. He is equally adept at finishing at full tilt or shifting gears (Euro-step or shoulder bumps) to dislodge defenders.</p>
    <blockquote>"The most bullish sign of Avdija's ascent might be his ability to draw contact. Avdija had one of the highest free throw attempt rates in the league... the only other non-bigs in his cohort were <b>Jimmy Butler</b> and <b>James Harden</b>."</blockquote>
    <h4>&#129516; Modern NBA Archetype: The Multidimensional Wing</h4>
    <p>Multidimensional wings are the lifeblood of the modern game. Avdija represents a high-reward venture that is paying off: incremental growth throughout his career, with breakthroughs in years 4 and 5. His vision and ballhandling got him noticed; his defense and rebounding instincts kept him on the floor long enough for it to pay off.</p>
  `,
};
const DEFAULT_SCOUTING = `<p>Scouting notes are still being written for this player &mdash; check back once more of the season is in the books. In the meantime, the Dashboard and Career Analysis pages track his per-game trends as they happen.</p>`;

function weightedAvg(rows, key) {
  const totalGP = sum(rows, "GP");
  return totalGP ? rows.reduce((a, r) => a + r[key] * r.GP, 0) / totalGP : 0;
}

function allstarThresholdChart(divId, playerName, playerStats, allstarStats) {
  const avgPts = weightedAvg(allstarStats, "PTS"), avgReb = weightedAvg(allstarStats, "REB"), avgAst = weightedAvg(allstarStats, "AST");
  const bottom4 = allstarStats.slice().sort((a, b) => a.PTS - b.PTS).slice(0, 4);
  const cats = ["PTS", "REB", "AST"];
  const pVals = [playerStats.PTS, playerStats.REB, playerStats.AST];
  const aVals = [avgPts, avgReb, avgAst];
  const traces = [
    { type: "bar", name: playerName, x: cats, y: pVals, marker: { color: COLOR.accent }, text: pVals.map(v => v.toFixed(1)), textposition: "outside" },
    { type: "bar", name: "All-Star Avg", x: cats, y: aVals, marker: { color: COLOR.info }, text: aVals.map(v => v.toFixed(1)), textposition: "outside" },
  ];
  bottom4.forEach(row => traces.push({ type: "bar", name: `${row.PLAYER_NAME} (Entry)`, x: cats, y: [row.PTS, row.REB, row.AST], opacity: 0.4, marker: { color: "gray" } }));
  plot(divId, traces, { title: { text: "The All-Star Threshold" }, barmode: "group", yaxis: { title: "Per Game", gridcolor: COLOR.mutedLine } });
}

function percentileBelow(rows, key, value) {
  if (!rows.length) return 0;
  const below = rows.filter(r => r[key] < value).length;
  return Math.floor((below / rows.length) * 100);
}

function renderVerdict(container, playerName, playerStats, allstarStats) {
  const first = firstName(playerName);
  const ptsP = percentileBelow(allstarStats, "PTS", playerStats.PTS);
  const rebP = percentileBelow(allstarStats, "REB", playerStats.REB);
  const astP = percentileBelow(allstarStats, "AST", playerStats.AST);
  const avgP = (ptsP + rebP + astP) / 3;
  let cls = "", msg = "";
  if (avgP > 50) { msg = `&#127942; <b>All-Star Caliber</b>: ${first} ranks in the top half of All-Stars (${ordinal(avgP)} percentile avg).`; }
  else if (avgP > 30) { cls = "warn"; msg = `&#9889; <b>Borderline</b>: ${first} is competitive with lower-tier All-Stars (${ordinal(avgP)} percentile avg).`; }
  else { cls = "info"; msg = `&#128200; <b>Developing</b>: ${first} shows flashes but trails the All-Star pack (${ordinal(avgP)} percentile avg).`; }
  container.innerHTML = `
    <h3>&#127919; The Analytical Verdict</h3>
    <div class="metric-row">
      ${kpiCard("Scoring Percentile", ordinal(ptsP))}
      ${kpiCard("Rebounding Percentile", ordinal(rebP))}
      ${kpiCard("Playmaking Percentile", ordinal(astP))}
    </div>
    <p style="color:var(--muted);font-size:13px">Compares ${first} against the selected All-Star roster. Being in the ${ordinal(ptsP)} percentile means he outscores ${ptsP}% of the NBA's elite.</p>
    <div class="verdict-box ${cls}">${msg}</div>
  `;
}

function tripleThreatChart(divId, playerName, playerStats, allstarStats) {
  const SIZE_FACTOR = 0.16;
  const ptsList = allstarStats.map(r => r.PTS).concat([playerStats.PTS]);
  const astList = allstarStats.map(r => r.AST).concat([playerStats.AST]);
  const rebList = allstarStats.map(r => r.REB).concat([playerStats.REB]);
  const namesList = allstarStats.map(r => r.PLAYER_NAME).concat([playerName]);

  const images = [];
  allstarStats.forEach(row => {
    const url = allstarHeadshot(row.PLAYER_NAME);
    if (url) {
      const size = Math.max(row.REB * SIZE_FACTOR, 0.5);
      images.push({ source: url, xref: "x", yref: "y", x: row.PTS, y: row.AST, sizex: size, sizey: size, xanchor: "center", yanchor: "middle", layer: "above" });
    }
  });
  const playerUrl = allstarHeadshot(playerName) || state.playerData.headshot;
  const playerSize = Math.max(playerStats.REB * SIZE_FACTOR, 0.5);
  if (playerUrl) images.push({ source: playerUrl, xref: "x", yref: "y", x: playerStats.PTS, y: playerStats.AST, sizex: playerSize, sizey: playerSize, xanchor: "center", yanchor: "middle", layer: "above" });

  const traces = [
    {
      type: "scatter", mode: "markers", x: ptsList, y: astList, text: namesList, customdata: rebList,
      marker: { size: rebList.map(r => r * 4), color: "rgba(0,0,0,0)", line: { width: 0 } },
      hovertemplate: "<b>%{text}</b><br>PTS: %{x:.1f}<br>AST: %{y:.1f}<br>REB: %{customdata:.1f}<extra></extra>", showlegend: false,
    },
    { type: "scatter", mode: "text", x: [playerStats.PTS], y: [playerStats.AST - playerSize * 0.6], text: [firstName(playerName)], textposition: "bottom center", textfont: { size: 14, color: COLOR.text, family: "Arial Black" }, showlegend: false },
  ];
  plot(divId, traces, {
    title: { text: "Triple Threat (2D): PTS vs AST (Face Size = REB)" },
    xaxis: { title: "Points Per Game", range: [Math.min(...ptsList) - 2, Math.max(...ptsList) + 2], gridcolor: COLOR.mutedLine },
    yaxis: { title: "Assists Per Game", range: [Math.min(...astList) - 1, Math.max(...astList) + 1], gridcolor: COLOR.mutedLine },
    images, height: 700, showlegend: false,
  });
}

function separationChart(divId, playerName, playerStats, allstarDetailed) {
  const others = allstarDetailed.filter(r => r.PLAYER_NAME !== playerName && r.USG_PCT !== null && r.TS_PCT !== null);
  const images = [];
  others.forEach(row => {
    const url = allstarHeadshot(row.PLAYER_NAME);
    if (url) images.push({ source: url, xref: "x", yref: "y", x: row.USG_PCT * 100, y: row.TS_PCT * 100, sizex: 1.5, sizey: 1.5, xanchor: "center", yanchor: "middle", layer: "above" });
  });
  const dx = playerStats.USG_PCT * 100, dy = playerStats.TS_PCT * 100;
  const playerUrl = allstarHeadshot(playerName) || state.playerData.headshot;
  if (playerUrl) images.push({ source: playerUrl, xref: "x", yref: "y", x: dx, y: dy, sizex: 1.5, sizey: 1.5, xanchor: "center", yanchor: "middle", layer: "above" });

  plot(divId, [
    { type: "scatter", mode: "text", x: [dx], y: [dy + 1.5], text: [firstName(playerName)], textposition: "top center", textfont: { size: 14, color: COLOR.text, family: "Arial Black" }, showlegend: false },
    { type: "scatter", mode: "markers", x: others.map(r => r.USG_PCT * 100), y: others.map(r => r.TS_PCT * 100), text: others.map(r => r.PLAYER_NAME), marker: { color: "rgba(0,0,0,0)", size: 30, line: { width: 0 } }, hoverinfo: "text+x+y", showlegend: false, name: "All-Stars" },
  ], { xaxis: { title: "Usage %", range: [18, 40], gridcolor: COLOR.mutedLine }, yaxis: { title: "True Shooting %", range: [48, 70], gridcolor: COLOR.mutedLine }, images, height: 700, showlegend: false });
}

function renderFullTable(container, tabId, playerName, playerStats, gp, allstarStats) {
  const metrics = ["PTS", "REB", "AST", "STL", "BLK", "TOV"];
  container.innerHTML = `
    <div class="control" style="margin-bottom:12px"><label class="field-label">&#127942; Rank Players By</label>
      <select id="rank-select-${tabId}">${metrics.map(m => `<option value="${m}">${m}</option>`).join("")}</select>
    </div>
    <div id="table-wrap-${tabId}"></div>
  `;
  const sel = container.querySelector(`#rank-select-${tabId}`);
  const wrap = container.querySelector(`#table-wrap-${tabId}`);
  function draw() {
    const metric = sel.value;
    let rows = allstarStats.map(r => ({ PLAYER_NAME: r.PLAYER_NAME, GP: r.GP, PTS: r.PTS, REB: r.REB, AST: r.AST, STL: r.STL, BLK: r.BLK, TOV: r.TOV }));
    rows.push({ PLAYER_NAME: playerName, GP: gp, PTS: playerStats.PTS, REB: playerStats.REB, AST: playerStats.AST, STL: playerStats.STL, BLK: playerStats.BLK, TOV: playerStats.TOV });
    rows.sort((a, b) => b[metric] - a[metric]);
    wrap.innerHTML = `<div class="table-wrap"><table class="data-table"><thead><tr><th>Rank</th><th>Player</th>${metrics.map(m => `<th>${m}</th>`).join("")}<th>GP</th></tr></thead><tbody>
      ${rows.map((r, i) => `<tr class="${r.PLAYER_NAME === playerName ? "highlight" : ""}"><td>${i + 1}</td><td>${r.PLAYER_NAME}</td>${metrics.map(m => `<td>${fmt1(r[m])}</td>`).join("")}<td>${r.GP}</td></tr>`).join("")}
    </tbody></table></div>`;
  }
  sel.addEventListener("change", draw);
  draw();
}

function radarChart(divId, playerName, playerStats, allstarStats) {
  const metrics = ["PTS", "REB", "AST", "STL", "BLK"];
  const dVals = metrics.map(m => playerStats[m] || 0);
  const avgVals = metrics.map(m => mean(allstarStats, m));
  const maxVals = metrics.map((m, i) => Math.max(Math.max(...allstarStats.map(r => r[m])), dVals[i]) || 1);
  const dNorm = dVals.map((v, i) => v / maxVals[i]);
  const aNorm = avgVals.map((v, i) => v / maxVals[i]);
  const theta = metrics.concat([metrics[0]]);
  plot(divId, [
    { type: "scatterpolar", r: aNorm.concat([aNorm[0]]), theta, fill: "toself", name: "All-Star Avg", line: { color: COLOR.info, width: 2 }, fillcolor: "rgba(159,208,255,0.18)" },
    { type: "scatterpolar", r: dNorm.concat([dNorm[0]]), theta, fill: "toself", name: playerName, line: { color: COLOR.accent, width: 3 }, fillcolor: "rgba(215,255,58,0.28)" },
  ], { polar: { radialaxis: { visible: false, range: [0, 1] }, bgcolor: "rgba(0,0,0,0)" }, title: { text: "The Multidimensional Wing (Normalized)" }, height: 500 });
}

function offensiveEngineChart(divId, playerName, playerStats, allstarStats) {
  let df = allstarStats.map(r => Object.assign({}, r, { PTS_CREATED: r.AST * 2.3, TOTAL_OUTPUT: r.PTS + r.AST * 2.3 }));
  if (!df.some(r => r.PLAYER_NAME === playerName)) {
    df.push({ PLAYER_NAME: playerName, PTS: playerStats.PTS, PTS_CREATED: playerStats.AST * 2.3, TOTAL_OUTPUT: playerStats.PTS + playerStats.AST * 2.3 });
  }
  df.sort((a, b) => b.TOTAL_OUTPUT - a.TOTAL_OUTPUT);
  let top15 = df.slice(0, 15);
  if (!top15.some(r => r.PLAYER_NAME === playerName)) {
    const row = df.find(r => r.PLAYER_NAME === playerName);
    top15 = top15.concat([row]).sort((a, b) => b.TOTAL_OUTPUT - a.TOTAL_OUTPUT);
  }
  const names = top15.map(r => r.PLAYER_NAME);
  plot(divId, [
    { type: "bar", name: "Points Scored", x: names, y: top15.map(r => r.PTS), marker: { color: names.map(n => n === playerName ? COLOR.accent : "#7f7f7f") } },
    { type: "bar", name: "Points Created (Est)", x: names, y: top15.map(r => r.PTS_CREATED), marker: { color: names.map(n => n === playerName ? COLOR.info : "#1f77b4") } },
  ], { barmode: "stack", title: { text: "The Offensive Engine (Scoring + Playmaking)" }, xaxis: { tickangle: -45 }, yaxis: { title: "Total Points Production" }, height: 500 });
}

function whatIfChart(divId) {
  const deniPpg = 25.6, deniRpg = 7.2, deniApg = 7.0, deniUsage = 28.0;
  const lukaPpg = 33.6, lukaRpg = 8.1, lukaApg = 8.7, lukaUsage = 37.9;
  const ratio = lukaUsage / deniUsage;
  const cats = ["Points", "Rebounds", "Assists"];
  const proj = [deniPpg * ratio, deniRpg * ratio, deniApg * ratio];
  plot(divId, [
    { type: "bar", name: "Deni (Actual - 28% USG)", x: cats, y: [deniPpg, deniRpg, deniApg], marker: { color: "rgb(160,160,160)" }, text: [deniPpg, deniRpg, deniApg], textposition: "auto" },
    { type: "bar", name: `Deni (Projected @ ${lukaUsage}% USG)`, x: cats, y: proj, marker: { color: COLOR.accent }, text: proj.map(v => v.toFixed(1)), textposition: "auto" },
    { type: "bar", name: "Luka (Actual)", x: cats, y: [lukaPpg, lukaRpg, lukaApg], marker: { color: "rgb(55,83,109)" }, text: [lukaPpg, lukaRpg, lukaApg], textposition: "auto" },
  ], { barmode: "group", title: { text: "Usage-Adjusted Efficiency: Deni vs. Luka" }, yaxis: { title: "Per Game Stats" }, legend: { orientation: "h", y: 1.15 }, height: 500 });
}

function renderFtLeaders(main, playerName) {
  const ft = state.shared.league_ft;
  if (!ft.leaders || !ft.leaders.length) return;
  main.appendChild(el("<hr/>"));
  main.appendChild(el(`<h3>&#128293; ${ft.season} Season: Free Throw Leaders (Top 10)</h3>`));
  let top10 = ft.leaders.slice().sort((a, b) => b.FTM - a.FTM).slice(0, 10);
  if (!top10.some(r => r.PLAYER_NAME === playerName)) {
    const own = ft.leaders.find(r => r.PLAYER_NAME === playerName);
    if (own) top10 = top10.concat([own]);
  }
  const controls = el(`<div class="control" style="margin-bottom:14px"><label class="field-label">Sort Leaderboard By</label>
    <select id="ft-sort"><option value="FTA">Total Attempts</option><option value="FTM">Total Made</option><option value="FT_PCT">FT%</option></select></div>`);
  main.appendChild(controls);
  const wrap = el(`<div id="ft-table"></div>`);
  main.appendChild(wrap);
  function draw() {
    const key = controls.querySelector("select").value;
    const rows = top10.slice().sort((a, b) => b[key] - a[key]);
    wrap.innerHTML = `<div class="table-wrap"><table class="data-table"><thead><tr><th>Rank</th><th>Player</th><th>Team</th><th>Games</th><th>Total Made</th><th>Total Attempts</th><th>FT%</th></tr></thead><tbody>
      ${rows.map((r, i) => `<tr class="${r.PLAYER_NAME === playerName ? "highlight" : ""}"><td>${i + 1}</td><td>${r.PLAYER_NAME}</td><td>${r.TEAM_ABBREVIATION}</td><td>${r.GP}</td><td>${r.FTM}</td><td>${r.FTA}</td><td>${fmtPct1(r.FT_PCT)}</td></tr>`).join("")}
    </tbody></table></div>`;
  }
  controls.querySelector("select").addEventListener("change", draw);
  draw();
}

function renderDeepDive() {
  const main = document.getElementById("page-content");
  const p = state.playerData;
  main.appendChild(el(`<h1>${p.player_name} &mdash; All-Star Comparison</h1>`));

  main.appendChild(el(`<details class="scouting"><summary>&#128203; READ: Scouting Report &amp; Analysis &mdash; ${p.player_name}</summary><div class="body">${SCOUTING_REPORTS[state.playerKey] || DEFAULT_SCOUTING}</div></details>`));

  renderFtLeaders(main, p.player_name);

  const logs = (p.game_logs || {})[state.meta.current_season] || [];
  if (!logs.length) {
    main.appendChild(el("<hr/>"));
    main.appendChild(el(`<div class="banner info">${p.player_name} hasn't played an NBA game yet this season &mdash; the All-Star comparison lab activates once box scores are logged.</div>`));
    return;
  }
  const playerStats = { PTS: mean(logs, "PTS"), REB: mean(logs, "REB"), AST: mean(logs, "AST"), STL: mean(logs, "STL"), BLK: mean(logs, "BLK"), TOV: mean(logs, "TOV") };
  const careerCur = (p.career || []).find(r => r.SEASON_ID === state.meta.current_season);
  if (careerCur) { playerStats.USG_PCT = careerCur.USG_PCT; playerStats.TS_PCT = careerCur.TS_PCT; }

  main.appendChild(el("<hr/>"));
  const tabs = el(`<div class="tabs">
    <button data-tab="bench" class="active">Benchmark (${seasonLabel(state.shared.allstar_bench.season)} All-Stars)</button>
    <button data-tab="race">The Race (${seasonLabel(state.shared.allstar_race.season)} All-Star Stats)</button>
  </div>`);
  main.appendChild(tabs);
  const body = el(`<div id="deepdive-body"></div>`);
  main.appendChild(body);

  function drawTab(tabName) {
    const cohort = tabName === "bench" ? state.shared.allstar_bench : state.shared.allstar_race;
    const stats = (cohort.stats || []).filter(r => r.PLAYER_NAME !== p.player_name);
    const detailed = (cohort.detailed || []).filter(r => r.PLAYER_NAME !== p.player_name);
    body.innerHTML = "";

    if (!stats.length) {
      body.appendChild(el(`<div class="banner warn">&#9888; No ${cohort.season} All-Star data found.</div>`));
      return;
    }

    body.appendChild(el(`<p style="color:var(--muted)">Comparing ${p.player_name}'s <b>current</b> stats against the ${tabName === "bench" ? "<b>final</b>" : `<b>current (${seasonLabel(cohort.season)})</b>`} stats of the All-Star cohort.</p>`));

    body.appendChild(el(`<h3>1. The ${tabName === "bench" ? "All-Star Threshold" : "Race Threshold"}</h3>`));
    const row1 = el(`<div class="two-col"><div id="thresh-${tabName}" style="height:420px"></div><div id="verdict-${tabName}"></div></div>`);
    body.appendChild(row1);
    allstarThresholdChart(`thresh-${tabName}`, p.player_name, playerStats, stats);
    renderVerdict(row1.querySelector(`#verdict-${tabName}`), p.player_name, playerStats, stats);

    body.appendChild(el("<hr/>"));
    body.appendChild(el(`<h3>2. The Triple Threat</h3>`));
    body.appendChild(el(`<div id="triple-${tabName}" style="height:700px"></div>`));
    tripleThreatChart(`triple-${tabName}`, p.player_name, playerStats, stats);

    if (detailed.length && playerStats.USG_PCT !== undefined && playerStats.USG_PCT !== null) {
      body.appendChild(el("<hr/>"));
      body.appendChild(el(`<h3>3. Separation Chart (Usage vs Efficiency)</h3>`));
      body.appendChild(el(`<div id="sep-${tabName}" style="height:700px;width:700px"></div>`));
      separationChart(`sep-${tabName}`, p.player_name, playerStats, detailed);
    }

    body.appendChild(el("<hr/>"));
    body.appendChild(el(`<h3>4. Full League Comparison Table</h3>`));
    const tableWrap = el(`<div></div>`);
    body.appendChild(tableWrap);
    renderFullTable(tableWrap, tabName, p.player_name, playerStats, logs.length, stats);

    body.appendChild(el("<hr/>"));
    body.appendChild(el(`<h3>5. Advanced Case Studies</h3>`));
    const advRow = el(`<div class="two-col"><div id="radar-${tabName}" style="height:500px"></div><div id="engine-${tabName}" style="height:500px"></div></div>`);
    body.appendChild(advRow);
    radarChart(`radar-${tabName}`, p.player_name, playerStats, stats);
    offensiveEngineChart(`engine-${tabName}`, p.player_name, playerStats, stats);

    if (tabName === "race" && state.playerKey === "deni_avdija") {
      body.appendChild(el("<hr/>"));
      body.appendChild(el(`<h3>6. What-If: Deni vs Luka Efficiency</h3><p style="color:var(--muted);font-size:13px">How would Deni compare if he had Luka Don&#269;i&#263;'s usage rate (37.9%)?</p>`));
      body.appendChild(el(`<div id="whatif-${tabName}" style="height:500px"></div>`));
      whatIfChart(`whatif-${tabName}`);
    }
  }

  tabs.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-tab]");
    if (!btn) return;
    tabs.querySelectorAll("button").forEach(b => b.classList.toggle("active", b === btn));
    drawTab(btn.dataset.tab);
  });

  drawTab("bench");
}

// -----------------------------
// About Me
// -----------------------------
function renderAbout() {
  const main = document.getElementById("page-content");
  main.innerHTML = `
    <h1>About the Creator</h1>
    <div style="display:flex;gap:32px;flex-wrap:wrap;align-items:flex-start">
      <div style="flex:0 0 220px;text-align:center">
        <img src="../profile_pic.png" alt="Ram Shiri" style="width:200px;height:200px;object-fit:cover;border:1px solid var(--line-strong)" onerror="this.style.display='none'"/>
        <h3 style="margin-top:12px">Ram Shiri</h3>
        <p><b>Data Engineering Student</b></p>
        <p><a href="https://www.linkedin.com/in/ram-shiri-1a1056304/?originalSubdomain=il" target="_blank">Connect on LinkedIn</a></p>
        <p><a href="https://github.com/RShiri/nba-dashboard" target="_blank">&#9733; View Source on GitHub</a></p>
      </div>
      <div style="flex:1 1 400px">
        <h3>&#128075; Hello!</h3>
        <p>I'm a <b>3rd year B.Sc. Data Engineering student</b> specializing in data science with a passion for building smart, practical solutions.</p>
        <p>I love combining creativity with technical skills to drive real-world impact&mdash;especially in the world of sports analytics.</p>
        <h3>&#128736; Skills &amp; Approach</h3>
        <ul>
          <li><b>Tech Stack:</b> Python, Java, SQL, Pandas, Streamlit, Plotly, JavaScript</li>
          <li><b>Soft Skills:</b> Creative thinking, fast learning, hands-on problem solving</li>
          <li><b>Philosophy:</b> Comfortable working with AI tools to accelerate development (like this dashboard!) while maintaining deep understanding of the core logic.</li>
        </ul>
        <h3>&#10084; Passions</h3>
        <p>&#127936; Basketball | &#9917; Football | &#127950; F1 Racing | &#129513; LEGO</p>
        <hr/>
        <div class="banner">&#128640; <b>Open to Work:</b> Actively seeking a student or full-time position in software or data engineering to grow, contribute, and thrive in a dynamic environment.</div>
      </div>
    </div>
  `;
}

boot();
