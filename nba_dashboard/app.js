/* Israeli NBA Players Analytics — static dashboard.
   Reads the JSON that build_data.py writes to data/, draws every chart with Plotly.
   Two looks (C / A) and two languages (EN / HE) share the switches of rshiri.github.io. */

const PAGES = ["dashboard", "career", "league", "raw", "shots", "deepdive", "project", "about"];
const PLAYER_HE = { "Deni Avdija": "דני אבדיה", "Ben Saraf": "בן שרף", "Danny Wolf": "דני וולף", "Emanuel Sharp": "עמנואל שארפ" };

const state = {
  meta: null,
  shared: null,
  zones: null,
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
// Language
// -----------------------------
function lang() { return document.documentElement.getAttribute("lang") === "he" ? "he" : "en"; }
function t(key, vars) {
  let s = (I18N[lang()] || {})[key];
  if (s === undefined) s = I18N.en[key];
  if (s === undefined) return key;
  if (vars) s = s.replace(/\{(\w+)\}/g, (m, k) => (vars[k] !== undefined ? vars[k] : m));
  return s;
}
function block(key) { return (I18N_BLOCKS[lang()] || {})[key] || I18N_BLOCKS.en[key] || ""; }
function displayName(name) { return lang() === "he" && PLAYER_HE[name] ? PLAYER_HE[name] : name; }
function pname() { return displayName(state.playerData.player_name); }

// -----------------------------
// Utilities
// -----------------------------
function seasonLabel(s) { return s ? `${s.slice(2, 4)}/${s.slice(5)}` : ""; }
function isNum(x) { return x !== null && x !== undefined && !isNaN(x); }
function fmt1(x) { return isNum(x) ? Number(x).toFixed(1) : "—"; }
function fmtPct1(x) { return isNum(x) ? (Number(x) * 100).toFixed(1) + "%" : "—"; }
function fmtInt(x) { return isNum(x) ? Math.round(x).toString() : "—"; }
function mean(arr, key) {
  const v = arr.map(r => r[key]).filter(isNum);
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0;
}
function sum(arr, key) { return arr.reduce((a, r) => a + (Number(r[key]) || 0), 0); }
function ordinal(n) {
  n = Math.floor(n);
  if (lang() === "he") return String(n);
  const s = ["th", "st", "nd", "rd"], v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}
function firstName(name) { return name.split(" ")[0]; }
function el(html) {
  const tpl = document.createElement("template");
  tpl.innerHTML = html.trim();
  return tpl.content.firstElementChild;
}
function ltr(s) { return `<span class="ltr">${s}</span>`; }
function allstarHeadshot(name) { return (state.shared.headshots || {})[name]; }
function main() { return document.getElementById("page-content"); }

// -----------------------------
// Plotly theme — colours and fonts come from the active look's CSS tokens
// -----------------------------
function tokens() {
  const cs = getComputedStyle(document.documentElement);
  const v = n => cs.getPropertyValue(n).trim();
  return {
    accent: v("--c"), accentInk: v("--accent-ink"), neg: v("--neg"), info: v("--info"), warn: v("--warn"),
    ink: v("--ink"), body: v("--body"), muted: v("--muted"), line: v("--line"), lineStrong: v("--line-strong"),
    pitch: v("--pitch"), court: v("--court-line"), font: v("--font"), text: v("--text"),
    style: document.documentElement.getAttribute("data-style"),
  };
}
function rgba(hex, a) {
  const h = hex.replace("#", "");
  if (h.length !== 6) return hex;
  return `rgba(${parseInt(h.slice(0, 2), 16)},${parseInt(h.slice(2, 4), 16)},${parseInt(h.slice(4, 6), 16)},${a})`;
}
function axis(T, extra) {
  return Object.assign({ gridcolor: T.line, zerolinecolor: T.line, linecolor: T.lineStrong, color: T.muted, tickfont: { family: T.font }, automargin: true }, extra || {});
}
function baseLayout(overrides) {
  const T = tokens();
  const o = Object.assign({}, overrides || {});
  const layout = Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: T.text, color: T.body, size: 12.5 },
    colorway: [T.accent, T.info, T.warn, T.neg, T.muted],
    margin: { t: 48, l: 52, r: 24, b: 48 },
    hoverlabel: { bgcolor: "#0a0e18", bordercolor: T.lineStrong, font: { family: T.text, color: "#e8f8ff" } },
    legend: { font: { color: T.muted } },
    autosize: true,
  }, o);
  layout.xaxis = axis(T, o.xaxis);
  layout.yaxis = axis(T, o.yaxis);
  if (o.yaxis2) layout.yaxis2 = axis(T, Object.assign({ showgrid: false }, o.yaxis2));
  if (o.title) {
    const title = typeof o.title === "string" ? { text: o.title } : o.title;
    layout.title = Object.assign({ x: 0.02, xanchor: "left", font: { family: T.font, color: T.ink, size: T.style === "a" ? 19 : 15 } }, title);
  }
  return layout;
}
function plotConfig(name) {
  return {
    displayModeBar: "hover", displaylogo: false, responsive: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
    toImageButtonOptions: { format: "png", filename: name || "israeli_nba_chart", height: 1000, width: 1400, scale: 2 },
  };
}
function plot(containerId, traces, layout, config) {
  const lay = baseLayout(layout);
  // phones: a side legend would squeeze the plot, so stack it under the chart
  const div = document.getElementById(containerId);
  if (div && div.clientWidth < 560 && lay.showlegend !== false && !(layout && layout.legend && layout.legend.orientation)) {
    lay.legend = Object.assign({}, lay.legend, { orientation: "h", x: 0, y: -0.12, yanchor: "top" });
  }
  return Plotly.newPlot(containerId, traces, lay, Object.assign(plotConfig(`${state.playerKey}_${containerId}`), config || {}));
}
function chartBlock(id, title, opts) {
  opts = opts || {};
  const cls = opts.cls || "plot";
  const style = opts.height ? `style="height:${opts.height}px"` : "";
  return `<div class="chart-block">${title ? `<h3>${title}</h3>` : ""}${opts.note ? `<p class="section-note">${opts.note}</p>` : ""}
    <div class="chart-panel bracket"><div class="${cls}" id="${id}" ${style}></div></div></div>`;
}

// -----------------------------
// Data loading + routing
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
function readHash() {
  const [page, player] = location.hash.replace(/^#/, "").split("/");
  return { page: PAGES.includes(page) ? page : null, player: player || null };
}
function writeHash() {
  const h = `#${state.page}/${state.playerKey}`;
  if (location.hash !== h) history.replaceState(null, "", h);
}

async function boot() {
  try {
    [state.meta, state.shared, state.zones] = await Promise.all([
      loadJSON("data/meta.json"), loadJSON("data/shared.json"), loadJSON("data/zones.json"),
    ]);
  } catch (e) {
    main().innerHTML = `<div class="banner warn">${t("load_error")}</div>`;
    return;
  }
  const keys = state.meta.players.map(p => p.key);
  const route = readHash();
  if (route.page) state.page = route.page;
  await loadPlayer(keys.includes(route.player) ? route.player : keys[0]);

  document.getElementById("player-select").addEventListener("change", async (e) => {
    await loadPlayer(e.target.value);
    state.shotSeason = null; state.shotSeasonB = null;
    renderAll();
  });
  document.getElementById("page-select").addEventListener("change", (e) => goPage(e.target.value));
  document.getElementById("nav-list").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-page]");
    if (btn) goPage(btn.dataset.page);
  });
  document.querySelector(".brand").addEventListener("click", (e) => { e.preventDefault(); goPage("dashboard"); });
  window.addEventListener("hashchange", async () => {
    const r = readHash();
    if (r.player && r.player !== state.playerKey && keys.includes(r.player)) await loadPlayer(r.player);
    if (r.page && r.page !== state.page) state.page = r.page;
    renderAll();
  });

  document.getElementById("styleSwitch").addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-style") === "a" ? "c" : "a";
    document.documentElement.setAttribute("data-style", next);
    try { localStorage.setItem("site-style", next); } catch (e) {}
    renderAll(true);
  });
  document.getElementById("langSwitch").addEventListener("click", () => {
    const next = lang() === "he" ? "en" : "he";
    document.documentElement.setAttribute("lang", next);
    document.documentElement.setAttribute("dir", next === "he" ? "rtl" : "ltr");
    try { localStorage.setItem("site-lang", next); } catch (e) {}
    renderAll(true);
  });

  renderAll();
}

function goPage(page) {
  state.page = page;
  renderAll();
  window.scrollTo(0, 0);
}

function renderAll(keepScroll) {
  const y = window.scrollY;
  writeHash();
  renderChrome();
  renderSidebarPlayer();
  renderPage();
  if (keepScroll) window.scrollTo(0, y);
}

function renderChrome() {
  document.querySelectorAll("[data-i18n]").forEach(n => { n.textContent = t(n.dataset.i18n); });
  document.title = `${t("page_" + state.page)} · ${t("brand_a")} ${t("brand_b")} - Ram Shiri`;
  document.getElementById("nav-list").innerHTML = PAGES.map(p =>
    `<li><button type="button" data-page="${p}" class="${p === state.page ? "active" : ""}" ${p === state.page ? 'aria-current="page"' : ""}>${t("page_" + p)}</button></li>`).join("");
  document.getElementById("page-select").innerHTML = PAGES.map(p =>
    `<option value="${p}" ${p === state.page ? "selected" : ""}>${t("page_" + p)}</option>`).join("");
  document.getElementById("player-select").innerHTML = state.meta.players.map(p =>
    `<option value="${p.key}" ${p.key === state.playerKey ? "selected" : ""}>${displayName(p.name)}</option>`).join("");
}

function renderSidebarPlayer() {
  const p = state.playerData;
  document.getElementById("topbar-player").textContent = `${pname()} · ${p.team_full}`;
  document.getElementById("player-card").innerHTML = `
    <img src="${p.headshot}" alt="" onerror="this.style.visibility='hidden'" />
    <div><div class="pc-name">${pname()}</div><div class="pc-role">${p.team_full}</div></div>`;
  const logs = (p.game_logs && p.game_logs[state.meta.current_season]) || [];
  let fetched = "—";
  if (state.meta.fetched_at) {
    fetched = new Date(state.meta.fetched_at).toLocaleDateString(lang() === "he" ? "he-IL" : "en-GB", { day: "numeric", month: "short", year: "numeric" });
  }
  document.getElementById("status-block").innerHTML = `
    <div>${t("data_updated")}: <b>${fetched}</b></div>
    <div>${t("games_tracked", { name: firstName(pname()) })}: <b class="num">${logs.length}</b></div>`;
}

function renderPage() {
  const m = main();
  Array.from(m.querySelectorAll(".js-plotly-plot")).forEach(d => Plotly.purge(d));
  m.innerHTML = "";
  const renderers = {
    dashboard: renderDashboard, career: renderCareer, league: renderLeague, raw: renderRaw,
    shots: renderShots, deepdive: renderDeepDive, project: renderProject, about: renderAbout,
  };
  renderers[state.page]();
}

// -----------------------------
// Shared bits
// -----------------------------
function heroHtml() {
  const p = state.playerData;
  return `
    <div class="hero bracket">
      <img src="${p.headshot}" alt="" onerror="this.style.visibility='hidden'" />
      <div>
        <div class="kicker">${p.team_full} · ${p.position}</div>
        <h1 class="name">${pname()}</h1>
        <div class="sub">${t("hero_sub", { season: ltr(state.meta.current_season) })}</div>
      </div>
    </div>`;
}

function kpiCard(label, value, delta) {
  let deltaHtml = "";
  if (delta !== undefined && delta !== null) {
    const cls = delta.startsWith("-") ? "down" : "up";
    const arrow = cls === "down" ? "▼" : "▲";
    deltaHtml = `<div class="kpi-delta ${cls}">${ltr(`${arrow} ${delta}`)}</div>`;
  }
  return `<div class="kpi-card"><div class="kpi-label">${label}</div><div class="kpi-value">${ltr(value)}</div>${deltaHtml}</div>`;
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
  if (!c) return false;
  const pr = seasonAggregates(prev);
  const d = (cv, pv, isPct) => {
    if (!pr) return null;
    const scaled = isPct ? (cv - pv) * 100 : cv - pv;
    return `${scaled >= 0 ? "+" : ""}${scaled.toFixed(1)}${isPct ? " pp" : ""}`;
  };
  container.appendChild(el(`<div><div class="kpi-grid">
    ${kpiCard(t("kpi_pts"), fmt1(c.pts), d(c.pts, pr && pr.pts))}
    ${kpiCard(t("kpi_reb"), fmt1(c.reb), d(c.reb, pr && pr.reb))}
    ${kpiCard(t("kpi_ast"), fmt1(c.ast), d(c.ast, pr && pr.ast))}
    ${kpiCard(t("kpi_min"), fmt1(c.min), d(c.min, pr && pr.min))}
  </div>
  <div class="kpi-grid">
    ${kpiCard(t("kpi_fg"), fmtPct1(c.fg), d(c.fg, pr && pr.fg, true))}
    ${kpiCard(t("kpi_3p"), fmtPct1(c.fg3), d(c.fg3, pr && pr.fg3, true))}
    ${kpiCard(t("kpi_ft"), fmtPct1(c.ft), d(c.ft, pr && pr.ft, true))}
    ${kpiCard(t("kpi_gp"), String(c.gp), pr ? `${c.gp - pr.gp >= 0 ? "+" : ""}${c.gp - pr.gp}` : null)}
  </div>${pr ? `<p class="section-note">${t("kpi_caption")}</p>` : ""}</div>`));
  return true;
}

// -----------------------------
// Dashboard
// -----------------------------
function renderDashboard() {
  const m = main();
  const p = state.playerData;
  const T = tokens();
  m.appendChild(el(heroHtml()));
  m.appendChild(el(`<p class="page-intro">${t("intro_dashboard", { name: pname() })}</p>`));

  const anyLogs = p.game_logs && Object.keys(p.game_logs).length > 0;
  if (!anyLogs) {
    m.appendChild(el(`<div class="banner info">${t("no_games_yet", { name: pname() })}</div>`));
  }
  if (state.playerKey === "deni_avdija") {
    m.appendChild(el(`<div class="banner">${t("deni_banner")}</div>`));
  }

  const hasKpis = renderKpiStrip(m);
  if (!hasKpis && anyLogs) {
    m.appendChild(el(`<div class="banner info">${t("no_games_season", { name: pname(), season: ltr(state.meta.current_season) })}</div>`));
  }

  const seasons = [state.meta.current_season, state.meta.prev_season, state.meta.prev2_season];
  seasons.forEach((season, i) => {
    const logs = ((p.game_logs || {})[season] || []).slice().sort((a, b) => a.GAME_DATE.localeCompare(b.GAME_DATE));
    if (!logs.length) return;
    m.appendChild(el(chartBlock(`dash-plot-${i}`, t("impact", { season: ltr(seasonLabel(season)) }), { height: 400 })));
    const avgPts = mean(logs, "PTS"), avgMin = mean(logs, "MIN");
    const dates = logs.map(r => r.GAME_DATE);
    plot(`dash-plot-${i}`, [
      { type: "bar", x: dates, y: logs.map(r => r.PTS), name: "PTS", marker: { color: logs.map(r => r.WL === "W" ? T.accent : T.neg) },
        customdata: logs.map(r => r.WL), hovertemplate: "%{x}<br>PTS %{y} (%{customdata})<extra></extra>", showlegend: false },
      { type: "bar", x: [null], y: [null], name: t("pts_win"), marker: { color: T.accent }, hoverinfo: "skip" },
      { type: "bar", x: [null], y: [null], name: t("pts_loss"), marker: { color: T.neg }, hoverinfo: "skip" },
      { type: "scatter", mode: "lines", x: dates, y: logs.map(r => r.MIN), name: "MIN", yaxis: "y2", line: { color: T.warn, width: 2 } },
      { type: "scatter", mode: "lines", x: dates, y: dates.map(() => avgMin), name: t("avg_min", { v: avgMin.toFixed(1) }), yaxis: "y2", line: { color: T.warn, width: 1, dash: "dot" } },
    ], {
      yaxis: { title: { text: t("points") } },
      yaxis2: { title: { text: t("minutes") }, overlaying: "y", side: "right", range: [0, 48] },
      legend: { orientation: "h", y: 1.12, font: { color: T.muted } },
      margin: { t: 36, l: 52, r: 52, b: 40 },
      shapes: [{ type: "line", x0: 0, x1: 1, xref: "paper", y0: avgPts, y1: avgPts, yref: "y", line: { color: T.muted, dash: "dash", width: 1 } }],
      annotations: [{ x: 0.01, xref: "paper", xanchor: "left", y: avgPts, yref: "y", yanchor: "bottom", text: t("avg_pts", { v: avgPts.toFixed(1) }), showarrow: false, font: { color: T.ink }, bgcolor: rgba(T.pitch, 0.8) }],
    });
  });
}

// -----------------------------
// Career Analysis
// -----------------------------
function renderCareer() {
  const m = main();
  const p = state.playerData;
  const T = tokens();
  m.appendChild(el(`<h1>${t("career_title", { name: pname() })}</h1>`));
  m.appendChild(el(`<p class="page-intro">${t("intro_career", { name: pname() })}</p>`));

  const career = p.career || [];
  if (!career.length) {
    m.appendChild(el(`<div class="banner info">${t("no_career")}</div>`));
    return;
  }
  const seasons = career.map(r => r.SEASON_ID);
  const labels = seasons.map(seasonLabel);

  m.appendChild(el(chartBlock("career-pergame", t("per_game"), { height: 400 })));
  plot("career-pergame", ["PTS", "REB", "AST"].map(k => ({ type: "bar", name: k, x: seasons, y: career.map(r => r[k]) })),
    { barmode: "group", title: "PTS / REB / AST", xaxis: { title: { text: t("season") }, showgrid: false, type: "category" }, yaxis: { dtick: 5 } });

  m.appendChild(el(chartBlock("career-per36", t("per36"), { height: 400 })));
  plot("career-per36", ["PTS", "REB", "AST", "STL", "TOV"].map(k => ({
    type: "bar", name: `${k}_36`, x: seasons, y: career.map(r => (r.MIN > 0 ? (r[k] / r.MIN) * 36 : 0)),
  })), { barmode: "group", title: "Per 36 Min", xaxis: { title: { text: t("season") }, showgrid: false, type: "category" }, yaxis: { dtick: 5 } });

  if (career.some(r => isNum(r.USG_PCT))) {
    m.appendChild(el(chartBlock("career-usg", t("usage"), { height: 360, note: t("usage_note") })));
    plot("career-usg", [{ type: "scatter", mode: "lines+markers", x: labels, y: career.map(r => r.USG_PCT), name: "USG%", line: { color: T.accent, width: 2.5 }, marker: { size: 8 } }], {
      title: "Usage %", xaxis: { title: { text: t("season") }, type: "category" }, yaxis: { title: { text: t("usage_axis") }, tickformat: ".0%" },
      shapes: [{ type: "line", x0: 0, x1: 1, xref: "paper", y0: 0.20, y1: 0.20, line: { dash: "dash", color: T.muted, width: 1 } }],
      annotations: [{ x: 0.99, xref: "paper", xanchor: "right", y: 0.20, yanchor: "bottom", text: t("league_avg_usg"), showarrow: false, font: { color: T.muted } }],
    });
  }

  if (career.some(r => isNum(r.TS_PCT))) {
    m.appendChild(el(chartBlock("career-ts", t("ts"), { height: 360, note: t("ts_note") })));
    plot("career-ts", [{ type: "scatter", mode: "lines+markers", x: labels, y: career.map(r => r.TS_PCT), name: "TS%", line: { color: T.info, width: 2.5 }, marker: { size: 8 } }], {
      title: "TS %", xaxis: { title: { text: t("season") }, type: "category" }, yaxis: { title: { text: t("ts_axis") }, tickformat: ".0%" },
      shapes: [{ type: "line", x0: 0, x1: 1, xref: "paper", y0: 0.58, y1: 0.58, line: { dash: "dash", color: T.muted, width: 1 } }],
      annotations: [{ x: 0.99, xref: "paper", xanchor: "right", y: 0.58, yanchor: "bottom", text: t("league_avg_ts"), showarrow: false, font: { color: T.muted } }],
    });
  }
}

// -----------------------------
// Raw Data
// -----------------------------
function renderRaw() {
  const m = main();
  const p = state.playerData;
  m.appendChild(el(`<h1>${t("raw_title", { name: pname() })}</h1>`));
  m.appendChild(el(`<p class="page-intro">${t("intro_raw", { name: pname() })}</p>`));

  const career = p.career || [];
  if (!career.length) {
    m.appendChild(el(`<div class="banner info">${t("no_raw")}</div>`));
    return;
  }
  const skip = new Set(["PLAYER_ID", "LEAGUE_ID", "TEAM_ID"]);
  const numericCols = Object.keys(career[0]).filter(k => !skip.has(k) && career.some(r => typeof r[k] === "number"));
  const defaults = numericCols.includes("PTS") ? ["PTS", "REB"] : [];

  m.appendChild(el(`<h3>${t("trend_viewer")}</h3>`));
  const controls = el(`<div class="controls-row"><div class="control">
      <label class="field-label" for="raw-metric-select">${t("select_metrics")}</label>
      <select id="raw-metric-select" multiple size="7" style="min-width:220px"></select>
    </div></div>`);
  m.appendChild(controls);
  const sel = controls.querySelector("#raw-metric-select");
  sel.innerHTML = numericCols.map(c => `<option value="${c}" ${defaults.includes(c) ? "selected" : ""}>${c}</option>`).join("");
  m.appendChild(el(chartBlock("raw-trend", "", { height: 400 })));

  const seasons = career.map(r => r.SEASON_ID);
  const drawTrend = () => {
    const chosen = Array.from(sel.selectedOptions).map(o => o.value);
    plot("raw-trend", chosen.map(c => ({ type: "scatter", mode: "lines+markers", name: c, x: seasons, y: career.map(r => r[c]) })),
      { title: t("custom_trends"), xaxis: { title: { text: "SEASON_ID" }, type: "category" }, legend: { orientation: "h", y: -0.2 } });
  };
  sel.addEventListener("change", drawTrend);
  drawTrend();

  m.appendChild(el("<hr/>"));
  m.appendChild(el(`<h3>${t("career_table")}</h3>`));
  const cols = Object.keys(career[0]).filter(c => !skip.has(c));
  const rename = { SEASON_ID: "Season", TEAM_ABBREVIATION: "TEAM", PLAYER_AGE: "AGE" };
  const intCols = new Set(["GP", "GS", "PLAYER_AGE"]);
  const textCols = new Set(["SEASON_ID", "TEAM_ABBREVIATION"]);
  const cell = (row, c) => {
    const v = row[c];
    if (v === null || v === undefined) return "—";
    if (c === "SEASON_ID") return seasonLabel(v);
    if (typeof v !== "number") return v;
    if (c.includes("PCT")) return fmtPct1(v);
    if (intCols.has(c)) return fmtInt(v);
    return Number.isInteger(v) ? v : v.toFixed(1);
  };
  m.appendChild(el(`<div class="table-wrap"><table class="data-table"><thead><tr>${cols.map(c => `<th class="${textCols.has(c) ? "" : "n"}">${rename[c] || c}</th>`).join("")}</tr></thead><tbody>
    ${career.map(row => `<tr>${cols.map(c => `<td class="${textCols.has(c) ? "" : "n"}">${cell(row, c)}</td>`).join("")}</tr>`).join("")}
  </tbody></table></div>`));
}

// -----------------------------
// League Trends (shared, not player-specific)
// -----------------------------
function renderLeague() {
  const m = main();
  const T = tokens();
  m.appendChild(el(`<h1>${t("league_title")}</h1>`));
  m.appendChild(el(`<p class="page-intro">${t("intro_league")}</p>`));

  const lt = state.shared.league_trends || {};
  m.appendChild(el(`<h3>${t("helio_title")}</h3>`));
  m.appendChild(el(`<p class="section-note">${t("helio_note")}</p>`));
  if (lt.heliocentric && lt.heliocentric.length) {
    m.appendChild(el(chartBlock("helio-chart", "", { height: 500 })));
    const h = lt.heliocentric;
    plot("helio-chart", [
      { type: "bar", x: h.map(r => r.TEAM_NAME), y: h.map(r => r.TEAM_PPG), name: t("team_ppg"), marker: { color: rgba(T.muted, 0.45) } },
      { type: "bar", x: h.map(r => r.TEAM_NAME), y: h.map(r => r.STAR_OUTPUT), name: t("star_output"), marker: { color: T.accent },
        text: h.map(r => r.STAR_NAME), textposition: "none", hovertemplate: "<b>%{text}</b><br>%{y:.1f}<extra></extra>" },
    ], { barmode: "overlay", title: t("helio_chart"), xaxis: { tickangle: -45 }, yaxis: { title: { text: t("points_prod") } }, legend: { orientation: "h", y: 1.08 } });
  } else {
    m.appendChild(el(`<div class="banner info">${t("no_league")}</div>`));
  }

  m.appendChild(el("<hr/>"));
  m.appendChild(el(`<h3>${t("leaderboards")}</h3>`));
  const cols = el(`<div class="three-col"></div>`);
  m.appendChild(cols);

  function tableCol(title, caption, rows, columns) {
    if (!rows || !rows.length) return `<div><h4>${title}</h4><p class="section-note">${caption}</p><p class="muted">${t("no_data")}</p></div>`;
    return `<div><h4>${title}</h4><p class="section-note">${caption}</p>
      <div class="table-wrap"><table class="data-table"><thead><tr><th class="n">${t("col_rank")}</th>${columns.map(c => `<th class="${c[2] ? "n" : ""}">${c[1]}</th>`).join("")}</tr></thead><tbody>
      ${rows.slice(0, 10).map((r, i) => `<tr><td class="n">${i + 1}</td>${columns.map(c => `<td class="${c[2] ? "n" : ""}">${c[2] ? c[2](r[c[0]]) : (r[c[0]] ?? "—")}</td>`).join("")}</tr>`).join("")}
      </tbody></table></div></div>`;
  }
  cols.innerHTML =
    tableCol(t("snipers"), t("snipers_note"), lt.passing, [["PLAYER_NAME", t("col_player")], ["TEAM_ABBREVIATION", t("col_team")], ["AST_3P", t("col_ast3"), fmt1], ["AST", t("col_ast"), fmtInt]]) +
    tableCol(t("magnets"), t("magnets_note"), lt.misc, [["PLAYER_NAME", t("col_player")], ["TEAM_ABBREVIATION", t("col_team")], ["PFD", t("col_pfd"), fmtInt]]) +
    tableCol(t("rim"), t("rim_note"), lt.drives, [["PLAYER_NAME", t("col_player")], ["TEAM_ABBREVIATION", t("col_team")], ["DRIVES", t("col_drives"), fmt1], ["DRIVE_PTS", t("col_drive_pts"), fmt1]]);
}

// -----------------------------
// Shot Maps
// -----------------------------
function courtShapes(T) {
  const c = T.court;
  const hoop = T.style === "a" ? T.accent : "#ec7607";
  return [
    { type: "rect", x0: -250, y0: -47.5, x1: 250, y1: 422.5, line: { color: c, width: 2 } },
    { type: "rect", x0: -80, y0: -47.5, x1: 80, y1: 142.5, line: { color: c, width: 2 } },
    { type: "rect", x0: -60, y0: -47.5, x1: 60, y1: 142.5, line: { color: c, width: 2 } },
    { type: "circle", x0: -7.5, y0: -7.5, x1: 7.5, y1: 7.5, line: { color: hoop, width: 2 } },
    { type: "line", x0: -30, y0: -40, x1: 30, y1: -40, line: { color: c, width: 2 } },
    { type: "line", x0: 0, y0: -40, x1: 0, y1: -7.5, line: { color: hoop, width: 2 } },
    { type: "line", x0: -220, y0: -47.5, x1: -220, y1: 92.5, line: { color: c, width: 2 } },
    { type: "line", x0: 220, y0: -47.5, x1: 220, y1: 92.5, line: { color: c, width: 2 } },
  ];
}
function courtArcTraces(T) {
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
    const a = Math.PI * (i / 49);
    ccX.push(60 * Math.cos(a)); ccY.push(422.5 + 60 * Math.sin(a));
  }
  return [
    { type: "scatter", mode: "lines", x: arcX, y: arcY, line: { color: T.court, width: 2 }, showlegend: false, hoverinfo: "skip" },
    { type: "scatter", mode: "lines", x: ccX, y: ccY, line: { color: T.court, width: 2 }, showlegend: false, hoverinfo: "skip" },
  ];
}
function courtLayout(T, extra) {
  return Object.assign({
    xaxis: { range: [-250, 250], showgrid: false, zeroline: false, visible: false, fixedrange: true },
    yaxis: { range: [-47.5, 422.5], scaleanchor: "x", scaleratio: 1, showgrid: false, zeroline: false, visible: false, fixedrange: true },
    shapes: courtShapes(T),
    margin: { l: 4, r: 4, t: 44, b: 4 },
    plot_bgcolor: T.pitch,
  }, extra || {});
}

function drawShotChart(divId, shotData, season) {
  const T = tokens();
  const traces = courtArcTraces(T);
  const has = shotData && shotData.points && shotData.points.length;
  if (has) {
    const made = shotData.points.filter(p => p.made);
    const missed = shotData.points.filter(p => !p.made);
    traces.push(
      { type: "scatter", mode: "markers", x: made.map(p => p.x), y: made.map(p => p.y), name: `${t("made")} (${made.length})`, marker: { color: T.accent, size: 6, opacity: 0.75, line: { width: 0 } }, hoverinfo: "skip" },
      { type: "scatter", mode: "markers", x: missed.map(p => p.x), y: missed.map(p => p.y), name: `${t("missed")} (${missed.length})`, marker: { color: T.neg, size: 6, opacity: 0.7, symbol: "x" }, hoverinfo: "skip" },
    );
  }
  const title = has ? t("shot_title", { season }) : t("no_data_title", { season });
  return plot(divId, traces, courtLayout(T, { title: { text: title, x: 0.5, xanchor: "center" }, legend: { orientation: "h", y: 1.0, yanchor: "bottom", x: 1, xanchor: "right", font: { color: T.muted } } }));
}

function drawZoneMap(divId, shotData, season) {
  const T = tokens();
  const byName = {};
  ((shotData && shotData.zones) || []).forEach(z => { byName[z.name] = z; });
  const traces = courtArcTraces(T);
  (state.zones || []).forEach(z => {
    const st = byName[z.name];
    let fill = "rgba(220,220,220,0.12)", hover = z.name, centroidText = null;
    if (st && st.fga > 0) {
      const pct = st.pct;
      fill = pct < 0.35 ? "rgba(69,117,180,0.85)" : pct < 0.45 ? "rgba(255,255,191,0.85)" : "rgba(215,48,39,0.85)";
      hover += `<br>${st.fgm}/${st.fga} (${(pct * 100).toFixed(1)}%)`;
      centroidText = `<b>${st.fgm}/${st.fga}</b><br><b>${(pct * 100).toFixed(1)}%</b>`;
    }
    traces.push({ type: "scatter", x: z.x, y: z.y, fill: "toself", mode: "lines", line: { color: "rgba(255,255,255,0.9)", width: 1.6 }, fillcolor: fill, hoverinfo: "text", text: hover, showlegend: false });
    if (centroidText) {
      const cx = z.x.reduce((a, b) => a + b, 0) / z.x.length;
      const cy = z.y.reduce((a, b) => a + b, 0) / z.y.length;
      traces.push({ type: "scatter", x: [cx], y: [cy], mode: "text", text: [centroidText], textfont: { size: 10, color: "#0a0b0e", family: T.font }, showlegend: false, hoverinfo: "skip" });
    }
  });
  return plot(divId, traces, courtLayout(T, { title: { text: t("zone_title", { season }), x: 0.5, xanchor: "center" } }));
}

function renderShots() {
  const m = main();
  const p = state.playerData;
  m.appendChild(el(`<h1>${t("shots_title", { name: pname() })}</h1>`));
  m.appendChild(el(`<p class="page-intro">${t("intro_shots", { name: pname() })}</p>`));

  const seasons = Object.keys(p.shot_charts || {}).sort().reverse();
  if (!seasons.length) {
    m.appendChild(el(`<div class="banner info">${t("no_shots", { name: pname() })}</div>`));
    return;
  }
  if (!state.shotSeason || !seasons.includes(state.shotSeason)) state.shotSeason = seasons[0];
  if (!state.shotSeasonB || !seasons.includes(state.shotSeasonB)) state.shotSeasonB = seasons[Math.min(1, seasons.length - 1)];
  const opts = cur => seasons.map(s => `<option value="${s}" ${s === cur ? "selected" : ""}>${s}</option>`).join("");

  const controls = el(`<div class="controls-row">
    <div class="control"><label class="checkbox-row"><input type="checkbox" id="shot-compare" ${state.shotCompare ? "checked" : ""} ${seasons.length < 2 ? "disabled" : ""}/> ${t("compare_mode")}</label></div>
    <div class="control"><label class="field-label" for="shot-season-a">${t("season_a")}</label><select id="shot-season-a">${opts(state.shotSeason)}</select></div>
    <div class="control" id="shot-season-b-wrap" style="display:${state.shotCompare ? "flex" : "none"}"><label class="field-label" for="shot-season-b">${t("season_b")}</label><select id="shot-season-b">${opts(state.shotSeasonB)}</select></div>
    <div class="control"><span class="field-label">${t("map_style")}</span>
      <div class="tabs" style="margin:0" role="group">
        <button type="button" data-view="chart" aria-pressed="${state.shotView === "chart"}"><span>${t("shot_chart")}</span></button>
        <button type="button" data-view="zone" aria-pressed="${state.shotView === "zone"}"><span>${t("zone_eff")}</span></button>
      </div>
    </div>
    <div class="control"><button type="button" class="btn" id="shot-download">${t("download_png")}</button></div>
  </div>`);
  m.appendChild(controls);

  const viewWrap = el(`<div class="shot-grid"></div>`);
  m.appendChild(viewWrap);

  function draw(id, season) {
    const sc = p.shot_charts[season];
    return state.shotView === "chart" ? drawShotChart(id, sc, season) : drawZoneMap(id, sc, season);
  }
  function redraw() {
    Array.from(viewWrap.querySelectorAll(".js-plotly-plot")).forEach(d => Plotly.purge(d));
    viewWrap.innerHTML = "";
    const ids = state.shotCompare ? ["shot-plot-a", "shot-plot-b"] : ["shot-plot-a"];
    ids.forEach(id => viewWrap.appendChild(el(`<div class="chart-panel bracket"><div class="plot court-plot" id="${id}"></div></div>`)));
    draw("shot-plot-a", state.shotSeason);
    if (state.shotCompare) draw("shot-plot-b", state.shotSeasonB);
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
    controls.querySelectorAll(".tabs button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
    redraw();
  }));
  controls.querySelector("#shot-download").addEventListener("click", () => {
    const view = state.shotView === "chart" ? "Shot_Chart" : "14-Zone_Efficiency";
    Plotly.downloadImage("shot-plot-a", { format: "png", width: 1000, height: 1040, scale: 2, filename: `${state.playerKey}_${state.shotSeason}_${view}` });
  });

  redraw();
}

// -----------------------------
// Deep Dive
// -----------------------------
function weightedAvg(rows, key) {
  const totalGP = sum(rows, "GP");
  return totalGP ? rows.reduce((a, r) => a + r[key] * r.GP, 0) / totalGP : 0;
}

function allstarThresholdChart(divId, playerName, playerStats, allstarStats) {
  const T = tokens();
  const avg = ["PTS", "REB", "AST"].map(k => weightedAvg(allstarStats, k));
  const bottom4 = allstarStats.slice().sort((a, b) => a.PTS - b.PTS).slice(0, 4);
  const cats = ["PTS", "REB", "AST"];
  const pVals = [playerStats.PTS, playerStats.REB, playerStats.AST];
  const traces = [
    { type: "bar", name: displayName(playerName), x: cats, y: pVals, marker: { color: T.accent }, text: pVals.map(v => v.toFixed(1)), textposition: "outside", cliponaxis: false },
    { type: "bar", name: t("allstar_avg"), x: cats, y: avg, marker: { color: T.info }, text: avg.map(v => v.toFixed(1)), textposition: "outside", cliponaxis: false },
  ];
  bottom4.forEach(row => traces.push({ type: "bar", name: t("entry", { name: row.PLAYER_NAME }), x: cats, y: [row.PTS, row.REB, row.AST], opacity: 0.4, marker: { color: T.muted } }));
  plot(divId, traces, { title: t("thresh_chart"), barmode: "group", yaxis: { title: { text: t("per_game_axis") } }, legend: { font: { size: 11, color: T.muted } } });
}

function percentileBelow(rows, key, value) {
  if (!rows.length) return 0;
  return Math.floor((rows.filter(r => r[key] < value).length / rows.length) * 100);
}

function renderVerdict(container, playerName, playerStats, allstarStats) {
  const first = firstName(displayName(playerName));
  const ptsP = percentileBelow(allstarStats, "PTS", playerStats.PTS);
  const rebP = percentileBelow(allstarStats, "REB", playerStats.REB);
  const astP = percentileBelow(allstarStats, "AST", playerStats.AST);
  const avgP = (ptsP + rebP + astP) / 3;
  let cls = "", key = "v_star";
  if (avgP > 50) key = "v_star";
  else if (avgP > 30) { cls = "warn"; key = "v_border"; }
  else { cls = "info"; key = "v_dev"; }
  container.innerHTML = `
    <h3>${t("verdict")}</h3>
    <div class="metric-row">
      ${kpiCard(t("pct_scoring"), ordinal(ptsP))}
      ${kpiCard(t("pct_reb"), ordinal(rebP))}
      ${kpiCard(t("pct_play"), ordinal(astP))}
    </div>
    <p class="section-note">${t("verdict_note", { first, p: ordinal(ptsP), n: ptsP })}</p>
    <div class="verdict-box ${cls}">${t(key, { first, p: ordinal(avgP) })}</div>`;
}

function tripleThreatChart(divId, playerName, playerStats, allstarStats) {
  const T = tokens();
  const SIZE_FACTOR = 0.16;
  const ptsList = allstarStats.map(r => r.PTS).concat([playerStats.PTS]);
  const astList = allstarStats.map(r => r.AST).concat([playerStats.AST]);
  const rebList = allstarStats.map(r => r.REB).concat([playerStats.REB]);
  const namesList = allstarStats.map(r => r.PLAYER_NAME).concat([displayName(playerName)]);

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

  plot(divId, [
    {
      type: "scatter", mode: "markers", x: ptsList, y: astList, text: namesList, customdata: rebList,
      // faint dots under the headshots, so the chart still reads if an image fails to load
      marker: { size: rebList.map(r => r * 4), color: namesList.map((n, i) => i === namesList.length - 1 ? rgba(T.accent, 0.5) : rgba(T.muted, 0.25)), line: { width: 0 } },
      hovertemplate: "<b>%{text}</b><br>PTS: %{x:.1f}<br>AST: %{y:.1f}<br>REB: %{customdata:.1f}<extra></extra>", showlegend: false,
    },
    { type: "scatter", mode: "text", x: [playerStats.PTS], y: [playerStats.AST - playerSize * 0.6], text: [firstName(displayName(playerName))], textposition: "bottom center", textfont: { size: 14, color: T.accent, family: T.font }, showlegend: false, hoverinfo: "skip" },
  ], {
    title: t("triple_chart"),
    xaxis: { title: { text: t("ppg_axis") }, range: [Math.min(...ptsList) - 2, Math.max(...ptsList) + 2] },
    yaxis: { title: { text: t("apg_axis") }, range: [Math.min(...astList) - 1, Math.max(...astList) + 1] },
    images, showlegend: false,
  });
}

function separationChart(divId, playerName, playerStats, allstarDetailed) {
  const T = tokens();
  const others = allstarDetailed.filter(r => r.PLAYER_NAME !== playerName && isNum(r.USG_PCT) && isNum(r.TS_PCT));
  const images = [];
  others.forEach(row => {
    const url = allstarHeadshot(row.PLAYER_NAME);
    if (url) images.push({ source: url, xref: "x", yref: "y", x: row.USG_PCT * 100, y: row.TS_PCT * 100, sizex: 1.5, sizey: 1.5, xanchor: "center", yanchor: "middle", layer: "above" });
  });
  const dx = playerStats.USG_PCT * 100, dy = playerStats.TS_PCT * 100;
  const playerUrl = allstarHeadshot(playerName) || state.playerData.headshot;
  if (playerUrl) images.push({ source: playerUrl, xref: "x", yref: "y", x: dx, y: dy, sizex: 1.5, sizey: 1.5, xanchor: "center", yanchor: "middle", layer: "above" });

  plot(divId, [
    { type: "scatter", mode: "text", x: [dx], y: [dy], text: [firstName(displayName(playerName))], textposition: "top center", textfont: { size: 14, color: T.accent, family: T.font }, showlegend: false, hoverinfo: "skip" },
    { type: "scatter", mode: "markers", x: others.map(r => r.USG_PCT * 100), y: others.map(r => r.TS_PCT * 100), text: others.map(r => r.PLAYER_NAME), marker: { color: rgba(T.muted, 0.25), size: 22, line: { width: 0 } }, hoverinfo: "text+x+y", showlegend: false, name: "All-Stars" },
  ], { xaxis: { title: { text: t("usg_axis") }, range: [18, 40] }, yaxis: { title: { text: t("ts_axis_full") }, range: [48, 70] }, images, showlegend: false, margin: { t: 24, l: 56, r: 16, b: 52 } });
}

function renderFullTable(container, tabId, playerName, playerStats, gp, allstarStats) {
  const metrics = ["PTS", "REB", "AST", "STL", "BLK", "TOV"];
  container.innerHTML = `
    <div class="control" style="margin-bottom:12px"><label class="field-label" for="rank-select-${tabId}">${t("rank_by")}</label>
      <select id="rank-select-${tabId}">${metrics.map(m => `<option value="${m}">${m}</option>`).join("")}</select>
    </div>
    <div id="table-wrap-${tabId}"></div>`;
  const sel = container.querySelector(`#rank-select-${tabId}`);
  const wrap = container.querySelector(`#table-wrap-${tabId}`);
  function draw() {
    const metric = sel.value;
    const rows = allstarStats.map(r => ({ PLAYER_NAME: r.PLAYER_NAME, GP: r.GP, PTS: r.PTS, REB: r.REB, AST: r.AST, STL: r.STL, BLK: r.BLK, TOV: r.TOV, me: false }));
    rows.splice(0, rows.length, ...rows.filter(r => r.PLAYER_NAME !== playerName));
    rows.push(Object.assign({ PLAYER_NAME: displayName(playerName), GP: gp, me: true }, playerStats));
    rows.sort((a, b) => b[metric] - a[metric]);
    wrap.innerHTML = `<div class="table-wrap"><table class="data-table"><thead><tr><th class="n">${t("col_rank")}</th><th>${t("col_player")}</th>${metrics.map(m => `<th class="n">${m}</th>`).join("")}<th class="n">GP</th></tr></thead><tbody>
      ${rows.map((r, i) => `<tr class="${r.me ? "highlight" : ""}"><td class="n">${i + 1}</td><td>${r.PLAYER_NAME}</td>${metrics.map(m => `<td class="n">${fmt1(r[m])}</td>`).join("")}<td class="n">${r.GP}</td></tr>`).join("")}
    </tbody></table></div>`;
  }
  sel.addEventListener("change", draw);
  draw();
}

function radarChart(divId, playerName, playerStats, allstarStats) {
  const T = tokens();
  const metrics = ["PTS", "REB", "AST", "STL", "BLK"];
  const dVals = metrics.map(m => playerStats[m] || 0);
  const avgVals = metrics.map(m => mean(allstarStats, m));
  const maxVals = metrics.map((m, i) => Math.max(Math.max(...allstarStats.map(r => r[m])), dVals[i]) || 1);
  const dNorm = dVals.map((v, i) => v / maxVals[i]);
  const aNorm = avgVals.map((v, i) => v / maxVals[i]);
  const theta = metrics.concat([metrics[0]]);
  plot(divId, [
    { type: "scatterpolar", r: aNorm.concat([aNorm[0]]), theta, fill: "toself", name: t("allstar_avg"), line: { color: T.info, width: 2 }, fillcolor: rgba(T.info, 0.16) },
    { type: "scatterpolar", r: dNorm.concat([dNorm[0]]), theta, fill: "toself", name: displayName(playerName), line: { color: T.accent, width: 3 }, fillcolor: rgba(T.accent, 0.24) },
  ], {
    polar: { radialaxis: { visible: false, range: [0, 1] }, angularaxis: { color: T.muted, gridcolor: T.line, linecolor: T.lineStrong }, bgcolor: "rgba(0,0,0,0)" },
    title: t("radar_chart"), legend: { orientation: "h", y: -0.08 },
  });
}

function offensiveEngineChart(divId, playerName, playerStats, allstarStats) {
  const T = tokens();
  const df = allstarStats.map(r => ({ PLAYER_NAME: r.PLAYER_NAME, PTS: r.PTS, PTS_CREATED: r.AST * 2.3, TOTAL_OUTPUT: r.PTS + r.AST * 2.3 }));
  if (!df.some(r => r.PLAYER_NAME === playerName)) {
    df.push({ PLAYER_NAME: playerName, PTS: playerStats.PTS, PTS_CREATED: playerStats.AST * 2.3, TOTAL_OUTPUT: playerStats.PTS + playerStats.AST * 2.3 });
  }
  df.sort((a, b) => b.TOTAL_OUTPUT - a.TOTAL_OUTPUT);
  let top15 = df.slice(0, 15);
  if (!top15.some(r => r.PLAYER_NAME === playerName)) {
    top15 = top15.concat([df.find(r => r.PLAYER_NAME === playerName)]).sort((a, b) => b.TOTAL_OUTPUT - a.TOTAL_OUTPUT);
  }
  const names = top15.map(r => r.PLAYER_NAME);
  const labels = names.map(displayName);
  plot(divId, [
    { type: "bar", name: t("pts_scored"), x: labels, y: top15.map(r => r.PTS), marker: { color: names.map(n => n === playerName ? T.accent : rgba(T.muted, 0.7)) } },
    { type: "bar", name: t("pts_created"), x: labels, y: top15.map(r => r.PTS_CREATED), marker: { color: names.map(n => n === playerName ? T.info : rgba(T.info, 0.35)) } },
  ], { barmode: "stack", title: t("engine_chart"), xaxis: { tickangle: -45 }, yaxis: { title: { text: t("total_prod") } }, legend: { orientation: "h", y: 1.1 } });
}

function whatIfChart(divId) {
  const T = tokens();
  const deni = [25.6, 7.2, 7.0], deniUsage = 28.0;
  const luka = [33.6, 8.1, 8.7], lukaUsage = 37.9;
  const ratio = lukaUsage / deniUsage;
  const cats = [t("cat_points"), t("cat_reb"), t("cat_ast")];
  const proj = deni.map(v => v * ratio);
  plot(divId, [
    { type: "bar", name: t("wi_actual"), x: cats, y: deni, marker: { color: rgba(T.muted, 0.75) }, text: deni.map(String), textposition: "auto" },
    { type: "bar", name: t("wi_proj", { u: lukaUsage }), x: cats, y: proj, marker: { color: T.accent }, text: proj.map(v => v.toFixed(1)), textposition: "auto", textfont: { color: T.accentInk } },
    { type: "bar", name: t("wi_luka"), x: cats, y: luka, marker: { color: T.info }, text: luka.map(String), textposition: "auto", textfont: { color: "#0a0b0e" } },
  ], { barmode: "group", title: t("whatif_chart"), yaxis: { title: { text: t("per_game_stats") } }, legend: { orientation: "h", y: 1.12 }, margin: { t: 80, l: 52, r: 24, b: 40 } });
}

function renderFtLeaders(m, playerName) {
  const ft = state.shared.league_ft || {};
  if (!ft.leaders || !ft.leaders.length) return;
  m.appendChild(el("<hr/>"));
  m.appendChild(el(`<h3>${t("ft_title", { season: ltr(ft.season) })}</h3>`));
  let top10 = ft.leaders.slice().sort((a, b) => b.FTM - a.FTM).slice(0, 10);
  if (!top10.some(r => r.PLAYER_NAME === playerName)) {
    const own = ft.leaders.find(r => r.PLAYER_NAME === playerName);
    if (own) top10 = top10.concat([own]);
  }
  const controls = el(`<div class="control" style="margin-bottom:14px"><label class="field-label" for="ft-sort">${t("sort_by")}</label>
    <select id="ft-sort"><option value="FTA">${t("ft_att")}</option><option value="FTM">${t("ft_made")}</option><option value="FT_PCT">${t("ft_pct")}</option></select></div>`);
  m.appendChild(controls);
  const wrap = el(`<div id="ft-table"></div>`);
  m.appendChild(wrap);
  function draw() {
    const key = controls.querySelector("select").value;
    const rows = top10.slice().sort((a, b) => b[key] - a[key]);
    wrap.innerHTML = `<div class="table-wrap"><table class="data-table"><thead><tr><th class="n">${t("col_rank")}</th><th>${t("col_player")}</th><th>${t("col_team")}</th><th class="n">${t("col_games")}</th><th class="n">${t("col_made")}</th><th class="n">${t("col_att")}</th><th class="n">FT%</th></tr></thead><tbody>
      ${rows.map((r, i) => `<tr class="${r.PLAYER_NAME === playerName ? "highlight" : ""}"><td class="n">${i + 1}</td><td>${displayName(r.PLAYER_NAME)}</td><td>${r.TEAM_ABBREVIATION}</td><td class="n">${r.GP}</td><td class="n">${r.FTM}</td><td class="n">${r.FTA}</td><td class="n">${fmtPct1(r.FT_PCT)}</td></tr>`).join("")}
    </tbody></table></div>`;
  }
  controls.querySelector("select").addEventListener("change", draw);
  draw();
}

function renderDeepDive() {
  const m = main();
  const p = state.playerData;
  m.appendChild(el(`<h1>${t("deep_title", { name: pname() })}</h1>`));
  m.appendChild(el(`<p class="page-intro">${t("intro_deep", { name: pname() })}</p>`));
  m.appendChild(el(`<details class="scouting"><summary>${t("scouting_summary", { name: pname() })}</summary><div class="body">${block("scouting_" + state.playerKey) || block("scouting_default")}</div></details>`));

  renderFtLeaders(m, p.player_name);

  const logs = (p.game_logs || {})[state.meta.current_season] || [];
  if (!logs.length) {
    m.appendChild(el("<hr/>"));
    m.appendChild(el(`<div class="banner info">${t("no_games_deep", { name: pname() })}</div>`));
    return;
  }
  const playerStats = { PTS: mean(logs, "PTS"), REB: mean(logs, "REB"), AST: mean(logs, "AST"), STL: mean(logs, "STL"), BLK: mean(logs, "BLK"), TOV: mean(logs, "TOV") };
  const careerCur = (p.career || []).find(r => r.SEASON_ID === state.meta.current_season);
  if (careerCur) { playerStats.USG_PCT = careerCur.USG_PCT; playerStats.TS_PCT = careerCur.TS_PCT; }

  const bench = state.shared.allstar_bench, race = state.shared.allstar_race;
  m.appendChild(el("<hr/>"));
  const tabs = el(`<div class="tabs underline" role="group">
    <button type="button" data-tab="bench" aria-pressed="${state.deepTab === "bench"}"><span>${t("tab_bench", { season: seasonLabel(bench.season) })}</span></button>
    <button type="button" data-tab="race" aria-pressed="${state.deepTab === "race"}"><span>${t("tab_race", { season: seasonLabel(race.season) })}</span></button>
  </div>`);
  m.appendChild(tabs);
  const body = el(`<div id="deepdive-body"></div>`);
  m.appendChild(body);

  function drawTab(tabName) {
    const isRace = tabName === "race";
    const cohort = isRace ? race : bench;
    const sl = ltr(seasonLabel(cohort.season));
    // Like app.py: the cohort keeps the player's own row (he can be an All-Star himself) for the
    // threshold, verdict and case-study charts; only the table and separation chart drop it.
    const stats = cohort.stats || [];
    const detailed = cohort.detailed || [];
    Array.from(body.querySelectorAll(".js-plotly-plot")).forEach(d => Plotly.purge(d));
    body.innerHTML = "";

    if (!stats.length) {
      body.appendChild(el(`<div class="banner warn">${t("no_allstar", { season: ltr(cohort.season) })}</div>`));
      return;
    }
    body.appendChild(el(`<p class="section-note">${t(isRace ? "comparing_race" : "comparing_bench", { name: pname(), season: sl })}</p>`));

    body.appendChild(el(`<h3>${t(isRace ? "s_thresh_race" : "s_thresh_bench")}</h3>`));
    const row1 = el(`<div class="two-col"><div class="chart-panel bracket"><div class="plot" id="thresh-${tabName}" style="height:420px"></div></div><div id="verdict-${tabName}"></div></div>`);
    body.appendChild(row1);
    allstarThresholdChart(`thresh-${tabName}`, p.player_name, playerStats, stats);
    renderVerdict(row1.querySelector(`#verdict-${tabName}`), p.player_name, playerStats, stats);

    body.appendChild(el("<hr/>"));
    body.appendChild(el(chartBlock(`triple-${tabName}`, isRace ? t("s_triple_race", { season: sl }) : t("s_triple"), { height: 640 })));
    tripleThreatChart(`triple-${tabName}`, p.player_name, playerStats, stats);

    if (detailed.length && isNum(playerStats.USG_PCT) && isNum(playerStats.TS_PCT)) {
      body.appendChild(el("<hr/>"));
      body.appendChild(el(chartBlock(`sep-${tabName}`, isRace ? t("s_sep_race", { season: sl }) : t("s_sep"), { cls: "plot square-plot" })));
      separationChart(`sep-${tabName}`, p.player_name, playerStats, detailed);
    }

    body.appendChild(el("<hr/>"));
    body.appendChild(el(`<h3>${isRace ? t("s_table_race", { season: sl }) : t("s_table")}</h3>`));
    const tableWrap = el(`<div></div>`);
    body.appendChild(tableWrap);
    renderFullTable(tableWrap, tabName, p.player_name, playerStats, logs.length, stats);

    body.appendChild(el("<hr/>"));
    body.appendChild(el(`<h3>${isRace ? t("s_cases_race", { season: sl }) : t("s_cases")}</h3>`));
    const advRow = el(`<div class="two-even"><div class="chart-panel bracket"><div class="plot" id="radar-${tabName}" style="height:480px"></div></div><div class="chart-panel bracket"><div class="plot" id="engine-${tabName}" style="height:480px"></div></div></div>`);
    body.appendChild(advRow);
    radarChart(`radar-${tabName}`, p.player_name, playerStats, stats);
    offensiveEngineChart(`engine-${tabName}`, p.player_name, playerStats, stats);

    if (isRace && state.playerKey === "deni_avdija") {
      body.appendChild(el("<hr/>"));
      body.appendChild(el(chartBlock(`whatif-${tabName}`, t("s_whatif"), { height: 480, note: t("whatif_note") })));
      whatIfChart(`whatif-${tabName}`);
    }
  }

  tabs.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-tab]");
    if (!btn) return;
    state.deepTab = btn.dataset.tab;
    tabs.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b === btn)));
    drawTab(state.deepTab);
  });
  drawTab(state.deepTab);
}

// -----------------------------
// About the project / About me
// -----------------------------
function renderProject() {
  const m = main();
  m.appendChild(el(`<h1>${t("page_project")}</h1>`));
  m.appendChild(el(`<div class="prose">${block("project")}</div>`));
}

function renderAbout() {
  const m = main();
  m.appendChild(el(`<h1>${t("about_title")}</h1>`));
  m.appendChild(el(`<div>${block("about")}</div>`));
}

boot();
