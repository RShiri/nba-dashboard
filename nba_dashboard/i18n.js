/* English / Hebrew strings for the static dashboard. {name}-style placeholders are filled by t(). */
const I18N = {
  en: {
    skip: "Skip to content",
    brand_a: "Israeli NBA",
    brand_b: "Analytics",
    style: "Style",
    player: "Player",
    page: "Page",
    source: "View source on GitHub",
    loading: "Loading…",
    load_error: "The dashboard data could not be loaded. Refresh the page to try again.",
    data_updated: "Data updated",
    games_tracked: "{name} games this season",

    page_dashboard: "Dashboard",
    page_career: "Career Analysis",
    page_league: "League Trends",
    page_raw: "Raw Data",
    page_shots: "Shot Maps",
    page_deepdive: "Research: Deep Dive",
    page_project: "About the Project",
    page_about: "About Me",

    // dashboard
    hero_sub: "360° performance analytics · {season} season",
    intro_dashboard: "Season at a glance for {name}: per-game averages with the change against last season, then every game of the last three seasons. Bars are points (accent colour for a win, red for a loss), the line is minutes played.",
    no_games_yet: "{name} hasn't logged an NBA game yet. Check back once the season tips off.",
    no_games_season: "{name} hasn't played a game in the {season} season yet, so the season cards will appear after the first box score. The charts below cover earlier seasons.",
    deni_banner: "<b>#1 in NBA free throw attempts</b> (250+ FTA)",
    kpi_pts: "Points / G", kpi_reb: "Rebounds / G", kpi_ast: "Assists / G", kpi_min: "Minutes / G",
    kpi_fg: "Field goal %", kpi_3p: "3-point %", kpi_ft: "Free throw %", kpi_gp: "Games played",
    kpi_caption: "▲▼ deltas compare the current season to the prior full season (per-game rates; shooting in percentage points).",
    impact: "{season} impact",
    points: "Points", minutes: "Minutes", pts_win: "PTS (win)", pts_loss: "PTS (loss)", avg_min: "Avg MIN ({v})", avg_pts: "Avg PTS: {v}",

    // career
    career_title: "{name} — Career Trajectory Analysis",
    intro_career: "How {name}'s game has grown season by season. Per-game numbers show raw output; per-36 numbers remove the effect of minutes, so a bench year and a starter year can be compared fairly. Usage and true shooting show how big the role is and how efficiently he scores in it.",
    no_career: "Not enough career data yet to chart a trajectory.",
    per_game: "Per game stats", per36: "Per 36 minutes",
    usage: "Usage rate", usage_note: "Share of team plays used by the player while on the floor. High (>30%): primary scorers such as Luka or Giannis. Low (<15%): role players.",
    usage_axis: "Usage %", league_avg_usg: "League avg (20%)",
    ts: "True shooting %", ts_note: "Scoring efficiency that counts a three as 1.5 shots' worth and includes free throws. Elite (>60%): Curry, Jokić. Average about 58%. Poor below 52%.",
    ts_axis: "TS %", league_avg_ts: "League avg (58%)", season: "Season",

    // league
    league_title: "League Trends & Advanced Metrics",
    intro_league: "League-wide context that isn't about one player: how much each team leans on its star, and who leads the NBA in creating threes, drawing fouls and attacking the rim.",
    helio_title: "1. Heliocentric offenses: team vs. star output",
    helio_note: "How much of a team's total offense comes from its biggest star? (points + created points)",
    helio_chart: "Offensive dependence (sorted by team offense)",
    team_ppg: "Team avg PPG", star_output: "Star player total output", points_prod: "Points production",
    no_league: "League trend data isn't in the current data snapshot yet. It fills in after the next daily refresh that reaches the NBA tracking endpoints.",
    leaderboards: "2. Specialized leaderboards",
    snipers: "Sniper finders", snipers_note: "Best at creating 3-point looks.",
    magnets: "Foul magnets", magnets_note: "Most fouls drawn (PFD).",
    rim: "Rim pressure", rim_note: "Most drives per game.",
    no_data: "No data",
    col_rank: "#", col_player: "Player", col_team: "Team", col_ast3: "Assists to 3P", col_ast: "Total AST",
    col_pfd: "Fouls drawn", col_drives: "Drives/G", col_drive_pts: "Drive PTS",

    // raw
    raw_title: "{name} — Raw Data & Custom Trends",
    intro_raw: "Every column the NBA stats API returns for {name}'s career, plus the advanced numbers merged in. Pick any metrics to plot them together over time (hold Ctrl or ⌘ to select several), or read the table below.",
    no_raw: "No career data yet.",
    trend_viewer: "Interactive trend viewer", select_metrics: "Select metrics", custom_trends: "Custom trends",
    career_table: "Career data table",

    // shots
    shots_title: "{name} — Shot Analysis",
    intro_shots: "Every field-goal attempt {name} took, plotted where it happened on a half court. Switch to the 14-zone view to see shooting percentage by area (blue below 35%, yellow 35–45%, red 45% and up), or turn on compare mode to put two seasons side by side.",
    no_shots: "No shot chart data available yet for {name}.",
    compare_mode: "Compare mode", season_a: "Season A", season_b: "Season B", map_style: "Map style",
    shot_chart: "Shot chart", zone_eff: "14-zone efficiency", made: "Made", missed: "Missed",
    shot_title: "{season} shot chart", zone_title: "{season} zone efficiency", no_data_title: "{season} - no data",
    download_png: "Download PNG",

    // deep dive
    deep_title: "{name} — All-Star Comparison",
    intro_deep: "A research lab that measures {name}'s current season against real All-Stars. The Benchmark tab uses the final 2024-25 numbers of that season's All-Stars, a fixed bar that never moves. The Race tab uses the same players' numbers this season, so it shows where he stands right now.",
    scouting_summary: "Read: scouting report & analysis — {name}",
    ft_title: "{season} season: free throw leaders (top 10)",
    sort_by: "Sort leaderboard by", ft_att: "Total attempts", ft_made: "Total made", ft_pct: "FT%",
    col_games: "Games", col_made: "Total made", col_att: "Total attempts",
    no_games_deep: "{name} hasn't played an NBA game yet this season, so the All-Star comparison lab activates once box scores are logged.",
    tab_bench: "Benchmark ({season} All-Stars)", tab_race: "The Race ({season} All-Star stats)",
    no_allstar: "No {season} All-Star data found yet.",
    comparing_bench: "Comparing {name}'s <b>current</b> stats against the <b>final</b> stats of the {season} All-Stars.",
    comparing_race: "Comparing {name}'s <b>current</b> stats against the <b>current ({season})</b> stats of the same All-Star cohort.",
    s_thresh_bench: "1. The All-Star threshold", s_thresh_race: "1. The race threshold",
    s_triple: "2. The triple threat", s_triple_race: "2. The triple threat ({season})",
    s_sep: "3. Separation chart (usage vs efficiency)", s_sep_race: "3. Separation chart ({season} performance)",
    s_table: "4. Full league comparison table", s_table_race: "4. {season} leaderboard",
    s_cases: "5. Advanced case studies", s_cases_race: "5. Advanced case studies ({season})",
    s_whatif: "6. What-if: Deni vs Luka efficiency", whatif_note: "How would Deni compare if he had Luka Dončić's usage rate (37.9%)?",
    thresh_chart: "The All-Star threshold", per_game_axis: "Per game", allstar_avg: "All-Star avg", entry: "{name} (entry)",
    verdict: "The analytical verdict", pct_scoring: "Scoring percentile", pct_reb: "Rebounding percentile", pct_play: "Playmaking percentile",
    verdict_note: "Compares {first} against the selected All-Star roster. Being in the {p} percentile means he outscores {n}% of the NBA's elite.",
    v_star: "<b>All-Star caliber</b>: {first} ranks in the top half of All-Stars ({p} percentile avg).",
    v_border: "<b>Borderline</b>: {first} is competitive with lower-tier All-Stars ({p} percentile avg).",
    v_dev: "<b>Developing</b>: {first} shows flashes but trails the All-Star pack ({p} percentile avg).",
    triple_chart: "Triple threat: PTS vs AST (face size = REB)", ppg_axis: "Points per game", apg_axis: "Assists per game",
    usg_axis: "Usage %", ts_axis_full: "True shooting %",
    rank_by: "Rank players by",
    radar_chart: "Type A: the multidimensional wing (normalized)",
    engine_chart: "Type B: the offensive engine (scoring + playmaking)", pts_scored: "Points scored", pts_created: "Points created (est.)", total_prod: "Total points production",
    whatif_chart: "Usage-adjusted efficiency: Deni vs. Luka", per_game_stats: "Per game stats",
    wi_actual: "Deni (actual - 28% USG)", wi_proj: "Deni (projected @ {u}% USG)", wi_luka: "Luka (actual)",
    cat_points: "Points", cat_reb: "Rebounds", cat_ast: "Assists",

    // about me
    about_title: "About the Creator",
  },

  he: {
    skip: "דלג לתוכן",
    brand_a: "ישראלים ב-NBA",
    brand_b: "אנליטיקס",
    style: "סגנון",
    player: "שחקן",
    page: "עמוד",
    source: "קוד המקור ב-GitHub",
    loading: "טוען…",
    load_error: "לא הצלחנו לטעון את נתוני הדשבורד. רעננו את העמוד כדי לנסות שוב.",
    data_updated: "הנתונים עודכנו",
    games_tracked: "משחקים של {name} העונה",

    page_dashboard: "לוח ראשי",
    page_career: "ניתוח קריירה",
    page_league: "מגמות בליגה",
    page_raw: "נתונים גולמיים",
    page_shots: "מפות זריקה",
    page_deepdive: "מחקר: צלילה לעומק",
    page_project: "על הפרויקט",
    page_about: "עליי",

    hero_sub: "ניתוח ביצועים 360° · עונת {season}",
    intro_dashboard: "העונה של {name} במבט אחד: ממוצעים למשחק עם השינוי לעומת העונה הקודמת, ואחריהם כל משחק משלוש העונות האחרונות. העמודות הן נקודות (בצבע ההדגשה לניצחון, באדום להפסד), והקו הוא דקות משחק.",
    no_games_yet: "{name} עוד לא שיחק משחק NBA. כדאי לחזור כשהעונה תיפתח.",
    no_games_season: "{name} עוד לא שיחק משחק בעונת {season}, ולכן כרטיסי העונה יופיעו אחרי המשחק הראשון. הגרפים למטה מציגים עונות קודמות.",
    deni_banner: "<b>מקום 1 ב-NBA בניסיונות זריקות עונשין</b> (250+ ניסיונות)",
    kpi_pts: "נקודות למשחק", kpi_reb: "ריבאונדים למשחק", kpi_ast: "אסיסטים למשחק", kpi_min: "דקות למשחק",
    kpi_fg: "אחוז מהשדה", kpi_3p: "אחוז לשלוש", kpi_ft: "אחוז מהעונשין", kpi_gp: "משחקים",
    kpi_caption: "▲▼ השינוי מושווה לעונה המלאה הקודמת (ממוצע למשחק; אחוזי קליעה בנקודות אחוז).",
    impact: "השפעה בעונת {season}",
    points: "נקודות", minutes: "דקות", pts_win: "נקודות (ניצחון)", pts_loss: "נקודות (הפסד)", avg_min: "ממוצע דקות ({v})", avg_pts: "ממוצע נקודות: {v}",

    career_title: "{name} — ניתוח מסלול הקריירה",
    intro_career: "איך המשחק של {name} התפתח עונה אחרי עונה. נתונים למשחק מראים תפוקה גולמית; נתונים ל-36 דקות מנטרלים את כמות הדקות, כך שאפשר להשוות בהגינות עונה מהספסל לעונה בחמישייה. שיעור השימוש ואחוז הקליעה האמיתי מראים כמה גדול התפקיד וכמה ביעילות הוא קולע בו.",
    no_career: "אין עדיין מספיק נתוני קריירה כדי לשרטט מגמה.",
    per_game: "נתונים למשחק", per36: "ל-36 דקות",
    usage: "שיעור שימוש (Usage)", usage_note: "החלק ממהלכי הקבוצה שהשחקן מסיים כשהוא על המגרש. גבוה (מעל 30%): קלעים ראשיים כמו לוקה או יאניס. נמוך (מתחת ל-15%): שחקני תפקיד.",
    usage_axis: "Usage %", league_avg_usg: "ממוצע ליגה (20%)",
    ts: "אחוז קליעה אמיתי (TS%)", ts_note: "מדד יעילות שמחשיב זריקה לשלוש כשווה פי 1.5 וכולל זריקות עונשין. עילית: מעל 60% (קרי, יוקיץ'). ממוצע: כ-58%. חלש: מתחת ל-52%.",
    ts_axis: "TS %", league_avg_ts: "ממוצע ליגה (58%)", season: "עונה",

    league_title: "מגמות ליגה ומדדים מתקדמים",
    intro_league: "הקשר מכלל הליגה שלא נוגע לשחקן אחד: כמה כל קבוצה נשענת על הכוכב שלה, ומי מוביל את ה-NBA ביצירת שלשות, בסחיטת עבירות ובחדירות לסל.",
    helio_title: "1. התקפות סביב כוכב: תפוקת הקבוצה מול הכוכב",
    helio_note: "כמה מההתקפה של הקבוצה מגיע מהכוכב הגדול שלה? (נקודות + נקודות שיצר)",
    helio_chart: "תלות התקפית (ממוין לפי התקפת הקבוצה)",
    team_ppg: "ממוצע נקודות קבוצתי", star_output: "תפוקה כוללת של הכוכב", points_prod: "תפוקת נקודות",
    no_league: "נתוני מגמות הליגה עוד לא נמצאים בתמונת הנתונים הנוכחית. הם יתמלאו אחרי הרענון היומי הבא שיגיע לממשקי המעקב של ה-NBA.",
    leaderboards: "2. טבלאות מובילים",
    snipers: "מייצרי שלשות", snipers_note: "הטובים ביותר ביצירת זריקות לשלוש לחברים.",
    magnets: "מגנטים לעבירות", magnets_note: "הכי הרבה עבירות שנסחטו (PFD).",
    rim: "לחץ על הטבעת", rim_note: "הכי הרבה חדירות למשחק.",
    no_data: "אין נתונים",
    col_rank: "#", col_player: "שחקן", col_team: "קבוצה", col_ast3: "אסיסטים לשלוש", col_ast: "סה״כ אסיסטים",
    col_pfd: "עבירות שנסחטו", col_drives: "חדירות למשחק", col_drive_pts: "נקודות מחדירות",

    raw_title: "{name} — נתונים גולמיים ומגמות לבחירה",
    intro_raw: "כל העמודות שממשק הסטטיסטיקה של ה-NBA מחזיר על הקריירה של {name}, יחד עם המדדים המתקדמים. בחרו מדדים כדי להציג אותם יחד לאורך זמן (Ctrl או ⌘ לבחירה מרובה), או קראו את הטבלה למטה.",
    no_raw: "אין עדיין נתוני קריירה.",
    trend_viewer: "מציג מגמות אינטראקטיבי", select_metrics: "בחירת מדדים", custom_trends: "מגמות לבחירה",
    career_table: "טבלת נתוני קריירה",

    shots_title: "{name} — ניתוח זריקות",
    intro_shots: "כל זריקה לסל של {name}, במקום שבו היא נלקחה על חצי מגרש. עברו לתצוגת 14 האזורים כדי לראות אחוזי קליעה לפי אזור (כחול מתחת ל-35%, צהוב 35–45%, אדום מ-45% ומעלה), או הפעילו מצב השוואה כדי להציג שתי עונות זו לצד זו.",
    no_shots: "אין עדיין נתוני זריקות עבור {name}.",
    compare_mode: "מצב השוואה", season_a: "עונה א׳", season_b: "עונה ב׳", map_style: "סוג מפה",
    shot_chart: "מפת זריקות", zone_eff: "יעילות ב-14 אזורים", made: "קלע", missed: "החטיא",
    shot_title: "מפת זריקות {season}", zone_title: "יעילות לפי אזור {season}", no_data_title: "{season} - אין נתונים",
    download_png: "הורדת PNG",

    deep_title: "{name} — השוואה לאולסטארים",
    intro_deep: "מעבדת מחקר שמודדת את העונה הנוכחית של {name} מול אולסטארים אמיתיים. לשונית ״רף ייחוס״ משתמשת בנתונים הסופיים של אולסטארי 2024-25, רף קבוע שלא זז. לשונית ״המרוץ״ משתמשת בנתונים של אותם שחקנים בעונה הנוכחית, ולכן מראה איפה הוא עומד ממש עכשיו.",
    scouting_summary: "לקריאה: דו״ח סקאוטינג וניתוח — {name}",
    ft_title: "עונת {season}: מובילי זריקות העונשין (10 הראשונים)",
    sort_by: "מיון הטבלה לפי", ft_att: "סה״כ ניסיונות", ft_made: "סה״כ קליעות", ft_pct: "אחוז",
    col_games: "משחקים", col_made: "סה״כ קליעות", col_att: "סה״כ ניסיונות",
    no_games_deep: "{name} עוד לא שיחק משחק NBA העונה, ולכן מעבדת ההשוואה לאולסטארים תופעל אחרי שיירשמו נתוני משחק.",
    tab_bench: "רף ייחוס (אולסטארי {season})", tab_race: "המרוץ (נתוני אולסטארים {season})",
    no_allstar: "עדיין לא נמצאו נתוני אולסטארים לעונת {season}.",
    comparing_bench: "משווה את הנתונים <b>הנוכחיים</b> של {name} לנתונים <b>הסופיים</b> של אולסטארי {season}.",
    comparing_race: "משווה את הנתונים <b>הנוכחיים</b> של {name} לנתונים <b>הנוכחיים ({season})</b> של אותה קבוצת אולסטארים.",
    s_thresh_bench: "1. רף האולסטאר", s_thresh_race: "1. רף המרוץ",
    s_triple: "2. האיום המשולש", s_triple_race: "2. האיום המשולש ({season})",
    s_sep: "3. גרף הפרדה (שימוש מול יעילות)", s_sep_race: "3. גרף הפרדה (ביצועי {season})",
    s_table: "4. טבלת השוואה מלאה", s_table_race: "4. טבלת {season}",
    s_cases: "5. מקרי בוחן מתקדמים", s_cases_race: "5. מקרי בוחן מתקדמים ({season})",
    s_whatif: "6. מה אם: יעילות דני מול לוקה", whatif_note: "איך דני היה נראה עם שיעור השימוש של לוקה דונצ׳יץ׳ (37.9%)?",
    thresh_chart: "רף האולסטאר", per_game_axis: "למשחק", allstar_avg: "ממוצע אולסטאר", entry: "{name} (רף כניסה)",
    verdict: "השורה התחתונה", pct_scoring: "אחוזון קליעה", pct_reb: "אחוזון ריבאונד", pct_play: "אחוזון מסירה",
    verdict_note: "משווה את {first} לסגל האולסטארים הנבחר. אחוזון {p} פירושו שהוא קולע יותר מ-{n}% מהעילית של ה-NBA.",
    v_star: "<b>ברמת אולסטאר</b>: {first} נמצא בחצי העליון של האולסטארים (אחוזון ממוצע {p}).",
    v_border: "<b>על הגבול</b>: {first} תחרותי מול האולסטארים בדרג התחתון (אחוזון ממוצע {p}).",
    v_dev: "<b>בהתפתחות</b>: {first} מראה הבזקים אבל עדיין מאחורי חבורת האולסטארים (אחוזון ממוצע {p}).",
    triple_chart: "האיום המשולש: נקודות מול אסיסטים (גודל הפנים = ריבאונדים)", ppg_axis: "נקודות למשחק", apg_axis: "אסיסטים למשחק",
    usg_axis: "Usage %", ts_axis_full: "אחוז קליעה אמיתי",
    rank_by: "דירוג השחקנים לפי",
    radar_chart: "סוג A: הכנף הרב-ממדי (מנורמל)",
    engine_chart: "סוג B: מנוע ההתקפה (קליעה + מסירה)", pts_scored: "נקודות שקלע", pts_created: "נקודות שיצר (הערכה)", total_prod: "תפוקת נקודות כוללת",
    whatif_chart: "יעילות מותאמת שימוש: דני מול לוקה", per_game_stats: "נתונים למשחק",
    wi_actual: "דני (בפועל - 28% שימוש)", wi_proj: "דני (תחזית ב-{u}% שימוש)", wi_luka: "לוקה (בפועל)",
    cat_points: "נקודות", cat_reb: "ריבאונדים", cat_ast: "אסיסטים",

    about_title: "על היוצר",
  },
};

/* Long-form blocks (HTML). Kept apart from the short strings so both languages read as prose. */
const I18N_BLOCKS = {
  en: {
    project: `
      <p class="page-intro">This dashboard follows every active Israeli NBA player (Deni Avdija, Ben Saraf, Danny Wolf and Emanuel Sharp) through the season. It started as a Streamlit app and now runs as a plain static site on GitHub Pages, so it opens instantly and never has to wake a server.</p>

      <h3>How the data gets here</h3>
      <ol class="pipeline">
        <li><b>Fetch</b><code>fetch_data.py</code> calls the official NBA stats API through <code>nba_api</code>: career totals, advanced stats, game logs, every shot with its court location, the All-Star cohort and league leaderboards.</li>
        <li><b>Store</b>Everything lands in one pickle, <code>nba_data.pkl</code>, versioned in the repo so any run can be reproduced.</li>
        <li><b>Build</b><code>build_data.py</code> turns that pickle into small JSON files: one per player plus shared league data and the court-zone geometry.</li>
        <li><b>Serve</b>This page (HTML, CSS and JavaScript with a bundled copy of Plotly) reads those JSON files and draws every chart in the browser.</li>
      </ol>
      <p>A GitHub Actions workflow runs steps 1 to 3 every day at 09:00 UTC, after the West Coast games finish, and commits the new data. The header shows when that last happened. The seasons roll over by themselves: from October the dashboard treats the new season as current.</p>

      <h3>What each page answers</h3>
      <ul>
        <li><b>Dashboard</b>: how is he doing this season compared with last? Eight season cards with deltas, then a game-by-game chart for each of the last three seasons.</li>
        <li><b>Career Analysis</b>: how has his role and efficiency grown? Per game, per 36 minutes, usage rate and true shooting by season.</li>
        <li><b>League Trends</b>: context from the whole league, such as how star-dependent each offense is and who leads in creating threes, drawing fouls and driving.</li>
        <li><b>Raw Data</b>: the full career table, and a trend viewer to plot any metric.</li>
        <li><b>Shot Maps</b>: where he shoots from and how well, as individual shots or as 14 court zones coloured by field-goal percentage, with a two-season compare mode.</li>
        <li><b>Research: Deep Dive</b>: is he an All-Star? His current numbers against a fixed 2024-25 All-Star benchmark and against the same All-Stars this season: percentile verdict, scatter plots with headshots, a ranking table, a radar profile and an estimate of total offensive output.</li>
      </ul>

      <h3>Metrics used</h3>
      <dl class="glossary">
        <dt>Per 36</dt><dd>A per-game stat scaled to 36 minutes (stat ÷ minutes × 36), so players with different minutes can be compared.</dd>
        <dt>USG%</dt><dd>Usage rate: the share of team possessions a player ends with a shot, free throws or a turnover while on the floor. League average is about 20%.</dd>
        <dt>TS%</dt><dd>True shooting: points ÷ (2 × (FGA + 0.44 × FTA)). One number for scoring efficiency that credits threes and free throws. League average is about 58%.</dd>
        <dt>Percentile</dt><dd>The share of the All-Star group a player beats in a stat. The verdict averages the scoring, rebounding and playmaking percentiles.</dd>
        <dt>Points created</dt><dd>Assists × 2.3, a common rough estimate of the points a pass leads to (most assists set up twos, some set up threes).</dd>
        <dt>14 zones</dt><dd>The NBA's own shot zones (restricted area, paint, five mid-range areas, two corners and three above-the-break areas) redrawn as polygons on the court.</dd>
      </dl>

      <h3>Built with</h3>
      <p>Python (pandas, NumPy, nba_api) for the data, GitHub Actions for the daily refresh, and plain HTML, CSS and JavaScript with Plotly.js for the front end. No framework and no build step. The original Streamlit version (<code>app.py</code>) is still in the repository.</p>

      <h3>Limits worth knowing</h3>
      <ul>
        <li>Numbers are as fresh as the last daily run, not live during a game.</li>
        <li>League tracking endpoints (drives, passing) sometimes fail from cloud runners; when they do, that section stays empty until a later run succeeds.</li>
        <li>The What-If chart uses fixed reference numbers for Deni and Luka; it illustrates the usage effect rather than tracking live stats.</li>
      </ul>`,

    about: `
      <div class="about-grid">
        <div>
          <img src="../profile_pic.png" alt="Ram Shiri" onerror="this.style.display='none'" />
          <h3 style="margin-top:12px">Ram Shiri</h3>
          <p><b>Data Engineering Student</b></p>
          <p><a href="https://rshiri.github.io/">Portfolio</a><br/>
          <a href="https://www.linkedin.com/in/ram-shiri-1a1056304/?originalSubdomain=il" target="_blank" rel="noopener">Connect on LinkedIn</a><br/>
          <a href="https://github.com/RShiri/nba-dashboard" target="_blank" rel="noopener">View source on GitHub</a></p>
        </div>
        <div>
          <h3>Hello!</h3>
          <p>I'm a <b>3rd year B.Sc. Data Engineering student</b> specializing in data science with a passion for building smart, practical solutions.</p>
          <p>I love combining creativity with technical skills to drive real-world impact, especially in the world of sports analytics.</p>
          <h3>Skills &amp; approach</h3>
          <ul>
            <li><b>Tech stack:</b> Python, Java, SQL, Pandas, Plotly, JavaScript</li>
            <li><b>Soft skills:</b> creative thinking, fast learning, hands-on problem solving</li>
            <li><b>Philosophy:</b> comfortable working with AI tools to accelerate development (like this dashboard!) while maintaining deep understanding of the core logic.</li>
          </ul>
          <h3>Passions</h3>
          <p>Basketball · Football · F1 racing · LEGO</p>
          <hr/>
          <div class="banner"><b>Open to work:</b> actively seeking a student or full-time position in software or data engineering to grow, contribute, and thrive in a dynamic environment.</div>
        </div>
      </div>`,

    scouting_deni_avdija: `
      <h4>Analysis: the expanded role</h4>
      <p><b>Early returns (11/26):</b> The most impressive aspect of Avdija's star-making season has been his capacity for scaling up his production to fit his expanded role.</p>
      <ul>
        <li><b>Pick-and-roll volume:</b> Avdija has already logged more possessions as a P&amp;R initiator than in his full years 2 or 3.</li>
        <li><b>Elite driving:</b> He is fully tapping into his physicality. No one in the league drives more often, and few pass out of drives more frequently.</li>
        <li><b>Free throw rate:</b> His downhill speed and "incessant drives" have him getting to the line at a rate on par with <b>Shai Gilgeous-Alexander</b>.</li>
      </ul>
      <h4>Defining trait: the one-man fast break</h4>
      <p>Avdija has become a reliable one-man fast break. He is equally adept at finishing at full tilt or shifting gears (Euro-step or shoulder bumps) to dislodge defenders.</p>
      <blockquote>"The most bullish sign of Avdija's ascent might be his ability to draw contact. Avdija had one of the highest free throw attempt rates in the league... the only other non-bigs in his cohort were <b>Jimmy Butler</b> and <b>James Harden</b>."</blockquote>
      <h4>Modern NBA archetype: the multidimensional wing</h4>
      <p>Multidimensional wings are the lifeblood of the modern game. Avdija represents a high-reward venture that is paying off:</p>
      <ul>
        <li><b>Growth:</b> incremental growth throughout his career, with breakthroughs in years 4 and 5.</li>
        <li><b>Foundation:</b> his vision and ballhandling got him noticed; his defense and rebounding instincts kept him on the floor long enough to see the fruits of his labor come to the fore.</li>
      </ul>`,

    scouting_default: `<p>Scouting notes are still being written for this player. Check back once more of the season is in the books. In the meantime, the Dashboard and Career Analysis pages track his per-game trends as they happen.</p>`,
  },

  he: {
    project: `
      <p class="page-intro">הדשבורד הזה עוקב אחרי כל השחקנים הישראלים הפעילים ב-NBA (דני אבדיה, בן שרף, דני וולף ועמנואל שארפ) לאורך העונה. הוא התחיל כאפליקציית Streamlit ועכשיו רץ כאתר סטטי פשוט ב-GitHub Pages, כך שהוא נפתח מיד ואף פעם לא צריך להעיר שרת.</p>

      <h3>איך הנתונים מגיעים לכאן</h3>
      <ol class="pipeline">
        <li><b>שליפה</b><code>fetch_data.py</code> פונה לממשק הסטטיסטיקה הרשמי של ה-NBA דרך <code>nba_api</code>: נתוני קריירה, מדדים מתקדמים, יומני משחק, כל זריקה עם המיקום שלה על המגרש, קבוצת האולסטארים וטבלאות המובילים בליגה.</li>
        <li><b>שמירה</b>הכול נשמר בקובץ pickle אחד, <code>nba_data.pkl</code>, שמנוהל בגרסאות בתוך הריפו כדי שכל הרצה תהיה ניתנת לשחזור.</li>
        <li><b>בנייה</b><code>build_data.py</code> הופך את הקובץ לקובצי JSON קטנים: אחד לכל שחקן, ועוד נתוני ליגה משותפים וגאומטריית אזורי המגרש.</li>
        <li><b>הצגה</b>העמוד הזה (HTML,‏ CSS ו-JavaScript עם עותק מקומי של Plotly) קורא את קובצי ה-JSON ומצייר כל גרף בדפדפן.</li>
      </ol>
      <p>תהליך GitHub Actions מריץ את שלבים 1 עד 3 כל יום ב-09:00 UTC, אחרי שמשחקי החוף המערבי נגמרים, ושומר את הנתונים החדשים. בראש העמוד מופיע מתי זה קרה לאחרונה. העונות מתחלפות לבד: מאוקטובר הדשבורד מתייחס לעונה החדשה כעונה הנוכחית.</p>

      <h3>על מה כל עמוד עונה</h3>
      <ul>
        <li><b>לוח ראשי</b>: איך הוא עושה העונה לעומת העונה שעברה? שמונה כרטיסי עונה עם השינוי, ואחריהם גרף משחק-אחר-משחק לכל אחת משלוש העונות האחרונות.</li>
        <li><b>ניתוח קריירה</b>: איך התפקיד והיעילות שלו גדלו? נתונים למשחק, ל-36 דקות, שיעור שימוש ואחוז קליעה אמיתי לפי עונה.</li>
        <li><b>מגמות בליגה</b>: הקשר מכל הליגה, למשל כמה כל התקפה תלויה בכוכב שלה ומי מוביל ביצירת שלשות, סחיטת עבירות וחדירות.</li>
        <li><b>נתונים גולמיים</b>: טבלת הקריירה המלאה, ומציג מגמות לכל מדד.</li>
        <li><b>מפות זריקה</b>: מאיפה הוא זורק וכמה טוב, כזריקות בודדות או כ-14 אזורים צבועים לפי אחוז קליעה, עם מצב השוואה בין שתי עונות.</li>
        <li><b>מחקר: צלילה לעומק</b>: האם הוא אולסטאר? הנתונים הנוכחיים שלו מול רף קבוע של אולסטארי 2024-25 ומול אותם אולסטארים העונה: אחוזונים ושורה תחתונה, גרפי פיזור עם תמונות, טבלת דירוג, פרופיל רדאר והערכה של התפוקה ההתקפית הכוללת.</li>
      </ul>

      <h3>המדדים</h3>
      <dl class="glossary">
        <dt>ל-36 דקות</dt><dd>נתון למשחק שמוגדל ל-36 דקות (נתון ÷ דקות × 36), כדי להשוות בין שחקנים שמשחקים כמות דקות שונה.</dd>
        <dt>USG%</dt><dd>שיעור שימוש: החלק מהפוזשנים של הקבוצה שהשחקן מסיים בזריקה, בזריקות עונשין או באיבוד כשהוא על המגרש. ממוצע הליגה בערך 20%.</dd>
        <dt>TS%</dt><dd>אחוז קליעה אמיתי: נקודות ÷ (2 × (ניסיונות מהשדה + 0.44 × ניסיונות עונשין)). מספר אחד ליעילות קליעה שמזכה שלשות וזריקות עונשין. ממוצע הליגה בערך 58%.</dd>
        <dt>אחוזון</dt><dd>החלק מקבוצת האולסטארים שהשחקן עוקף בנתון מסוים. השורה התחתונה היא ממוצע אחוזוני הקליעה, הריבאונד והמסירה.</dd>
        <dt>נקודות שיצר</dt><dd>אסיסטים × 2.3, הערכה מקובלת וגסה לנקודות שמסירה מובילה אליהן (רוב האסיסטים מובילים לשתיים, חלקם לשלוש).</dd>
        <dt>14 אזורים</dt><dd>אזורי הזריקה הרשמיים של ה-NBA (אזור מוגבל, צבע, חמישה אזורי חצי מרחק, שתי פינות ושלושה אזורים מעל הקשת) מצוירים כמצולעים על המגרש.</dd>
      </dl>

      <h3>טכנולוגיות</h3>
      <p>Python (pandas,‏ NumPy,‏ nba_api) לנתונים, GitHub Actions לרענון היומי, ו-HTML,‏ CSS ו-JavaScript פשוטים עם Plotly.js לצד הלקוח. בלי פריימוורק ובלי שלב בנייה. גרסת ה-Streamlit המקורית (<code>app.py</code>) עדיין נמצאת בריפו.</p>

      <h3>מגבלות שכדאי להכיר</h3>
      <ul>
        <li>המספרים עדכניים לפי ההרצה היומית האחרונה, לא בזמן אמת במהלך משחק.</li>
        <li>ממשקי המעקב של הליגה (חדירות, מסירות) נכשלים לפעמים משרתי ענן; כשזה קורה החלק הזה נשאר ריק עד הרצה מוצלחת.</li>
        <li>גרף ה״מה אם״ משתמש במספרי ייחוס קבועים לדני וללוקה; הוא ממחיש את השפעת שיעור השימוש ולא עוקב אחרי נתונים חיים.</li>
      </ul>`,

    about: `
      <div class="about-grid">
        <div>
          <img src="../profile_pic.png" alt="רם שירי" onerror="this.style.display='none'" />
          <h3 style="margin-top:12px">רם שירי</h3>
          <p><b>סטודנט להנדסת נתונים</b></p>
          <p><a href="https://rshiri.github.io/">תיק העבודות</a><br/>
          <a href="https://www.linkedin.com/in/ram-shiri-1a1056304/?originalSubdomain=il" target="_blank" rel="noopener">לפרופיל הלינקדאין</a><br/>
          <a href="https://github.com/RShiri/nba-dashboard" target="_blank" rel="noopener">קוד המקור ב-GitHub</a></p>
        </div>
        <div>
          <h3>שלום!</h3>
          <p>אני <b>סטודנט שנה שלישית לתואר ראשון בהנדסת נתונים</b> עם התמחות במדעי הנתונים, ואוהב לבנות פתרונות חכמים ומעשיים.</p>
          <p>אני אוהב לשלב יצירתיות עם יכולות טכניות כדי להשפיע בעולם האמיתי, ובמיוחד בעולם האנליטיקה של הספורט.</p>
          <h3>כישורים וגישה</h3>
          <ul>
            <li><b>טכנולוגיות:</b> Python,‏ Java,‏ SQL,‏ Pandas,‏ Plotly,‏ JavaScript</li>
            <li><b>כישורים רכים:</b> חשיבה יצירתית, למידה מהירה, פתרון בעיות בידיים</li>
            <li><b>גישה:</b> עובד בנוחות עם כלי AI כדי להאיץ פיתוח (כמו הדשבורד הזה!) תוך הבנה עמוקה של הלוגיקה שבבסיס.</li>
          </ul>
          <h3>תחביבים</h3>
          <p>כדורסל · כדורגל · פורמולה 1 · לגו</p>
          <hr/>
          <div class="banner"><b>פתוח להצעות:</b> מחפש באופן פעיל משרת סטודנט או משרה מלאה בהנדסת תוכנה או נתונים, כדי לצמוח, לתרום ולהצליח בסביבה דינמית.</div>
        </div>
      </div>`,

    scouting_deni_avdija: `
      <h4>ניתוח: התפקיד המורחב</h4>
      <p><b>תוצאות ראשונות (11/26):</b> הדבר המרשים ביותר בעונת הפריצה של אבדיה הוא היכולת שלו להגדיל את התפוקה כך שתתאים לתפקיד המורחב.</p>
      <ul>
        <li><b>נפח פיק-אנד-רול:</b> אבדיה כבר רשם יותר פוזשנים כמוביל פיק-אנד-רול מאשר בעונות 2 או 3 המלאות שלו.</li>
        <li><b>חדירות ברמת עילית:</b> הוא מנצל עד הסוף את הפיזיות שלו. אף אחד בליגה לא חודר יותר ממנו, ומעטים מוסרים מתוך חדירה בתדירות גבוהה יותר.</li>
        <li><b>קצב הגעה לקו:</b> המהירות שלו בירידה לסל וה״חדירות הבלתי פוסקות״ מביאות אותו לקו העונשין בקצב שדומה לזה של <b>שיי גילג׳ס-אלכסנדר</b>.</li>
      </ul>
      <h4>התכונה המגדירה: מתפרצת של איש אחד</h4>
      <p>אבדיה הפך למתפרצת אמינה של איש אחד. הוא טוב באותה מידה בסיום במהירות מלאה ובהחלפת הילוכים (יורו-סטפ או מכת כתף) כדי להזיז את המגנים.</p>
      <blockquote>״הסימן האופטימי ביותר לעלייה של אבדיה הוא אולי היכולת שלו לסחוט מגע. לאבדיה היה אחד משיעורי ניסיונות העונשין הגבוהים בליגה... השחקנים היחידים שאינם גבוהים בקבוצה הזו היו <b>ג׳ימי באטלר</b> ו<b>ג׳יימס הארדן</b>.״</blockquote>
      <h4>ארכיטיפ NBA מודרני: הכנף הרב-ממדי</h4>
      <p>כנפיים רב-ממדיים הם הדם של המשחק המודרני. אבדיה הוא השקעה עתירת סיכוי שמשתלמת:</p>
      <ul>
        <li><b>צמיחה:</b> התקדמות הדרגתית לאורך הקריירה, עם פריצות בשנים 4 ו-5.</li>
        <li><b>בסיס:</b> הראייה והכדרור שלו הם מה שהפנה אליו תשומת לב; ההגנה וחוש הריבאונד החזיקו אותו על המגרש מספיק זמן כדי לראות את פירות העבודה.</li>
      </ul>`,

    scouting_default: `<p>הערות הסקאוטינג על השחקן הזה עדיין נכתבות. כדאי לחזור כשיותר מהעונה יהיה מאחורינו. בינתיים, עמודי הלוח הראשי וניתוח הקריירה עוקבים אחרי המגמות שלו למשחק בזמן שהן קורות.</p>`,
  },
};
