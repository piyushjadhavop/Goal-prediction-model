import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GoalIQ · XGBoost Goal Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Bebas+Neue&display=swap');

:root {
    --ink: #eaf0e8;
    --muted: #91a098;
    --lime: #c8f36a;
    --cyan: #66d8d1;
    --line: rgba(209, 229, 216, 0.11);
    --panel: rgba(17, 26, 21, 0.88);
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    color: var(--ink);
    background:
        radial-gradient(ellipse at 86% 0%, rgba(37, 91, 61, 0.20), transparent 34%),
        linear-gradient(145deg, #080d0a 0%, #0b110e 50%, #080c0a 100%);
}

header[data-testid="stHeader"] {
    background: rgba(8, 13, 10, 0.72);
}

div[data-testid="stAppViewContainer"] > .main .block-container {
    max-width: 1560px;
    padding: 1.4rem 2.4rem 3rem;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101813 0%, #0b110e 100%);
    border-right: 1px solid var(--line);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.2rem;
}

section[data-testid="stSidebar"] h3 {
    color: var(--lime);
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

div[data-testid="stMetric"] {
    min-height: 112px;
    padding: 18px 20px;
    background: linear-gradient(145deg, rgba(22, 33, 27, 0.94), rgba(15, 23, 19, 0.9));
    border: 1px solid var(--line);
    border-radius: 10px;
    box-shadow: 0 14px 34px rgba(0, 0, 0, 0.16);
    transition: border-color 180ms ease, transform 180ms ease;
}

div[data-testid="stMetric"]:hover {
    border-color: rgba(200, 243, 106, 0.36);
    transform: translateY(-2px);
}

div[data-testid="stMetricLabel"] {
    color: var(--muted);
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

div[data-testid="stMetricValue"] {
    color: var(--ink);
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    letter-spacing: 0.02em;
}

.stSlider > div > div > div > div {
    background: var(--lime) !important;
}

.stButton > button {
    min-height: 44px;
    background: var(--lime);
    color: #10170e;
    font-weight: 700;
    font-family: 'Space Grotesk', sans-serif;
    border: none;
    border-radius: 7px;
    padding: 10px 18px;
    font-size: 14px;
    transition: background 180ms ease, box-shadow 180ms ease, transform 180ms ease;
    width: 100%;
}

.stButton > button:hover {
    background: #d7ff83;
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(161, 210, 79, 0.18);
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 5px;
    gap: 3px;
}

.stTabs [data-baseweb="tab"] {
    min-height: 42px;
    border-radius: 5px;
    color: var(--muted);
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    transition: color 160ms ease, background 160ms ease;
}

.stTabs [aria-selected="true"] {
    background: rgba(200, 243, 106, 0.11) !important;
    color: var(--lime) !important;
}

h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; }
h1, h2, h3 { color: var(--ink); }
h4 { color: var(--lime); }

div[data-baseweb="select"] > div,
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    background: rgba(255, 255, 255, 0.045);
    border-color: rgba(209, 229, 216, 0.14);
    border-radius: 6px;
    color: var(--ink);
}

hr { border-color: var(--line); }

.result-box {
    background: linear-gradient(145deg, rgba(200, 243, 106, 0.09), rgba(102, 216, 209, 0.045));
    border: 1px solid rgba(200, 243, 106, 0.32);
    border-radius: 10px;
    padding: 26px 28px;
    text-align: center;
    margin: 16px 0;
    box-shadow: inset 0 1px rgba(255,255,255,0.04), 0 18px 40px rgba(0,0,0,0.18);
}

.result-number {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 76px;
    line-height: 1;
    color: var(--lime);
    background: linear-gradient(135deg, var(--lime), var(--cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.result-label {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 4px;
}

.rating-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 100px;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.03em;
    margin-top: 12px;
}

.player-header {
    background: linear-gradient(120deg, rgba(102, 216, 209, 0.08), rgba(255,255,255,0.02));
    border: 1px solid var(--line);
    border-radius: 9px;
    padding: 18px 20px;
    margin-bottom: 16px;
}

.info-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
}

.stDataFrame { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    padding: 0 0 16px;
    border-bottom: 1px solid var(--line);
    margin-bottom: 24px;
}
.brand-lockup { display: flex; align-items: center; gap: 11px; }
.brand-mark {
    display: grid;
    place-items: center;
    width: 38px;
    height: 38px;
    border-radius: 8px;
    background: rgba(200, 243, 106, 0.12);
    color: var(--lime);
    font-size: 20px;
}
.brand-name { color: var(--ink); font-size: 17px; font-weight: 700; letter-spacing: 0.02em; }
.brand-subtitle { color: var(--muted); font-size: 11px; }
.model-status {
    color: var(--muted);
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.model-status span { color: var(--lime); }
.welcome-kicker {
    color: var(--lime);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}
.welcome-title { margin: 6px 0 4px; color: var(--ink); font-size: 29px; font-weight: 700; }
.welcome-copy { color: var(--muted); font-size: 14px; margin-bottom: 18px; }

@media (max-width: 760px) {
    div[data-testid="stAppViewContainer"] > .main .block-container { padding: 1rem 1rem 2rem; }
    .topbar { align-items: flex-start; }
    .model-status { text-align: right; }
    .welcome-title { font-size: 24px; }
    .stTabs [data-baseweb="tab"] { font-size: 12px; }
}
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

FEATURES = ['Matches_Played', 'Mins', 'xG Per Avg Match',
            'Shots', 'OnTarget', 'Shots Per Avg Match', 'On Target Per Avg Match']

FEAT_LABELS = {
    'Matches_Played': 'Matches Played',
    'Mins': 'Minutes Played',
    'xG Per Avg Match': 'xG per Match',
    'Shots': 'Total Shots',
    'OnTarget': 'Shots on Target',
    'Shots Per Avg Match': 'Shots per Match',
    'On Target Per Avg Match': 'On Target per Match',
}

FEAT_ICONS = {
    'Matches_Played': '🎮',
    'Mins': '⏱️',
    'xG Per Avg Match': '📊',
    'Shots': '🎯',
    'OnTarget': '✅',
    'Shots Per Avg Match': '⚡',
    'On Target Per Avg Match': '🔥',
}

FI = model.feature_importances_
FI_DICT = {f: v for f, v in zip(FEATURES, FI)}

def predict(inputs: dict) -> float:
    X = pd.DataFrame([inputs], columns=FEATURES)
    return float(model.predict(X)[0])

def rating(goals: float) -> tuple:
    if goals >= 20:   return "World Class ⭐⭐⭐", "#39FF14"
    if goals >= 14:   return "Elite 🔥", "#00D4FF"
    if goals >= 8:    return "Quality Player 💎", "#FFE600"
    if goals >= 4:    return "Decent 👍", "#FF6B35"
    return "Needs Improvement 📈", "#FF2D78"

def _normalized_column_name(column: str) -> str:
    return "".join(character for character in str(column).casefold() if character.isalnum())

def _find_player_column(data: pd.DataFrame):
    name_columns = {"playername", "playernames", "name", "player"}
    return next((column for column in data.columns
                 if _normalized_column_name(column) in name_columns), None)

@st.cache_data(show_spinner=False)
def load_comparison_data():
    data_dir = Path(__file__).resolve().parent
    source_path = data_dir / "Data.csv"
    data = pd.read_csv(source_path)
    player_column = _find_player_column(data)

    if player_column is None:
        raise ValueError("Data.csv does not contain a player-name column.")

    data[player_column] = data[player_column].astype("string").str.strip()
    data = data.dropna(subset=[player_column])
    data = data[data[player_column] != ""].drop_duplicates()
    return data, player_column, source_path.name

def aggregate_comparison_records(data: pd.DataFrame, player_column: str) -> pd.DataFrame:
    year_column = next((column for column in data.columns
                        if _normalized_column_name(column) in {"year", "season", "seasonx"}), None)
    numeric_columns = [column for column in data.select_dtypes(include="number").columns
                       if column != year_column]
    rate_columns = [column for column in numeric_columns
                    if "per" in _normalized_column_name(column)
                    or "avg" in _normalized_column_name(column)]
    derived_rates = {
        "xgperavgmatch": ("xG", "Matches_Played"),
        "shotsperavgmatch": ("Shots", "Matches_Played"),
        "ontargetperavgmatch": ("OnTarget", "Matches_Played"),
    }
    records = []

    for player, player_rows in data.groupby(player_column, sort=False):
        record = {player_column: player}
        for column in numeric_columns:
            if column not in rate_columns:
                record[column] = pd.to_numeric(player_rows[column], errors="coerce").sum(min_count=1)

        for column in rate_columns:
            normalized_column = _normalized_column_name(column)
            numerator_column, denominator_column = derived_rates.get(normalized_column, (None, None))
            if numerator_column in record and denominator_column in record:
                denominator = record[denominator_column]
                record[column] = (record[numerator_column] / denominator
                                  if pd.notna(denominator) and denominator != 0 else np.nan)
            elif "Matches_Played" in player_rows.columns:
                values = pd.to_numeric(player_rows[column], errors="coerce")
                weights = pd.to_numeric(player_rows["Matches_Played"], errors="coerce")
                valid = values.notna() & weights.notna() & (weights > 0)
                record[column] = (np.average(values[valid], weights=weights[valid])
                                  if valid.any() else np.nan)
            else:
                record[column] = pd.to_numeric(player_rows[column], errors="coerce").mean()

        for column in data.columns:
            if column == player_column or column in numeric_columns:
                continue
            values = player_rows[column].dropna().astype(str).unique()
            record[column] = " / ".join(values)
        if year_column and year_column in numeric_columns:
            years = player_rows[year_column].dropna().astype(str).unique()
            record[year_column] = " / ".join(years)
        records.append(record)

    return pd.DataFrame(records)

def render_player_comparison():
    st.markdown("#### Player Performance Comparison")
    try:
        data, player_column, source_name = load_comparison_data()
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        st.error(f"Unable to load player comparison data: {error}")
        return

    st.caption(f"Data source: {source_name} · Exact duplicate records are removed.")

    filtered_data = data.copy()
    year_column = next((column for column in filtered_data.columns
                        if _normalized_column_name(column) in {"year", "season", "seasonx"}), None)
    league_column = next((column for column in filtered_data.columns
                          if _normalized_column_name(column) in {"league", "competition"}), None)
    filter_columns = st.columns(2)

    if year_column:
        seasons = sorted(filtered_data[year_column].dropna().unique().tolist(), reverse=True)
        season_options = ["All seasons", *seasons]
        chosen_season = filter_columns[0].selectbox(
            "Season", season_options, index=1 if seasons else 0
        )
        if chosen_season != "All seasons":
            filtered_data = filtered_data[filtered_data[year_column] == chosen_season]

    if league_column:
        competitions = sorted(filtered_data[league_column].dropna().astype(str).unique().tolist())
        competition_options = ["All competitions", *competitions]
        chosen_competition = filter_columns[1].selectbox("Competition", competition_options)
        if chosen_competition != "All competitions":
            filtered_data = filtered_data[
                filtered_data[league_column].astype(str) == chosen_competition
            ]

    player_options = sorted(filtered_data[player_column].dropna().unique().tolist(), key=str.casefold)
    if not player_options:
        st.warning("No players are available for the selected season and competition.")
        return

    selected_players = st.multiselect(
        "Search and select players",
        options=player_options,
        placeholder="Type a player name, then select at least two players",
        help="Search by name and select two or more players to compare their recorded statistics.",
    )
    if selected_players:
        st.caption("Selected players: " + " · ".join(selected_players))
    if len(selected_players) < 2:
        st.info("Select at least two players to see the comparison dashboard.")
        return

    selected_data = filtered_data[filtered_data[player_column].isin(selected_players)]
    comparison = aggregate_comparison_records(selected_data, player_column)
    if len(comparison) < 2:
        st.warning("At least two selected players need records in the chosen filters.")
        return

    stat_columns = [column for column in comparison.select_dtypes(include="number").columns
                    if column != year_column]
    if not stat_columns:
        st.warning("The selected records do not contain numerical performance statistics.")
        return

    st.markdown("##### Player Profiles")
    profile_columns = st.columns(min(len(selected_players), 3))
    for index, player in enumerate(selected_players):
        player_record = comparison[comparison[player_column] == player].iloc[0]
        with profile_columns[index % len(profile_columns)]:
            with st.expander(player, expanded=True):
                profile_values = player_record.drop(labels=[player_column]).map(
                    lambda value: "" if pd.isna(value) else str(value)
                ).astype("string").rename("Value")
                st.dataframe(profile_values.to_frame(), use_container_width=True)

    st.markdown("##### Side-by-Side Statistics")
    stats_table = comparison.set_index(player_column)[stat_columns].transpose()
    stats_table.index.name = "Statistic"
    st.dataframe(stats_table, use_container_width=True)

    preferred_metrics = {"goals", "assists", "shots", "xg", "mins", "matchesplayed", "ontarget"}
    default_metrics = [column for column in stat_columns
                       if _normalized_column_name(column) in preferred_metrics][:6]
    if not default_metrics:
        default_metrics = stat_columns[:min(6, len(stat_columns))]
    chart_metrics = st.multiselect(
        "Metrics to chart",
        options=stat_columns,
        default=default_metrics,
    )

    if chart_metrics:
        chart_data = comparison[[player_column, *chart_metrics]].melt(
            id_vars=player_column, var_name="Metric", value_name="Value"
        )
        fig_comparison = px.bar(
            chart_data,
            x="Metric",
            y="Value",
            color=player_column,
            barmode="group",
            title="Recorded Statistics",
        )
        fig_comparison.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Player",
            xaxis_title="Statistic",
            yaxis_title="Recorded value",
        )
        st.plotly_chart(fig_comparison, use_container_width=True)

        values = comparison[chart_metrics].apply(pd.to_numeric, errors="coerce")
        minimums = values.min()
        spans = values.max() - minimums
        normalized = values.subtract(minimums).divide(spans.where(spans != 0))
        for metric in chart_metrics:
            if pd.notna(spans[metric]) and spans[metric] == 0:
                normalized[metric] = values[metric].where(values[metric].isna(), 0.5)

        fig_radar = go.Figure()
        for row_index, player in enumerate(comparison[player_column]):
            radar_values = normalized.iloc[row_index].tolist()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_values + [radar_values[0]],
                theta=chart_metrics + [chart_metrics[0]],
                fill="toself",
                name=player,
            ))
        fig_radar.update_layout(
            template="plotly_dark",
            title="Normalized Comparison (0–1 across selected players)",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=55, b=30, l=50, r=50),
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption(
            "Radar values are normalized only for visual comparison among these players; "
            "they are not official player ratings."
        )

        st.markdown("##### Metric Leaders")
        for metric in chart_metrics:
            valid_values = comparison[[player_column, metric]].dropna(subset=[metric])
            if valid_values.empty:
                st.write(f"**{metric}:** no recorded values for the selected players.")
                continue
            highest = valid_values[metric].max()
            leaders = valid_values.loc[valid_values[metric] == highest, player_column].tolist()
            st.write(f"**{metric}:** {', '.join(leaders)} ({highest:g})")
        st.caption("A metric leader is not an overall ranking; compare the statistics in context.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px;'>
        <div style='font-size:48px;'>⚽</div>
        <div style='font-family:"Space Grotesk",sans-serif; font-size:26px; font-weight:800;
                    background:linear-gradient(135deg,#39FF14,#00D4FF);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    letter-spacing:3px;'>GoalIQ</div>
        <div style='font-size:11px; color:rgba(255,255,255,0.4); letter-spacing:2px;
                    text-transform:uppercase; margin-top:2px;'>XGBoost Predictor</div>
    </div>
    <hr style='margin:12px 0;'>
    """, unsafe_allow_html=True)

    st.markdown("### 🧑‍💼 Player Profile")
    player_name = st.text_input("Player Name", placeholder="e.g. Erling Haaland", label_visibility="collapsed")
    player_name = player_name if player_name else "Unknown Player"

    position = st.selectbox("Position", ["⚽ Forward", "🎯 Attacking Mid", "🏃 Winger", "🔗 Midfielder"])
    team = st.text_input("Club / Team", placeholder="e.g. Manchester City")

    st.markdown("<hr style='margin:8px 0;'>", unsafe_allow_html=True)
    st.markdown("### 📐 Stat Inputs")

    matches  = st.slider("🎮 Matches Played",        min_value=1,   max_value=1000,  value=20, step=1)
    mins     = st.slider("⏱️ Minutes Played",         min_value=90,  max_value=3420, value=1800, step=90)
    xg       = st.slider("📊 xG per Match",          min_value=0.0, max_value=1.5,  value=0.35, step=0.01, format="%.2f")
    shots    = st.slider("🎯 Total Shots",            min_value=0,   max_value=200,  value=60, step=1)
    ontarget = st.slider("✅ Shots on Target",        min_value=0,   max_value=100,  value=25, step=1)
    spm      = st.slider("⚡ Shots per Match",        min_value=0.0, max_value=8.0,  value=3.0, step=0.01, format="%.1f")
    otpm     = st.slider("🔥 On Target per Match",   min_value=0.0, max_value=5.0,  value=1.2, step=0.01, format="%.1f")

    st.markdown("<hr style='margin:8px 0;'>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ Predict Goals", use_container_width=True)

# ── Build inputs dict ─────────────────────────────────────────────────────────
inputs = {
    'Matches_Played':          matches,
    'Mins':                    mins,
    'xG Per Avg Match':        xg,
    'Shots':                   shots,
    'OnTarget':                ontarget,
    'Shots Per Avg Match':     spm,
    'On Target Per Avg Match': otpm,
}

# ── Main Content ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="brand-lockup">
        <div class="brand-mark">⚽</div>
        <div><div class="brand-name">GOALIQ</div><div class="brand-subtitle">Football intelligence</div></div>
    </div>
    <div class="model-status"><span>●</span> XGBoost model active</div>
</div>
<div class="welcome-kicker">Performance workspace</div>
<div class="welcome-title">Football, measured with clarity.</div>
<div class="welcome-copy">Explore recorded player performance, compare seasons, and model goal output.</div>
""", unsafe_allow_html=True)

try:
    analytics_data, analytics_player_column, analytics_source = load_comparison_data()
except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
    analytics_data = pd.DataFrame()
    analytics_player_column = None
    analytics_source = None

if not analytics_data.empty and analytics_player_column:
    normalized_columns = {
        _normalized_column_name(column): column for column in analytics_data.columns
    }
    goals_column = normalized_columns.get("goals")
    season_column = next((normalized_columns[key] for key in ("year", "season", "seasonx")
                          if key in normalized_columns), None)
    competition_column = next((normalized_columns[key] for key in ("league", "competition")
                               if key in normalized_columns), None)
    kpi_columns = st.columns(4)
    kpi_columns[0].metric("Player records", f"{len(analytics_data):,}")
    kpi_columns[1].metric("Unique players", f"{analytics_data[analytics_player_column].nunique():,}")
    kpi_columns[2].metric("Seasons", f"{analytics_data[season_column].nunique():,}" if season_column else "—")
    kpi_columns[3].metric("Competitions", f"{analytics_data[competition_column].nunique():,}" if competition_column else "—")

    chart_columns = st.columns([1.25, 1], gap="large")
    if goals_column and season_column:
        with chart_columns[0]:
            st.markdown("#### Scoring across seasons")
            season_goals = analytics_data.groupby(season_column, as_index=False)[goals_column].sum()
            season_goals = season_goals.sort_values(season_column)
            season_figure = px.line(
                season_goals,
                x=season_column,
                y=goals_column,
                markers=True,
                labels={season_column: "Season", goals_column: "Recorded goals"},
            )
            season_figure.update_traces(
                line_color="#c8f36a", line_width=3,
                marker=dict(size=8, color="#66d8d1"),
            )
            season_figure.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=280,
                margin=dict(t=16, b=35, l=12, r=12),
                font=dict(family="Space Grotesk", color="#aab7ae"),
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
            )
            st.plotly_chart(season_figure, use_container_width=True, config={"displayModeBar": False})
    if goals_column:
        with chart_columns[1]:
            st.markdown("#### Leading scorers in this dataset")
            top_scorers = (
                analytics_data.groupby(analytics_player_column, as_index=False)[goals_column]
                .sum().nlargest(8, goals_column).sort_values(goals_column)
            )
            scorer_figure = px.bar(
                top_scorers,
                x=goals_column,
                y=analytics_player_column,
                orientation="h",
                labels={goals_column: "Recorded goals", analytics_player_column: "Player"},
                color_discrete_sequence=["#66d8d1"],
            )
            scorer_figure.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=280,
                margin=dict(t=16, b=35, l=12, r=12),
                font=dict(family="Space Grotesk", color="#aab7ae"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
                yaxis=dict(showgrid=False),
                showlegend=False,
            )
            st.plotly_chart(scorer_figure, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        f"Overview uses recorded values from {analytics_source}; totals reflect records in this dataset, not career totals."
    )
else:
    st.warning("The dataset overview is unavailable. Check that Data.csv is present and includes player and goals columns.")

st.markdown("#### Analysis tools")

# Auto-predict on any input change
predicted_goals = predict(inputs)
r_label, r_color = rating(predicted_goals)

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if predict_btn:
    st.session_state.prediction_history.insert(0, {
        "Recorded at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "Player": player_name,
        "Predicted goals": round(predicted_goals, 1),
        "Matches": matches,
        "Minutes": mins,
        "Shots": shots,
        "Shots on target": ontarget,
    })
    st.session_state.prediction_history = st.session_state.prediction_history[:8]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Prediction",
    "📊 Feature Importance",
    "🔬 Scenario Builder",
    "📋 Player Report",
    "👥 Player Comparison",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — Prediction
# ══════════════════════════════════════════════════════════════
with tab1:
    col_res, col_stats = st.columns([1, 1.6], gap="large")

    with col_res:
        st.markdown(f"""
        <div class='player-header'>
            <div style='font-size:13px; color:rgba(255,255,255,0.45); letter-spacing:2px; text-transform:uppercase;'>Player</div>
            <div style='font-size:22px; font-weight:700; color:white; margin:4px 0;'>{player_name}</div>
            <div style='font-size:13px; color:rgba(255,255,255,0.5);'>{position} {"· " + team if team else ""}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='result-box'>
            <div class='result-number'>{predicted_goals:.1f}</div>
            <div class='result-label'>Predicted Goals</div>
            <div class='rating-badge' style='background:rgba(255,255,255,0.07); color:{r_color}; border:1px solid {r_color}40;'>
                {r_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_goals,
            number={'font': {'size': 36, 'color': '#39FF14', 'family': 'Space Grotesk'},
                    'suffix': ' G'},
            gauge={
                'axis': {'range': [0, 35], 'tickcolor': 'rgba(255,255,255,0.3)',
                         'tickfont': {'color': 'rgba(255,255,255,0.4)', 'size': 10}},
                'bar': {'color': '#39FF14', 'thickness': 0.25},
                'bgcolor': 'rgba(255,255,255,0.04)',
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 4],   'color': 'rgba(255,45,120,0.15)'},
                    {'range': [4, 8],   'color': 'rgba(255,107,53,0.15)'},
                    {'range': [8, 14],  'color': 'rgba(255,230,0,0.15)'},
                    {'range': [14, 20], 'color': 'rgba(0,212,255,0.15)'},
                    {'range': [20, 35], 'color': 'rgba(57,255,20,0.15)'},
                ],
                'threshold': {'line': {'color': '#00D4FF', 'width': 2}, 'value': predicted_goals},
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}, height=200, margin=dict(t=20, b=10, l=20, r=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

    with col_stats:
        st.markdown("#### 📋 Current Input Stats")

        # Radar chart
        categories = [FEAT_LABELS[f] for f in FEATURES]
        # Normalize values 0-1 for radar
        ranges = {
            'Matches_Played': (1, 38), 'Mins': (90, 3420),
            'xG Per Avg Match': (0, 1.5), 'Shots': (0, 200),
            'OnTarget': (0, 100), 'Shots Per Avg Match': (0, 8),
            'On Target Per Avg Match': (0, 5),
        }
        norm = [(inputs[f] - ranges[f][0]) / (ranges[f][1] - ranges[f][0]) for f in FEATURES]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=norm + [norm[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(57,255,20,0.12)',
            line=dict(color='#39FF14', width=2),
            name=player_name,
            hovertemplate='%{theta}<br>Value: %{r:.2f}<extra></extra>',
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0,1], tickfont=dict(size=9, color='rgba(255,255,255,0.3)'),
                                gridcolor='rgba(255,255,255,0.08)', linecolor='rgba(255,255,255,0.08)'),
                angularaxis=dict(tickfont=dict(size=11, color='rgba(255,255,255,0.6)'),
                                 gridcolor='rgba(255,255,255,0.08)', linecolor='rgba(255,255,255,0.1)'),
                bgcolor='rgba(0,0,0,0)',
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=320,
            margin=dict(t=30, b=30, l=40, r=40),
            showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

        # Stat summary table
        stat_data = []
        for f in FEATURES:
            stat_data.append({
                "": FEAT_ICONS[f],
                "Stat": FEAT_LABELS[f],
                "Value": str(inputs[f]) if isinstance(inputs[f], int) else f"{inputs[f]:.2f}",
                "Importance": f"{FI_DICT[f]*100:.1f}%"
            })
        df_stats = pd.DataFrame(stat_data)
        st.dataframe(df_stats, use_container_width=True, hide_index=True,
                     column_config={"Importance": st.column_config.ProgressColumn(
                         "Model Weight", min_value=0, max_value=100, format="%.1f%%")})


# ══════════════════════════════════════════════════════════════
# TAB 2 — Feature Importance
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### 🧠 What drives the model?")

    col_fi1, col_fi2 = st.columns([1.4, 1], gap="large")

    with col_fi1:
        fi_df = pd.DataFrame({
            'Feature': [FEAT_ICONS[f] + " " + FEAT_LABELS[f] for f in FEATURES],
            'Importance': FI,
            'Color': ['#39FF14','#00D4FF','#FF2D78','#FFE600','#FF6B35','#7B2FBE','#C17FA0']
        }).sort_values('Importance', ascending=True)

        fig_bar = go.Figure(go.Bar(
            x=fi_df['Importance'],
            y=fi_df['Feature'],
            orientation='h',
            marker=dict(color=fi_df['Color'], opacity=0.85,
                        line=dict(color='rgba(255,255,255,0.1)', width=0.5)),
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>',
        ))
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Importance Score', gridcolor='rgba(255,255,255,0.06)',
                       color='rgba(255,255,255,0.5)', showline=False),
            yaxis=dict(color='rgba(255,255,255,0.7)', showgrid=False),
            font=dict(color='white', family='Space Grotesk'),
            height=340, margin=dict(t=20, b=40, l=10, r=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

    with col_fi2:
        # Donut
        fig_pie = go.Figure(go.Pie(
            labels=[FEAT_LABELS[f] for f in FEATURES],
            values=FI,
            hole=0.55,
            marker=dict(colors=['#39FF14','#00D4FF','#FF2D78','#FFE600','#FF6B35','#7B2FBE','#C17FA0'],
                        line=dict(color='#0A0A0F', width=2)),
            hovertemplate='<b>%{label}</b><br>%{percent}<extra></extra>',
            textfont=dict(size=10, color='white'),
        ))
        fig_pie.add_annotation(text="Model<br>Weights", x=0.5, y=0.5,
                               font=dict(size=13, color='rgba(255,255,255,0.5)', family='Space Grotesk'),
                               showarrow=False)
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Space Grotesk'),
            height=340, margin=dict(t=20, b=20, l=0, r=0),
            legend=dict(font=dict(size=10, color='rgba(255,255,255,0.6)'),
                        bgcolor='rgba(0,0,0,0)'),
            showlegend=True,
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})

    st.markdown("""
    <div class='info-card'>
        <div style='font-size:13px; font-weight:700; color:#39FF14; margin-bottom:8px;'>💡 Key Insight</div>
        <div style='font-size:14px; color:rgba(255,255,255,0.7); line-height:1.7;'>
            <strong>Shots on Target</strong> is overwhelmingly the most important feature (~79% importance),
            which aligns with football analytics — a player who consistently hits the target converts goals
            at a much higher rate. <strong>xG per Match</strong> ranks second (~12%), capturing shot quality.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — Scenario Builder
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 🔬 What-if Scenario Analysis")
    st.markdown("<div style='color:rgba(255,255,255,0.5); font-size:13px; margin-bottom:20px;'>Sweep a single feature while keeping everything else constant — see how goals change.</div>", unsafe_allow_html=True)

    sweep_feature = st.selectbox(
        "Feature to sweep",
        options=FEATURES,
        format_func=lambda f: FEAT_ICONS[f] + " " + FEAT_LABELS[f],
        index=4,  # OnTarget by default
    )

    sweep_range = {
        'Matches_Played': np.arange(1, 39, 1),
        'Mins': np.arange(90, 3421, 90),
        'xG Per Avg Match': np.round(np.arange(0.0, 1.51, 0.05), 2),
        'Shots': np.arange(0, 201, 5),
        'OnTarget': np.arange(0, 101, 2),
        'Shots Per Avg Match': np.round(np.arange(0.0, 8.1, 0.2), 1),
        'On Target Per Avg Match': np.round(np.arange(0.0, 5.1, 0.1), 1),
    }

    xs = sweep_range[sweep_feature]
    ys = []
    for v in xs:
        tmp = inputs.copy()
        tmp[sweep_feature] = v
        ys.append(predict(tmp))

    current_x = inputs[sweep_feature]
    current_y = predicted_goals

    fig_line = go.Figure()

    # Shade regions
    for lo, hi, col, lbl in [(0,4,'rgba(255,45,120,0.08)','Needs Work'),
                              (4,8,'rgba(255,107,53,0.08)','Decent'),
                              (8,14,'rgba(255,230,0,0.08)','Quality'),
                              (14,20,'rgba(0,212,255,0.08)','Elite'),
                              (20,50,'rgba(57,255,20,0.08)','World Class')]:
        fig_line.add_hrect(y0=lo, y1=hi, fillcolor=col, line_width=0, annotation_text=lbl,
                           annotation_position="right", annotation_font_size=9,
                           annotation_font_color='rgba(255,255,255,0.3)')

    fig_line.add_trace(go.Scatter(
        x=xs, y=ys,
        mode='lines',
        line=dict(color='#39FF14', width=3),
        fill='tozeroy', fillcolor='rgba(57,255,20,0.07)',
        name='Predicted Goals',
        hovertemplate=f'{FEAT_LABELS[sweep_feature]}: %{{x}}<br>Goals: %{{y:.1f}}<extra></extra>',
    ))

    # Current point
    fig_line.add_trace(go.Scatter(
        x=[current_x], y=[current_y],
        mode='markers',
        marker=dict(color='#FF2D78', size=12, line=dict(color='white', width=2)),
        name='Current Value',
        hovertemplate=f'Current: {current_x}<br>Goals: {current_y:.1f}<extra></extra>',
    ))

    fig_line.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title=FEAT_LABELS[sweep_feature], color='rgba(255,255,255,0.5)',
                   gridcolor='rgba(255,255,255,0.06)'),
        yaxis=dict(title='Predicted Goals', color='rgba(255,255,255,0.5)',
                   gridcolor='rgba(255,255,255,0.06)'),
        font=dict(color='white', family='Space Grotesk'),
        height=380, margin=dict(t=20, b=40, l=50, r=120),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='rgba(255,255,255,0.6)')),
        hovermode='x unified',
    )
    st.plotly_chart(fig_line, use_container_width=True, config={'displayModeBar': False})

    # Quick stats row
    max_goals = max(ys)
    min_goals = min(ys)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📌 Current", f"{current_y:.1f} G")
    c2.metric("🔝 Maximum", f"{max_goals:.1f} G", delta=f"+{max_goals-current_y:.1f}")
    c3.metric("📉 Minimum", f"{min_goals:.1f} G", delta=f"{min_goals-current_y:.1f}")
    c4.metric("📊 Range", f"{max_goals-min_goals:.1f} G")


# ══════════════════════════════════════════════════════════════
# TAB 4 — Player Report
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f"#### 📋 Season Report — {player_name}")

    # Comparison benchmarks
    benchmarks = {
        "World Class (≥20G)": {
            'Matches_Played':30, 'Mins':2700, 'xG Per Avg Match':0.75,
            'Shots':120, 'OnTarget':65, 'Shots Per Avg Match':4.0, 'On Target Per Avg Match':2.2,
        },
        "Elite (14-19G)": {
            'Matches_Played':28, 'Mins':2520, 'xG Per Avg Match':0.55,
            'Shots':95, 'OnTarget':48, 'Shots Per Avg Match':3.4, 'On Target Per Avg Match':1.7,
        },
        "Quality (8-13G)": {
            'Matches_Played':26, 'Mins':2340, 'xG Per Avg Match':0.38,
            'Shots':72, 'OnTarget':32, 'Shots Per Avg Match':2.8, 'On Target Per Avg Match':1.2,
        },
        "Average (4-7G)": {
            'Matches_Played':22, 'Mins':1800, 'xG Per Avg Match':0.22,
            'Shots':50, 'OnTarget':18, 'Shots Per Avg Match':2.2, 'On Target Per Avg Match':0.8,
        },
    }

    # Build comparison df
    rows = []
    for tier, bvals in benchmarks.items():
        p = predict(bvals)
        rows.append({"Tier": tier, **{FEAT_LABELS[f]: bvals[f] for f in FEATURES},
                     "Predicted Goals": round(p, 1)})

    # Add current player
    rows.insert(0, {"Tier": f"🟢 {player_name}", **{FEAT_LABELS[f]: inputs[f] for f in FEATURES},
                    "Predicted Goals": round(predicted_goals, 1)})

    comp_df = pd.DataFrame(rows)

    st.markdown("**📊 Benchmark Comparison**")
    st.dataframe(comp_df, use_container_width=True, hide_index=True,
                 column_config={
                     "Predicted Goals": st.column_config.ProgressColumn(
                         "Predicted Goals ⚽", min_value=0, max_value=35, format="%.1f"),
                 })

    # Summary analysis
    st.markdown("<br>", unsafe_allow_html=True)
    top_fi_feature = max(FI_DICT, key=FI_DICT.get)
    low_fi_feature = min(FI_DICT, key=FI_DICT.get)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"""
        <div class='info-card'>
            <div style='font-size:13px; font-weight:700; color:#39FF14; margin-bottom:10px;'>✅ Strengths</div>
            <div style='font-size:13px; color:rgba(255,255,255,0.7); line-height:1.8;'>
                • Rating: <strong style='color:{r_color}'>{r_label}</strong><br>
                • {FEAT_ICONS[top_fi_feature]} Most impactful stat: <strong>{FEAT_LABELS[top_fi_feature]}</strong><br>
                • On-target shots: <strong>{ontarget}</strong> ({ontarget/max(shots,1)*100:.0f}% conversion rate)<br>
                • xG per match: <strong>{xg:.2f}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        improvement_needed = max(0, 14 - predicted_goals)
        st.markdown(f"""
        <div class='info-card'>
            <div style='font-size:13px; font-weight:700; color:#00D4FF; margin-bottom:10px;'>📈 To Reach Elite Tier</div>
            <div style='font-size:13px; color:rgba(255,255,255,0.7); line-height:1.8;'>
                • Need <strong style='color:#FFE600'>{improvement_needed:.1f} more goals</strong><br>
                • Increase shots on target from <strong>{ontarget}</strong> → <strong>{ontarget + 15}</strong><br>
                • Improve xG per match from <strong>{xg:.2f}</strong> → <strong>{min(xg+0.2, 1.5):.2f}</strong><br>
                • Play <strong>{max(0, 28-matches)}</strong> more matches this season
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Mini sparklines for each stat
    st.markdown("<br>**📉 Stat vs. Benchmarks (Normalized)**", unsafe_allow_html=True)

    bench_vals = benchmarks["World Class (≥20G)"]
    norm_current = [(inputs[f] - ranges[f][0]) / (ranges[f][1] - ranges[f][0]) for f in FEATURES]
    norm_wc = [(bench_vals[f] - ranges[f][0]) / (ranges[f][1] - ranges[f][0]) for f in FEATURES]

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Bar(
        name=player_name, x=[FEAT_LABELS[f] for f in FEATURES], y=norm_current,
        marker_color='#39FF14', opacity=0.8,
    ))
    fig_compare.add_trace(go.Bar(
        name="World Class Benchmark", x=[FEAT_LABELS[f] for f in FEATURES], y=norm_wc,
        marker_color='rgba(0,212,255,0.4)', opacity=0.8,
    ))
    fig_compare.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(color='rgba(255,255,255,0.5)', gridcolor='rgba(255,255,255,0.06)',
                   tickangle=-20),
        yaxis=dict(title='Normalized Score', color='rgba(255,255,255,0.5)',
                   gridcolor='rgba(255,255,255,0.06)', range=[0, 1.1]),
        font=dict(color='white', family='Space Grotesk'),
        height=300, margin=dict(t=20, b=60, l=50, r=20),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='rgba(255,255,255,0.6)')),
    )
    st.plotly_chart(fig_compare, use_container_width=True, config={'displayModeBar': False})

    st.markdown("#### Recent prediction history")
    if st.session_state.prediction_history:
        st.dataframe(
            pd.DataFrame(st.session_state.prediction_history),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Your saved predictions will appear here after you select Predict Goals in the sidebar.")

with tab5:
    render_player_comparison()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='margin-top:40px; border-color:rgba(255,255,255,0.06);'>
<div style='text-align:center; padding:12px 0; font-size:12px; color:rgba(255,255,255,0.25);
            letter-spacing:1px; font-family:"Space Grotesk",sans-serif;'>
    GoalIQ · XGBoost Goal Predictor · Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
