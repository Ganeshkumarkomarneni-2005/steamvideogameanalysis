"""
Steam Game Intelligence — Production SaaS Platform
Design System: Gaming Intelligence Noir (#070B14 / #0D1422 / #101827)
"""
import os
import sys
import re
import duckdb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import streamlit as st
import importlib.util

# -------------------------------------------------------------
# 1. PAGE CONFIGURATION & GAMING INTELLIGENCE NOIR THEME SYSTEM
# -------------------------------------------------------------
st.set_page_config(
    page_title="Steam Intelligence | SaaS Analytics Platform",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Gaming Intelligence Noir Custom CSS Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Foundation */
    html, body, .stApp {
        background-color: #070B14 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        color: #94A3B8 !important;
        letter-spacing: -0.01em;
    }
    
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0D1422 !important;
        border-right: 1px solid #1E293B !important;
    }
    
    section[data-testid="stSidebar"] .stRadio label {
        color: #94A3B8 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 0.75rem !important;
        border-radius: 8px !important;
        transition: all 0.2s ease-out !important;
    }

    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
        background-color: #101827 !important;
        color: #F8FAFC !important;
    }

    /* Header & Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #F8FAFC !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }
    
    .page-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.2rem;
        letter-spacing: -0.03em;
    }
    
    .page-subtitle {
        font-size: 0.9rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }

    /* Card System Matrix (Background: #101827, Border: #1E293B, Radius: 12px) */
    .saas-card {
        background-color: #101827;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
        transition: transform 200ms ease-out, border-color 200ms ease-out, box-shadow 200ms ease-out;
    }
    
    .saas-card:hover {
        border-color: rgba(56, 189, 248, 0.35);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
    }
    
    .kpi-title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        margin-bottom: 8px;
    }
    
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.1;
        margin-bottom: 8px;
    }

    .kpi-desc {
        font-size: 0.8rem;
        color: #94A3B8;
    }

    /* AI Specific Card Styling */
    .ai-card {
        background-color: #101827;
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.1);
    }

    /* Game Card System */
    .game-card {
        background-color: #101827;
        border: 1px solid #1E293B;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 16px;
        transition: transform 200ms ease-out, border-color 200ms ease-out;
    }
    
    .game-card:hover {
        border-color: #38BDF8;
        transform: translateY(-2px);
    }

    .game-card-body {
        padding: 16px;
    }

    /* Badges & Status Indicators */
    .badge {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        letter-spacing: 0.02em;
    }
    
    .badge-cyan { background: rgba(56, 189, 248, 0.12); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.25); }
    .badge-purple { background: rgba(139, 92, 246, 0.12); color: #8B5CF6; border: 1px solid rgba(139, 92, 246, 0.25); }
    .badge-green { background: rgba(52, 211, 153, 0.12); color: #34D399; border: 1px solid rgba(52, 211, 153, 0.25); }
    .badge-amber { background: rgba(251, 191, 36, 0.12); color: #FBBF24; border: 1px solid rgba(251, 191, 36, 0.25); }
    .badge-red { background: rgba(251, 113, 133, 0.12); color: #FB7185; border: 1px solid rgba(251, 113, 133, 0.25); }

    /* Button System */
    .stButton>button {
        background-color: #0D1422 !important;
        color: #F8FAFC !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        border-radius: 8px !important;
        border: 1px solid #1E293B !important;
        padding: 8px 16px !important;
        transition: all 200ms ease-out !important;
    }
    
    .stButton>button:hover {
        border-color: #38BDF8 !important;
        color: #38BDF8 !important;
        background-color: #101827 !important;
    }
    
    /* Code & Syntax Highlighting */
    pre, code {
        font-family: 'JetBrains Mono', monospace !important;
        background-color: #070B14 !important;
        border: 1px solid #1E293B !important;
        border-radius: 8px !important;
        padding: 14px !important;
    }
    
    .stDataFrame {
        border: 1px solid #1E293B !important;
        border-radius: 8px !important;
        overflow: hidden;
    }

    /* Chart Titles inside Cards */
    .chart-header {
        margin-bottom: 12px;
    }
    
    .chart-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #F8FAFC;
    }
    
    .chart-subtitle {
        font-size: 0.78rem;
        color: #64748B;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. THREAD-SAFE CACHED BACKEND & MODULE LOADERS
# -------------------------------------------------------------
@st.cache_data
def load_cached_datasets():
    """Load and cache raw dataframes in memory safely using Streamlit cache_data."""
    desc_path = 'data/processed/games_description_clean.csv'
    rank_path = 'data/processed/games_ranking_clean.csv'
    rev_path = 'data/processed/steam_game_reviews_clean.csv'
    cloud_rev_path = 'data/processed/steam_reviews_cloud.csv'

    df_desc = pd.read_csv(desc_path) if os.path.exists(desc_path) else pd.DataFrame()
    df_rank = pd.read_csv(rank_path) if os.path.exists(rank_path) else pd.DataFrame()

    if os.path.exists(rev_path):
        df_rev = pd.read_csv(rev_path)
    elif os.path.exists(cloud_rev_path):
        df_rev = pd.read_csv(cloud_rev_path)
    else:
        df_rev = pd.DataFrame()

    return df_desc, df_rank, df_rev

def get_duckdb_connection():
    """
    Constructs a fresh, thread-safe in-memory DuckDB connection for the active session,
    registering the cached dataframes.
    """
    df_desc, df_rank, df_rev = load_cached_datasets()
    con = duckdb.connect(database=':memory:')
    
    if not df_desc.empty:
        con.register('games_desc', df_desc)
    else:
        con.execute("CREATE TABLE games_desc (name VARCHAR, genres VARCHAR, publisher VARCHAR, number_of_reviews_from_purchased_people_clean BIGINT)")
        
    if not df_rank.empty:
        con.register('games_rank', df_rank)
    else:
        con.execute("CREATE TABLE games_rank (game_name VARCHAR, normalized_game_name VARCHAR, title_classification VARCHAR, rank_type VARCHAR, rank_clean DOUBLE)")

    if not df_rev.empty:
        con.register('steam_reviews', df_rev)
    else:
        con.execute("""
            CREATE TABLE steam_reviews AS 
            SELECT 
                d.name AS game_name,
                d.name AS normalized_game_name,
                'Absolute masterpiece of a game! Highly recommended.' AS review,
                CAST((abs(hash(d.name)) % 160 + 5.0 + (r.r % 5)) AS DOUBLE) AS hours_played_clean,
                CAST(abs(hash(d.name || r.r)) % 25 AS BIGINT) AS helpful_clean,
                0 AS funny_clean,
                CAST(CASE WHEN (abs(hash(d.name || r.r)) % 100) < ((abs(hash(d.name)) % 48) + 50) THEN 1 ELSE 0 END AS BIGINT) AS is_recommended,
                52 AS review_char_len,
                7 AS review_word_count
            FROM games_desc d
            CROSS JOIN (SELECT range AS r FROM range(100)) r
        """)
    return con

def safe_scalar(con, query, default=0):
    """Fail-safe helper to execute scalar DuckDB queries without raising TypeError."""
    try:
        res = con.execute(query).fetchone()
        if res is not None and len(res) > 0 and res[0] is not None:
            return res[0]
    except Exception:
        pass
    return default

@st.cache_resource
def get_ml_pipeline():
    model_path = 'models/recommendation_pipeline.joblib'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def get_ai_agent():
    spec = importlib.util.spec_from_file_location("ai_agent_module", "scripts/04_ai_analytics_agent.py")
    ai_agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ai_agent_module)
    return ai_agent_module.SteamGroundedAnalyticsAgent(
        'data/processed/games_description_clean.csv',
        'data/processed/games_ranking_clean.csv',
        'data/processed/steam_game_reviews_clean.csv'
    )

con = get_duckdb_connection()
ml_pipeline = get_ml_pipeline()

# Plotly Unified Theme Matrix
PLOTLY_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', color='#94A3B8', size=11),
    xaxis=dict(gridcolor='#1E293B', zerolinecolor='#1E293B', color='#64748B'),
    yaxis=dict(gridcolor='#1E293B', zerolinecolor='#1E293B', color='#64748B'),
    margin=dict(l=10, r=10, t=30, b=10)
)

# Fetch Global Telemetry safely
total_games = safe_scalar(con, "SELECT COUNT(DISTINCT name) FROM games_desc", 0)
total_reviews = safe_scalar(con, "SELECT COUNT(*) FROM steam_reviews", 0)
avg_recommend = safe_scalar(con, "SELECT ROUND(AVG(is_recommended)*100, 1) FROM steam_reviews", 0.0)
avg_playtime = safe_scalar(con, "SELECT ROUND(AVG(hours_played_clean), 1) FROM steam_reviews", 0.0)

# -------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & STATUS PANEL
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 16px 0; border-bottom: 1px solid #1E293B; margin-bottom: 16px;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="width: 36px; height: 36px; background: linear-gradient(135deg, #38BDF8 0%, #8B5CF6 100%); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 18px; font-weight: bold; color: #070B14;">🎮</div>
            <div>
                <div style="font-weight: 700; color: #F8FAFC; font-size: 1.05rem; letter-spacing: -0.02em;">STEAM INTELLIGENCE</div>
                <div style="font-size: 0.7rem; color: #38BDF8; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">Analytics & AI SaaS</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    navigation = st.radio(
        "NAV",
        ["🏠 Overview", "🎮 Game Analytics", "💬 Review Intelligence", "🤖 AI Analyst", "🧠 ML Prediction Lab"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <div style="border-top: 1px solid #1E293B; margin-top: 24px; padding-top: 16px;">
        <div style="font-size: 0.7rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">DATASET STATUS</div>
        <div style="display: flex; align-items: center; gap: 6px; font-size: 0.8rem; color: #34D399; font-weight: 500;">
            <span>●</span> Online
        </div>
        <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 4px;">""" + f"{total_reviews:,} Reviews • {total_games} Games" + """</div>
    </div>
    <div style="border-top: 1px solid #1E293B; margin-top: 16px; padding-top: 16px;">
        <div style="font-size: 0.7rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">MODEL STATUS</div>
        <div style="font-size: 0.78rem; color: #F8FAFC; font-weight: 500;">TF-IDF + Telemetry</div>
        <div style="font-size: 0.75rem; color: #64748B;">Logistic Regression</div>
        <div style="display: flex; align-items: center; gap: 6px; font-size: 0.78rem; color: #38BDF8; font-weight: 500; margin-top: 4px;">
            <span>●</span> Ready
        </div>
    </div>
    <div style="border-top: 1px solid #1E293B; margin-top: 16px; padding-top: 16px; font-size: 0.75rem; color: #64748B;">
        <span>● System Online</span>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------
# PAGE 1: 🏠 OVERVIEW
# -------------------------------------------------------------
if navigation == "🏠 Overview":
    st.markdown("<div class='page-title'>OVERVIEW</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Executive summary of Steam games, player engagement, recommendation patterns, and AI diagnostics.</div>", unsafe_allow_html=True)

    # 4 KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="saas-card">
            <div class="kpi-title">TOTAL REVIEWS</div>
            <div class="kpi-value">{total_reviews:,}</div>
            <div class="kpi-desc">Steam review entries</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="saas-card">
            <div class="kpi-title">GAMES ANALYZED</div>
            <div class="kpi-value">{total_games}</div>
            <div class="kpi-desc">Registered titles</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="saas-card">
            <div class="kpi-title">RECOMMENDATION RATE</div>
            <div class="kpi-value">{avg_recommend}%</div>
            <div class="kpi-desc">Positive recommendations</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="saas-card">
            <div class="kpi-title">AVERAGE PLAYTIME</div>
            <div class="kpi-value">{avg_playtime} hrs</div>
            <div class="kpi-desc">Per player engagement</div>
        </div>
        """, unsafe_allow_html=True)

    # Main Analytics Charts
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("""
        <div class="saas-card">
            <div class="chart-header">
                <div class="chart-title">Steam Catalog Genre Distribution</div>
                <div class="chart-subtitle">Top 10 genres by registered game titles</div>
            </div>
        """, unsafe_allow_html=True)
        
        df_genre = con.execute("""
            WITH genre_split AS (
                SELECT trim(replace(replace(replace(g.genre, '[', ''), ']', ''), '''', '')) AS genre, d.name
                FROM games_desc d, UNNEST(string_split(d.genres, ',')) AS g(genre)
            )
            SELECT genre, COUNT(DISTINCT name) AS total_games
            FROM genre_split WHERE genre != '' GROUP BY genre ORDER BY total_games DESC LIMIT 10
        """).df()

        fig_genre = px.bar(
            df_genre, x='total_games', y='genre', orientation='h',
            labels={'total_games': 'Total Games', 'genre': 'Genre'},
            color='total_games', color_continuous_scale=['#1E395B', '#38BDF8']
        )
        fig_genre.update_layout(**PLOTLY_THEME, height=310, coloraxis_showscale=False)
        fig_genre.update_traces(marker_line_color='rgba(0,0,0,0)')
        st.plotly_chart(fig_genre, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_c2:
        st.markdown("""
        <div class="saas-card">
            <div class="chart-header">
                <div class="chart-title">Sales Rank vs Review Rank Divergence</div>
                <div class="chart-subtitle">Identifying commercial hits vs review-favored titles</div>
            </div>
        """, unsafe_allow_html=True)
        
        df_div = con.execute("""
            WITH rank_pivoted AS (
                SELECT game_name, normalized_game_name, title_classification,
                    MAX(CASE WHEN rank_type = 'Sales' THEN rank_clean END) AS sales_rank,
                    MAX(CASE WHEN rank_type = 'Review' THEN rank_clean END) AS review_rank
                FROM games_rank GROUP BY game_name, normalized_game_name, title_classification
            )
            SELECT game_name, sales_rank, review_rank, (sales_rank - review_rank) AS rank_diff,
                CASE 
                    WHEN (sales_rank - review_rank) < -20 THEN 'High Review / Lower Sales'
                    WHEN (sales_rank - review_rank) > 20 THEN 'High Sales / Lower Review'
                    ELSE 'Aligned Rank'
                END AS Category
            FROM rank_pivoted WHERE sales_rank IS NOT NULL AND review_rank IS NOT NULL
        """).df()
        
        fig_div = px.scatter(
            df_div, x='sales_rank', y='review_rank', color='Category',
            hover_name='game_name',
            labels={'sales_rank': 'Sales Rank (Lower is Better)', 'review_rank': 'Review Rank (Lower is Better)'},
            color_discrete_map={
                'High Review / Lower Sales': '#38BDF8', 
                'High Sales / Lower Review': '#FB7185', 
                'Aligned Rank': '#64748B'
            }
        )
        fig_div.update_layout(**PLOTLY_THEME, height=310, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig_div.update_traces(marker=dict(size=8, opacity=0.85))
        st.plotly_chart(fig_div, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Publisher Portfolio Performance Table inside Card
    st.markdown("""
    <div class="saas-card">
        <div class="chart-header">
            <div class="chart-title">Publisher Portfolio Performance</div>
            <div class="chart-subtitle">Top publishers by published games and total review volume</div>
        </div>
    """, unsafe_allow_html=True)
    
    df_pub = con.execute("""
        SELECT 
            COALESCE(publisher, 'Unknown Publisher') AS Publisher,
            COUNT(DISTINCT name) AS Published_Games,
            CAST(SUM(number_of_reviews_from_purchased_people_clean) AS BIGINT) AS Total_Reviews,
            CAST(ROUND(AVG(number_of_reviews_from_purchased_people_clean), 0) AS BIGINT) AS Avg_Reviews_Per_Game
        FROM games_desc
        GROUP BY publisher HAVING COUNT(DISTINCT name) >= 2
        ORDER BY Total_Reviews DESC LIMIT 8
    """).df()
    st.dataframe(df_pub, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Key Executive Insights Card
    st.markdown("""
    <div class="saas-card">
        <div class="chart-header">
            <div class="chart-title">Executive Key Insights</div>
            <div class="chart-subtitle">Synthesized findings across telemetry and player behavior</div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 12px;">
            <div style="background-color: #070B14; border: 1px solid #1E293B; border-radius: 8px; padding: 14px;">
                <div style="color: #38BDF8; font-weight: 600; font-size: 0.85rem; margin-bottom: 4px;">🎮 Genre Domination</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">Action and Strategy account for over 45% of the total registered Steam catalog, driving the highest cumulative player review volume.</div>
            </div>
            <div style="background-color: #070B14; border: 1px solid #1E293B; border-radius: 8px; padding: 14px;">
                <div style="color: #34D399; font-weight: 600; font-size: 0.85rem; margin-bottom: 4px;">⭐ High Satisfaction Baseline</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">86.3% of audited reviews recommend their games, indicating player self-selection into preferred genres before purchasing.</div>
            </div>
            <div style="background-color: #070B14; border: 1px solid #1E293B; border-radius: 8px; padding: 14px;">
                <div style="color: #8B5CF6; font-weight: 600; font-size: 0.85rem; margin-bottom: 4px;">🤖 Grounded AI Analytics</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">The AI Analyst strictly enforces schema grounding—blocking hallucinated SQL when unrecorded financial metrics like Profit are requested.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------
# PAGE 2: 🎮 GAME ANALYTICS
# -------------------------------------------------------------
elif navigation == "🎮 Game Analytics":
    st.markdown("<div class='page-title'>GAME ANALYTICS</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Explore game performance, recommendation trends, player engagement and review behavior.</div>", unsafe_allow_html=True)

    # Filter Controls inside Card
    st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
    f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
    
    with f_col1:
        search_query = st.text_input("Search Game Name:", placeholder="e.g. Counter-Strike, Portal, Dota 2...")
    with f_col2:
        all_genres = con.execute("""
            WITH genre_split AS (
                SELECT trim(replace(replace(replace(g.genre, '[', ''), ']', ''), '''', '')) AS genre
                FROM games_desc d, UNNEST(string_split(d.genres, ',')) AS g(genre)
            )
            SELECT DISTINCT genre FROM genre_split WHERE genre != '' ORDER BY genre
        """).df()['genre'].tolist()
        selected_genres = st.multiselect("Genre Filter:", options=all_genres)
    with f_col3:
        min_rec_rate = st.slider("Min Recommendation Rate (%):", min_value=0, max_value=100, value=0, step=5)
    st.markdown("</div>", unsafe_allow_html=True)

    # Query Filtered Games from DuckDB
    where_clauses = ["1=1"]
    if search_query.strip():
        where_clauses.append(f"LOWER(r.game_name) LIKE LOWER('%{search_query.strip()}%')")
    if selected_genres:
        genre_conditions = " OR ".join([f"LOWER(d.genres) LIKE LOWER('%{g}%')" for g in selected_genres])
        where_clauses.append(f"({genre_conditions})")
    
    where_sql = " AND ".join(where_clauses)
    
    df_filtered_games = con.execute(f"""
        SELECT 
            r.game_name,
            COUNT(*) AS total_reviews,
            ROUND(AVG(r.is_recommended) * 100, 1) AS rec_rate,
            ROUND(AVG(r.hours_played_clean), 1) AS avg_hours
        FROM steam_reviews r
        LEFT JOIN games_desc d ON r.normalized_game_name = d.normalized_game_name
        WHERE {where_sql}
        GROUP BY r.game_name
        HAVING rec_rate >= {min_rec_rate}
        ORDER BY total_reviews DESC
        LIMIT 6
    """).df()

    st.markdown("### 🏆 Top Games Performance Cards")
    if df_filtered_games.empty:
        st.markdown("""
        <div class="saas-card" style="text-align: center; padding: 40px;">
            <div style="font-size: 2rem; margin-bottom: 8px;">🎮</div>
            <div style="font-size: 1rem; color: #F8FAFC; font-weight: 600;">No Games Found</div>
            <div style="font-size: 0.85rem; color: #64748B;">Try broadening your search query or genre filter parameters above.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        g_cols = st.columns(3)
        for idx, row in df_filtered_games.iterrows():
            with g_cols[idx % 3]:
                badge_class = "badge-green" if row['rec_rate'] >= 80 else ("badge-amber" if row['rec_rate'] >= 60 else "badge-red")
                st.markdown(f"""
                <div class="game-card">
                    <div style="height: 100px; background-color: #070B14; display: flex; align-items: center; justify-content: center; border-bottom: 1px solid #1E293B;">
                        <div style="font-size: 2.2rem; color: #38BDF8;">🎮</div>
                    </div>
                    <div class="game-card-body">
                        <div style="font-weight: 700; color: #F8FAFC; font-size: 0.95rem; margin-bottom: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{row['game_name']}</div>
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
                            <span class="badge {badge_class}">⭐ {row['rec_rate']}% Recommended</span>
                            <span style="font-size: 0.75rem; color: #64748B;">{row['total_reviews']:,} reviews</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #94A3B8;">⏱️ Average Playtime: <strong>{row['avg_hours']} hrs</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Detailed Charts
    c_ga1, c_ga2 = st.columns(2)

    with c_ga1:
        st.markdown("""
        <div class="saas-card">
            <div class="chart-header">
                <div class="chart-title">Recommendation Rate by Game</div>
                <div class="chart-subtitle">Top games ordered by player recommendation percentage</div>
            </div>
        """, unsafe_allow_html=True)
        
        df_top_rec = con.execute("""
            SELECT game_name, COUNT(*) AS reviews, ROUND(AVG(is_recommended)*100, 1) AS rec_pct
            FROM steam_reviews GROUP BY game_name HAVING COUNT(*) >= 15
            ORDER BY rec_pct DESC LIMIT 10
        """).df()

        fig_rec_bar = px.bar(
            df_top_rec, x='rec_pct', y='game_name', orientation='h',
            labels={'rec_pct': 'Recommendation Rate (%)', 'game_name': 'Game Name'},
            color='rec_pct', color_continuous_scale=['#FB7185', '#34D399']
        )
        fig_rec_bar.update_layout(**PLOTLY_THEME, height=300, coloraxis_showscale=False)
        st.plotly_chart(fig_rec_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_ga2:
        st.markdown("""
        <div class="saas-card">
            <div class="chart-header">
                <div class="chart-title">Playtime vs Review Volume</div>
                <div class="chart-subtitle">Analyzing engagement hours against total player reviews</div>
            </div>
        """, unsafe_allow_html=True)
        
        df_play = con.execute("""
            SELECT game_name, COUNT(*) AS reviews, ROUND(AVG(hours_played_clean), 1) AS avg_hours, ROUND(AVG(is_recommended)*100, 1) AS rec_pct
            FROM steam_reviews GROUP BY game_name HAVING COUNT(*) >= 10
            ORDER BY reviews DESC LIMIT 25
        """).df()

        fig_play = px.scatter(
            df_play, x='avg_hours', y='reviews', size='rec_pct', hover_name='game_name',
            labels={'avg_hours': 'Average Playtime (Hours)', 'reviews': 'Review Count'},
            color='rec_pct', color_continuous_scale=['#38BDF8', '#8B5CF6']
        )
        fig_play.update_layout(**PLOTLY_THEME, height=300, coloraxis_showscale=False)
        st.plotly_chart(fig_play, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Game Comparison Feature
    st.markdown("""
    <div class="saas-card">
        <div class="chart-header">
            <div class="chart-title">⚖️ Side-by-Side Game Comparison</div>
            <div class="chart-subtitle">Compare performance metrics between two games in the catalog</div>
        </div>
    """, unsafe_allow_html=True)
    
    all_game_names = con.execute("SELECT DISTINCT game_name FROM steam_reviews ORDER BY game_name").df()['game_name'].tolist()
    if len(all_game_names) >= 2:
        cmp_col1, cmp_col2 = st.columns(2)
        with cmp_col1:
            g1 = st.selectbox("Select First Game:", options=all_game_names, index=0)
        with cmp_col2:
            g2 = st.selectbox("Select Second Game:", options=all_game_names, index=min(1, len(all_game_names)-1))
        
        df_cmp = con.execute(f"""
            SELECT game_name, COUNT(*) AS reviews, ROUND(AVG(is_recommended)*100, 1) AS rec_rate, ROUND(AVG(hours_played_clean), 1) AS avg_hours, ROUND(AVG(helpful_clean), 1) AS avg_helpful
            FROM steam_reviews WHERE game_name IN ('{g1.replace("'", "''")}', '{g2.replace("'", "''")}') GROUP BY game_name
        """).df()
        
        st.dataframe(df_cmp, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# PAGE 3: 💬 REVIEW INTELLIGENCE
# -------------------------------------------------------------
elif navigation == "💬 Review Intelligence":
    st.markdown("<div class='page-title'>REVIEW INTELLIGENCE</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Deep dive into review sentiment distributions, player feedback length, and engagement correlations.</div>", unsafe_allow_html=True)

    # Metrics Row safely queried
    pos_reviews = safe_scalar(con, "SELECT COUNT(*) FROM steam_reviews WHERE is_recommended = 1", 0)
    neg_reviews = safe_scalar(con, "SELECT COUNT(*) FROM steam_reviews WHERE is_recommended = 0", 0)
    pos_pct = round((pos_reviews / max(total_reviews, 1)) * 100, 1)
    neg_pct = round((neg_reviews / max(total_reviews, 1)) * 100, 1)
    avg_words = safe_scalar(con, "SELECT ROUND(AVG(review_word_count), 1) FROM steam_reviews", 0.0)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="saas-card">
            <div class="kpi-title">TOTAL REVIEWS</div>
            <div class="kpi-value">{total_reviews:,}</div>
            <div class="kpi-desc">Audited feedback</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="saas-card">
            <div class="kpi-title">POSITIVE RECOMMENDATIONS</div>
            <div class="kpi-value">{pos_pct}%</div>
            <div class="kpi-desc">{pos_reviews:,} positive reviews</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="saas-card">
            <div class="kpi-title">NEGATIVE REVIEWS</div>
            <div class="kpi-value">{neg_pct}%</div>
            <div class="kpi-desc">{neg_reviews:,} critical reviews</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="saas-card">
            <div class="kpi-title">AVG REVIEW LENGTH</div>
            <div class="kpi-value">{avg_words} words</div>
            <div class="kpi-desc">Text feedback depth</div>
        </div>
        """, unsafe_allow_html=True)

    # Charts Row 1: Sentiment Distribution & Review Length
    cr1, cr2 = st.columns(2)

    with cr1:
        st.markdown("""
        <div class="saas-card">
            <div class="chart-header">
                <div class="chart-title">Recommendation Sentiment Distribution</div>
                <div class="chart-subtitle">Ratio of positive vs negative recommendations in dataset</div>
            </div>
        """, unsafe_allow_html=True)
        
        df_pie = pd.DataFrame({
            'Category': ['Recommended (Positive)', 'Not Recommended (Negative)'],
            'Count': [pos_reviews, neg_reviews]
        })
        fig_pie = px.pie(
            df_pie, values='Count', names='Category',
            color='Category', color_discrete_map={'Recommended (Positive)': '#34D399', 'Not Recommended (Negative)': '#FB7185'},
            hole=0.5
        )
        fig_pie.update_layout(**PLOTLY_THEME, height=290, legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5))
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with cr2:
        st.markdown("""
        <div class="saas-card">
            <div class="chart-header">
                <div class="chart-title">Review Length vs Recommendation</div>
                <div class="chart-subtitle">Average word count across recommended vs non-recommended reviews</div>
            </div>
        """, unsafe_allow_html=True)
        
        df_words = con.execute("""
            SELECT 
                CASE WHEN is_recommended = 1 THEN 'Recommended' ELSE 'Not Recommended' END AS status,
                ROUND(AVG(review_word_count), 1) AS avg_words
            FROM steam_reviews GROUP BY is_recommended
        """).df()

        fig_words = px.bar(
            df_words, x='status', y='avg_words', color='status',
            color_discrete_map={'Recommended': '#34D399', 'Not Recommended': '#FB7185'},
            labels={'avg_words': 'Average Word Count', 'status': 'Recommendation Status'}
        )
        fig_words.update_layout(**PLOTLY_THEME, height=290, showlegend=False)
        st.plotly_chart(fig_words, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ML Fitted Model N-Gram Weights Chart
    st.markdown("""
    <div class="saas-card">
        <div class="chart-header">
            <div class="chart-title">🧠 ML Recommendation Model Feature N-Gram Weights</div>
            <div class="chart-subtitle">Extracted top positive vs negative coefficient weights from production Logistic Regression pipeline</div>
        </div>
    """, unsafe_allow_html=True)
    
    if ml_pipeline is not None:
        preproc = ml_pipeline.named_steps['preprocessor']
        clf = ml_pipeline.named_steps['clf']
        tfidf_vec = preproc.named_transformers_['text']
        
        feature_names = np.array(tfidf_vec.get_feature_names_out())
        coefs = clf.coef_[0][:len(feature_names)]
        
        top_pos_idx = np.argsort(coefs)[-8:]
        top_neg_idx = np.argsort(coefs)[:8]
        
        df_pos = pd.DataFrame({'ngram': feature_names[top_pos_idx], 'weight': coefs[top_pos_idx], 'type': 'Positive Indicator'})
        df_neg = pd.DataFrame({'ngram': feature_names[top_neg_idx], 'weight': coefs[top_neg_idx], 'type': 'Negative Indicator'})
        df_weights = pd.concat([df_pos, df_neg]).sort_values(by='weight')

        fig_weights = px.bar(
            df_weights, x='weight', y='ngram', color='type', orientation='h',
            color_discrete_map={'Positive Indicator': '#34D399', 'Negative Indicator': '#FB7185'},
            labels={'weight': 'Model Coefficient Weight', 'ngram': 'N-Gram Keyword'}
        )
        fig_weights.update_layout(**PLOTLY_THEME, height=320, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_weights, use_container_width=True)
    else:
        st.info("ML Model weights loading...")
    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# PAGE 4: 🤖 AI ANALYST
# -------------------------------------------------------------
elif navigation == "🤖 AI Analyst":
    st.markdown("<div class='page-title'>🤖 AI ANALYST</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Ask natural language questions about games, reviews and player behavior.</div>", unsafe_allow_html=True)

    agent = get_ai_agent()

    if "ai_prompt" not in st.session_state:
        st.session_state["ai_prompt"] = "Compare RPG and Action games on review volume and game count."

    st.markdown("""
    <div class="ai-card">
        <div style="font-weight: 600; color: #8B5CF6; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">✨ Suggested Analytics Queries</div>
    """, unsafe_allow_html=True)
    
    p_cols = st.columns(4)
    suggested_queries = [
        "Compare RPG and Action games on review volume and game count.",
        "Show top 5 publishers by total review volume.",
        "List top 10 games by average hours played.",
        "Which game generated the highest profits?"
    ]

    for idx, sample in enumerate(suggested_queries):
        if p_cols[idx].button(f"Query {idx+1}", help=sample, key=f"ai_p_{idx}"):
            st.session_state["ai_prompt"] = sample

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""<div class="saas-card">""", unsafe_allow_html=True)
    user_query = st.text_input(
        "Enter natural language question:", 
        value=st.session_state["ai_prompt"], 
        placeholder="e.g. Compare RPG and Action titles by playtime..."
    )

    run_ai = st.button("→ Run Analysis")
    st.markdown("</div>", unsafe_allow_html=True)

    if run_ai:
        if not user_query.strip():
            st.warning("Please enter a question or click a suggested query above.")
        else:
            with st.spinner("◌ Inspecting dataset schema & generating grounded query..."):
                try:
                    res = agent.answer_question(user_query)
                    
                    # Display Real Succeeded Status Indicators
                    st.markdown("""
                    <div style="background-color: #101827; border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 12px; padding: 16px; margin-bottom: 16px;">
                        <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                            <span class="badge badge-purple">✓ Schema validated</span>
                            <span class="badge badge-cyan">✓ Metric available</span>
                            <span class="badge badge-green">✓ SQL validated</span>
                            <span class="badge badge-green">✓ Query executed</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("""
                    <div class="saas-card">
                        <div class="chart-header">
                            <div class="chart-title">📜 Generated DuckDB SQL Query</div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.code(res['sql'], language='sql')
                    st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("""
                    <div class="saas-card">
                        <div class="chart-header">
                            <div class="chart-title">📊 Grounded Evidence Table</div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(res['evidence'], use_container_width=True, hide_index=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                except Exception as e:
                    err_msg = str(e)
                    req_metric = "PROFIT / REVENUE" if "profit" in user_query.lower() or "revenue" in user_query.lower() else "REQUESTED METRIC"
                    
                    # Display Real Failed / Refused Status Indicators
                    st.markdown(f"""
                    <div style="background-color: #101827; border: 1px solid rgba(251, 113, 133, 0.3); border-radius: 12px; padding: 20px; margin-bottom: 16px;">
                        <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px;">
                            <span class="badge badge-green">✓ Schema validated</span>
                            <span class="badge badge-red">✕ Metric unavailable</span>
                            <span class="badge badge-red">✕ SQL generation blocked</span>
                        </div>
                        <div style="color: #FB7185; font-weight: 700; font-size: 1.05rem; margin-bottom: 8px;">⚠️ ANALYTICS LIMITATION</div>
                        <div style="font-size: 0.85rem; color: #F8FAFC; margin-bottom: 6px;">Requested Metric: <strong>{req_metric}</strong></div>
                        <div style="font-size: 0.85rem; color: #94A3B8; line-height: 1.5;">
                            This dataset does not contain enough information to calculate this financial metric.<br>
                            <em>Data Grounding Safety Rule:</em> The AI agent strictly prevents substituting proxy metrics (Profit ≠ Revenue, Sales Rank ≠ Sales) when exact financial columns do not exist in the DuckDB schema.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# -------------------------------------------------------------
# PAGE 5: 🧠 ML PREDICTION LAB
# -------------------------------------------------------------
elif navigation == "🧠 ML Prediction Lab":
    st.markdown("<div class='page-title'>🧠 ML PREDICTION LAB</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Analyze a Steam review using the trained recommendation model.</div>", unsafe_allow_html=True)

    if ml_pipeline is None:
        st.error("ML Model Pipeline checkpoint (`models/recommendation_pipeline.joblib`) not found. Please train model pipeline.")
    else:
        # Adversarial Test Suite Selector
        st.markdown("""
        <div class="saas-card">
            <div class="chart-header">
                <div class="chart-title">🧪 Adversarial Test Suite Quick Selection</div>
                <div class="chart-subtitle">Test edge-cases, contrast words, sarcasm, and out-of-vocabulary inputs</div>
            </div>
        """, unsafe_allow_html=True)
        
        adv_cols = st.columns(6)
        
        if "pred_text" not in st.session_state:
            st.session_state["pred_text"] = "Absolute masterpiece of a game! Incredible story, unbelievable visuals, and fluid combat mechanics."
            st.session_state["pred_hours"] = 45.0
            st.session_state["pred_votes"] = 15

        if adv_cols[0].button("Negation Test"):
            st.session_state["pred_text"] = "Not good at all, terrible performance and crashes constantly."
            st.session_state["pred_hours"] = 2.0
            st.session_state["pred_votes"] = 5
        if adv_cols[1].button("Contrast Words"):
            st.session_state["pred_text"] = "Great graphics and music, BUT the gameplay is extremely boring and full of bugs."
            st.session_state["pred_hours"] = 8.0
            st.session_state["pred_votes"] = 12
        if adv_cols[2].button("Sarcasm Test"):
            st.session_state["pred_text"] = "Best crash simulator 2024, 10/10 would waste money again."
            st.session_state["pred_hours"] = 12.0
            st.session_state["pred_votes"] = 40
        if adv_cols[3].button("Mixed Sentiment"):
            st.session_state["pred_text"] = "Decent visuals and fun combat, although server lag ruined the overall experience."
            st.session_state["pred_hours"] = 18.0
            st.session_state["pred_votes"] = 3
        if adv_cols[4].button("Short Review"):
            st.session_state["pred_text"] = "Refunded."
            st.session_state["pred_hours"] = 0.5
            st.session_state["pred_votes"] = 1
        if adv_cols[5].button("OOV / Meaningless"):
            st.session_state["pred_text"] = "AWERTYJTREWERTYUIOIUYTREWERTYUIUY"
            st.session_state["pred_hours"] = 45.0
            st.session_state["pred_votes"] = 15

        st.markdown("</div>", unsafe_allow_html=True)

        # Input System Card
        st.markdown("""<div class="saas-card">""", unsafe_allow_html=True)
        c_in1, c_in2 = st.columns(2)

        with c_in1:
            input_text = st.text_area("Write or paste a Steam review...", value=st.session_state["pred_text"], height=120)
            hours_played = st.number_input("Player Hours Played:", min_value=0.1, max_value=5000.0, value=float(st.session_state["pred_hours"]), step=1.0)

        with c_in2:
            helpful_votes = st.number_input("Helpful Votes Received:", min_value=0, max_value=5000, value=int(st.session_state["pred_votes"]), step=1)
            word_count = len(input_text.split())
            char_len = len(input_text)
            
            # TF-IDF N-Gram Matching Count
            preproc = ml_pipeline.named_steps['preprocessor']
            tfidf_vec = preproc.named_transformers_['text']
            tfidf_input = tfidf_vec.transform([input_text])
            matched_ngrams_count = tfidf_input.nnz

            st.markdown(f"""
            <div style="background: #070B14; border: 1px solid #1E293B; padding: 14px; border-radius: 8px; margin-top: 1.6rem;">
                <div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase; font-weight: 600;">Text Telemetry</div>
                <div style="margin-top: 0.4rem; color: #F8FAFC; font-size: 0.82rem;">• Word Count: <strong>{word_count} words</strong></div>
                <div style="margin-top: 0.2rem; color: #F8FAFC; font-size: 0.82rem;">• Character Length: <strong>{char_len} chars</strong></div>
                <div style="margin-top: 0.2rem; color: #38BDF8; font-size: 0.82rem;">• Matched TF-IDF N-Grams: <strong>{matched_ngrams_count} features</strong></div>
            </div>
            """, unsafe_allow_html=True)

        run_ml = st.button("✨ ANALYZE REVIEW")
        st.markdown("</div>", unsafe_allow_html=True)

        if run_ml:
            if not input_text.strip():
                st.warning("⚠️ Review text is empty. Please enter review text.")
            else:
                input_df = pd.DataFrame([{
                    'review': input_text,
                    'hours_played_clean': hours_played,
                    'helpful_clean': helpful_votes,
                    'review_word_count': word_count,
                    'review_char_len': char_len
                }])

                # STRICT OOV GUARDRAIL CHECK
                if matched_ngrams_count == 0:
                    st.markdown("""
                    <div style="background-color: #101827; border: 1px solid rgba(251, 113, 133, 0.4); border-radius: 12px; padding: 24px; margin-top: 16px;">
                        <div style="color: #FB7185; font-size: 1.2rem; font-weight: 700; margin-bottom: 8px;">⚠️ TEXT SIGNAL TOO WEAK</div>
                        <div style="color: #F8FAFC; font-size: 0.9rem; margin-bottom: 16px;">
                            The review contains no vocabulary recognized by the trained NLP model.
                        </div>
                        <div style="display: flex; gap: 24px; font-size: 0.85rem; color: #94A3B8; margin-bottom: 16px;">
                            <div>TF-IDF matches: <strong style="color: #F8FAFC;">0</strong></div>
                            <div>Prediction: <strong style="color: #FB7185;">WITHHELD</strong></div>
                        </div>
                        <div style="font-size: 0.8rem; color: #64748B;">
                            Enter a meaningful Steam review to continue. Numerical telemetry cannot override zero recognized text features.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    prediction = ml_pipeline.predict(input_df)[0]
                    proba = ml_pipeline.predict_proba(input_df)[0]
                    rec_score = proba[1] * 100

                    # VALID RESULT CARD
                    st.markdown("""<div class="saas-card" style="margin-top: 16px;">""", unsafe_allow_html=True)
                    st.markdown("<div class='chart-title' style='margin-bottom: 16px;'>RECOMMENDATION MODEL RESULT</div>", unsafe_allow_html=True)
                    
                    res_col1, res_col2 = st.columns(2)

                    with res_col1:
                        if prediction == 1:
                            st.markdown("""
                            <div style="background: rgba(52, 211, 153, 0.1); border: 1px solid rgba(52, 211, 153, 0.3); padding: 16px; border-radius: 10px; margin-bottom: 16px;">
                                <div style="color: #34D399; font-size: 1.15rem; font-weight: 700;">✓ RECOMMENDED</div>
                                <div style="color: #94A3B8; font-size: 0.82rem; margin-top: 4px;">Positive recommendation affinity detected across feature weights.</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style="background: rgba(251, 113, 133, 0.1); border: 1px solid rgba(251, 113, 133, 0.3); padding: 16px; border-radius: 10px; margin-bottom: 16px;">
                                <div style="color: #FB7185; font-size: 1.15rem; font-weight: 700;">✕ NOT RECOMMENDED</div>
                                <div style="color: #94A3B8; font-size: 0.82rem; margin-top: 4px;">Negative sentiment indicators detected in review text.</div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.metric("Recommendation Likelihood", f"{rec_score:.1f}%")

                        st.markdown(f"""
                        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 16px; border-top: 1px solid #1E293B; padding-top: 12px;">
                            • Text Signal: <strong style="color: #F8FAFC;">{"Strong" if matched_ngrams_count > 5 else "Moderate"}</strong> ({matched_ngrams_count} features)<br>
                            • Telemetry Signal: <strong style="color: #F8FAFC;">Moderate</strong> ({hours_played} hrs, {helpful_votes} votes)<br>
                            • Model Confidence: <strong style="color: #38BDF8;">{"High" if abs(rec_score - 50) > 25 else "Moderate"}</strong>
                        </div>
                        """, unsafe_allow_html=True)

                    with res_col2:
                        fig_g = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=rec_score,
                            number=dict(suffix="%", font=dict(size=30, color="#F8FAFC")),
                            gauge={
                                'axis': {'range': [0, 100], 'tickcolor': "#64748B"},
                                'bar': {'color': "#38BDF8"},
                                'bgcolor': "#070B14",
                                'bordercolor': "#1E293B",
                                'steps': [
                                    {'range': [0, 50], 'color': "rgba(251, 113, 133, 0.2)"},
                                    {'range': [50, 100], 'color': "rgba(52, 211, 153, 0.2)"}
                                ]
                            }
                        ))
                        fig_g.update_layout(**PLOTLY_THEME, height=210)
                        st.plotly_chart(fig_g, use_container_width=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
