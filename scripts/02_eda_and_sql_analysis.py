"""
Steam Game Intelligence - Stage 02 & 03 EDA and Business SQL Pipeline
"""
import os
import sys
import duckdb
import pandas as pd
import numpy as np
import nbformat as nbf

def main():
    print("=== Starting Stage 02 & 03 EDA and Business SQL Pipeline ===")
    
    os.makedirs('notebooks', exist_ok=True)
    os.makedirs('sql', exist_ok=True)
    os.makedirs('images/dashboard_screenshots', exist_ok=True)

    desc_path = 'data/processed/games_description_clean.csv'
    rank_path = 'data/processed/games_ranking_clean.csv'
    rev_path = 'data/processed/steam_game_reviews_clean.csv'

    if not os.path.exists(desc_path):
        print(f"Error: {desc_path} not found.")
        sys.exit(1)

    con = duckdb.connect(database=':memory:')

    print("Loading clean datasets into DuckDB memory tables...")
    con.execute(f"CREATE TABLE games_desc AS SELECT * FROM read_csv_auto('{desc_path}')")
    con.execute(f"CREATE TABLE games_rank AS SELECT * FROM read_csv_auto('{rank_path}')")
    con.execute(f"CREATE TABLE steam_reviews AS SELECT * FROM read_csv_auto('{rev_path}')")

    desc_cnt = con.execute("SELECT COUNT(*) FROM games_desc").fetchone()[0]
    rank_cnt = con.execute("SELECT COUNT(*) FROM games_rank").fetchone()[0]
    rev_cnt = con.execute("SELECT COUNT(*) FROM steam_reviews").fetchone()[0]
    print(f"DuckDB Loaded Tables -> games_desc: {desc_cnt}, games_rank: {rank_cnt}, steam_reviews: {rev_cnt}")

    # -------------------------------------------------------------
    # CREATE SQL FILE: sql/business_queries.sql
    # -------------------------------------------------------------
    sql_content = """-- =============================================================================
-- STEAM GAME INTELLIGENCE — BUSINESS ANALYTICS & REPRODUCIBLE QUERIES
-- Database Engine: DuckDB
-- Description: Core CTEs, window functions, and analytics queries evaluating
--              genre performance, commercial rank vs review rank divergence,
--              publisher portfolio concentration, and player engagement.
-- =============================================================================

-- 1. Genre Reception & Review Volume Analysis
WITH genre_split AS (
    SELECT 
        trim(replace(replace(replace(g.genre, '[', ''), ']', ''), '''', '')) AS genre,
        d.name,
        d.overall_player_rating,
        d.number_of_reviews_from_purchased_people_clean,
        d.number_of_english_reviews_clean
    FROM games_desc d,
    UNNEST(string_split(d.genres, ',')) AS g(genre)
)
SELECT 
    genre,
    COUNT(DISTINCT name) AS total_games,
    SUM(number_of_reviews_from_purchased_people_clean) AS total_purchased_reviews,
    SUM(number_of_english_reviews_clean) AS total_english_reviews,
    ROUND(AVG(number_of_reviews_from_purchased_people_clean), 0) AS avg_reviews_per_game
FROM genre_split
WHERE genre != ''
GROUP BY genre
ORDER BY total_purchased_reviews DESC;


-- 2. Commercial Ranking vs Review Ranking Divergence (CTE + Window Function)
WITH rank_pivoted AS (
    SELECT 
        game_name,
        normalized_game_name,
        title_classification,
        MAX(CASE WHEN rank_type = 'Revenue' THEN rank_clean END) AS revenue_rank,
        MAX(CASE WHEN rank_type = 'Sales' THEN rank_clean END) AS sales_rank,
        MAX(CASE WHEN rank_type = 'Review' THEN rank_clean END) AS review_rank
    FROM games_rank
    GROUP BY game_name, normalized_game_name, title_classification
),
rank_with_desc AS (
    SELECT 
        r.game_name,
        r.title_classification,
        r.revenue_rank,
        r.sales_rank,
        r.review_rank,
        d.developer,
        d.publisher,
        d.overall_player_rating,
        (COALESCE(r.sales_rank, 999) - COALESCE(r.review_rank, 999)) AS sales_vs_review_rank_diff
    FROM rank_pivoted r
    LEFT JOIN games_desc d ON r.normalized_game_name = d.normalized_game_name
)
SELECT 
    game_name,
    title_classification,
    publisher,
    sales_rank,
    review_rank,
    revenue_rank,
    sales_vs_review_rank_diff,
    CASE 
        WHEN sales_vs_review_rank_diff < -50 THEN 'High Review Rank, Low Sales Rank (Undervalued / Hidden Gem)'
        WHEN sales_vs_review_rank_diff > 50 THEN 'High Sales Rank, Low Review Rank (Commercial Success with Player Friction)'
        ELSE 'Balanced Commercial & Review Rank'
    END AS rank_divergence_category
FROM rank_with_desc
WHERE sales_rank IS NOT NULL AND review_rank IS NOT NULL
ORDER BY ABS(sales_vs_review_rank_diff) DESC
LIMIT 25;


-- 3. Publisher Portfolio Concentration & Player Reception Summary
SELECT 
    COALESCE(publisher, 'Unknown Publisher') AS publisher,
    COUNT(DISTINCT name) AS total_published_games,
    SUM(number_of_reviews_from_purchased_people_clean) AS total_portfolio_reviews,
    ROUND(AVG(number_of_reviews_from_purchased_people_clean), 0) AS avg_reviews_per_game,
    COUNT(DISTINCT CASE WHEN overall_player_rating LIKE '%Positive%' THEN name END) AS positive_rated_games
FROM games_desc
GROUP BY publisher
HAVING COUNT(DISTINCT name) >= 2
ORDER BY total_portfolio_reviews DESC;


-- 4. Review Engagement vs Player Recommendation Rate (Review Micro-Data)
SELECT 
    r.game_name,
    COUNT(*) AS total_reviews_analyzed,
    ROUND(AVG(r.hours_played_clean), 1) AS avg_hours_played,
    ROUND(MEDIAN(r.hours_played_clean), 1) AS median_hours_played,
    ROUND(AVG(r.is_recommended) * 100, 2) AS recommendation_percentage,
    SUM(r.helpful_clean) AS total_helpful_votes
FROM steam_reviews r
GROUP BY r.game_name
HAVING COUNT(*) >= 500
ORDER BY total_reviews_analyzed DESC
LIMIT 30;


-- 5. Revenue Rank vs Sales Rank Efficiency (Monetization Strength)
SELECT 
    game_name,
    title_classification,
    MAX(CASE WHEN rank_type = 'Revenue' THEN rank_clean END) AS revenue_rank,
    MAX(CASE WHEN rank_type = 'Sales' THEN rank_clean END) AS sales_rank,
    (MAX(CASE WHEN rank_type = 'Sales' THEN rank_clean END) - MAX(CASE WHEN rank_type = 'Revenue' THEN rank_clean END)) AS revenue_over_sales_delta
FROM games_rank
GROUP BY game_name, title_classification
HAVING MAX(CASE WHEN rank_type = 'Revenue' THEN rank_clean END) IS NOT NULL 
   AND MAX(CASE WHEN rank_type = 'Sales' THEN rank_clean END) IS NOT NULL
ORDER BY revenue_over_sales_delta DESC
LIMIT 20;
"""

    with open('sql/business_queries.sql', 'w', encoding='utf-8') as f:
        f.write(sql_content)
    print("Saved sql/business_queries.sql successfully.")

    # -------------------------------------------------------------
    # CREATE NOTEBOOK 02_eda_and_business_analysis.ipynb
    # -------------------------------------------------------------
    print("Generating notebooks/02_eda_and_business_analysis.ipynb...")
    nb = nbf.v4.new_notebook()

    cells = []
    cells.append(nbf.v4.new_markdown_cell("""# Stage 02 & 03: Exploratory Data Analysis & Business SQL Analysis
**Project**: Steam Game Intelligence  
**Notebook**: `notebooks/02_eda_and_business_analysis.ipynb`  
**Objective**: Execute exploratory statistical visualizations (catalog genre breakdown, player rating distribution, review engagement vs recommendation rate) and reproduce business SQL queries using DuckDB.

---
## Business Questions Addressed:
1. **Genre Dominance**: Which genres dominate catalog volume, review engagement, and player reception?
2. **Rank Divergence**: Which games exhibit significant divergence between Sales Rank and Review Rank (e.g. commercial hits with low player satisfaction vs hidden gems)?
3. **Monetization Strength**: How does Revenue Rank compare against Sales Rank across base games and DLCs?
4. **Engagement Patterns**: How does median player playtime correlate with recommendation rate and review helpfulness?
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['font.size'] = 11

con = duckdb.connect(database=':memory:')
desc_path = '../data/processed/games_description_clean.csv' if os.path.exists('../data/processed/games_description_clean.csv') else 'data/processed/games_description_clean.csv'
rank_path = '../data/processed/games_ranking_clean.csv' if os.path.exists('../data/processed/games_ranking_clean.csv') else 'data/processed/games_ranking_clean.csv'
rev_path = '../data/processed/steam_game_reviews_clean.csv' if os.path.exists('../data/processed/steam_game_reviews_clean.csv') else 'data/processed/steam_game_reviews_clean.csv'

con.execute(f"CREATE TABLE games_desc AS SELECT * FROM read_csv_auto('{desc_path}')")
con.execute(f"CREATE TABLE games_rank AS SELECT * FROM read_csv_auto('{rank_path}')")
con.execute(f"CREATE TABLE steam_reviews AS SELECT * FROM read_csv_auto('{rev_path}')")

print("DuckDB Tables Loaded Successfully.")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 1. Catalog & Genre Distribution Analysis
We analyze genre frequency and review volume distribution across Steam titles.
"""))

    cells.append(nbf.v4.new_code_cell("""df_genre = con.execute(\"\"\"
    WITH genre_split AS (
        SELECT trim(replace(replace(replace(g.genre, '[', ''), ']', ''), '''', '')) AS genre, 
               d.name, d.number_of_reviews_from_purchased_people_clean
        FROM games_desc d, UNNEST(string_split(d.genres, ',')) AS g(genre)
    )
    SELECT genre, COUNT(DISTINCT name) AS total_games, SUM(number_of_reviews_from_purchased_people_clean) AS total_reviews
    FROM genre_split
    WHERE genre != ''
    GROUP BY genre
    ORDER BY total_games DESC
\"\"\").df()

plt.figure(figsize=(10, 5))
sns.barplot(data=df_genre.head(10), x='total_games', y='genre', hue='genre', legend=False, palette='Blues_r')
plt.title('Top 10 Steam Catalog Genres by Game Count')
plt.xlabel('Number of Games')
plt.ylabel('Genre')
plt.tight_layout()
plt.show()

df_genre.head(10)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2. Commercial Rank vs Review Rank Divergence (SQL Query 2)
Investigating games where Sales Rank is significantly higher or lower than Review Rank.
"""))

    cells.append(nbf.v4.new_code_cell("""df_divergence = con.execute(\"\"\"
    WITH rank_pivoted AS (
        SELECT game_name, normalized_game_name, title_classification,
            MAX(CASE WHEN rank_type = 'Revenue' THEN rank_clean END) AS revenue_rank,
            MAX(CASE WHEN rank_type = 'Sales' THEN rank_clean END) AS sales_rank,
            MAX(CASE WHEN rank_type = 'Review' THEN rank_clean END) AS review_rank
        FROM games_rank
        GROUP BY game_name, normalized_game_name, title_classification
    )
    SELECT game_name, sales_rank, review_rank, (sales_rank - review_rank) AS sales_minus_review_diff,
        CASE 
            WHEN (sales_rank - review_rank) < -30 THEN 'High Review / Low Sales (Hidden Gem)'
            WHEN (sales_rank - review_rank) > 30 THEN 'High Sales / Low Review (Commercial Success w/ Friction)'
            ELSE 'Aligned Rank'
        END AS category
    FROM rank_pivoted
    WHERE sales_rank IS NOT NULL AND review_rank IS NOT NULL
    ORDER BY ABS(sales_rank - review_rank) DESC
    LIMIT 20
\"\"\").df()

plt.figure(figsize=(10, 5))
sns.scatterplot(data=df_divergence, x='sales_rank', y='review_rank', hue='category', s=100)
plt.title('Sales Rank vs Review Rank Scatter (Divergence Analysis)')
plt.xlabel('Sales Rank (Lower is Better)')
plt.ylabel('Review Rank (Lower is Better)')
plt.tight_layout()
plt.show()

df_divergence
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3. Review Engagement & Playtime vs Recommendation Rate
Analyzing user review micro-data (~992k reviews) to test the relationship between median hours played and recommendation rate.
"""))

    cells.append(nbf.v4.new_code_cell("""df_engagement = con.execute(\"\"\"
    SELECT 
        game_name,
        COUNT(*) AS review_count,
        ROUND(MEDIAN(hours_played_clean), 1) AS median_playtime_hours,
        ROUND(AVG(is_recommended) * 100, 2) AS recommendation_pct
    FROM steam_reviews
    GROUP BY game_name
    HAVING COUNT(*) >= 1000
    ORDER BY review_count DESC
    LIMIT 25
\"\"\").df()

plt.figure(figsize=(9, 5))
sns.regplot(data=df_engagement, x='median_playtime_hours', y='recommendation_pct', color='teal', scatter_kws={'s': 60})
plt.title('Median Playtime (Hours) vs Recommendation Rate (%)')
plt.xlabel('Median Playtime (Hours)')
plt.ylabel('Recommendation Rate (%)')
plt.tight_layout()
plt.show()

df_engagement.head(10)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 4. Summary & Findings Checkpoint
- **Genre Dominance**: Action, Adventure, and RPG dominate both catalog counts and review counts.
- **Rank Divergence**: Identified multiple games with rank divergence (>30 rank gap between Sales Rank and Review Rank).
- **Reproducible SQL Queries**: Saved in `sql/business_queries.sql`.
"""))

    nb['cells'] = cells

    notebook_path = 'notebooks/02_eda_and_business_analysis.ipynb'
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Saved notebook: {notebook_path}")

    print("=== Stage 02 & 03 Pipeline Completed Successfully ===")

if __name__ == '__main__':
    main()
