-- =============================================================================
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
