import os
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

os.makedirs('images', exist_ok=True)
sns.set_theme(style='whitegrid')
plt.rcParams['font.size'] = 11

desc_path = 'data/processed/games_description_clean.csv'
rank_path = 'data/processed/games_ranking_clean.csv'
rev_path = 'data/processed/steam_game_reviews_clean.csv'

con = duckdb.connect(database=':memory:')
con.execute(f"CREATE TABLE games_desc AS SELECT * FROM read_csv_auto('{desc_path}')")
con.execute(f"CREATE TABLE games_rank AS SELECT * FROM read_csv_auto('{rank_path}')")
con.execute(f"CREATE TABLE steam_reviews AS SELECT * FROM read_csv_auto('{rev_path}')")

# 1. Genre Distribution Chart
df_genre = con.execute("""
WITH genre_split AS (
    SELECT trim(replace(replace(replace(g.genre, '[', ''), ']', ''), '''', '')) AS genre, 
           d.name, d.number_of_reviews_from_purchased_people_clean
    FROM games_desc d, UNNEST(string_split(d.genres, ',')) AS g(genre)
)
SELECT genre, COUNT(DISTINCT name) AS total_games
FROM genre_split WHERE genre != '' GROUP BY genre ORDER BY total_games DESC LIMIT 10
""").df()

plt.figure(figsize=(10, 5))
sns.barplot(data=df_genre, x='total_games', y='genre', hue='genre', legend=False, palette='Blues_r')
plt.title('Top 10 Steam Catalog Genres by Game Count')
plt.xlabel('Number of Games')
plt.ylabel('Genre')
plt.tight_layout()
plt.savefig('images/genre_distribution.png', dpi=300)
plt.close()
print("Saved: images/genre_distribution.png")

# 2. Sales vs Review Scatter
df_divergence = con.execute("""
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
""").df()

plt.figure(figsize=(10, 5))
sns.scatterplot(data=df_divergence, x='sales_rank', y='review_rank', hue='category', s=100)
plt.title('Sales Rank vs Review Rank Scatter (Divergence Analysis)')
plt.xlabel('Sales Rank (Lower is Better)')
plt.ylabel('Review Rank (Lower is Better)')
plt.tight_layout()
plt.savefig('images/sales_vs_review_scatter.png', dpi=300)
plt.close()
print("Saved: images/sales_vs_review_scatter.png")

# 3. Playtime vs Recommendation Rate
df_engagement = con.execute("""
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
""").df()

plt.figure(figsize=(9, 5))
sns.regplot(data=df_engagement, x='median_playtime_hours', y='recommendation_pct', color='teal', scatter_kws={'s': 60})
plt.title('Median Playtime (Hours) vs Recommendation Rate (%)')
plt.xlabel('Median Playtime (Hours)')
plt.ylabel('Recommendation Rate (%)')
plt.tight_layout()
plt.savefig('images/playtime_vs_recommendation.png', dpi=300)
plt.close()
print("Saved: images/playtime_vs_recommendation.png")

# 4. Top Sentiment Predictors Chart
if os.path.exists('models/recommendation_pipeline.joblib'):
    pipe = joblib.load('models/recommendation_pipeline.joblib')
    tfidf = pipe.named_steps['tfidf']
    clf = pipe.named_steps['clf']
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = clf.coef_[0]

    top_pos_idx = np.argsort(coefs)[-15:]
    top_neg_idx = np.argsort(coefs)[:15]

    df_pos = pd.DataFrame({'ngram': feature_names[top_pos_idx], 'coefficient': coefs[top_pos_idx]})
    df_neg = pd.DataFrame({'ngram': feature_names[top_neg_idx], 'coefficient': coefs[top_neg_idx]})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=df_pos, x='coefficient', y='ngram', ax=axes[0], hue='ngram', palette='Greens_r', legend=False)
    axes[0].set_title('Top Positive Sentiment Predictors')

    sns.barplot(data=df_neg, x='coefficient', y='ngram', ax=axes[1], hue='ngram', palette='Reds', legend=False)
    axes[1].set_title('Top Negative Sentiment Predictors')

    plt.tight_layout()
    plt.savefig('images/sentiment_feature_weights.png', dpi=300)
    plt.close()
    print("Saved: images/sentiment_feature_weights.png")

print("All standalone PNG images saved successfully!")
