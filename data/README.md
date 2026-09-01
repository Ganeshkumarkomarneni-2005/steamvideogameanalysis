# Data Directory

This directory stores the data pipeline artifacts for the Steam Game Intelligence project.

## Structure
- `processed/`: Processed, cleaned, and standardized CSV datasets output from `01_data_cleaning.ipynb`.
  - `games_description_clean.csv`: Cleaned metadata for 290 Steam games.
  - `games_ranking_clean.csv`: Cleaned ranking records (672 rows across Revenue, Sales, and Review rank types).
  - `steam_game_reviews_clean.csv`: Processed review records (992,153 reviews) with normalized sentiment binary flags, text statistics, and numeric parsing.

## Raw Data Usage
Raw data files (`archive.zip`, `games_description.csv`, `games_ranking.csv`, `steam_game_reviews.csv`) are ignored by Git to adhere to dataset licensing and GitHub file size limits (~454 MB review file). Raw data can be downloaded from Kaggle's Steam Dataset repository.
