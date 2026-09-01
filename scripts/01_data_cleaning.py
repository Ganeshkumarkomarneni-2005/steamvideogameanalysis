"""
Steam Game Intelligence - Stage 01 Data Understanding & Cleaning Pipeline
"""
import os
import zipfile
import re
import sys
import pandas as pd
import numpy as np
import nbformat as nbf

def extract_num(val):
    if pd.isna(val):
        return 0
    nums = re.findall(r'[\d,]+', str(val))
    clean_nums = [int(n.replace(',', '')) for n in nums if n.replace(',', '').isdigit()]
    return max(clean_nums) if clean_nums else 0

def main():
    print("=== Starting Stage 01 Data Understanding & Cleaning Pipeline ===")
    
    zip_path = 'archive.zip'
    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)

    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} not found.")
        sys.exit(1)

    z = zipfile.ZipFile(zip_path)

    # -------------------------------------------------------------
    # 1. GAMES DESCRIPTION CLEANING & PROFILING
    # -------------------------------------------------------------
    print("\n[1/3] Processing games_description.csv...")
    df_desc = pd.read_csv(z.open('games_description.csv'))
    desc_raw_count = len(df_desc)
    print(f"Raw games_description count: {desc_raw_count} rows, {len(df_desc.columns)} columns.")

    def normalize_string(val):
        if pd.isna(val):
            return ""
        val = str(val).lower().strip()
        val = re.sub(r'[^\w\s]', '', val)
        val = re.sub(r'\s+', ' ', val)
        return val

    df_desc['normalized_game_name'] = df_desc['name'].apply(normalize_string)
    df_desc['release_date_clean'] = pd.to_datetime(df_desc['release_date'], errors='coerce')
    
    df_desc['number_of_reviews_from_purchased_people_clean'] = df_desc['number_of_reviews_from_purchased_people'].apply(extract_num)
    df_desc['number_of_english_reviews_clean'] = df_desc['number_of_english_reviews'].apply(extract_num)

    df_desc['missing_short_desc'] = df_desc['short_description'].isna()

    desc_clean_path = os.path.join(processed_dir, 'games_description_clean.csv')
    df_desc.to_csv(desc_clean_path, index=False)
    print(f"Saved: {desc_clean_path} ({len(df_desc)} rows)")

    # -------------------------------------------------------------
    # 2. GAMES RANKING CLEANING & PROFILING
    # -------------------------------------------------------------
    print("\n[2/3] Processing games_ranking.csv...")
    df_rank = pd.read_csv(z.open('games_ranking.csv'))
    rank_raw_count = len(df_rank)
    print(f"Raw games_ranking count: {rank_raw_count} rows, {len(df_rank.columns)} columns.")

    df_rank['normalized_game_name'] = df_rank['game_name'].apply(normalize_string)
    df_rank['rank_clean'] = pd.to_numeric(df_rank['rank'], errors='coerce').fillna(0).astype(int)

    dlc_keywords = ['dlc', 'pack', 'expansion', 'reserve', 'pass', 'deluxe', 'edition']
    def classify_title(name):
        n = str(name).lower()
        if any(k in n for k in dlc_keywords):
            return 'DLC / Content / Pass'
        return 'Base Game Title'

    df_rank['title_classification'] = df_rank['game_name'].apply(classify_title)

    desc_norms = set(df_desc['normalized_game_name'])
    df_rank['is_in_games_desc'] = df_rank['normalized_game_name'].isin(desc_norms)

    unmatched_rank = df_rank[~df_rank['is_in_games_desc']]['game_name'].unique()
    print(f"Games Ranking matched titles: {df_rank['is_in_games_desc'].sum()}/{len(df_rank)} rows ({df_rank['is_in_games_desc'].mean()*100:.2f}%)")
    print(f"Unmatched Ranking titles ({len(unmatched_rank)}): {list(unmatched_rank)}")

    rank_clean_path = os.path.join(processed_dir, 'games_ranking_clean.csv')
    df_rank.to_csv(rank_clean_path, index=False)
    print(f"Saved: {rank_clean_path} ({len(df_rank)} rows)")

    # -------------------------------------------------------------
    # 3. STEAM REVIEWS CHUNKED PROCESSING
    # -------------------------------------------------------------
    print("\n[3/3] Processing steam_game_reviews.csv in 100k chunks...")
    chunk_size = 100000
    review_chunks = []
    total_reviews_processed = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(z.open('steam_game_reviews.csv'), chunksize=chunk_size, low_memory=False)):
        chunk['normalized_game_name'] = chunk['game_name'].apply(normalize_string)
        chunk['hours_played_clean'] = pd.to_numeric(chunk['hours_played'].astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
        chunk['helpful_clean'] = pd.to_numeric(chunk['helpful'].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)
        chunk['funny_clean'] = pd.to_numeric(chunk['funny'].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)
        chunk['is_recommended'] = (chunk['recommendation'].astype(str).str.strip() == 'Recommended').astype(int)
        chunk['review'] = chunk['review'].fillna('')
        chunk['review_char_len'] = chunk['review'].astype(str).str.len()
        chunk['review_word_count'] = chunk['review'].astype(str).apply(lambda x: len(x.split()))
        
        total_reviews_processed += len(chunk)
        review_chunks.append(chunk)
        print(f"   Chunk {chunk_idx+1}: processed {len(chunk)} rows (cumulative: {total_reviews_processed})")

    df_reviews_clean = pd.concat(review_chunks, ignore_index=True)
    df_reviews_clean['is_in_games_desc'] = df_reviews_clean['normalized_game_name'].isin(desc_norms)

    unmatched_rev = df_reviews_clean[~df_reviews_clean['is_in_games_desc']]['game_name'].unique()
    print(f"\nTotal Reviews Cleaned: {len(df_reviews_clean)} rows")
    print(f"Reviews matched titles: {df_reviews_clean['is_in_games_desc'].sum()}/{len(df_reviews_clean)} rows ({df_reviews_clean['is_in_games_desc'].mean()*100:.2f}%)")
    print(f"Unmatched Review titles ({len(unmatched_rev)}): {list(unmatched_rev)}")

    rev_clean_path = os.path.join(processed_dir, 'steam_game_reviews_clean.csv')
    df_reviews_clean.to_csv(rev_clean_path, index=False)
    print(f"Saved: {rev_clean_path} ({len(df_reviews_clean)} rows)")

    # -------------------------------------------------------------
    # 4. GENERATE NOTEBOOK 01_data_cleaning.ipynb
    # -------------------------------------------------------------
    print("\nGenerating notebooks/01_data_cleaning.ipynb...")
    nb = nbf.v4.new_notebook()

    cells = []
    cells.append(nbf.v4.new_markdown_cell("""# Stage 01: Data Understanding & Cleaning
**Project**: Steam Game Intelligence  
**Notebook**: `notebooks/01_data_cleaning.ipynb`  
**Objective**: Profile schemas, audit join quality, handle missing/invalid values, process the large reviews CSV in chunks, and export clean datasets to `data/processed/`.
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import zipfile
import re
import pandas as pd
import numpy as np

zip_path = '../archive.zip' if os.path.exists('../archive.zip') else 'archive.zip'
z = zipfile.ZipFile(zip_path)
print(f"Archive contents: {z.namelist()}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 1. Games Description Dataset Profiling & Cleaning
"""))

    cells.append(nbf.v4.new_code_cell("""df_desc = pd.read_csv(z.open('games_description.csv'))
print(f"Raw shape: {df_desc.shape}")

def extract_num(val):
    if pd.isna(val): return 0
    nums = re.findall(r'[\\d,]+', str(val))
    clean_nums = [int(n.replace(',', '')) for n in nums if n.replace(',', '').isdigit()]
    return max(clean_nums) if clean_nums else 0

def normalize_string(val):
    if pd.isna(val): return ""
    val = str(val).lower().strip()
    val = re.sub(r'[^\\w\\s]', '', val)
    val = re.sub(r'\\s+', ' ', val)
    return val

df_desc['normalized_game_name'] = df_desc['name'].apply(normalize_string)
df_desc['release_date_clean'] = pd.to_datetime(df_desc['release_date'], errors='coerce')
df_desc['number_of_reviews_from_purchased_people_clean'] = df_desc['number_of_reviews_from_purchased_people'].apply(extract_num)
df_desc['number_of_english_reviews_clean'] = df_desc['number_of_english_reviews'].apply(extract_num)
df_desc['missing_short_desc'] = df_desc['short_description'].isna()

df_desc[['name', 'number_of_reviews_from_purchased_people_clean', 'number_of_english_reviews_clean']].head(5)
"""))

    nb['cells'] = cells

    notebook_path = 'notebooks/01_data_cleaning.ipynb'
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Saved notebook: {notebook_path}")

    print("\n=== Stage 01 Pipeline Execution Completed Successfully ===")

if __name__ == '__main__':
    main()
