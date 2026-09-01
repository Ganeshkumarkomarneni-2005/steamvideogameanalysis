"""
Steam Game Intelligence - Stage 07 AI Analytics Agent Pipeline
"""
import os
import sys
import duckdb
import pandas as pd
import numpy as np
import nbformat as nbf

class SteamGroundedAnalyticsAgent:
    """
    Grounded AI Analytics Agent for Steam Game Intelligence.
    Translates structured intents into read-only DuckDB SQL queries,
    executes them against actual dataset tables, and returns grounded evidence.
    """
    def __init__(self, desc_path, rank_path, rev_path):
        self.con = duckdb.connect(database=':memory:')
        self.con.execute(f"CREATE TABLE games_desc AS SELECT * FROM read_csv_auto('{desc_path}')")
        self.con.execute(f"CREATE TABLE games_rank AS SELECT * FROM read_csv_auto('{rank_path}')")
        self.con.execute(f"CREATE TABLE steam_reviews AS SELECT * FROM read_csv_auto('{rev_path}')")

    def validate_and_execute_sql(self, sql_query):
        clean_sql = sql_query.strip()
        upper_sql = clean_sql.upper()

        forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
        for keyword in forbidden_keywords:
            if f" {keyword} " in f" {upper_sql} ":
                raise ValueError(f"Security Violation: Query contains forbidden non-read-only keyword '{keyword}'.")

        if not upper_sql.startswith('SELECT') and not upper_sql.startswith('WITH'):
            raise ValueError("Security Violation: Only SELECT or WITH read-only queries are permitted.")

        df_res = self.con.execute(clean_sql).df()
        return df_res

    def answer_question(self, question):
        q = question.lower()
        
        # Intent 1: Compare Genres
        if 'compare' in q and ('rpg' in q or 'action' in q):
            sql = """
            WITH genre_split AS (
                SELECT 
                    lower(trim(replace(replace(replace(g.genre, '[', ''), ']', ''), '''', ''))) AS genre, 
                    d.name, 
                    d.number_of_reviews_from_purchased_people_clean
                FROM games_desc d, UNNEST(string_split(d.genres, ',')) AS g(genre)
            )
            SELECT 
                genre, 
                COUNT(DISTINCT name) AS total_games, 
                SUM(number_of_reviews_from_purchased_people_clean) AS total_reviews
            FROM genre_split
            WHERE genre IN ('rpg', 'action')
            GROUP BY genre;
            """
            evidence = self.validate_and_execute_sql(sql)
            explanation = f"Grounded Result:\n{evidence.to_string(index=False)}"
            return {'question': question, 'intent': 'Genre Comparison', 'sql': sql.strip(), 'evidence': evidence, 'explanation': explanation}

        # Intent 2: Publishers reception
        elif 'publisher' in q or 'publishers' in q:
            sql = """
            SELECT COALESCE(publisher, 'Unknown') AS publisher, COUNT(DISTINCT name) AS game_count,
                   SUM(number_of_reviews_from_purchased_people_clean) AS total_reviews
            FROM games_desc
            GROUP BY publisher
            HAVING COUNT(DISTINCT name) >= 2
            ORDER BY total_reviews DESC LIMIT 10;
            """
            evidence = self.validate_and_execute_sql(sql)
            explanation = f"Grounded Result:\n{evidence.to_string(index=False)}"
            return {'question': question, 'intent': 'Publisher Analysis', 'sql': sql.strip(), 'evidence': evidence, 'explanation': explanation}

        # Default: General Ranking Overview
        else:
            sql = """
            SELECT rank_type, COUNT(*) AS record_count, MIN(rank_clean) AS top_rank, MAX(rank_clean) AS max_rank
            FROM games_rank GROUP BY rank_type;
            """
            evidence = self.validate_and_execute_sql(sql)
            explanation = f"Grounded Result:\n{evidence.to_string(index=False)}"
            return {'question': question, 'intent': 'Ranking Summary', 'sql': sql.strip(), 'evidence': evidence, 'explanation': explanation}


def main():
    print("=== Starting Stage 07 AI Analytics Agent Pipeline ===")
    
    os.makedirs('notebooks', exist_ok=True)

    desc_path = 'data/processed/games_description_clean.csv'
    rank_path = 'data/processed/games_ranking_clean.csv'
    rev_path = 'data/processed/steam_game_reviews_clean.csv'

    agent = SteamGroundedAnalyticsAgent(desc_path, rank_path, rev_path)

    test_q = "Compare RPG and Action games on review volume and recommendation rate."
    print(f"\nTesting Agent with Question: '{test_q}'")
    res = agent.answer_question(test_q)
    print(f"Intent: {res['intent']}")
    print(f"Executed SQL:\n{res['sql']}")
    print(f"\n{res['explanation']}")

    # -------------------------------------------------------------
    # GENERATE NOTEBOOK 04_ai_analytics_agent.ipynb
    # -------------------------------------------------------------
    print("\nGenerating notebooks/04_ai_analytics_agent.ipynb...")
    nb = nbf.v4.new_notebook()

    cells = []
    cells.append(nbf.v4.new_markdown_cell("""# Stage 07: Grounded AI Analytics Agent
**Project**: Steam Game Intelligence  
**Notebook**: `notebooks/04_ai_analytics_agent.ipynb`  
**Objective**: Build a grounded analytics assistant flow that parses natural language questions, validates read-only DuckDB SQL queries, and returns evidence-backed structured outputs without hallucinating data.

---
## Agent Design & Guardrails:
1. **Flow**: User Question -> Intent Classification -> Safe Read-Only SQL -> Data Execution -> Grounded Synthesis.
2. **Security**: Enforces strict read-only execution (disallowing `DROP`, `DELETE`, `UPDATE`, `INSERT`).
3. **No Hallucination**: Every number traces to DuckDB query results.
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import duckdb
import pandas as pd

class SteamGroundedAnalyticsAgent:
    def __init__(self, desc_path, rank_path, rev_path):
        self.con = duckdb.connect(database=':memory:')
        self.con.execute(f"CREATE TABLE games_desc AS SELECT * FROM read_csv_auto('{desc_path}')")
        self.con.execute(f"CREATE TABLE games_rank AS SELECT * FROM read_csv_auto('{rank_path}')")
        self.con.execute(f"CREATE TABLE steam_reviews AS SELECT * FROM read_csv_auto('{rev_path}')")

    def validate_and_execute_sql(self, sql_query):
        clean_sql = sql_query.strip()
        upper_sql = clean_sql.upper()
        forbidden = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
        for k in forbidden:
            if f" {k} " in f" {upper_sql} ":
                raise ValueError(f"Forbidden keyword '{k}' detected.")
        if not upper_sql.startswith('SELECT') and not upper_sql.startswith('WITH'):
            raise ValueError("Only SELECT/WITH queries allowed.")
        return self.con.execute(clean_sql).df()

    def query(self, question):
        q = question.lower()
        if 'rpg' in q or 'action' in q:
            sql = \"\"\"
            WITH genre_split AS (
                SELECT lower(trim(replace(replace(replace(g.genre, '[', ''), ']', ''), '''', ''))) AS genre, 
                       d.name, d.number_of_reviews_from_purchased_people_clean
                FROM games_desc d, UNNEST(string_split(d.genres, ',')) AS g(genre)
            )
            SELECT genre, COUNT(DISTINCT name) AS total_games, SUM(number_of_reviews_from_purchased_people_clean) AS total_reviews
            FROM genre_split WHERE genre IN ('rpg', 'action') GROUP BY genre;
            \"\"\"
        else:
            sql = "SELECT rank_type, COUNT(*) AS record_count FROM games_rank GROUP BY rank_type;"
        
        df_res = self.validate_and_execute_sql(sql)
        return {'question': question, 'sql': sql.strip(), 'evidence': df_res}

desc_p = '../data/processed/games_description_clean.csv' if os.path.exists('../data/processed/games_description_clean.csv') else 'data/processed/games_description_clean.csv'
rank_p = '../data/processed/games_ranking_clean.csv' if os.path.exists('../data/processed/games_ranking_clean.csv') else 'data/processed/games_ranking_clean.csv'
rev_p = '../data/processed/steam_game_reviews_clean.csv' if os.path.exists('../data/processed/steam_game_reviews_clean.csv') else 'data/processed/steam_game_reviews_clean.csv'

agent = SteamGroundedAnalyticsAgent(desc_p, rank_p, rev_p)
result = agent.query("Compare RPG and Action games")
print("Executed SQL:")
print(result['sql'])
print("\\nCalculated Evidence:")
display(result['evidence'])
"""))

    nb['cells'] = cells

    notebook_path = 'notebooks/04_ai_analytics_agent.ipynb'
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Saved notebook: {notebook_path}")

    print("=== Stage 07 Pipeline Completed Successfully ===")

if __name__ == '__main__':
    main()
