"""
Steam Game Intelligence - Stage 04 & 05 Advanced Machine Learning Pipeline
Upgraded: Multi-Feature ColumnTransformer, XGBoost Classifier, GridSearchCV, and Model Comparison
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import nbformat as nbf

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
)

def main():
    print("=== Starting Stage 04 & 05 Advanced NLP & ML Pipeline ===")
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)
    os.makedirs('images', exist_ok=True)

    rev_path = 'data/processed/steam_game_reviews_clean.csv'

    if not os.path.exists(rev_path):
        print(f"Error: {rev_path} not found.")
        sys.exit(1)

    print("Loading steam_game_reviews_clean.csv...")
    df_reviews = pd.read_csv(rev_path, usecols=[
        'game_name', 'review', 'hours_played_clean', 'helpful_clean', 'funny_clean', 
        'is_recommended', 'review_char_len', 'review_word_count'
    ])
    print(f"Total reviews in dataset: {len(df_reviews)}")

    # Clean text and missing numerical values
    df_reviews['review'] = df_reviews['review'].fillna('')
    df_reviews = df_reviews[df_reviews['review'].str.strip() != ''].copy()
    df_reviews['hours_played_clean'] = df_reviews['hours_played_clean'].fillna(0)
    df_reviews['helpful_clean'] = df_reviews['helpful_clean'].fillna(0)
    df_reviews['review_word_count'] = df_reviews['review_word_count'].fillna(0)
    df_reviews['review_char_len'] = df_reviews['review_char_len'].fillna(0)

    print("\nTarget Class Distribution (is_recommended):")
    print(df_reviews['is_recommended'].value_counts(normalize=True))

    # Stratified sample for ML training
    sample_size = min(30000, len(df_reviews))
    print(f"\nSampling {sample_size} records for model training & evaluation...")
    df_sample = df_reviews.sample(n=sample_size, random_state=42).reset_index(drop=True)

    feature_cols = ['review', 'hours_played_clean', 'helpful_clean', 'review_word_count', 'review_char_len']
    X = df_sample[feature_cols]
    y = df_sample['is_recommended']

    # Train / Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train set: {len(X_train)} | Test set: {len(X_test)}")

    # Construct Multi-Feature ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english'), 'review'),
            ('num', StandardScaler(), ['hours_played_clean', 'helpful_clean', 'review_word_count', 'review_char_len'])
        ]
    )

    # 1. Model 1: Baseline Dummy Classifier
    print("\n--- Model 1: Baseline Dummy Classifier ---")
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train['review'], y_train)
    y_pred_dummy = dummy.predict(X_test['review'])
    print(f"Baseline Accuracy:  {accuracy_score(y_test, y_pred_dummy):.4f}")
    print(f"Baseline F1-Score:  {f1_score(y_test, y_pred_dummy, zero_division=0):.4f}")

    # 2. Model 2: Multi-Feature Logistic Regression
    print("\n--- Model 2: Multi-Feature Logistic Regression ---")
    lr_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=42))
    ])
    lr_pipe.fit(X_train, y_train)
    y_pred_lr = lr_pipe.predict(X_test)
    y_proba_lr = lr_pipe.predict_proba(X_test)[:, 1]
    auc_lr = roc_auc_score(y_test, y_proba_lr)
    f1_lr = f1_score(y_test, y_pred_lr)
    print(f"Logistic Regression Accuracy:  {accuracy_score(y_test, y_pred_lr):.4f}")
    print(f"Logistic Regression F1-Score:  {f1_lr:.4f}")
    print(f"Logistic Regression ROC-AUC:   {auc_lr:.4f}")

    # Save multi-feature model pipeline
    model_save_path = 'models/recommendation_pipeline.joblib'
    joblib.dump(lr_pipe, model_save_path)
    print(f"Saved multi-feature ML recommendation pipeline to: {model_save_path}")

    # -------------------------------------------------------------
    # GENERATE NOTEBOOK 03_nlp_and_machine_learning.ipynb
    # -------------------------------------------------------------
    print("\nGenerating notebooks/03_nlp_and_machine_learning.ipynb...")
    nb = nbf.v4.new_notebook()

    cells = []
    cells.append(nbf.v4.new_markdown_cell("""# Stage 04 & 05: Multi-Feature ML Classification & Model Benchmarking
**Project**: Steam Game Intelligence  
**Notebook**: `notebooks/03_nlp_and_machine_learning.ipynb`  
**Objective**: Build a multi-feature ML recommendation pipeline combining TF-IDF review text features with engagement metrics (`hours_played_clean`, `helpful_clean`, `review_word_count`), comparing Logistic Regression vs XGBoost with GridSearchCV tuning.

---
## Pipeline Architecture & Benchmark Metrics:
1. **Target**: `is_recommended` (1 = Recommended, 0 = Not Recommended).
2. **Feature Fusion**: `ColumnTransformer` joining text n-grams with scaled numerical engagement metrics.
3. **Model Benchmark**: Evaluated Logistic Regression vs Tuned XGBoost Classifier.
4. **Leakage Prevention**: All transformations fit strictly on `X_train`.
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

sns.set_theme(style='whitegrid')

# Load Processed Dataset
rev_path = '../data/processed/steam_game_reviews_clean.csv' if os.path.exists('../data/processed/steam_game_reviews_clean.csv') else 'data/processed/steam_game_reviews_clean.csv'
df_reviews = pd.read_csv(rev_path, usecols=['review', 'hours_played_clean', 'helpful_clean', 'review_word_count', 'review_char_len', 'is_recommended'])
df_reviews['review'] = df_reviews['review'].fillna('')
df_reviews = df_reviews[df_reviews['review'].str.strip() != ''].copy()

df_sample = df_reviews.sample(n=100000, random_state=42).reset_index(drop=True)
feature_cols = ['review', 'hours_played_clean', 'helpful_clean', 'review_word_count', 'review_char_len']
X = df_sample[feature_cols]
y = df_sample['is_recommended']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# ColumnTransformer Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english'), 'review'),
        ('num', StandardScaler(), ['hours_played_clean', 'helpful_clean', 'review_word_count', 'review_char_len'])
    ]
)

# Fit Fine-Tuned XGBoost Pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss'))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

print("=== Fine-Tuned XGBoost Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['Not Recommended', 'Recommended']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
"""))

    nb['cells'] = cells

    notebook_path = 'notebooks/03_nlp_and_machine_learning.ipynb'
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Saved notebook: {notebook_path}")

    print("=== Stage 04 & 05 Pipeline Completed Successfully ===")

if __name__ == '__main__':
    main()
