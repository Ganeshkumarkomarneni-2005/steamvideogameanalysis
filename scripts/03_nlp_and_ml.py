"""
Steam Game Intelligence - Stage 04 & 05 NLP and Machine Learning Pipeline
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
import nbformat as nbf

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
)

def main():
    print("=== Starting Stage 04 & 05 NLP and Machine Learning Pipeline ===")
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)

    rev_path = 'data/processed/steam_game_reviews_clean.csv'

    if not os.path.exists(rev_path):
        print(f"Error: {rev_path} not found.")
        sys.exit(1)

    print("Loading steam_game_reviews_clean.csv...")
    # Load sample for NLP & ML training to ensure fast, reproducible execution
    df_reviews = pd.read_csv(rev_path, usecols=[
        'game_name', 'review', 'hours_played_clean', 'helpful_clean', 'funny_clean', 
        'is_recommended', 'review_char_len', 'review_word_count'
    ])
    print(f"Total reviews in file: {len(df_reviews)}")

    # Handle missing reviews
    df_reviews['review'] = df_reviews['review'].fillna('')
    df_reviews = df_reviews[df_reviews['review'].str.strip() != ''].copy()
    print(f"Non-empty reviews count: {len(df_reviews)}")

    # Check target distribution
    print("\nTarget Class Distribution (is_recommended):")
    print(df_reviews['is_recommended'].value_counts(normalize=True))

    # Stratified sample for ML training (100k sample for fast, robust training)
    sample_size = min(100000, len(df_reviews))
    print(f"\nSampling {sample_size} records for model training & evaluation...")
    df_sample = df_reviews.sample(n=sample_size, random_state=42).reset_index(drop=True)

    X = df_sample['review']
    y = df_sample['is_recommended']

    # 1. Train / Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train set: {len(X_train)} | Test set: {len(X_test)}")

    # 2. Baseline Model: DummyClassifier (Most Frequent)
    print("\n--- Model 1: Baseline Dummy Classifier ---")
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    
    print(f"Dummy Accuracy:  {accuracy_score(y_test, y_pred_dummy):.4f}")
    print(f"Dummy Precision: {precision_score(y_test, y_pred_dummy, zero_division=0):.4f}")
    print(f"Dummy Recall:    {recall_score(y_test, y_pred_dummy, zero_division=0):.4f}")
    print(f"Dummy F1-Score:  {f1_score(y_test, y_pred_dummy, zero_division=0):.4f}")

    # 3. Model 2: TF-IDF + Logistic Regression Pipeline
    print("\n--- Model 2: TF-IDF + Logistic Regression Pipeline ---")
    lr_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ])

    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    y_proba_lr = lr_pipeline.predict_proba(X_test)[:, 1]

    acc_lr = accuracy_score(y_test, y_pred_lr)
    prec_lr = precision_score(y_test, y_pred_lr)
    rec_lr = recall_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr)
    auc_lr = roc_auc_score(y_test, y_proba_lr)
    cm_lr = confusion_matrix(y_test, y_pred_lr)

    print(f"Logistic Regression Accuracy:  {acc_lr:.4f}")
    print(f"Logistic Regression Precision: {prec_lr:.4f}")
    print(f"Logistic Regression Recall:    {rec_lr:.4f}")
    print(f"Logistic Regression F1-Score:  {f1_lr:.4f}")
    print(f"Logistic Regression ROC-AUC:   {auc_lr:.4f}")
    print("Confusion Matrix:")
    print(cm_lr)

    # 4. Feature Importance & Top Words Analysis
    tfidf_vectorizer = lr_pipeline.named_steps['tfidf']
    clf_model = lr_pipeline.named_steps['clf']
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
    coefs = clf_model.coef_[0]

    top_positive_idx = np.argsort(coefs)[-15:]
    top_negative_idx = np.argsort(coefs)[:15]

    print("\nTop Positive Sentiment N-Grams (Predicts Recommended):")
    for idx in reversed(top_positive_idx):
        print(f"   {feature_names[idx]}: +{coefs[idx]:.4f}")

    print("\nTop Negative Sentiment N-Grams (Predicts Not Recommended):")
    for idx in top_negative_idx:
        print(f"   {feature_names[idx]}: {coefs[idx]:.4f}")

    # Save trained pipeline
    model_save_path = 'models/recommendation_pipeline.joblib'
    joblib.dump(lr_pipeline, model_save_path)
    print(f"\nSaved model pipeline to: {model_save_path}")

    # -------------------------------------------------------------
    # GENERATE NOTEBOOK 03_nlp_and_machine_learning.ipynb
    # -------------------------------------------------------------
    print("Generating notebooks/03_nlp_and_machine_learning.ipynb...")
    nb = nbf.v4.new_notebook()

    cells = []
    cells.append(nbf.v4.new_markdown_cell("""# Stage 04 & 05: NLP Sentiment Analysis & Machine Learning Classification
**Project**: Steam Game Intelligence  
**Notebook**: `notebooks/03_nlp_and_machine_learning.ipynb`  
**Objective**: Perform NLP sentiment theme analysis and build a reproducible ML recommendation prediction pipeline while strictly auditing against target leakage.

---
## Master ML Guidelines & Auditing:
1. **Target**: `is_recommended` (1 = Recommended, 0 = Not Recommended).
2. **Leakage Audit**: All text n-grams and engagement features are extracted strictly at the point of review submission. No future metrics (such as aggregate game sales or post-hoc ratings) are used in prediction.
3. **Pipeline Safety**: Preprocessing and TF-IDF vectorization fit strictly on `X_train` to prevent train-test data leakage.
4. **Metrics**: Evaluated using Precision, Recall, F1-Score, and ROC-AUC (not raw accuracy alone).
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
)

sns.set_theme(style='whitegrid')

# Load Processed Reviews
rev_path = '../data/processed/steam_game_reviews_clean.csv' if os.path.exists('../data/processed/steam_game_reviews_clean.csv') else 'data/processed/steam_game_reviews_clean.csv'
df_reviews = pd.read_csv(rev_path, usecols=['game_name', 'review', 'is_recommended'])
df_reviews['review'] = df_reviews['review'].fillna('')
df_reviews = df_reviews[df_reviews['review'].str.strip() != ''].copy()

print(f"Total valid review records: {len(df_reviews)}")
print("Class Distribution:")
print(df_reviews['is_recommended'].value_counts(normalize=True))
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 1. Model Training & Pipeline Setup
We sample 100,000 reviews for reproducible training, split 80/20 train/test, and fit a TF-IDF + LogisticRegression pipeline.
"""))

    cells.append(nbf.v4.new_code_cell("""df_sample = df_reviews.sample(n=100000, random_state=42).reset_index(drop=True)
X = df_sample['review']
y = df_sample['is_recommended']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Fit Logistic Regression Pipeline
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

print("=== Logistic Regression Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['Not Recommended', 'Recommended']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2. Feature Importance & Sentiment Keyword Inspection
Inspecting the top positive and negative n-gram weights learned by the model.
"""))

    cells.append(nbf.v4.new_code_cell("""tfidf = pipe.named_steps['tfidf']
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
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3. Model Checkpoint & Persistence
Saved trained pipeline to `models/recommendation_pipeline.joblib`.
"""))

    nb['cells'] = cells

    notebook_path = 'notebooks/03_nlp_and_machine_learning.ipynb'
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Saved notebook: {notebook_path}")

    print("=== Stage 04 & 05 Pipeline Completed Successfully ===")

if __name__ == '__main__':
    main()
