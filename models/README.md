# Models Directory

This directory contains trained machine learning models, preprocessors, and model performance benchmarks for the Steam Game Intelligence project.

## Models Overview
- `recommendation_pipeline.joblib`: Scikit-learn baseline and production classification pipelines (TF-IDF + Classifier) predicting review recommendation status.
- `leakage_audit_log.json`: Audit log verifying feature temporal availability and preventing target leakage.
