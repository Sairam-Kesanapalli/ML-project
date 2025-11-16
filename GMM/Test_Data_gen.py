import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

#@title 🔵 BLOCK 6 — CREATE MATCHED EXTERNAL EXAMPLE DATASET
# Kindly note that the features names, order of features and dimensions in this example dataset is same as in the original dataset
means = X.mean()
stds  = X.std().replace(0, 1.0)

rows = [(means + np.random.randn(len(means)) * stds).values for _ in range(10)]
example_df = pd.DataFrame(rows, columns=list(X.columns))

EXAMPLE_CSV = "/content/external_data_example_matched.csv"
example_df.to_csv(EXAMPLE_CSV, index=False)

print("Created example:", EXAMPLE_CSV)
print(example_df.head())

#@title 🔵 BLOCK 7 — EXTERNAL PREDICTION WITH GMM
EXAMPLE_CSV = "/content/external_data_example_matched.csv"
OUT_PRED = "/content/output/external_gmm_predictions_matched.csv"

scaler = joblib.load("/content/output/scaler_gmm_shuttle.joblib")
gmms   = joblib.load("/content/output/gmm_class_model_shuttle.joblib")

df_ext = pd.read_csv(EXAMPLE_CSV)
X_ext_s = scaler.transform(df_ext)

preds, scores = gmm_predict_from_dict(gmms, X_ext_s)

df_out = df_ext.copy()
df_out["predicted_class"] = preds
df_out.to_csv(OUT_PRED, index=False)

print("Saved predictions to:", OUT_PRED)
print(df_out.head())

scores_df = pd.DataFrame(scores, columns=[str(c) for c in sorted(gmms.keys())])
print("\nPer-class log-likelihoods (first 5 rows):")
print(scores_df.head())
