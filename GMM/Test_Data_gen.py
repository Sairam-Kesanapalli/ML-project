#!/usr/bin/env python3
"""
Test_Data_gen.py
Generate an example external CSV whose columns and shapes match training features.
Produces: external/example_matched.csv
"""

import os
import numpy as np
import pandas as pd
import joblib

EXTERNAL_DIR = "external"
OUT_CSV = os.path.join(EXTERNAL_DIR, "example_matched.csv")
os.makedirs(EXTERNAL_DIR, exist_ok=True)

# Load training columns and (optionally) training statistics if available
try:
    cols = joblib.load("output/training_columns_shuttle.joblib")
except Exception:
    # Fallback: small default (will fail only if MAIN train hasn't been run)
    raise RuntimeError("training_columns_shuttle.joblib not found. Run `python MAIN.py train` first.")

# If you have scaler saved, use its mean/std to make more realistic samples:
if os.path.exists("output/scaler_gmm_shuttle.joblib"):
    scaler = joblib.load("output/scaler_gmm_shuttle.joblib")
    # scaler.mean_ exists for StandardScaler fitted on training set
    means = scaler.mean_
    # get scale (std)
    scale = scaler.scale_
    # create 10 rows around training distribution
    rows = (means + np.random.randn(10, len(means)) * scale)
    df = pd.DataFrame(rows, columns=cols)
else:
    # Fallback random
    rows = np.random.randn(10, len(cols))
    df = pd.DataFrame(rows, columns=cols)

df.to_csv(OUT_CSV, index=False)
print("Created example external CSV at:", OUT_CSV)
print(df.head())
