#INSTALL & IMPORTS
!pip install -q ucimlrepo joblib matplotlib scikit-learn pandas numpy seaborn

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

os.makedirs("/content/output", exist_ok=True)
print("Environment ready.")

#LOAD STATLOG SHUTTLE DATASET
data = fetch_ucirepo(id=148)
X = data.data.features.copy()
y = data.data.targets.squeeze().copy()

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Class distribution:\n", y.value_counts())

#TRAIN/TEST SPLIT + STANDARDIZATION
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler_sh = StandardScaler().fit(X_train)
X_train_s = scaler_sh.transform(X_train)
X_test_s  = scaler_sh.transform(X_test)

joblib.dump(scaler_sh, "/content/output/scaler_gmm_shuttle.joblib")
joblib.dump(list(X.columns), "/content/output/training_columns_shuttle.joblib")

print("Scaler + training column list saved.")

#TRAIN GMM MODELS (ONE PER CLASS)
classes = sorted(y_train.unique())
gmms_sh = {}

for c in classes:
    Xc = X_train_s[y_train == c]
    n_comp = 1 if len(Xc) < 50 else 2
    gmm = GaussianMixture(
        n_components=n_comp,
        covariance_type='full',
        random_state=42,
        max_iter=200
    )
    gmm.fit(Xc)
    gmms_sh[c] = gmm
    print(f"Trained GMM for class {c}  (n_components={n_comp})")

joblib.dump(gmms_sh, "/content/output/gmm_class_model_shuttle.joblib")
print("Saved all GMMs.")
