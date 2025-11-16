#!/usr/bin/env python3
"""
MAIN.py
Train per-class GMMs on Statlog Shuttle dataset, save models, and allow external prediction.
Usage:
  python MAIN.py train
  python MAIN.py predict --external external/example_matched.csv
  python MAIN.py evaluate
"""

import os
import argparse
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# If you use ucimlrepo fetcher in your environment:
try:
    from ucimlrepo import fetch_ucirepo
    _HAS_UCIML = True
except Exception:
    _HAS_UCIML = False

# --- config ---
OUTPUT_DIR = "output"
EXTERNAL_DIR = "external"
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler_gmm_shuttle.joblib")
COLS_PATH   = os.path.join(OUTPUT_DIR, "training_columns_shuttle.joblib")
GMMS_PATH   = os.path.join(OUTPUT_DIR, "gmm_class_model_shuttle.joblib")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXTERNAL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_shuttle():
    if _HAS_UCIML:
        logging.info("Loading Statlog Shuttle via ucimlrepo.fetch_ucirepo(id=148)")
        data = fetch_ucirepo(id=148)
        X = data.data.features.copy()
        y = data.data.targets.squeeze().copy()
        return X, y
    else:
        raise RuntimeError("ucimlrepo not installed in this environment. Please install or supply local CSV.")


def train_and_save(random_state=42):
    X, y = load_shuttle()
    logging.info("X shape: %s, y shape: %s", X.shape, y.shape)
    logging.info("Class distribution:\n%s", y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(list(X.columns), COLS_PATH)
    logging.info("Saved scaler and training column list to %s and %s", SCALER_PATH, COLS_PATH)

    classes = sorted(y_train.unique())
    gmms = {}
    for c in classes:
        Xc = X_train_s[y_train == c]
        # Choose components sensibly
        n_comp = 1 if len(Xc) < 50 else 2
        gmm = GaussianMixture(
            n_components=n_comp,
            covariance_type="full",
            random_state=random_state,
            max_iter=300
        )
        gmm.fit(Xc)
        gmms[c] = gmm
        logging.info("Trained GMM for class %s  (n_components=%d, samples=%d)", c, n_comp, len(Xc))

    joblib.dump(gmms, GMMS_PATH)
    logging.info("Saved GMM models to %s", GMMS_PATH)

    # Evaluate on test set
    y_pred, scores = gmm_predict_from_dict(gmms, X_test_s)
    acc = accuracy_score(y_test, y_pred)
    logging.info("GMM Classification Accuracy on test split: %.4f", acc)
    logging.info("Classification report:\n%s", classification_report(y_test, y_pred))
    logging.info("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred))

    return {"accuracy": acc}


def gmm_predict_from_dict(gmms_dict, X_scaled):
    classes_list = sorted(gmms_dict.keys())
    scores = np.vstack([gmms_dict[c].score_samples(X_scaled) for c in classes_list]).T
    preds_idx = np.argmax(scores, axis=1)
    return np.array([classes_list[i] for i in preds_idx]), scores


def predict_external(external_csv_path, out_path=None):
    if not os.path.exists(SCALER_PATH) or not os.path.exists(GMMS_PATH) or not os.path.exists(COLS_PATH):
        raise RuntimeError("Required artifacts not found. Run 'python MAIN.py train' first.")

    scaler = joblib.load(SCALER_PATH)
    gmms = joblib.load(GMMS_PATH)
    cols = joblib.load(COLS_PATH)

    df_ext = pd.read_csv(external_csv_path)
    # Ensure columns match training order
    df_ext = df_ext.loc[:, cols]
    X_ext_s = scaler.transform(df_ext.values)

    preds, scores = gmm_predict_from_dict(gmms, X_ext_s)

    df_out = df_ext.copy()
    df_out["predicted_class"] = preds

    out_path = out_path or os.path.join(OUTPUT_DIR, "external_gmm_predictions_matched.csv")
    df_out.to_csv(out_path, index=False)
    logging.info("Saved predictions to: %s", out_path)

    scores_df = pd.DataFrame(scores, columns=[str(c) for c in sorted(gmms.keys())])
    logging.info("Per-class log-likelihoods (first 5 rows):\n%s", scores_df.head().to_string())
    return out_path


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub_train = sub.add_parser("train", help="Train GMMs and save artifacts")
    sub_train.add_argument("--random-state", type=int, default=42)

    sub_predict = sub.add_parser("predict", help="Predict on external CSV")
    sub_predict.add_argument("--external", required=True, help="Path to external CSV (features must match)")
    sub_predict.add_argument("--out", required=False, help="Output CSV path")

    args = parser.parse_args()

    if args.cmd == "train":
        train_and_save(random_state=args.random_state)
    elif args.cmd == "predict":
        predict_external(args.external, out_path=args.out)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
