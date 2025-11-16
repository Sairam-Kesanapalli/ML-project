#!/usr/bin/env bash
set -e

REPO_URL="https://github.com/Sairam-Kesanapalli/ML-project.git"
REPO_DIR="ML-project"
PY=python3
VENV_DIR=".venv"

echo "1/ Cloning or updating GitHub repository"
if [ ! -d "$REPO_DIR" ]; then
  git clone "$REPO_URL"
else
  cd "$REPO_DIR"
  git pull
  cd ..
fi

echo "2/ Creating virtualenv (if missing)"
if [ ! -d "$VENV_DIR" ]; then
  $PY -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "3/ Installing requirements"
if [ -f "$REPO_DIR/requirements.txt" ]; then
  pip install -r "$REPO_DIR/requirements.txt"
else
  pip install numpy pandas scikit-learn matplotlib joblib ucimlrepo
fi

echo "4/ Running MAIN.py from GitHub repo"
cd "$REPO_DIR/GMM"

python MAIN.py train

echo "5/ Generating external test data"
python Test_Data_gen.py

echo "6/ Running prediction"
python MAIN.py predict --external external/example_matched.csv --out output/external_gmm_predictions_matched.csv

echo "✔ Done. Predictions saved to output/external_gmm_predictions_matched.csv"
