#!/usr/bin/env bash
set -e

PY=python3
VENV_DIR=".venv"

echo "1/ Creating virtualenv (if missing)"
if [ ! -d "$VENV_DIR" ]; then
  $PY -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "2/ Installing requirements"
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  pip install --upgrade pip
  pip install numpy pandas scikit-learn matplotlib joblib ucimlrepo
fi

echo "3/ Train models"
python MAIN.py train

echo "4/ Generate example external CSV"
python Test_Data_gen.py

echo "5/ Run external prediction"
python MAIN.py predict --external external/example_matched.csv --out output/external_gmm_predictions_matched.csv

echo "Done. Predictions at output/external_gmm_predictions_matched.csv"
