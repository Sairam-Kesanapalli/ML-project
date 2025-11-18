@echo off
setlocal enabledelayedexpansion

set REPO_URL=https://github.com/Sairam-Kesanapalli/ML-project.git
set REPO_DIR=ML-project
set PY=python
set VENV_DIR=.venv

echo 1/ Cloning or updating GitHub repository
if not exist "%REPO_DIR%" (
    git clone "%REPO_URL%" || exit /b 1
) else (
    cd "%REPO_DIR%" || exit /b 1
    git pull || exit /b 1
    cd .. || exit /b 1
)

echo 2/ Creating virtualenv ^(if missing^)
if not exist "%VENV_DIR%" (
    %PY% -m venv "%VENV_DIR%" || exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat" || exit /b 1

echo 3/ Installing requirements
if exist "%REPO_DIR%\requirements.txt" (
    pip install -r "%REPO_DIR%\requirements.txt" || exit /b 1
) else (
    pip install numpy pandas scikit-learn matplotlib joblib ucimlrepo || exit /b 1
)

echo 4/ Running MAIN.py from GitHub repo
cd "%REPO_DIR%\GMM" || exit /b 1
python MAIN.py train || exit /b 1

echo 5/ Generating external test data
python Test_Data_gen.py || exit /b 1

echo 6/ Running prediction
python MAIN.py predict --external external\example_matched.csv --out output\external_gmm_predictions_matched.csv || exit /b 1

echo ✔ Done. Predictions saved to output\external_gmm_predictions_matched.csv

endlocal
