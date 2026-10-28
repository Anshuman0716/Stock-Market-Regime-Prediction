@echo off
echo ===================================================
echo Setting up the environment and running everything
echo ===================================================

REM requirements.txt pins numpy<2.0, which has no wheels for Python 3.13,
REM so the environment is built on Python 3.12 with uv.
where uv >nul 2>nul
if errorlevel 1 (
    echo uv not found. Install it from https://docs.astral.sh/uv/ and re-run.
    exit /b 1
)

echo [1/5] Creating virtual environment (.venv, Python 3.12) if missing...
if not exist .venv uv venv .venv --python 3.12
if errorlevel 1 exit /b 1

echo [2/5] Installing dependencies...
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
if errorlevel 1 exit /b 1

echo [3/5] Running the pipeline (train, backtest, analysis, database)...
.venv\Scripts\python.exe scripts\run_pipeline.py
if errorlevel 1 exit /b 1

echo [4/5] Regenerating the README results from the new store...
.venv\Scripts\python.exe scripts\update_readme.py
if errorlevel 1 exit /b 1

echo [5/5] Running test suite...
.venv\Scripts\python.exe -m pytest tests\ -q
if errorlevel 1 exit /b 1

echo ===================================================
echo ALL DONE.
echo ===================================================
echo Daily refresh without retraining:  .venv\Scripts\python.exe scripts\run_pipeline.py --reuse-models
echo Dashboard:                         .venv\Scripts\streamlit.exe run app.py
