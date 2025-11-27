@echo off
REM reproduce.bat - Complete pipeline reproduction script for Windows
REM Rebuilds entire pipeline with a single command

setlocal enabledelayedexpansion

echo ========================================
echo Data Versioning Pipeline - Full Reproduce
echo ========================================
echo.

REM Check if virtual environment is activated
if "%VIRTUAL_ENV%"=="" (
    echo [WARNING] Virtual environment not activated
    echo    Run: venv\Scripts\activate
    echo.
    set /p confirm="Continue anyway? (y/n): "
    if /i not "!confirm!"=="y" (
        exit /b 1
    )
)

echo Step 1: Validating data...
echo -----------------------------------
python src/data/validate.py
if errorlevel 1 (
    echo [FAILED] Data validation failed
    exit /b 1
)
echo [OK] Data validation passed
echo.

echo Step 2: Preprocessing data...
echo -----------------------------------
python src/data/preprocess.py
if errorlevel 1 (
    echo [FAILED] Data preprocessing failed
    exit /b 1
)
echo [OK] Data preprocessing completed
echo.

echo Step 3: Training model...
echo -----------------------------------
python simple_train.py
if errorlevel 1 (
    echo [FAILED] Model training failed
    exit /b 1
)
echo [OK] Model training completed
echo.

echo Step 4: Registering model...
echo -----------------------------------
python src/models/registry.py
if errorlevel 1 (
    echo [WARNING] Model registration failed ^(continuing...^)
) else (
    echo [OK] Model registered
)
echo.

echo ========================================
echo Pipeline Execution Complete!
echo ========================================
echo.
echo Summary:
echo   [OK] Data validated and preprocessed
echo   [OK] Model trained with MLflow tracking
echo   [OK] Model registered ^(if successful^)
echo.
echo Generated outputs:
echo   - data/processed/train.csv
echo   - data/processed/test.csv
echo   - models/model.pkl
echo   - metrics.json
echo   - mlruns/ ^(MLflow tracking^)
echo.
echo Next steps:
echo   1. View MLflow UI:  mlflow ui
echo   2. View metrics:    dvc metrics show
echo   3. Push to remote:  dvc push ^&^& git push
echo.

endlocal