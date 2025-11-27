# Makefile for Data Versioning Pipeline
# Simplifies common workflow commands

.PHONY: help setup install clean reproduce train evaluate register test lint format run-pipeline mlflow-ui dvc-status commit-all

# Default target
help:
	@echo "Data Versioning Pipeline - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup          - Create virtual environment and install dependencies"
	@echo "  make install        - Install dependencies in existing environment"
	@echo ""
	@echo "Pipeline Execution:"
	@echo "  make reproduce      - Run complete DVC pipeline (validate -> train -> evaluate)"
	@echo "  make train          - Train model only"
	@echo "  make evaluate       - Evaluate model only"
	@echo "  make register       - Register best model to MLflow registry"
	@echo ""
	@echo "Development:"
	@echo "  make test           - Run all tests"
	@echo "  make lint           - Check code quality"
	@echo "  make format         - Format code with black"
	@echo ""
	@echo "Utilities:"
	@echo "  make mlflow-ui      - Start MLflow UI"
	@echo "  make dvc-status     - Check DVC pipeline status"
	@echo "  make clean          - Remove temporary files and caches"
	@echo "  make commit-all     - Stage and commit all changes"

# Setup virtual environment and install dependencies
setup:
	@echo "Creating virtual environment..."
	python -m venv venv
	@echo "Installing dependencies..."
	.\venv\Scripts\pip install --upgrade pip
	.\venv\Scripts\pip install -r requirements.txt
	@echo "✓ Setup complete! Activate with: venv\Scripts\activate"

# Install dependencies in existing environment
install:
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "✓ Dependencies installed!"

# Run complete DVC pipeline
reproduce:
	@echo "Running complete DVC pipeline..."
	dvc repro
	@echo "✓ Pipeline execution complete!"
	@echo ""
	@echo "View metrics: dvc metrics show"
	@echo "View MLflow UI: make mlflow-ui"

# Train model (using simple_train.py for proper MLflow logging)
train:
	@echo "Training model..."
	python simple_train.py
	@echo "✓ Training complete!"
	@echo ""
	@echo "View in MLflow UI: make mlflow-ui"

# Evaluate model
evaluate:
	@echo "Evaluating model..."
	python src/models/evaluate.py
	@echo "✓ Evaluation complete!"
	@echo ""
	@echo "View metrics: type metrics.json"

# Register best model to MLflow registry
register:
	@echo "Registering best model..."
	python src/models/registry.py
	@echo "✓ Model registered!"
	@echo ""
	@echo "View in MLflow UI: make mlflow-ui"

# Run all tests
test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov-report=html
	@echo "✓ Tests complete!"
	@echo ""
	@echo "View coverage report: start htmlcov\index.html"

# Check code quality with flake8
lint:
	@echo "Checking code quality..."
	flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503
	@echo "✓ Linting complete!"

# Format code with black
format:
	@echo "Formatting code with black..."
	black src/ tests/
	@echo "✓ Code formatted!"

# Start MLflow UI
mlflow-ui:
	@echo "Starting MLflow UI..."
	@echo "Access at: http://127.0.0.1:5000"
	mlflow ui

# Check DVC pipeline status
dvc-status:
	@echo "DVC Pipeline Status:"
	@echo ""
	dvc status
	@echo ""
	@echo "DVC Metrics:"
	dvc metrics show

# Clean temporary files and caches
clean:
	@echo "Cleaning temporary files..."
	@if exist __pycache__ rmdir /s /q __pycache__
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@if exist htmlcov rmdir /s /q htmlcov
	@if exist .coverage del /q .coverage
	@if exist *.pyc del /q *.pyc
	@for /r %%i in (__pycache__) do @if exist "%%i" rmdir /s /q "%%i"
	@echo "✓ Cleaned!"

# Commit all changes
commit-all:
	@echo "Staging all changes..."
	git add .
	@echo ""
	@set /p msg="Enter commit message: " && git commit -m "!msg!"
	@echo "✓ Changes committed!"

# Run complete workflow: reproduce -> register -> push
run-pipeline: reproduce register
	@echo ""
	@echo "========================================"
	@echo "Complete Pipeline Executed!"
	@echo "========================================"
	@echo ""
	@echo "Summary:"
	@echo "  ✓ Data validated and preprocessed"
	@echo "  ✓ Model trained and evaluated"
	@echo "  ✓ Best model registered"
	@echo ""
	@echo "Next steps:"
	@echo "  1. View results: make mlflow-ui"
	@echo "  2. Push to remote: dvc push && git push"
	@echo "  3. Deploy model to production"
