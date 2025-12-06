# Data Versioning Pipeline

A production-ready MLOps pipeline for complete data and model lifecycle management using DVC, MLflow, Great Expectations, and automated CI/CD.

## 🚀 Features

- **Data Validation**: Automated data quality checks using Great Expectations.
- **Drift Detection**: Schema and distribution drift detection with alerting.
- **Data Versioning**: DVC-powered data and model versioning.
- **Experiment Tracking**: MLflow integration for tracking experiments.
- **Model Registry**: Centralized model version management.
- **Automated Pipeline**: Reproducible ML pipeline with single command.
- **CI/CD**: GitHub Actions for automated testing, validation, and reporting.
- **API Deployment**: FastAPI endpoint for model serving.

## 📁 Project Structure
```
data-versioning-pipeline/
├── .github/workflows/    # CI/CD workflows
├── data/                 # Data directory
│   ├── raw/              # Raw datasets
│   └── processed/        # Processed datasets
├── gx/                   # Great Expectations configuration
├── models/               # Trained models
├── notebooks/            # Jupyter notebooks
├── src/                  # Source code
│   ├── data/             # Data processing & validation modules
│   ├── models/           # Model training/evaluation
│   ├── utils/            # Utility functions
│   └── api/              # FastAPI application
├── tests/                # Unit and integration tests
├── configs/              # Configuration files
└── README.md
```

## 🔧 Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize Great Expectations:**
   ```bash
   python src/data/create_dummy_data.py
   python src/data/create_expectation_suite.py
   ```

## 📊 Usage

**Run the Pipeline:**
The pipeline is automated via GitHub Actions, but you can run individual steps locally:

1. **Validate Data:**
   ```bash
   python src/data/validate_gx.py
   ```

2. **Simulate Bad Data (Drift/Quality Issues):**
   ```bash
   python src/data/simulate_bad_data.py
   python src/data/validate_gx.py data/raw/dataset_bad.csv
   ```

3. **Train Model:**
   ```bash
   dvc repro
   ```

## 👤 Author

**Harshith**
- GitHub: [@harshithluc073](https://github.com/harshithluc073)
- Email: chitikeshiharshith@gmail.com

## 📝 License

MIT License
