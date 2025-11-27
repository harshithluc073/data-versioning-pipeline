"""
SIMPLIFIED Model Training - Guaranteed to work with registry
"""

import pandas as pd
import yaml
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from datetime import datetime


def train_and_log_model():
    """Train model with guaranteed MLflow logging"""
    
    print("\n" + "="*60)
    print("TRAINING MODEL WITH PROPER MLFLOW LOGGING")
    print("="*60 + "\n")
    
    # Load params
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    train_config = params.get('train', {})
    mlflow_config = params.get('mlflow', {})
    
    # Setup MLflow
    mlflow.set_tracking_uri(mlflow_config.get('tracking_uri', './mlruns'))
    mlflow.set_experiment(mlflow_config.get('experiment_name', 'data-versioning-pipeline'))
    
    # Load training data
    print("Loading training data...")
    train_data = pd.read_csv("data/processed/train.csv")
    X_train = train_data.drop(columns=['target'])
    y_train = train_data['target']
    print(f"✓ Loaded {len(X_train)} training samples")
    
    # Load test data
    print("Loading test data...")
    test_data = pd.read_csv("data/processed/test.csv")
    X_test = test_data.drop(columns=['target'])
    y_test = test_data['target']
    print(f"✓ Loaded {len(X_test)} test samples")
    
    # Create model
    print("\nCreating model...")
    model = RandomForestClassifier(
        n_estimators=train_config.get('n_estimators', 100),
        max_depth=train_config.get('max_depth', 10),
        min_samples_split=train_config.get('min_samples_split', 2),
        min_samples_leaf=train_config.get('min_samples_leaf', 1),
        random_state=train_config.get('random_state', 42),
        n_jobs=-1
    )
    print(f"✓ Created Random Forest with {train_config.get('n_estimators', 100)} trees")
    
    # Start MLflow run
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\nStarting MLflow run: {run_name}")
    
    with mlflow.start_run(run_name=run_name):
        
        # Train model
        print("\nTraining model...")
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate metrics
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        print(f"✓ Training completed in {training_time:.2f}s")
        print(f"✓ Train accuracy: {train_accuracy:.4f}")
        print(f"✓ Test accuracy: {test_accuracy:.4f}")
        
        # Log parameters
        mlflow.log_params(train_config)
        print("\n✓ Logged parameters to MLflow")
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("training_time", training_time)
        print("✓ Logged metrics to MLflow")
        
        # Save model locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")
        print("✓ Saved model to models/model.pkl")
        
        # Log model to MLflow - THE CRITICAL PART
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",  # MUST be exactly "model"
            registered_model_name=None  # We'll register manually
        )
        print("✓ Logged model to MLflow with artifact_path='model'")
        
        run_id = mlflow.active_run().info.run_id
        print(f"\n✓ MLflow Run ID: {run_id}")
        print(f"✓ This run CAN be registered! ✅")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. python diagnostic_check_runs.py  (verify model logged)")
    print("  2. python src/models/registry.py    (register model)")
    

if __name__ == "__main__":
    train_and_log_model()