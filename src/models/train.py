"""
Model Training Module
Trains ML model with MLflow tracking
"""

import pandas as pd
import numpy as np
import yaml
import os
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
from datetime import datetime


class ModelTrainer:
    """Train ML models with MLflow tracking"""
    
    def __init__(self, params_path="params.yaml"):
        """
        Initialize trainer
        
        Args:
            params_path: Path to parameters file
        """
        self.params = self._load_params(params_path)
        self.train_config = self.params.get('train', {})
        self.mlflow_config = self.params.get('mlflow', {})
        self.model = None
        
    def _load_params(self, params_path):
        """Load parameters from YAML file"""
        with open(params_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        print(f"\n{'='*50}")
        print("Setting up MLflow")
        print(f"{'='*50}\n")
        
        # Set tracking URI
        tracking_uri = self.mlflow_config.get('tracking_uri', './mlruns')
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment
        experiment_name = self.mlflow_config.get('experiment_name', 'data-versioning-pipeline')
        mlflow.set_experiment(experiment_name)
        
        print(f"✓ Tracking URI: {tracking_uri}")
        print(f"✓ Experiment: {experiment_name}")
        
        return experiment_name
    
    def load_training_data(self, train_path="data/processed/train.csv"):
        """Load training data"""
        print(f"\n{'='*50}")
        print("Loading Training Data")
        print(f"{'='*50}\n")
        
        self.train_data = pd.read_csv(train_path)
        
        # Split features and target
        self.X_train = self.train_data.drop(columns=['target'])
        self.y_train = self.train_data['target']
        
        print(f"✓ Loaded training data: {self.train_data.shape}")
        print(f"✓ Features: {self.X_train.shape}")
        print(f"✓ Target: {self.y_train.shape}")
        print(f"✓ Feature columns: {list(self.X_train.columns)}")
        print(f"✓ Target distribution: {dict(self.y_train.value_counts().sort_index())}")
        
        return self.X_train, self.y_train
    
    def create_model(self):
        """Create model based on configuration"""
        print(f"\n{'='*50}")
        print("Creating Model")
        print(f"{'='*50}\n")
        
        model_type = self.train_config.get('model_type', 'random_forest')
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=self.train_config.get('n_estimators', 100),
                max_depth=self.train_config.get('max_depth', 10),
                min_samples_split=self.train_config.get('min_samples_split', 2),
                min_samples_leaf=self.train_config.get('min_samples_leaf', 1),
                random_state=self.train_config.get('random_state', 42),
                n_jobs=-1
            )
            print("✓ Model type: Random Forest Classifier")
            
        elif model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(
                max_depth=self.train_config.get('max_depth', 10),
                min_samples_split=self.train_config.get('min_samples_split', 2),
                min_samples_leaf=self.train_config.get('min_samples_leaf', 1),
                random_state=self.train_config.get('random_state', 42)
            )
            print("✓ Model type: Decision Tree Classifier")
            
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=self.train_config.get('random_state', 42)
            )
            print("✓ Model type: Logistic Regression")
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Print model parameters
        print(f"✓ Model parameters:")
        for param, value in self.model.get_params().items():
            print(f"  • {param}: {value}")
        
        return self.model
    
    def train_model(self):
        """Train the model"""
        print(f"\n{'='*50}")
        print("Training Model")
        print(f"{'='*50}\n")
        
        # Start timer
        start_time = datetime.now()
        
        # Train model
        self.model.fit(self.X_train, self.y_train)
        
        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Training accuracy
        train_score = self.model.score(self.X_train, self.y_train)
        
        print(f"✓ Training completed!")
        print(f"✓ Training time: {training_time:.2f} seconds")
        print(f"✓ Training accuracy: {train_score:.4f}")
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n✓ Feature Importance (Top 5):")
            for idx, row in feature_importance.head().iterrows():
                print(f"  • {row['feature']}: {row['importance']:.4f}")
            
            self.feature_importance = feature_importance
        
        self.train_score = train_score
        self.training_time = training_time
        
        return self.model
    
    def save_model(self, output_path="models/model.pkl"):
        """Save trained model"""
        print(f"\n{'='*50}")
        print("Saving Model")
        print(f"{'='*50}\n")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save model
        joblib.dump(self.model, output_path)
        
        print(f"✓ Model saved: {output_path}")
        
        # Save feature importance if available
        if hasattr(self, 'feature_importance'):
            importance_path = output_path.replace('.pkl', '_feature_importance.csv')
            self.feature_importance.to_csv(importance_path, index=False)
            print(f"✓ Feature importance saved: {importance_path}")
        
        return output_path
    
    def log_to_mlflow(self):
        """Log parameters, metrics, and model to MLflow"""
        print(f"\n{'='*50}")
        print("Logging to MLflow")
        print(f"{'='*50}\n")
        
        # Start MLflow run
        run_name = self.mlflow_config.get('run_name_prefix', 'run') + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_params(self.train_config)
            print("✓ Logged parameters")
            
            # Log metrics
            mlflow.log_metric("train_accuracy", self.train_score)
            mlflow.log_metric("training_time", self.training_time)
            print("✓ Logged metrics")
            
            # Log model
            mlflow.sklearn.log_model(
                self.model, 
                "model",
                registered_model_name=self.train_config.get('model_type', 'model')
            )
            print("✓ Logged model to MLflow")
            
            # Log feature importance if available
            if hasattr(self, 'feature_importance'):
                importance_path = "models/model_feature_importance.csv"
                mlflow.log_artifact(importance_path, "feature_importance")
                print("✓ Logged feature importance")
            
            # Get run ID
            run_id = mlflow.active_run().info.run_id
            print(f"✓ MLflow Run ID: {run_id}")
            
            return run_id
    
    def run_training(self, train_path="data/processed/train.csv", output_path="models/model.pkl"):
        """Run complete training pipeline"""
        print(f"\n{'='*60}")
        print("STARTING MODEL TRAINING PIPELINE")
        print(f"{'='*60}\n")
        
        # Setup MLflow
        self.setup_mlflow()
        
        # Load data
        self.load_training_data(train_path)
        
        # Create model
        self.create_model()
        
        # Train model
        self.train_model()
        
        # Save model
        self.save_model(output_path)
        
        # Log to MLflow
        self.log_to_mlflow()
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE!")
        print(f"{'='*60}\n")
        
        # Summary
        print("Summary:")
        print(f"  • Model type: {self.train_config.get('model_type', 'random_forest')}")
        print(f"  • Training samples: {len(self.X_train)}")
        print(f"  • Features: {self.X_train.shape[1]}")
        print(f"  • Training accuracy: {self.train_score:.4f}")
        print(f"  • Training time: {self.training_time:.2f}s")
        print(f"  • Model saved: {output_path}")
        print()


def main():
    """Main training function"""
    # Initialize trainer
    trainer = ModelTrainer(params_path="params.yaml")
    
    # Run training pipeline
    trainer.run_training(
        train_path="data/processed/train.csv",
        output_path="models/model.pkl"
    )
    
    print("✓ Model training completed successfully!")
    print("\nTo view MLflow UI, run: mlflow ui")


if __name__ == "__main__":
    main()