"""
Model Evaluation Module
Evaluates trained model on test data and generates metrics
"""

import pandas as pd
import numpy as np
import yaml
import json
import os
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Evaluate trained ML model"""
    
    def __init__(self, params_path="params.yaml"):
        """
        Initialize evaluator
        
        Args:
            params_path: Path to parameters file
        """
        self.params = self._load_params(params_path)
        self.evaluate_config = self.params.get('evaluate', {})
        self.mlflow_config = self.params.get('mlflow', {})
        self.metrics = {}
        
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
    
    def load_model(self, model_path="models/model.pkl"):
        """Load trained model"""
        print(f"\n{'='*50}")
        print("Loading Trained Model")
        print(f"{'='*50}\n")
        
        self.model = joblib.load(model_path)
        
        print(f"✓ Model loaded: {model_path}")
        print(f"✓ Model type: {type(self.model).__name__}")
        
        return self.model
    
    def load_test_data(self, test_path="data/processed/test.csv"):
        """Load test data"""
        print(f"\n{'='*50}")
        print("Loading Test Data")
        print(f"{'='*50}\n")
        
        self.test_data = pd.read_csv(test_path)
        
        # Split features and target
        self.X_test = self.test_data.drop(columns=['target'])
        self.y_test = self.test_data['target']
        
        print(f"✓ Loaded test data: {self.test_data.shape}")
        print(f"✓ Features: {self.X_test.shape}")
        print(f"✓ Target: {self.y_test.shape}")
        print(f"✓ Target distribution: {dict(self.y_test.value_counts().sort_index())}")
        
        return self.X_test, self.y_test
    
    def make_predictions(self):
        """Generate predictions on test data"""
        print(f"\n{'='*50}")
        print("Making Predictions")
        print(f"{'='*50}\n")
        
        # Predictions
        self.y_pred = self.model.predict(self.X_test)
        
        # Prediction probabilities (if available)
        if hasattr(self.model, 'predict_proba'):
            self.y_pred_proba = self.model.predict_proba(self.X_test)
            print(f"✓ Predictions generated with probabilities")
        else:
            self.y_pred_proba = None
            print(f"✓ Predictions generated")
        
        print(f"✓ Predicted distribution: {dict(pd.Series(self.y_pred).value_counts().sort_index())}")
        
        return self.y_pred
    
    def calculate_metrics(self):
        """Calculate evaluation metrics"""
        print(f"\n{'='*50}")
        print("Calculating Metrics")
        print(f"{'='*50}\n")
        
        # Accuracy
        accuracy = accuracy_score(self.y_test, self.y_pred)
        self.metrics['accuracy'] = float(accuracy)
        print(f"✓ Accuracy: {accuracy:.4f}")
        
        # Precision, Recall, F1 (weighted for multiclass)
        precision = precision_score(self.y_test, self.y_pred, average='weighted')
        recall = recall_score(self.y_test, self.y_pred, average='weighted')
        f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        
        self.metrics['precision'] = float(precision)
        self.metrics['recall'] = float(recall)
        self.metrics['f1_score'] = float(f1)
        
        print(f"✓ Precision: {precision:.4f}")
        print(f"✓ Recall: {recall:.4f}")
        print(f"✓ F1-Score: {f1:.4f}")
        
        # Per-class metrics
        print(f"\n✓ Per-Class Metrics:")
        precision_per_class = precision_score(self.y_test, self.y_pred, average=None)
        recall_per_class = recall_score(self.y_test, self.y_pred, average=None)
        f1_per_class = f1_score(self.y_test, self.y_pred, average=None)
        
        for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
            print(f"  Class {i}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}")
            self.metrics[f'precision_class_{i}'] = float(p)
            self.metrics[f'recall_class_{i}'] = float(r)
            self.metrics[f'f1_class_{i}'] = float(f)
        
        return self.metrics
    
    def generate_confusion_matrix(self, save_path="outputs/confusion_matrix.png"):
        """Generate and save confusion matrix"""
        print(f"\n{'='*50}")
        print("Generating Confusion Matrix")
        print(f"{'='*50}\n")
        
        if not self.evaluate_config.get('generate_confusion_matrix', True):
            print("⚠ Confusion matrix generation disabled")
            return None
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Create directory
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=sorted(self.y_test.unique()),
            yticklabels=sorted(self.y_test.unique())
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"✓ Confusion matrix saved: {save_path}")
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Store in metrics
        self.metrics['confusion_matrix'] = cm.tolist()
        
        return cm
    
    def generate_classification_report(self):
        """Generate detailed classification report"""
        print(f"\n{'='*50}")
        print("Classification Report")
        print(f"{'='*50}\n")
        
        report = classification_report(
            self.y_test, 
            self.y_pred,
            target_names=[f"Class {i}" for i in sorted(self.y_test.unique())]
        )
        
        print(report)
        
        # Store as dict for MLflow
        report_dict = classification_report(
            self.y_test, 
            self.y_pred,
            target_names=[f"Class {i}" for i in sorted(self.y_test.unique())],
            output_dict=True
        )
        
        return report_dict
    
    def save_metrics(self, output_path="metrics.json"):
        """Save metrics to JSON file"""
        print(f"\n{'='*50}")
        print("Saving Metrics")
        print(f"{'='*50}\n")
        
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        print(f"✓ Metrics saved: {output_path}")
        
        return output_path
    
    def log_to_mlflow(self):
        """Log evaluation metrics to MLflow"""
        print(f"\n{'='*50}")
        print("Logging to MLflow")
        print(f"{'='*50}\n")
        
        # Get latest run or create new one
        try:
            # Try to use the last active run from training
            with mlflow.start_run():
                # Log test metrics
                mlflow.log_metrics({
                    "test_accuracy": self.metrics['accuracy'],
                    "test_precision": self.metrics['precision'],
                    "test_recall": self.metrics['recall'],
                    "test_f1_score": self.metrics['f1_score']
                })
                print("✓ Logged test metrics to MLflow")
                
                # Log confusion matrix if exists
                if os.path.exists("outputs/confusion_matrix.png"):
                    mlflow.log_artifact("outputs/confusion_matrix.png", "evaluation")
                    print("✓ Logged confusion matrix")
                
                # Log metrics.json
                if os.path.exists("metrics.json"):
                    mlflow.log_artifact("metrics.json", "evaluation")
                    print("✓ Logged metrics.json")
                
                run_id = mlflow.active_run().info.run_id
                print(f"✓ MLflow Run ID: {run_id}")
                
                return run_id
        except Exception as e:
            print(f"⚠ Could not log to existing run: {e}")
            print("  Metrics are still saved locally in metrics.json")
    
    def run_evaluation(self, model_path="models/model.pkl", test_path="data/processed/test.csv"):
        """Run complete evaluation pipeline"""
        print(f"\n{'='*60}")
        print("STARTING MODEL EVALUATION PIPELINE")
        print(f"{'='*60}\n")
        
        # Setup MLflow
        self.setup_mlflow()
        
        # Load model
        self.load_model(model_path)
        
        # Load test data
        self.load_test_data(test_path)
        
        # Make predictions
        self.make_predictions()
        
        # Calculate metrics
        self.calculate_metrics()
        
        # Generate confusion matrix
        self.generate_confusion_matrix()
        
        # Classification report
        self.generate_classification_report()
        
        # Save metrics
        self.save_metrics()
        
        # Log to MLflow
        self.log_to_mlflow()
        
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE!")
        print(f"{'='*60}\n")
        
        # Summary
        print("Summary:")
        print(f"  • Test samples: {len(self.X_test)}")
        print(f"  • Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"  • Precision: {self.metrics['precision']:.4f}")
        print(f"  • Recall: {self.metrics['recall']:.4f}")
        print(f"  • F1-Score: {self.metrics['f1_score']:.4f}")
        print(f"  • Metrics saved: metrics.json")
        print()


def main():
    """Main evaluation function"""
    # Initialize evaluator
    evaluator = ModelEvaluator(params_path="params.yaml")
    
    # Run evaluation pipeline
    evaluator.run_evaluation(
        model_path="models/model.pkl",
        test_path="data/processed/test.csv"
    )
    
    print("✓ Model evaluation completed successfully!")


if __name__ == "__main__":
    main()