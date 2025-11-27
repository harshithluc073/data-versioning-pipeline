"""
MLflow Utilities Module
Helper functions for MLflow experiment tracking and model registry
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from datetime import datetime
import pandas as pd
import yaml


class MLflowManager:
    """Manage MLflow experiments and model registry"""
    
    def __init__(self, tracking_uri="./mlruns", experiment_name="data-versioning-pipeline"):
        """
        Initialize MLflow manager
        
        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: Name of experiment
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.client = None
        self.setup()
    
    def setup(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.client = MlflowClient()
        
        print(f"✓ MLflow Manager initialized")
        print(f"  • Tracking URI: {self.tracking_uri}")
        print(f"  • Experiment: {self.experiment_name}")
    
    def get_experiment_id(self):
        """Get current experiment ID"""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment:
            return experiment.experiment_id
        return None
    
    def list_all_runs(self, max_results=10):
        """
        List all runs in current experiment
        
        Args:
            max_results: Maximum number of runs to return
        
        Returns:
            DataFrame with run information
        """
        experiment_id = self.get_experiment_id()
        
        if not experiment_id:
            print("⚠ No experiment found")
            return pd.DataFrame()
        
        runs = self.client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["start_time DESC"],
            max_results=max_results
        )
        
        if not runs:
            print("⚠ No runs found")
            return pd.DataFrame()
        
        # Extract run data
        run_data = []
        for run in runs:
            run_info = {
                'run_id': run.info.run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
                'status': run.info.status,
                'start_time': datetime.fromtimestamp(run.info.start_time / 1000),
                'duration': (run.info.end_time - run.info.start_time) / 1000 if run.info.end_time else 0
            }
            
            # Add metrics
            for key, value in run.data.metrics.items():
                run_info[key] = value
            
            # Add key parameters
            for key, value in run.data.params.items():
                run_info[f"param_{key}"] = value
            
            run_data.append(run_info)
        
        df = pd.DataFrame(run_data)
        
        print(f"\n✓ Found {len(df)} runs")
        return df
    
    def get_best_run(self, metric="test_accuracy", ascending=False):
        """
        Get best run based on metric
        
        Args:
            metric: Metric to optimize
            ascending: Sort order (False for maximize)
        
        Returns:
            Best run info
        """
        runs_df = self.list_all_runs(max_results=100)
        
        if runs_df.empty:
            print("⚠ No runs available")
            return None
        
        if metric not in runs_df.columns:
            print(f"⚠ Metric '{metric}' not found in runs")
            available_metrics = [col for col in runs_df.columns if not col.startswith('param_')]
            print(f"  Available metrics: {available_metrics}")
            return None
        
        # Sort and get best
        sorted_df = runs_df.sort_values(metric, ascending=ascending)
        best_run = sorted_df.iloc[0]
        
        print(f"\n✓ Best run by {metric}:")
        print(f"  • Run ID: {best_run['run_id']}")
        print(f"  • Run Name: {best_run['run_name']}")
        print(f"  • {metric}: {best_run[metric]:.4f}")
        
        return best_run
    
    def compare_runs(self, run_ids=None, metrics=None):
        """
        Compare multiple runs
        
        Args:
            run_ids: List of run IDs to compare (None for latest runs)
            metrics: List of metrics to compare (None for all)
        
        Returns:
            Comparison DataFrame
        """
        if run_ids is None:
            # Get latest runs
            runs_df = self.list_all_runs(max_results=5)
            if runs_df.empty:
                return pd.DataFrame()
        else:
            # Get specific runs
            run_data = []
            for run_id in run_ids:
                run = self.client.get_run(run_id)
                run_info = {
                    'run_id': run_id,
                    'run_name': run.data.tags.get('mlflow.runName', 'N/A')
                }
                run_info.update(run.data.metrics)
                run_info.update({f"param_{k}": v for k, v in run.data.params.items()})
                run_data.append(run_info)
            
            runs_df = pd.DataFrame(run_data)
        
        if metrics:
            # Select specific metrics
            cols = ['run_id', 'run_name'] + [m for m in metrics if m in runs_df.columns]
            runs_df = runs_df[cols]
        
        print(f"\n✓ Comparing {len(runs_df)} runs")
        print(runs_df.to_string(index=False))
        
        return runs_df
    
    def get_run_artifacts(self, run_id):
        """
        List artifacts for a run
        
        Args:
            run_id: Run ID
        
        Returns:
            List of artifact paths
        """
        artifacts = self.client.list_artifacts(run_id)
        
        artifact_paths = [artifact.path for artifact in artifacts]
        
        print(f"\n✓ Artifacts for run {run_id}:")
        for path in artifact_paths:
            print(f"  • {path}")
        
        return artifact_paths
    
    def load_model_from_run(self, run_id, model_path="model"):
        """
        Load model from specific run
        
        Args:
            run_id: Run ID
            model_path: Path to model within run artifacts
        
        Returns:
            Loaded model
        """
        model_uri = f"runs:/{run_id}/{model_path}"
        model = mlflow.sklearn.load_model(model_uri)
        
        print(f"\n✓ Model loaded from run: {run_id}")
        print(f"  • Model type: {type(model).__name__}")
        
        return model
    
    def delete_run(self, run_id):
        """
        Delete a run
        
        Args:
            run_id: Run ID to delete
        """
        self.client.delete_run(run_id)
        print(f"✓ Deleted run: {run_id}")
    
    def create_experiment_summary(self, output_path="experiment_summary.csv"):
        """
        Create summary of all experiments
        
        Args:
            output_path: Path to save summary
        
        Returns:
            Summary DataFrame
        """
        runs_df = self.list_all_runs(max_results=100)
        
        if runs_df.empty:
            print("⚠ No runs to summarize")
            return pd.DataFrame()
        
        # Save to CSV
        runs_df.to_csv(output_path, index=False)
        
        print(f"\n✓ Experiment summary saved: {output_path}")
        print(f"  • Total runs: {len(runs_df)}")
        
        # Print summary statistics
        metric_cols = [col for col in runs_df.columns 
                      if col not in ['run_id', 'run_name', 'status', 'start_time', 'duration']
                      and not col.startswith('param_')]
        
        if metric_cols:
            print(f"\n  Metric Summary:")
            for col in metric_cols:
                if runs_df[col].notna().any():
                    print(f"    • {col}: mean={runs_df[col].mean():.4f}, "
                          f"min={runs_df[col].min():.4f}, max={runs_df[col].max():.4f}")
        
        return runs_df


class ModelRegistry:
    """Manage model registry operations"""
    
    def __init__(self, tracking_uri="./mlruns"):
        """
        Initialize model registry
        
        Args:
            tracking_uri: MLflow tracking URI
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        
        print(f"✓ Model Registry initialized")
        print(f"  • Tracking URI: {tracking_uri}")
    
    def register_model(self, run_id, model_name, model_path="model"):
        """
        Register model from run
        
        Args:
            run_id: Run ID containing model
            model_name: Name for registered model
            model_path: Path to model in run artifacts
        
        Returns:
            Model version
        """
        model_uri = f"runs:/{run_id}/{model_path}"
        
        model_version = mlflow.register_model(model_uri, model_name)
        
        print(f"\n✓ Model registered:")
        print(f"  • Name: {model_name}")
        print(f"  • Version: {model_version.version}")
        print(f"  • Run ID: {run_id}")
        
        return model_version
    
    def list_registered_models(self):
        """
        List all registered models
        
        Returns:
            List of registered models
        """
        models = self.client.search_registered_models()
        
        if not models:
            print("⚠ No registered models found")
            return []
        
        print(f"\n✓ Found {len(models)} registered models:")
        
        for model in models:
            print(f"\n  • {model.name}")
            print(f"    Latest versions:")
            for version in model.latest_versions:
                print(f"      - Version {version.version}: {version.current_stage}")
        
        return models
    
    def get_model_versions(self, model_name):
        """
        Get all versions of a model
        
        Args:
            model_name: Name of registered model
        
        Returns:
            List of model versions
        """
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            print(f"\n✓ Versions for model '{model_name}':")
            for version in versions:
                print(f"  • Version {version.version}: {version.current_stage}")
            
            return versions
        except Exception as e:
            print(f"⚠ Error getting versions: {e}")
            return []
    
    def transition_model_stage(self, model_name, version, stage):
        """
        Transition model to different stage
        
        Args:
            model_name: Name of registered model
            version: Model version number
            stage: Target stage (Staging, Production, Archived)
        
        Returns:
            Updated model version
        """
        valid_stages = ["Staging", "Production", "Archived"]
        
        if stage not in valid_stages:
            print(f"⚠ Invalid stage. Must be one of: {valid_stages}")
            return None
        
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        
        print(f"\n✓ Model transitioned:")
        print(f"  • Model: {model_name}")
        print(f"  • Version: {version}")
        print(f"  • New Stage: {stage}")
        
        return self.client.get_model_version(model_name, version)
    
    def load_production_model(self, model_name):
        """
        Load production model
        
        Args:
            model_name: Name of registered model
        
        Returns:
            Production model
        """
        model_uri = f"models:/{model_name}/Production"
        
        try:
            model = mlflow.sklearn.load_model(model_uri)
            print(f"\n✓ Loaded production model: {model_name}")
            print(f"  • Model type: {type(model).__name__}")
            return model
        except Exception as e:
            print(f"⚠ Error loading production model: {e}")
            print(f"  • Make sure a model is in Production stage")
            return None
    
    def delete_model_version(self, model_name, version):
        """
        Delete a model version
        
        Args:
            model_name: Name of registered model
            version: Version to delete
        """
        self.client.delete_model_version(model_name, version)
        print(f"✓ Deleted model version: {model_name} v{version}")


def main():
    """Demonstrate MLflow utilities"""
    print("="*60)
    print("MLflow Utilities Demo")
    print("="*60)
    
    # Initialize manager
    manager = MLflowManager()
    
    # List all runs
    print("\n" + "="*60)
    print("All Runs:")
    print("="*60)
    runs_df = manager.list_all_runs()
    if not runs_df.empty:
        print(runs_df[['run_name', 'train_accuracy', 'test_accuracy']].to_string(index=False))
    
    # Get best run
    print("\n" + "="*60)
    print("Best Run:")
    print("="*60)
    best_run = manager.get_best_run(metric="test_accuracy")
    
    # Create summary
    print("\n" + "="*60)
    print("Creating Experiment Summary:")
    print("="*60)
    manager.create_experiment_summary()
    
    # Model registry
    print("\n" + "="*60)
    print("Model Registry:")
    print("="*60)
    registry = ModelRegistry()
    registry.list_registered_models()


if __name__ == "__main__":
    main()