"""
Model Registry Module
Manages model registration, versioning, and promotion workflow
"""

import os
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.mlflow_utils import MLflowManager, ModelRegistry


class ModelRegistryWorkflow:
    """Automated workflow for model registry operations"""
    
    def __init__(self, params_path="params.yaml"):
        """
        Initialize registry workflow
        
        Args:
            params_path: Path to parameters file
        """
        self.params = self._load_params(params_path)
        self.mlflow_config = self.params.get('mlflow', {})
        self.manager = MLflowManager()
        self.registry = ModelRegistry()
        
        print(f"✓ Model Registry Workflow initialized")
    
    def _load_params(self, params_path):
        """Load parameters from YAML file"""
        with open(params_path, 'r') as f:
            return yaml.safe_load(f)
    
    def find_best_model(self, metric="test_accuracy", min_threshold=0.8):
        """
        Find best model that meets minimum threshold
        
        Args:
            metric: Metric to optimize
            min_threshold: Minimum acceptable value
        
        Returns:
            Best run info or None
        """
        print(f"\n{'='*60}")
        print("Finding Best Model")
        print(f"{'='*60}\n")
        
        print(f"Criteria:")
        print(f"  • Metric: {metric}")
        print(f"  • Minimum threshold: {min_threshold}")
        
        # Get all runs
        runs_df = self.manager.list_all_runs(max_results=100)
        
        if runs_df.empty:
            print("\n⚠ No runs found")
            return None
        
        # Filter by metric existence
        if metric not in runs_df.columns:
            print(f"\n⚠ Metric '{metric}' not found in runs")
            return None
        
        # Filter by threshold
        qualified_runs = runs_df[runs_df[metric] >= min_threshold]
        
        if qualified_runs.empty:
            print(f"\n⚠ No runs meet threshold {min_threshold}")
            print(f"  • Best available: {runs_df[metric].max():.4f}")
            return None
        
        # Get best
        best_run = qualified_runs.loc[qualified_runs[metric].idxmax()]
        
        print(f"\n✓ Found best model:")
        print(f"  • Run ID: {best_run['run_id']}")
        print(f"  • Run Name: {best_run['run_name']}")
        print(f"  • {metric}: {best_run[metric]:.4f}")
        print(f"  • Qualified runs: {len(qualified_runs)}/{len(runs_df)}")
        
        return best_run
    
    def register_best_model(self, model_name=None, metric="test_accuracy", min_threshold=0.8):
        """
        Find and register best model
        
        Args:
            model_name: Name for registered model
            metric: Metric to optimize
            min_threshold: Minimum acceptable value
        
        Returns:
            Model version info
        """
        print(f"\n{'='*60}")
        print("Registering Best Model")
        print(f"{'='*60}\n")
        
        # Find best model
        best_run = self.find_best_model(metric, min_threshold)
        
        if best_run is None:
            print("\n⚠ Cannot register: No suitable model found")
            return None
        
        # Use default model name if not provided
        if model_name is None:
            model_type = self.params.get('train', {}).get('model_type', 'model')
            model_name = f"{model_type}_classifier"
        
        # Register model
        print(f"\nRegistering model: {model_name}")
        model_version = self.registry.register_model(
            run_id=best_run['run_id'],
            model_name=model_name
        )
        
        # Add description
        self.registry.client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=f"Registered on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                       f"with {metric}={best_run[metric]:.4f}"
        )
        
        print(f"\n✓ Model registered successfully")
        print(f"  • Model: {model_name}")
        print(f"  • Version: {model_version.version}")
        print(f"  • Performance: {metric}={best_run[metric]:.4f}")
        
        return model_version
    
    def promote_to_staging(self, model_name, version=None):
        """
        Promote model to staging
        
        Args:
            model_name: Name of registered model
            version: Version to promote (latest if None)
        
        Returns:
            Updated model version
        """
        print(f"\n{'='*60}")
        print("Promoting to Staging")
        print(f"{'='*60}\n")
        
        # Get latest version if not specified
        if version is None:
            versions = self.registry.get_model_versions(model_name)
            if not versions:
                print("⚠ No versions found")
                return None
            version = max([v.version for v in versions])
        
        # Transition to staging
        updated_version = self.registry.transition_model_stage(
            model_name=model_name,
            version=version,
            stage="Staging"
        )
        
        print(f"\n✓ Model in staging - ready for testing")
        
        return updated_version
    
    def promote_to_production(self, model_name, version=None, auto_archive=True):
        """
        Promote model to production
        
        Args:
            model_name: Name of registered model
            version: Version to promote (latest staging if None)
            auto_archive: Archive current production version
        
        Returns:
            Updated model version
        """
        print(f"\n{'='*60}")
        print("Promoting to Production")
        print(f"{'='*60}\n")
        
        # Get versions
        versions = self.registry.get_model_versions(model_name)
        if not versions:
            print("⚠ No versions found")
            return None
        
        # Auto-archive current production version
        if auto_archive:
            for v in versions:
                if v.current_stage == "Production":
                    print(f"Archiving current production version {v.version}")
                    self.registry.transition_model_stage(
                        model_name=model_name,
                        version=v.version,
                        stage="Archived"
                    )
        
        # Get version to promote
        if version is None:
            # Find latest staging version
            staging_versions = [v for v in versions if v.current_stage == "Staging"]
            if not staging_versions:
                print("⚠ No staging versions found")
                return None
            version = max([v.version for v in staging_versions])
        
        # Transition to production
        updated_version = self.registry.transition_model_stage(
            model_name=model_name,
            version=version,
            stage="Production"
        )
        
        print(f"\n✓ Model deployed to production!")
        print(f"  • Ready for serving")
        
        return updated_version
    
    def full_deployment_workflow(self, model_name=None, metric="test_accuracy", 
                                 min_threshold=0.8, skip_staging=False):
        """
        Complete deployment workflow: register → staging → production
        
        Args:
            model_name: Name for registered model
            metric: Metric to optimize
            min_threshold: Minimum acceptable value
            skip_staging: Skip staging and go directly to production
        
        Returns:
            Final model version info
        """
        print(f"\n{'='*60}")
        print("FULL DEPLOYMENT WORKFLOW")
        print(f"{'='*60}\n")
        
        print(f"Workflow: Register → {'Production' if skip_staging else 'Staging → Production'}")
        
        # Step 1: Register best model
        model_version = self.register_best_model(model_name, metric, min_threshold)
        
        if model_version is None:
            print("\n⚠ Workflow stopped: Model registration failed")
            return None
        
        final_model_name = model_version.name
        final_version = model_version.version
        
        # Step 2: Promote to staging (unless skipped)
        if not skip_staging:
            self.promote_to_staging(final_model_name, final_version)
            
            # Optional: Add validation step here
            print(f"\n⚠ Manual validation recommended before production")
            print(f"  • Test model in staging environment")
            print(f"  • Run: python -c \"from src.models.registry import *; workflow = ModelRegistryWorkflow(); workflow.promote_to_production('{final_model_name}')\"")
            
            return model_version
        
        # Step 3: Promote to production
        production_version = self.promote_to_production(final_model_name, final_version)
        
        print(f"\n{'='*60}")
        print("DEPLOYMENT COMPLETE!")
        print(f"{'='*60}\n")
        
        print(f"Summary:")
        print(f"  • Model: {final_model_name}")
        print(f"  • Version: {final_version}")
        print(f"  • Stage: Production")
        print(f"  • Performance: {metric}={min_threshold}")
        
        return production_version
    
    def compare_model_versions(self, model_name):
        """
        Compare all versions of a model
        
        Args:
            model_name: Name of registered model
        
        Returns:
            Comparison DataFrame
        """
        print(f"\n{'='*60}")
        print(f"Comparing Versions: {model_name}")
        print(f"{'='*60}\n")
        
        versions = self.registry.get_model_versions(model_name)
        
        if not versions:
            print("⚠ No versions found")
            return None
        
        # Get run info for each version
        version_data = []
        for v in versions:
            run = self.manager.client.get_run(v.run_id)
            version_info = {
                'version': v.version,
                'stage': v.current_stage,
                'created': datetime.fromtimestamp(v.creation_timestamp / 1000),
            }
            version_info.update(run.data.metrics)
            version_data.append(version_info)
        
        import pandas as pd
        df = pd.DataFrame(version_data)
        
        print(df.to_string(index=False))
        
        return df
    
    def rollback_to_version(self, model_name, version):
        """
        Rollback to previous model version
        
        Args:
            model_name: Name of registered model
            version: Version to rollback to
        
        Returns:
            Updated model version
        """
        print(f"\n{'='*60}")
        print("Rolling Back Model")
        print(f"{'='*60}\n")
        
        print(f"Rolling back {model_name} to version {version}")
        
        # Promote specified version to production
        updated_version = self.promote_to_production(
            model_name=model_name,
            version=version,
            auto_archive=True
        )
        
        print(f"\n✓ Rollback complete")
        
        return updated_version


def main():
    """Main registry workflow demonstration"""
    print("="*60)
    print("Model Registry Workflow Demo")
    print("="*60)
    
    # Initialize workflow
    workflow = ModelRegistryWorkflow()
    
    # Option 1: Register best model
    print("\n" + "="*60)
    print("Option 1: Register Best Model")
    print("="*60)
    model_version = workflow.register_best_model(
        model_name="classification_model",
        metric="test_accuracy",
        min_threshold=0.9
    )
    
    if model_version:
        # Option 2: Promote to staging
        print("\n" + "="*60)
        print("Option 2: Promote to Staging")
        print("="*60)
        workflow.promote_to_staging(
            model_name=model_version.name,
            version=model_version.version
        )
        
        print("\n✓ Model is now in staging!")
        print("  Test the model, then promote to production when ready.")
    
    # List registered models
    print("\n" + "="*60)
    print("Registered Models:")
    print("="*60)
    workflow.registry.list_registered_models()


if __name__ == "__main__":
    main()