"""
Data Validation Module
Validates raw data for schema, missing values, and data types
"""

import pandas as pd
import json
import yaml
import os
from pathlib import Path


class DataValidator:
    """Validates data quality and schema"""
    
    def __init__(self, data_path, params_path="params.yaml"):
        """
        Initialize validator
        
        Args:
            data_path: Path to raw data file
            params_path: Path to parameters file
        """
        self.data_path = data_path
        self.params = self._load_params(params_path)
        self.validation_config = self.params.get('validate', {})
        self.validation_report = {}
        
    def _load_params(self, params_path):
        """Load parameters from YAML file"""
        with open(params_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self):
        """Load data from CSV file"""
        try:
            self.data = pd.read_csv(self.data_path)
            self.validation_report['data_loaded'] = True
            self.validation_report['total_rows'] = len(self.data)
            self.validation_report['total_columns'] = len(self.data.columns)
            print(f"✓ Data loaded: {len(self.data)} rows, {len(self.data.columns)} columns")
            return True
        except Exception as e:
            self.validation_report['data_loaded'] = False
            self.validation_report['error'] = str(e)
            print(f"✗ Failed to load data: {e}")
            return False
    
    def validate_schema(self):
        """Validate that required columns exist"""
        required_columns = self.validation_config.get('required_columns', [])
        
        if not required_columns:
            print("⚠ No required columns specified in params.yaml")
            self.validation_report['schema_validation'] = 'skipped'
            return True
        
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            self.validation_report['schema_validation'] = 'failed'
            self.validation_report['missing_columns'] = missing_columns
            print(f"✗ Schema validation failed: Missing columns {missing_columns}")
            return False
        
        self.validation_report['schema_validation'] = 'passed'
        self.validation_report['columns_present'] = list(self.data.columns)
        print(f"✓ Schema validation passed: All required columns present")
        return True
    
    def check_missing_values(self):
        """Check for missing values"""
        missing_threshold = self.validation_config.get('missing_threshold', 0.1)
        
        missing_counts = self.data.isnull().sum()
        missing_percentages = (missing_counts / len(self.data)) * 100
        
        # Find columns exceeding threshold
        problematic_columns = missing_percentages[missing_percentages > (missing_threshold * 100)]
        
        self.validation_report['missing_values'] = {
            'total_missing': int(missing_counts.sum()),
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentages': {k: f"{v:.2f}%" for k, v in missing_percentages[missing_percentages > 0].items()}
        }
        
        if len(problematic_columns) > 0:
            self.validation_report['missing_values']['status'] = 'warning'
            self.validation_report['missing_values']['columns_exceeding_threshold'] = problematic_columns.to_dict()
            print(f"⚠ Missing values warning: {len(problematic_columns)} columns exceed {missing_threshold*100}% threshold")
        else:
            self.validation_report['missing_values']['status'] = 'passed'
            print(f"✓ Missing values check passed")
        
        return True
    
    def check_duplicates(self):
        """Check for duplicate rows"""
        if not self.validation_config.get('duplicate_check', False):
            self.validation_report['duplicate_check'] = 'skipped'
            return True
        
        duplicates = self.data.duplicated().sum()
        
        self.validation_report['duplicate_check'] = {
            'total_duplicates': int(duplicates),
            'duplicate_percentage': f"{(duplicates / len(self.data)) * 100:.2f}%"
        }
        
        if duplicates > 0:
            self.validation_report['duplicate_check']['status'] = 'warning'
            print(f"⚠ Found {duplicates} duplicate rows ({(duplicates / len(self.data)) * 100:.2f}%)")
        else:
            self.validation_report['duplicate_check']['status'] = 'passed'
            print(f"✓ No duplicate rows found")
        
        return True
    
    def validate_data_types(self):
        """Validate data types"""
        data_types = self.data.dtypes.astype(str).to_dict()
        
        self.validation_report['data_types'] = data_types
        print(f"✓ Data types validated")
        
        return True
    
    def generate_statistics(self):
        """Generate basic statistics"""
        stats = {
            'numerical_summary': self.data.describe().to_dict(),
            'categorical_columns': list(self.data.select_dtypes(include=['object']).columns),
            'numerical_columns': list(self.data.select_dtypes(include=['number']).columns)
        }
        
        self.validation_report['statistics'] = stats
        print(f"✓ Statistics generated")
        
        return True
    
    def run_validation(self):
        """Run all validation checks"""
        print("\n" + "="*50)
        print("Starting Data Validation")
        print("="*50 + "\n")
        
        # Load data
        if not self.load_data():
            return False
        
        # Run validations
        self.validate_schema()
        self.check_missing_values()
        self.check_duplicates()
        self.validate_data_types()
        self.generate_statistics()
        
        # Overall status
        self.validation_report['overall_status'] = 'completed'
        
        print("\n" + "="*50)
        print("Validation Complete")
        print("="*50 + "\n")
        
        return True
    
    def save_report(self, output_path="data/validation_report.json"):
        """Save validation report to JSON"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.validation_report, f, indent=4)
        
        print(f"✓ Validation report saved to: {output_path}")


def main():
    """Main validation function"""
    # Define paths
    data_path = "data/raw/dataset.csv"
    params_path = "params.yaml"
    output_path = "data/validation_report.json"
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"✗ Error: Data file not found at {data_path}")
        return
    
    # Create validator and run
    validator = DataValidator(data_path, params_path)
    validator.run_validation()
    validator.save_report(output_path)
    
    print("\n✓ Data validation completed successfully!")


if __name__ == "__main__":
    main()