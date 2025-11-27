"""
Unit Tests for Data Processing
Tests for validation and preprocessing functions
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.validate import DataValidator
from src.data.preprocess import DataPreprocessor


class TestDataValidator:
    """Test suite for DataValidator"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing"""
        data = pd.DataFrame({
            'feature1': [5.1, 4.9, 4.7],
            'feature2': [3.5, 3.0, 3.2],
            'feature3': [1.4, 1.4, 1.3],
            'feature4': [0.2, 0.2, 0.2],
            'feature5': [2.3, 2.1, 1.9],
            'target': [0, 0, 0]
        })
        return data
    
    @pytest.fixture
    def validator(self, tmp_path, sample_data):
        """Create validator with sample data"""
        # Save sample data
        data_path = tmp_path / "test_data.csv"
        sample_data.to_csv(data_path, index=False)
        
        # Create params.yaml
        params_path = tmp_path / "params.yaml"
        with open(params_path, 'w') as f:
            f.write("""
validate:
  required_columns:
    - feature1
    - feature2
    - target
  missing_threshold: 0.1
  duplicate_check: true
""")
        
        return DataValidator(str(data_path), str(params_path))
    
    def test_load_data(self, validator):
        """Test data loading"""
        result = validator.load_data()
        
        assert result is True
        assert validator.data is not None
        assert len(validator.data) == 3
        assert len(validator.data.columns) == 6
    
    def test_validate_schema_success(self, validator):
        """Test schema validation with correct columns"""
        validator.load_data()
        result = validator.validate_schema()
        
        assert result is True
        assert validator.validation_report['schema_validation'] == 'passed'
    
    def test_validate_schema_missing_column(self, validator, tmp_path):
        """Test schema validation with missing column"""
        # Create data without required column
        data = pd.DataFrame({
            'feature1': [5.1, 4.9],
            'feature2': [3.5, 3.0],
            # Missing 'target' column
        })
        
        data_path = tmp_path / "invalid_data.csv"
        data.to_csv(data_path, index=False)
        
        params_path = tmp_path / "params.yaml"
        validator_invalid = DataValidator(str(data_path), str(params_path))
        validator_invalid.load_data()
        
        result = validator_invalid.validate_schema()
        
        assert result is False
        assert validator_invalid.validation_report['schema_validation'] == 'failed'
    
    def test_check_missing_values_none(self, validator):
        """Test missing values check with clean data"""
        validator.load_data()
        validator.check_missing_values()
        
        assert validator.validation_report['missing_values']['total_missing'] == 0
        assert validator.validation_report['missing_values']['status'] == 'passed'
    
    def test_check_missing_values_present(self, validator, tmp_path):
        """Test missing values check with missing data"""
        # Create data with missing values
        data = pd.DataFrame({
            'feature1': [5.1, np.nan, 4.7],
            'feature2': [3.5, 3.0, np.nan],
            'feature3': [1.4, 1.4, 1.3],
            'feature4': [0.2, 0.2, 0.2],
            'feature5': [2.3, 2.1, 1.9],
            'target': [0, 0, 0]
        })
        
        data_path = tmp_path / "missing_data.csv"
        data.to_csv(data_path, index=False)
        
        params_path = tmp_path / "params.yaml"
        validator_missing = DataValidator(str(data_path), str(params_path))
        validator_missing.load_data()
        validator_missing.check_missing_values()
        
        assert validator_missing.validation_report['missing_values']['total_missing'] == 2
    
    def test_check_duplicates_none(self, validator):
        """Test duplicate check with unique data"""
        validator.load_data()
        validator.check_duplicates()
        
        assert validator.validation_report['duplicate_check']['total_duplicates'] == 0
        assert validator.validation_report['duplicate_check']['status'] == 'passed'
    
    def test_check_duplicates_present(self, validator, tmp_path):
        """Test duplicate check with duplicate rows"""
        # Create data with duplicates
        data = pd.DataFrame({
            'feature1': [5.1, 5.1, 4.7],  # First two rows identical
            'feature2': [3.5, 3.5, 3.2],
            'feature3': [1.4, 1.4, 1.3],
            'feature4': [0.2, 0.2, 0.2],
            'feature5': [2.3, 2.3, 1.9],
            'target': [0, 0, 0]
        })
        
        data_path = tmp_path / "duplicate_data.csv"
        data.to_csv(data_path, index=False)
        
        params_path = tmp_path / "params.yaml"
        validator_dup = DataValidator(str(data_path), str(params_path))
        validator_dup.load_data()
        validator_dup.check_duplicates()
        
        assert validator_dup.validation_report['duplicate_check']['total_duplicates'] == 1


class TestDataPreprocessor:
    """Test suite for DataPreprocessor"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset"""
        data = pd.DataFrame({
            'feature1': [5.1, 4.9, 4.7, 6.2, 5.9, 6.1],
            'feature2': [3.5, 3.0, 3.2, 2.9, 3.0, 2.8],
            'feature3': [1.4, 1.4, 1.3, 4.3, 5.1, 5.6],
            'feature4': [0.2, 0.2, 0.2, 1.3, 1.8, 1.4],
            'feature5': [2.3, 2.1, 1.9, 3.7, 4.2, 4.5],
            'target': [0, 0, 0, 1, 2, 2]
        })
        return data
    
    @pytest.fixture
    def preprocessor(self, tmp_path, sample_data):
        """Create preprocessor with sample data"""
        # Save sample data
        data_path = tmp_path / "raw" / "dataset.csv"
        os.makedirs(data_path.parent, exist_ok=True)
        sample_data.to_csv(data_path, index=False)
        
        # Create params.yaml
        params_path = tmp_path / "params.yaml"
        with open(params_path, 'w') as f:
            f.write("""
preprocess:
  test_size: 0.3
  random_state: 42
  validation_split: 0.1
""")
        
        return DataPreprocessor(str(params_path))
    
    def test_load_raw_data(self, preprocessor, tmp_path):
        """Test loading raw data"""
        data_path = tmp_path / "raw" / "dataset.csv"
        data = preprocessor.load_raw_data(str(data_path))
        
        assert data is not None
        assert len(data) == 6
        assert 'target' in data.columns
    
    def test_handle_missing_values_clean(self, preprocessor, tmp_path):
        """Test handling missing values with clean data"""
        data_path = tmp_path / "raw" / "dataset.csv"
        preprocessor.load_raw_data(str(data_path))
        
        initial_nulls = preprocessor.data.isnull().sum().sum()
        preprocessor.handle_missing_values()
        final_nulls = preprocessor.data.isnull().sum().sum()
        
        assert initial_nulls == 0
        assert final_nulls == 0
    
    def test_remove_duplicates(self, preprocessor, tmp_path):
        """Test removing duplicate rows"""
        data_path = tmp_path / "raw" / "dataset.csv"
        preprocessor.load_raw_data(str(data_path))
        
        initial_rows = len(preprocessor.data)
        preprocessor.remove_duplicates()
        final_rows = len(preprocessor.data)
        
        assert final_rows <= initial_rows
    
    def test_feature_engineering(self, preprocessor, tmp_path):
        """Test feature engineering creates new features"""
        data_path = tmp_path / "raw" / "dataset.csv"
        preprocessor.load_raw_data(str(data_path))
        
        initial_columns = len(preprocessor.data.columns)
        preprocessor.feature_engineering()
        final_columns = len(preprocessor.data.columns)
        
        assert final_columns > initial_columns
        assert 'feature_ratio' in preprocessor.data.columns
        assert 'feature_sum' in preprocessor.data.columns
    
    def test_split_features_target(self, preprocessor, tmp_path):
        """Test splitting features and target"""
        data_path = tmp_path / "raw" / "dataset.csv"
        preprocessor.load_raw_data(str(data_path))
        
        X, y = preprocessor.split_features_target()
        
        assert 'target' not in X.columns
        assert len(X) == len(y)
        assert len(X.columns) == 5  # Original features only
    
    def test_train_test_split_ratio(self, preprocessor, tmp_path):
        """Test train-test split maintains correct ratio"""
        data_path = tmp_path / "raw" / "dataset.csv"
        preprocessor.load_raw_data(str(data_path))
        preprocessor.split_features_target()
        
        X_train, X_test, y_train, y_test = preprocessor.train_test_split_data()
        
        total_samples = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total_samples
        
        assert abs(test_ratio - 0.3) < 0.1  # Within 10% of target
    
    def test_scale_features(self, preprocessor, tmp_path):
        """Test feature scaling"""
        data_path = tmp_path / "raw" / "dataset.csv"
        preprocessor.load_raw_data(str(data_path))
        preprocessor.split_features_target()
        preprocessor.train_test_split_data()
        
        X_train_scaled, X_test_scaled = preprocessor.scale_features()
        
        # Check mean is approximately 0 and std is approximately 1
        assert abs(X_train_scaled.mean().mean()) < 1.0
        assert abs(X_train_scaled.std().mean() - 1.0) < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])