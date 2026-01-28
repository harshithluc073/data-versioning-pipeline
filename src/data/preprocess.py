"""
Data Preprocessing Module
Handles data cleaning, feature engineering, and train-test split
"""

import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


class DataPreprocessor:
    """Preprocess raw data for model training"""
    
    def __init__(self, params_path="params.yaml"):
        """
        Initialize preprocessor
        
        Args:
            params_path: Path to parameters file
        """
        self.params = self._load_params(params_path)
        self.preprocess_config = self.params.get('preprocess', {})
        self.scaler = StandardScaler()
        
    def _load_params(self, params_path):
        """Load parameters from YAML file"""
        with open(params_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_raw_data(self, data_path="data/raw/dataset.csv"):
        """Load raw data from CSV"""
        print(f"\n{'='*50}")
        print("Loading Raw Data")
        print(f"{'='*50}\n")
        
        self.data = pd.read_csv(data_path)
        print(f"✓ Loaded data: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        print(f"✓ Columns: {list(self.data.columns)}")
        
        return self.data
    
    def handle_missing_values(self):
        """Handle missing values in dataset"""
        print(f"\n{'='*50}")
        print("Handling Missing Values")
        print(f"{'='*50}\n")
        
        missing_before = self.data.isnull().sum().sum()
        
        if missing_before > 0:
            print(f"⚠ Found {missing_before} missing values")
            
            # Fill numerical columns with median
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns

            # Identify columns with missing values
            has_nans = self.data[numerical_cols].isnull().any()
            cols_with_nans = has_nans[has_nans].index

            if len(cols_with_nans) > 0:
                # Calculate medians for columns with missing values
                medians = self.data[cols_with_nans].median()

                # Fill missing values efficiently
                self.data.fillna(medians, inplace=True)

                # Log filled columns to match original behavior
                for col in cols_with_nans:
                    print(f"  • Filled {col} with median: {medians[col]:.2f}")
            
            # Fill categorical columns with mode
            categorical_cols = self.data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.data[col].isnull().any():
                    mode_value = self.data[col].mode()[0]
                    self.data[col].fillna(mode_value, inplace=True)
                    print(f"  • Filled {col} with mode: {mode_value}")
            
            missing_after = self.data.isnull().sum().sum()
            print(f"\n✓ Missing values handled: {missing_before} → {missing_after}")
        else:
            print("✓ No missing values found")
        
        return self.data
    
    def remove_duplicates(self):
        """Remove duplicate rows"""
        print(f"\n{'='*50}")
        print("Removing Duplicates")
        print(f"{'='*50}\n")
        
        rows_before = len(self.data)
        self.data.drop_duplicates(inplace=True)
        rows_after = len(self.data)
        
        removed = rows_before - rows_after
        
        if removed > 0:
            print(f"⚠ Removed {removed} duplicate rows")
        else:
            print("✓ No duplicate rows found")
        
        print(f"✓ Dataset size: {rows_before} → {rows_after} rows")
        
        return self.data
    
    def feature_engineering(self):
        """Create new features if needed"""
        print(f"\n{'='*50}")
        print("Feature Engineering")
        print(f"{'='*50}\n")
        
        # Example: Create interaction features
        # This is dataset-specific, adjust based on your data
        
        # For this sample dataset, we'll create a simple derived feature
        if 'feature1' in self.data.columns and 'feature2' in self.data.columns:
            self.data['feature_ratio'] = self.data['feature1'] / (self.data['feature2'] + 1e-10)
            print("✓ Created feature: feature_ratio (feature1 / feature2)")
        
        if 'feature3' in self.data.columns and 'feature4' in self.data.columns:
            self.data['feature_sum'] = self.data['feature3'] + self.data['feature4']
            print("✓ Created feature: feature_sum (feature3 + feature4)")
        
        print(f"\n✓ Total features: {self.data.shape[1]}")
        
        return self.data
    
    def split_features_target(self, target_column='target'):
        """Split data into features and target"""
        print(f"\n{'='*50}")
        print("Splitting Features and Target")
        print(f"{'='*50}\n")
        
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        self.X = self.data.drop(columns=[target_column])
        self.y = self.data[target_column]
        
        print(f"✓ Features (X): {self.X.shape}")
        print(f"✓ Target (y): {self.y.shape}")
        print(f"✓ Target classes: {sorted(self.y.unique())}")
        
        return self.X, self.y
    
    def train_test_split_data(self):
        """Split data into train and test sets"""
        print(f"\n{'='*50}")
        print("Train-Test Split")
        print(f"{'='*50}\n")
        
        test_size = self.preprocess_config.get('test_size', 0.2)
        random_state = self.preprocess_config.get('random_state', 42)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y  # Ensure balanced split
        )
        
        print(f"✓ Test size: {test_size * 100}%")
        print(f"✓ Random state: {random_state}")
        print(f"✓ Train set: {self.X_train.shape}")
        print(f"✓ Test set: {self.X_test.shape}")
        
        # Show class distribution
        print(f"\n  Class Distribution:")
        print(f"  Train: {dict(self.y_train.value_counts().sort_index())}")
        print(f"  Test:  {dict(self.y_test.value_counts().sort_index())}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self):
        """Scale numerical features"""
        print(f"\n{'='*50}")
        print("Feature Scaling")
        print(f"{'='*50}\n")
        
        # Fit scaler on training data
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✓ Features scaled using StandardScaler")
        print(f"✓ Mean: {self.scaler.mean_[:3]} ...")
        print(f"✓ Std: {self.scaler.scale_[:3]} ...")
        
        # Convert back to DataFrame for easier handling
        self.X_train_scaled = pd.DataFrame(
            self.X_train_scaled,
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        
        self.X_test_scaled = pd.DataFrame(
            self.X_test_scaled,
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        
        return self.X_train_scaled, self.X_test_scaled
    
    def save_processed_data(self, output_dir="data/processed"):
        """Save processed train and test data"""
        print(f"\n{'='*50}")
        print("Saving Processed Data")
        print(f"{'='*50}\n")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Combine features and target for saving
        train_data = self.X_train_scaled.copy()
        train_data['target'] = self.y_train.values
        
        test_data = self.X_test_scaled.copy()
        test_data['target'] = self.y_test.values
        
        # Save to CSV
        train_path = os.path.join(output_dir, "train.csv")
        test_path = os.path.join(output_dir, "test.csv")
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        print(f"✓ Saved train data: {train_path} ({train_data.shape})")
        print(f"✓ Saved test data: {test_path} ({test_data.shape})")
        
        # Save scaler for later use
        scaler_path = os.path.join(output_dir, "scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Saved scaler: {scaler_path}")
        
        return train_path, test_path
    
    def run_preprocessing(self, input_path="data/raw/dataset.csv", output_dir="data/processed"):
        """Run complete preprocessing pipeline"""
        print(f"\n{'='*60}")
        print("STARTING DATA PREPROCESSING PIPELINE")
        print(f"{'='*60}\n")
        
        # Execute preprocessing steps
        self.load_raw_data(input_path)
        self.handle_missing_values()
        self.remove_duplicates()
        self.feature_engineering()
        self.split_features_target()
        self.train_test_split_data()
        self.scale_features()
        self.save_processed_data(output_dir)
        
        print(f"\n{'='*60}")
        print("PREPROCESSING COMPLETE!")
        print(f"{'='*60}\n")
        
        # Summary
        print("Summary:")
        print(f"  • Input: {input_path}")
        print(f"  • Train samples: {len(self.X_train_scaled)}")
        print(f"  • Test samples: {len(self.X_test_scaled)}")
        print(f"  • Features: {self.X_train_scaled.shape[1]}")
        print(f"  • Output: {output_dir}/")
        print()


def main():
    """Main preprocessing function"""
    # Initialize preprocessor
    preprocessor = DataPreprocessor(params_path="params.yaml")
    
    # Run preprocessing pipeline
    preprocessor.run_preprocessing(
        input_path="data/raw/dataset.csv",
        output_dir="data/processed"
    )
    
    print("✓ Data preprocessing completed successfully!")


if __name__ == "__main__":
    main()