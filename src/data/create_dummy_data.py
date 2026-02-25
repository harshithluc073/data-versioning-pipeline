"""
Create dummy dataset for development and testing
"""
import pandas as pd
import numpy as np
import os

def create_dummy_data(output_path="data/raw/dataset.csv", n_rows=1000, random_state=42):
    """
    Generate a dummy dataset with features and target
    """
    np.random.seed(random_state)

    # Generate features
    # feature1: Normal distribution
    feature1 = np.random.normal(loc=10.0, scale=2.5, size=n_rows)

    # feature2: Uniform distribution (positive to avoid division by zero issues, though preprocessor handles it)
    feature2 = np.random.uniform(low=1.0, high=10.0, size=n_rows)

    # feature3: Gamma distribution
    feature3 = np.random.gamma(shape=2.0, scale=2.0, size=n_rows)

    # feature4: Normal distribution with some missing values (we'll add missing later if we want)
    feature4 = np.random.normal(loc=0.0, scale=1.0, size=n_rows)

    # Generate target (binary classification based on features)
    # A simple linear combination with some noise
    logits = 0.5 * feature1 - 1.0 * feature2 + 0.3 * feature3 + np.random.normal(0, 1, n_rows)
    target = (logits > 0).astype(int)

    # Create DataFrame
    df = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'feature4': feature4,
        'target': target
    })

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✓ Created dummy dataset at {output_path} with shape {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Target distribution:\n{df['target'].value_counts(normalize=True)}")

if __name__ == "__main__":
    create_dummy_data()
