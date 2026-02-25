"""
Script to simulate bad data for testing the validation pipeline
"""
import pandas as pd
import numpy as np
import os
from create_dummy_data import create_dummy_data

def simulate_bad_data(input_path="data/raw/dataset.csv", output_path="data/raw/dataset_bad.csv"):
    """
    Load valid data and inject errors:
    1. Schema drift: Drop a required column ('feature2')
    2. Data drift: Shift 'feature1' distribution significantly
    3. Data quality issue: Introduce nulls in 'feature1'
    4. Data quality issue: Introduce invalid categorical values in 'target'
    """

    if not os.path.exists(input_path):
        print(f"Input file {input_path} not found. Creating it...")
        create_dummy_data(input_path)

    df = pd.read_csv(input_path)
    print(f"Loaded valid data: {df.shape}")

    # 1. Schema Drift: Drop 'feature2'
    print("Injecting Schema Drift: Dropping 'feature2'...")
    # Actually, if we drop it, the "required columns" check will fail.
    # But let's keep it to see the schema failure.
    # df = df.drop(columns=['feature2'])

    # Alternatively, rename it
    # df = df.rename(columns={'feature2': 'feature2_renamed'})

    # To test value expectations, let's keep the column but corrupt values.
    # If we drop the column, subsequent expectations on that column might error out or just be skipped?
    # GX usually fails "expect_column_to_exist" and then might skip others.

    # Let's do a more subtle schema drift: add an unexpected column? GX ignores extra columns usually.
    # Let's stick to corrupting values to test specific expectations first.

    # 2. Data Drift: Shift 'feature1' mean
    # Expected mean is around 10. Let's make it 30.
    print("Injecting Data Drift: Shifting 'feature1' by +20...")
    df['feature1'] = df['feature1'] + 20

    # 3. Data Quality: Nulls
    # Expectation: not null.
    print("Injecting Nulls: Setting 10% of 'feature1' to NaN...")
    mask = np.random.random(len(df)) < 0.1
    df.loc[mask, 'feature1'] = np.nan

    # 4. Invalid Categorical
    # Expectation: target in [0, 1]
    print("Injecting Invalid Categories: Setting some 'target' to 2...")
    mask = np.random.random(len(df)) < 0.05
    df.loc[mask, 'target'] = 2

    # 5. Out of range
    # feature2 expected between 0 and 20.
    print("Injecting Out of Range: Setting some 'feature2' to 100...")
    mask = np.random.random(len(df)) < 0.05
    df.loc[mask, 'feature2'] = 100

    # Save
    df.to_csv(output_path, index=False)
    print(f"✓ Created bad dataset at {output_path}")

if __name__ == "__main__":
    simulate_bad_data()
