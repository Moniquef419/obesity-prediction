import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'Obesity.csv')

def main():
    df = pd.read_csv(DATA_PATH)
    print('Dataset shape:', df.shape)
    print('\nColumns:')
    print(df.columns.tolist())
    print('\nFirst 5 rows:')
    print(df.head())

    # Basic target distribution
    target_col = 'Obesity'
    print('\nTarget distribution:')
    print(df[target_col].value_counts())

    # Missing values
    print('\nMissing values per column:')
    print(df.isnull().sum())

    # Add BMI
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    print('\nBMI stats:')
    print(df['BMI'].describe())

    # Save a small sample for quick inspection
    sample_path = os.path.join(os.path.dirname(__file__), '..', 'data_sample.csv')
    df.sample(200, random_state=42).to_csv(sample_path, index=False)
    print(f"\nSaved a 200-row sample to {sample_path}")

if __name__ == '__main__':
    main()
