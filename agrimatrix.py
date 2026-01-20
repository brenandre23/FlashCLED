import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import yaml
import sys
import os
from pathlib import Path

def load_submodels(config_path='configs/models.yaml'):
    """Loads submodel definitions from the YAML config."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('submodels', {})
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None

def analyze_submodel(submodel_name, submodels, df, output_dir):
    """Performs correlation and feature importance analysis for a given submodel."""
    if submodel_name not in submodels:
        print(f"Error: Submodel '{submodel_name}' not found in configuration.")
        print("Available submodels are:", list(submodels.keys()))
        return

    print(f"Analyzing submodel: {submodel_name}...")

    # 2. Select feature columns for the submodel
    feature_cols = submodels[submodel_name].get('features', [])
    if not feature_cols:
        print(f"Warning: No features defined for submodel '{submodel_name}'.")
        return

    target_col = 'target_fatalities_1_step'
    analysis_cols = feature_cols + [target_col]

    # Filter for columns that actually exist in your matrix
    available_cols = [c for c in analysis_cols if c in df.columns]
    missing_cols = set(analysis_cols) - set(available_cols)
    if missing_cols:
        print(f"Warning: The following columns were not found in the feature matrix and will be ignored:")
        for col in sorted(list(missing_cols)):
            print(f"- {col}")

    if target_col not in available_cols:
        print(f"Critical Error: Target column '{target_col}' not found in the feature matrix. Cannot proceed with {submodel_name}.")
        return

    # Ensure there are features to analyze
    features_in_df = [c for c in available_cols if c != target_col]
    if not features_in_df:
        print(f"Error: None of the specified features for this submodel exist in the feature matrix.")
        return
        
    corr_matrix = df[available_cols].corr()

    # 3. Visualize the "Conflict Signal"
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix[[target_col]].sort_values(by=target_col, ascending=False), 
                annot=True, cmap='RdYlGn', center=0, fmt=".2f")
    plt.title(f"Correlation of {submodel_name.title()} Features with Next-Step Fatalities")
    
    output_filename = output_dir / f"{submodel_name}_correlation.png"
    plt.savefig(output_filename)
    print(f"Correlation heatmap saved to {output_filename}")
    plt.close()

    # 4. Print Specific Insights
    print(f"\nTop Predictors for Conflict from '{submodel_name}':")
    print(corr_matrix[target_col].sort_values(ascending=False).head(10))

    # 5. Non-Linear Feature Importance
    X = df[features_in_df].copy()
    y = df[target_col].copy()

    # Handle potential NaN values in target or features
    if y.isnull().any() or X.isnull().values.any():
        print("\nWarning: NaN values detected. Dropping rows with NaNs for modeling.")
        # Align X and y before dropping NaNs
        combined = pd.concat([X, y], axis=1)
        combined.dropna(inplace=True)
        X = combined[features_in_df]
        y = combined[target_col]

    if X.empty or y.empty:
        print("Error: No data available for modeling after handling NaNs.")
        return

    print("\nFitting RandomForest model for non-linear feature importance...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nNon-Linear Feature Importance:")
    print(importances)
    print("-" * 50)

def main():
    """Main function to run the analysis for all submodels."""
    output_dir = Path("agrimatrix_outputs")
    output_dir.mkdir(exist_ok=True)
    
    submodels = load_submodels()
    if submodels is None:
        sys.exit(1)

    # 1. Load the final matrix once
    try:
        df = pd.read_parquet("data/processed/feature_matrix.parquet")
    except FileNotFoundError:
        print("Error: data/processed/feature_matrix.parquet not found. Please generate the feature matrix first.")
        sys.exit(1)

    for submodel_name in submodels.keys():
        # Only analyze enabled submodels
        if submodels[submodel_name].get('enabled', False):
             analyze_submodel(submodel_name, submodels, df, output_dir)
        else:
            print(f"Skipping disabled submodel: {submodel_name}")


if __name__ == "__main__":
    main()