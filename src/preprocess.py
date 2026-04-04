# =============================================================================
# src/preprocess.py — DVC Pipeline Stage 1: Data Preprocessing
# =============================================================================
# WHAT THIS SCRIPT DOES:
#   Reads the raw Titanic CSV, cleans it, engineers features, and saves a
#   clean feature matrix to data/processed/features.csv
#
# HOW IT FITS INTO THE PIPELINE:
#   Raw CSV (DVC tracked) → [THIS SCRIPT] → features.csv → train.py
#
# HOW TO RUN (standalone):
#   python src/preprocess.py
#
# HOW DVC RUNS IT:
#   dvc repro  →  DVC reads dvc.yaml  →  runs this script if inputs changed
#
# KEY TEACHING POINT:
#   DVC hashes the input file (titanic.csv) and params (features.use_columns).
#   If neither changed since the last run, DVC SKIPS this stage entirely.
#   Change one column name in params.yaml → DVC re-runs this stage only.
# =============================================================================

import pandas as pd
import numpy as np
import yaml
import os
import sys

# ---------------------------------------------------------------------------
# Load pipeline parameters from the central params.yaml
# ---------------------------------------------------------------------------
# WHY YAML AND NOT ARGPARSE?
#   Both work. But yaml means all configs are in one place (params.yaml),
#   which DVC can watch for changes automatically. No magic numbers in code.
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Extract relevant sections
data_params    = params["data"]
feature_params = params["features"]

RAW_PATH       = data_params["raw_path"]
PROCESSED_PATH = data_params["processed_path"]
USE_COLUMNS    = feature_params["use_columns"]


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load the raw Titanic CSV file.

    Parameters
    ----------
    path : str
        Path to the raw CSV (e.g. data/raw/titanic.csv)

    Returns
    -------
    pd.DataFrame
        The raw dataframe as downloaded from Kaggle
    """
    if not os.path.exists(path):
        print(f"[ERROR] Raw data not found at: {path}")
        print("  → Please place titanic.csv in data/raw/")
        print("  → Download from: https://www.kaggle.com/c/titanic/data")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"[preprocess] Loaded raw data: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the raw Titanic dataset.

    Missing value strategy:
      - Age     → fill with MEDIAN (robust to outliers like very old passengers)
      - Embarked → fill with MODE (most common port: Southampton)
      - Fare    → fill with MEDIAN (rare missing, one passenger in test set)

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with possible NaN values

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with no missing values in used columns
    """
    df = df.copy()

    # Age: ~20% missing — use median imputation
    # DISCUSSION POINT: mean vs median — why median is better here?
    age_median = df["Age"].median()
    df["Age"] = df["Age"].fillna(age_median)
    print(f"[preprocess] Filled {df['Age'].isna().sum()} missing Age values with median ({age_median:.1f})")

    # Embarked: only 2 rows missing — use mode (most frequent value)
    embarked_mode = df["Embarked"].mode()[0]
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)
    print(f"[preprocess] Filled missing Embarked values with mode ('{embarked_mode}')")

    # Fare: very rarely missing — use median
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    return df


def engineer_features(df: pd.DataFrame, use_columns: list) -> pd.DataFrame:
    """
    Select and encode features for the Decision Tree model.

    Encoding strategy:
      - Sex      → Label encode: male=0, female=1
      - Embarked → One-hot encode: creates Embarked_C, Embarked_Q, Embarked_S
      - All others → keep as-is (DT handles scale natively, no normalization needed)

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe
    use_columns : list
        List of column names from params.yaml (features.use_columns)

    Returns
    -------
    pd.DataFrame
        Feature matrix X with encoded columns + target column 'Survived'
    """
    df = df.copy()

    # --- Encode Sex: male=0, female=1 ---
    # Map is explicit so we know exactly what encoding was applied
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    print("[preprocess] Encoded 'Sex': male=0, female=1")

    # --- One-hot encode Embarked ---
    # drop_first=True drops Embarked_C to avoid multicollinearity
    # (less critical for DTs but good practice to teach)
    if "Embarked" in use_columns:
        embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked", drop_first=True)
        # Python 3.12+ returns bool dtype from get_dummies — cast to int
        # so sklearn does not raise a ValueError during model training
        embarked_dummies = embarked_dummies.astype(int)
        df = pd.concat([df, embarked_dummies], axis=1)
        # Replace the original Embarked column with the dummies in use_columns
        use_columns = [c for c in use_columns if c != "Embarked"]
        use_columns += list(embarked_dummies.columns)
        print(f"[preprocess] One-hot encoded 'Embarked': {list(embarked_dummies.columns)}")

    # --- Select only the columns we want (from params.yaml) ---
    # TEACHING POINT: changing use_columns in params.yaml and running
    # `dvc repro` will re-run this stage with the new feature set automatically
    feature_df = df[use_columns].copy()

    # --- Attach the target column ---
    feature_df["Survived"] = df["Survived"]

    print(f"[preprocess] Final feature columns: {list(feature_df.columns)}")
    return feature_df


def save_processed(df: pd.DataFrame, path: str) -> None:
    """
    Save the processed feature matrix to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Processed feature matrix with target column
    path : str
        Output path (e.g. data/processed/features.csv)
    """
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[preprocess] Saved processed data to: {path} ({df.shape[0]} rows, {df.shape[1]} cols)")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("STAGE 1: Preprocessing")
    print("=" * 60)

    # Step 1: Load raw data
    df_raw = load_raw_data(RAW_PATH)

    # Step 2: Handle missing values
    df_clean = clean_data(df_raw)

    # Step 3: Feature engineering + selection
    df_features = engineer_features(df_clean, USE_COLUMNS.copy())

    # Step 4: Save to disk (DVC will hash this output for the next stage)
    save_processed(df_features, PROCESSED_PATH)

    print("\n[preprocess] Done. Run `python src/train.py` or `dvc repro` next.")