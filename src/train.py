# =============================================================================
# src/train.py — DVC Pipeline Stage 2: Model Training + MLflow Tracking
# =============================================================================
# WHAT THIS SCRIPT DOES:
#   1. Loads the processed feature matrix from preprocess.py
#   2. Trains a Decision Tree classifier
#   3. Logs EVERYTHING to MLflow: params, metrics, the model artifact
#   4. Saves the trained model to models/ (tracked by DVC)
#
# HOW IT FITS INTO THE PIPELINE:
#   features.csv → [THIS SCRIPT] → mlruns/ (MLflow) + models/dt_model.pkl
#
# HOW TO RUN (standalone):
#   python src/train.py
#   Then open: mlflow ui  (in a new terminal)
#   Open browser: http://localhost:5000
#
# KEY TEACHING POINT:
#   Every time this runs, MLflow creates a new "run" with a unique ID.
#   You can change max_depth in params.yaml, run again, and compare
#   both runs side by side in the MLflow UI — without overwriting anything.
# =============================================================================

import pandas as pd
import numpy as np
import yaml
import os
import pickle

import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Load parameters from the central params.yaml
# ---------------------------------------------------------------------------
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

data_params   = params["data"]
model_params  = params["model"]
mlflow_params = params["mlflow"]

PROCESSED_PATH  = data_params["processed_path"]
TEST_SIZE       = data_params["test_size"]
RANDOM_STATE    = data_params["random_state"]
MODEL_SAVE_PATH = "models/dt_model.pkl"


def load_features(path: str):
    """
    Load the processed feature matrix and split into X (features) and y (target).

    Parameters
    ----------
    path : str
        Path to the processed CSV (output of preprocess.py)

    Returns
    -------
    X : pd.DataFrame — feature matrix (all columns except 'Survived')
    y : pd.Series   — target column ('Survived': 0 or 1)
    """
    df = pd.read_csv(path)
    print(f"[train] Loaded features: {df.shape[0]} rows, {df.shape[1]} cols")

    # Separate features from target
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    print(f"[train] Features used: {list(X.columns)}")
    print(f"[train] Class distribution — Survived: {y.sum()} ({y.mean()*100:.1f}%), Died: {(~y.astype(bool)).sum()}")
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series):
    """
    Split data into train and test sets.

    Parameters use test_size and random_state from params.yaml so the
    split is reproducible and logged to MLflow automatically.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,    # Ensures both splits have the same class ratio
    )
    print(f"[train] Train size: {len(X_train)} | Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """
    Train a Decision Tree using hyperparameters from params.yaml.

    All hyperparams come from params.yaml — NOT hardcoded here.
    This means changing params.yaml + `dvc repro` = new experiment automatically.

    Parameters
    ----------
    X_train, y_train : training features and labels

    Returns
    -------
    DecisionTreeClassifier : fitted model
    """
    model = DecisionTreeClassifier(
        max_depth        = model_params["max_depth"],
        criterion        = model_params["criterion"],
        min_samples_split= model_params["min_samples_split"],
        min_samples_leaf = model_params["min_samples_leaf"],
        random_state     = model_params["random_state"],
    )

    model.fit(X_train, y_train)
    print(f"[train] Model trained. Tree depth used: {model.get_depth()} / max allowed: {model_params['max_depth']}")
    return model


def compute_metrics(model: DecisionTreeClassifier, X_test, y_test) -> dict:
    """
    Compute evaluation metrics on the held-out test set.

    Returns a dict of metric_name → value, which gets logged to MLflow.

    Metrics explained:
      accuracy  → overall correct predictions / total predictions
      precision → of all predicted Survived, how many actually survived?
      recall    → of all actual survivors, how many did we catch?
      f1        → harmonic mean of precision and recall (balanced metric)
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall"   : round(recall_score(y_test, y_pred), 4),
        "f1_score" : round(f1_score(y_test, y_pred), 4),
    }

    # Confusion matrix for manual inspection
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n[train] Confusion Matrix:")
    print(f"         Predicted: 0    1")
    print(f"  Actual 0:      {cm[0][0]:4d} {cm[0][1]:4d}   (TN, FP)")
    print(f"  Actual 1:      {cm[1][0]:4d} {cm[1][1]:4d}   (FN, TP)")

    return metrics


def save_model(model: DecisionTreeClassifier, path: str) -> None:
    """
    Save the trained model as a pickle file.

    DVC tracks this file in models/ — so every version of the model
    is stored in DVC cache and can be retrieved by git checkout + dvc checkout.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[train] Model saved to: {path}")


# ---------------------------------------------------------------------------
# Main execution — this is where MLflow tracking happens
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("STAGE 2: Training")
    print("=" * 60)

    # Step 1: Load processed features
    X, y = load_features(PROCESSED_PATH)

    # Step 2: Train/test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ------------------------------------------------------------------
    # MLflow Tracking Block
    # ------------------------------------------------------------------
    # TEACHING POINT: Everything inside this `with` block is tracked.
    # MLflow assigns a unique run_id and stores:
    #   - params   → what settings did we use?
    #   - metrics  → how well did the model perform?
    #   - artifacts → the trained model file itself
    #
    # After running this script, open a terminal and type:
    #   mlflow ui
    # Then visit http://localhost:5000 to see all runs side by side.
    # ------------------------------------------------------------------

    # Tell MLflow where to store run data (local folder: mlruns/)
    mlflow.set_tracking_uri(mlflow_params["tracking_uri"])

    # Group this run under the experiment name (creates it if it doesn't exist)
    mlflow.set_experiment(mlflow_params["experiment_name"])

    with mlflow.start_run() as run:
        print(f"\n[train] MLflow Run ID: {run.info.run_id}")
        print(f"[train] Experiment  : {mlflow_params['experiment_name']}")

        # ---- Step 3: Train the model ----
        model = train_model(X_train, y_train)

        # ---- Step 4: Log hyperparameters to MLflow ----
        # mlflow.log_params() takes a dict — log everything from params.yaml
        # so a run is 100% reproducible from just its MLflow record
        mlflow.log_params({
            # Model hyperparams
            "max_depth"         : model_params["max_depth"],
            "criterion"         : model_params["criterion"],
            "min_samples_split" : model_params["min_samples_split"],
            "min_samples_leaf"  : model_params["min_samples_leaf"],
            # Data params
            "test_size"         : TEST_SIZE,
            "random_state"      : RANDOM_STATE,
            # Features used (log as a string for readability)
            "features"          : str(list(X.columns)),
        })
        print("[train] Logged params to MLflow")

        # ---- Step 5: Compute and log metrics ----
        metrics = compute_metrics(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        print(f"\n[train] Metrics:")
        for k, v in metrics.items():
            print(f"  {k:12s}: {v:.4f}")

        # ---- Step 6: Log the model as an MLflow artifact ----
        # mlflow.sklearn.log_model() saves the model in a standard format
        # that can later be loaded by name from the registry:
        #   mlflow.pyfunc.load_model("models:/titanic-decision-tree/Production")
        mlflow.sklearn.log_model(
            sk_model        = model,
            artifact_path   = "decision-tree-model",
            # registered_model_name=mlflow_params["model_name"]  # Uncomment in Phase 4
        )
        print(f"[train] Model artifact logged to MLflow")

        # ---- Step 7: Also save as pickle for DVC tracking ----
        # DVC tracks this file so model history is tied to data + code history
        save_model(model, MODEL_SAVE_PATH)

        # Write run_id to a file so evaluate.py can load the same run's model
        with open("models/run_id.txt", "w") as f:
            f.write(run.info.run_id)
        print(f"[train] Run ID saved to models/run_id.txt")

    print("\n[train] Done!")
    print(f"[train] Run `mlflow ui` in terminal, then open http://localhost:5000")