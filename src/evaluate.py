# =============================================================================
# src/evaluate.py — DVC Pipeline Stage 3: Evaluation + Quality Gate
# =============================================================================
# WHAT THIS SCRIPT DOES:
#   1. Loads the trained model saved by train.py
#   2. Evaluates it on the test set
#   3. Enforces a QUALITY GATE — fails with exit code 1 if accuracy is too low
#   4. Writes metrics to metrics.json (DVC can read this for `dvc metrics show`)
#
# THE QUALITY GATE IS THE KEY TEACHING POINT:
#   In GitHub Actions CI, this script runs after every push.
#   If the model doesn't meet the minimum accuracy, the CI workflow FAILS.
#   This prevents a bad model from ever reaching the registry.
#
#   Think of it as: unit tests for your model's performance.
#
# HOW TO RUN (standalone):
#   python src/evaluate.py
#
# EXPECTED OUTPUT:
#   Accuracy 0.8150 >= threshold 0.78 → PASS
# =============================================================================

import pandas as pd
import numpy as np
import yaml
import json
import pickle
import sys
import os

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Load parameters
# ---------------------------------------------------------------------------
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

data_params = params["data"]
eval_params = params["evaluation"]

PROCESSED_PATH  = data_params["processed_path"]
TEST_SIZE       = data_params["test_size"]
RANDOM_STATE    = data_params["random_state"]
MODEL_PATH      = "models/dt_model.pkl"
METRICS_PATH    = "metrics.json"
MIN_ACCURACY    = eval_params["min_accuracy"]


def load_model(path: str):
    """
    Load the trained Decision Tree model from disk.

    Parameters
    ----------
    path : str
        Path to the saved pickle file

    Returns
    -------
    DecisionTreeClassifier : loaded model
    """
    if not os.path.exists(path):
        print(f"[evaluate] ERROR: Model not found at {path}")
        print("  → Run `python src/train.py` or `dvc repro` first")
        sys.exit(1)

    with open(path, "rb") as f:
        model = pickle.load(f)

    print(f"[evaluate] Loaded model from: {path}")
    return model


def load_test_data(path: str):
    """
    Load processed features and reconstruct the same test split used in training.

    IMPORTANT: We use the same test_size and random_state as train.py.
    This guarantees we evaluate on the exact same test rows every time.
    """
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(path)
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # Must match the split in train.py exactly
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"[evaluate] Test set: {len(X_test)} rows")
    return X_test, y_test


def evaluate(model, X_test, y_test) -> dict:
    """
    Compute detailed evaluation metrics and print a classification report.

    Returns
    -------
    dict : metrics dictionary (will be saved to metrics.json)
    """
    y_pred = model.predict(X_test)

    accuracy  = round(accuracy_score(y_test, y_pred), 4)
    cm        = confusion_matrix(y_test, y_pred)

    # Full classification report (precision, recall, f1 per class)
    report = classification_report(y_test, y_pred, target_names=["Died", "Survived"])

    print(f"\n[evaluate] Classification Report:")
    print(report)

    print(f"[evaluate] Confusion Matrix:")
    print(f"           Predicted: Died  Survived")
    print(f"  Actual Died      :  {cm[0][0]:4d}      {cm[0][1]:4d}")
    print(f"  Actual Survived  :  {cm[1][0]:4d}      {cm[1][1]:4d}")

    # Feature importances — great to show students
    feature_names = list(X_test.columns)
    importances   = model.feature_importances_
    sorted_idx    = np.argsort(importances)[::-1]

    print(f"\n[evaluate] Feature Importances:")
    for i in sorted_idx:
        bar = "█" * int(importances[i] * 40)
        print(f"  {feature_names[i]:15s}: {importances[i]:.4f}  {bar}")

    metrics = {
        "accuracy"        : accuracy,
        "test_size"       : len(X_test),
        "tree_depth_used" : model.get_depth(),
        "n_leaves"        : model.get_n_leaves(),
    }

    return metrics


def check_quality_gate(accuracy: float, threshold: float) -> None:
    """
    Enforce the model quality gate.

    If accuracy is below the threshold defined in params.yaml, this function
    exits with code 1, which causes GitHub Actions CI to MARK THE JOB AS FAILED.

    This is the 'automated gatekeeper' that prevents bad models from being
    registered or deployed.

    Parameters
    ----------
    accuracy  : float — model accuracy on the test set
    threshold : float — minimum acceptable accuracy (from params.yaml)
    """
    print(f"\n[evaluate] Quality Gate Check:")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Threshold: {threshold:.4f}")

    if accuracy >= threshold:
        print(f"  Result   : PASS ✓  ({accuracy:.4f} >= {threshold:.4f})")
    else:
        print(f"  Result   : FAIL ✗  ({accuracy:.4f} < {threshold:.4f})")
        print(f"\n  [evaluate] Model did not meet the quality gate.")
        print(f"  → Investigate: did you remove an important feature?")
        print(f"  → Or lower min_accuracy in params.yaml if threshold is too strict.")
        # Exit code 1 = failure — GitHub Actions will mark the CI step as failed
        sys.exit(1)


def save_metrics(metrics: dict, path: str) -> None:
    """
    Save metrics to a JSON file.

    WHY SAVE METRICS TO A FILE?
      DVC has a built-in command `dvc metrics show` that reads this file
      and compares metrics across git branches/commits.
      This means students can do:
        git checkout main
        dvc metrics show
        git checkout feature-branch
        dvc metrics show
      ...and see the performance difference without running anything.
    """
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[evaluate] Metrics saved to: {path}")
    print(f"  → Run `dvc metrics show` to compare across runs")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("STAGE 3: Evaluation + Quality Gate")
    print("=" * 60)

    # Step 1: Load trained model
    model = load_model(MODEL_PATH)

    # Step 2: Load test data (same split as training)
    X_test, y_test = load_test_data(PROCESSED_PATH)

    # Step 3: Evaluate
    metrics = evaluate(model, X_test, y_test)

    # Step 4: Save metrics for DVC
    save_metrics(metrics, METRICS_PATH)

    # Step 5: Quality gate — this is what CI checks
    # If this fails, the script exits with code 1 → CI workflow fails
    check_quality_gate(metrics["accuracy"], MIN_ACCURACY)

    print("\n[evaluate] All checks passed. Model is ready for registration.")
    print("[evaluate] Next step → Phase 4: Register model in MLflow Registry")