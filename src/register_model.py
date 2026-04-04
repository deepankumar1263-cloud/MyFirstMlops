# =============================================================================
# src/register_model.py — Phase 4: MLflow Model Registry
# =============================================================================
# WHAT THIS SCRIPT DOES:
#   Takes the best run from MLflow and registers it in the Model Registry.
#   Demonstrates the full model lifecycle:
#     None → Staging → Production → Archived
#
# WHEN TO RUN THIS:
#   After running train.py at least once and verifying the model is good.
#   This is intentionally NOT part of the automated DVC pipeline —
#   promotion to Production is a deliberate human decision.
#
# HOW TO RUN:
#   python src/register_model.py
#
#   Optional: specify a run ID directly
#   python src/register_model.py --run-id <run_id_from_mlflow_ui>
#
# KEY TEACHING POINT:
#   After running this, show students how to LOAD a model by stage name:
#     import mlflow.pyfunc
#     model = mlflow.pyfunc.load_model("models:/titanic-decision-tree/Production")
#   This is how real production systems work — no file paths, just registry names.
# =============================================================================

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import yaml
import argparse
import sys

# ---------------------------------------------------------------------------
# Load parameters
# ---------------------------------------------------------------------------
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

mlflow_params = params["mlflow"]
eval_params   = params["evaluation"]

TRACKING_URI  = mlflow_params["tracking_uri"]
EXPERIMENT    = mlflow_params["experiment_name"]
MODEL_NAME    = mlflow_params["model_name"]
MIN_ACCURACY  = eval_params["min_accuracy"]


def get_best_run(client: MlflowClient, experiment_name: str) -> str:
    """
    Find the best MLflow run by accuracy metric.

    Searches all runs in the experiment and returns the run_id of the
    run with the highest test accuracy.

    Parameters
    ----------
    client : MlflowClient
    experiment_name : str

    Returns
    -------
    str : run_id of the best run
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"[register] ERROR: Experiment '{experiment_name}' not found.")
        print("  → Run `python src/train.py` first to create runs.")
        sys.exit(1)

    # Search runs sorted by accuracy (descending)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1,
    )

    if not runs:
        print("[register] ERROR: No runs found in the experiment.")
        sys.exit(1)

    best_run = runs[0]
    accuracy  = best_run.data.metrics.get("accuracy", 0)

    print(f"[register] Best run found:")
    print(f"  Run ID  : {best_run.info.run_id}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Params  : max_depth={best_run.data.params.get('max_depth')}, "
          f"criterion={best_run.data.params.get('criterion')}")

    return best_run.info.run_id, accuracy


def register_model(client: MlflowClient, run_id: str, model_name: str) -> str:
    """
    Register a model version from an existing MLflow run.

    Creates a new version under the given model name.
    Initial stage is 'None' — we manually promote it below.

    Returns
    -------
    str : the new version number
    """
    model_uri = f"runs:/{run_id}/decision-tree-model"

    print(f"\n[register] Registering model...")
    print(f"  Model URI: {model_uri}")
    print(f"  Name     : {model_name}")

    model_version = mlflow.register_model(
        model_uri  = model_uri,
        name       = model_name,
    )

    version = model_version.version
    print(f"  Version  : {version}")
    print(f"  Stage    : None → will be promoted next")

    return version


def promote_to_staging(client: MlflowClient, model_name: str, version: str) -> None:
    """
    Move a model version to 'Staging' stage.

    Staging = "ready for QA / validation — not yet in production"
    """
    client.transition_model_version_stage(
        name    = model_name,
        version = version,
        stage   = "Staging",
    )
    print(f"\n[register] Version {version} promoted to: Staging")


def promote_to_production(client: MlflowClient, model_name: str, version: str,
                           accuracy: float) -> None:
    """
    Move a model version to 'Production' stage if it passes the quality gate.

    Also archives the previous Production version automatically.

    TEACHING POINT:
      In real systems, this promotion would be triggered by CI after
      a successful canary deploy evaluation — not just accuracy on test set.
    """
    if accuracy < MIN_ACCURACY:
        print(f"\n[register] BLOCKED: accuracy {accuracy:.4f} < threshold {MIN_ACCURACY}")
        print("  → Model stays in Staging. Fix the model before promoting to Production.")
        return

    # archive_existing_versions=True → old Production version becomes Archived
    client.transition_model_version_stage(
        name                    = model_name,
        version                 = version,
        stage                   = "Production",
        archive_existing_versions=True,   # Automatically archive the previous Production
    )
    print(f"\n[register] Version {version} promoted to: Production")
    print(f"  → Previous Production version (if any) is now Archived")
    print(f"\n  Load this model anywhere with:")
    print(f"    import mlflow.pyfunc")
    print(f"    model = mlflow.pyfunc.load_model('models:/{model_name}/Production')")


def demo_load_from_registry(model_name: str) -> None:
    """
    Demonstrate loading a model by stage name from the registry.

    This is the 'aha moment' — no file paths, no pickle files.
    Just ask the registry for the Production model by name.
    """
    print(f"\n[register] Demo: Loading Production model from registry...")
    try:
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
        print(f"  Model loaded successfully!")
        print(f"  Type: {type(model)}")
        print(f"  → In production code you would call: model.predict(X_new)")
    except Exception as e:
        print(f"  Could not load model: {e}")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Optional: accept a specific run_id via command line
    parser = argparse.ArgumentParser(description="Register MLflow model to registry")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Specific MLflow run ID to register (default: best run by accuracy)")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 4: MLflow Model Registry")
    print("=" * 60)

    # Set up MLflow client
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # Step 1: Find or use specified run
    if args.run_id:
        run      = client.get_run(args.run_id)
        run_id   = args.run_id
        accuracy = run.data.metrics.get("accuracy", 0)
        print(f"[register] Using specified run: {run_id} (accuracy: {accuracy:.4f})")
    else:
        run_id, accuracy = get_best_run(client, EXPERIMENT)

    # Step 2: Register the model (creates version 1, 2, 3... automatically)
    version = register_model(client, run_id, MODEL_NAME)

    # Step 3: Promote to Staging
    promote_to_staging(client, MODEL_NAME, version)

    # Step 4: Promote to Production (quality gate enforced inside)
    promote_to_production(client, MODEL_NAME, version, accuracy)

    # Step 5: Demo load from registry
    demo_load_from_registry(MODEL_NAME)

    print("\n[register] Done!")
    print(f"[register] Open MLflow UI to see the registry:")
    print(f"  mlflow ui → http://localhost:5000 → Models tab")