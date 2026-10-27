from pathlib import Path
import json
import os


def log_existing_run():
    import mlflow
    root = Path(__file__).resolve().parents[1]
    metrics_dir = root / "artifacts" / "metrics"
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:" + str(root / "mlruns")))
    mlflow.set_experiment("credit-risk-ai")
    with mlflow.start_run(run_name="cs-training-xgboost-final"):
        comparison = __import__("pandas").read_csv(metrics_dir / "model_comparison.csv")
        business = json.loads((metrics_dir / "business_metrics.json").read_text())
        champion = json.loads((metrics_dir / "champion_metrics.json").read_text())
        mlflow.log_metrics({"roc_auc": champion["roc_auc"], "pr_auc": champion["pr_auc"],
                            "npa_reduction": business["npa_reduction"], "approval_rate": business["approval_rate"]})
        mlflow.log_param("champion", "XGBoost")
        mlflow.log_param("threshold", business["threshold"])
        mlflow.log_param("dataset", "cs-training.csv")
        mlflow.log_artifact(str(metrics_dir / "model_comparison.csv"))
        mlflow.log_artifact(str(metrics_dir / "business_metrics.json"))
        return mlflow.active_run().info.run_id

if __name__ == "__main__":
    print(log_existing_run())
