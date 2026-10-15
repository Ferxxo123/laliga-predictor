import mlflow

from src.predictor.evaluate import accuracy, brier_score, log_loss


def log_run(name, probs, results, params=None):
    """Save one model's scores (and settings) as an MLflow run."""
    mlflow.set_experiment("laliga-predictor")

    with mlflow.start_run(run_name=name):
        mlflow.log_params(params or {})
        mlflow.log_metric("log_loss", log_loss(probs, results))
        mlflow.log_metric("brier", brier_score(probs, results))
        mlflow.log_metric("accuracy", accuracy(probs, results))