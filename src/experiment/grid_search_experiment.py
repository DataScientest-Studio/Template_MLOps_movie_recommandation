from mlflow import MlflowClient
import mlflow
import requests
import os
import json
from itertools import product
import yaml

API_URL = "http://api:8000"
#API_URL = "http://localhost:8000"

def call_train(hyperparams):
    response = requests.post(f"{API_URL}/train", json={"hyperparams": hyperparams})
    response.raise_for_status()
    print("✅ Model trained successfully.")
    return response.json()

def call_predict(user_ids=None, n_recommendations=None):
    json_data = {}
    if user_ids:
        json_data["user_ids"] = user_ids
    if n_recommendations:
        json_data["n_recommendations"] = n_recommendations

    if json_data:
        response = requests.post(f"{API_URL}/predict", json=json_data)
    else:
        response = requests.post(f"{API_URL}/predict")
    response.raise_for_status()
    print("✅ Predictions generated and saved.")
    return response.json()

def call_evaluate():
    response = requests.post(f"{API_URL}/evaluate")
    response.raise_for_status()
    print("✅ Evaluation complete.")
    return response.json()

def main():
    # Define tracking_uri
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()

    # Define parameters grid
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hyperparameters.yaml')
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    hyperparam_grid = config.get("hyperparameters", {})
    keys = list(hyperparam_grid.keys())
    param_combinations = list(product(*hyperparam_grid.values()))

    # For tracking best run
    best_coverage = -1
    best_run_id = None
    best_params = None

    for values in param_combinations:
        hyperparams = dict(zip(keys, values))
        print(f"\nTraining with hyperparameters: {hyperparams}")

        call_train(hyperparams)
        call_predict()
        call_evaluate()

        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        metrics_path = os.path.join(BASE_DIR, "metrics", "scores.json")
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        # Store information in tracking server
        with mlflow.start_run() as run:
            mlflow.log_params(hyperparams)
            mlflow.log_metrics(metrics)
            run_id = run.info.run_id
            print(f"Logged run_id: {run_id} with metrics: {metrics}")

            # Track best coverage
            if metrics.get('coverage_10', 0) > best_coverage:
                best_coverage = metrics['coverage_10']
                best_run_id = run_id
                best_params = hyperparams

    print("\n=== Best run based on 'coverage' ===")
    print(f"Run ID: {best_run_id}")
    print(f"Best Params: {best_params}")
    print(f"Best Coverage: {best_coverage}")

if __name__ == "__main__":
    main()