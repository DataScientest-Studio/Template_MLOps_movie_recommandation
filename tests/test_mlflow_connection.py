import pytest
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import time

def test_mlflow_tracking():
    # Initialize the MLflow client
    client = MlflowClient()

    # Start an MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Log fak e run with a parameter and metric
        mlflow.log_param("test_param", 123)
        mlflow.log_metric("test_metric", 0.456)

    # Retrieve the run data to confirm logging
    run_data = client.get_run(run_id).data

    # Assertions to check if the parameter and metric were logged
    assert run_data.params.get("test_param") == "123", "Parameter 'test_param' was not logged correctly."
    assert run_data.metrics.get("test_metric") == 0.456, "Metric 'test_metric' was not logged correctly."

    # Cleanup
    client.delete_run(run_id)

def test_mlflow_model_registry():

    client = MlflowClient()

    # Start an MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Log a dummy model 
        from sklearn.dummy import DummyRegressor

        model = DummyRegressor(strategy="mean")
        model.fit([[0]], [0])

        # Log the model
        mlflow.sklearn.log_model(model, artifact_path="test_model")

        # Register the model
        result = mlflow.register_model(
            model_uri=f"runs:/{run_id}/test_model",
            name="Test_Model"
        )

        for _ in range(10):
            model_version_details = client.get_model_version(name="Test_Model", version=result.version)
            if model_version_details.status == 'READY':
                break
            time.sleep(1)

        assert model_version_details.status == 'READY', "Model version is not ready."
        assert model_version_details.name == "Test_Model", "Model name mismatch."
        assert int(model_version_details.version) >= 1, "Unexpected model version number."

    # Simple Cleanup
    client.delete_registered_model(name="Test_Model")
    client.delete_run(run_id)
