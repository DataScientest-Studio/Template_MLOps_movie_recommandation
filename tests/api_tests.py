import pytest
import requests

API_BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test the /health endpoint for API health check."""
    response = requests.get(f"{API_BASE_URL}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_import_raw_data_endpoint():
    """Test the /import_raw_data endpoint."""
    response = requests.post(f"{API_BASE_URL}/import_raw_data")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "success"

def test_build_features_endpoint():
    """Test the /build_features endpoint."""
    response = requests.post(f"{API_BASE_URL}/build_features")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "success"

def test_train_endpoint():
    """Test the /train endpoint."""
    response = requests.post(f"{API_BASE_URL}/train")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "success"

def test_predict_endpoint():
    """Test the /predict endpoint with valid input."""
    response = requests.post(f"{API_BASE_URL}/predict")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "success"
    assert "logs" in response.json()  # Ensures the predict service logs are returned

def test_evaluate_endpoint():
    """Test the /evaluate endpoint."""
    response = requests.post(f"{API_BASE_URL}/evaluate")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "success"