import pytest
from fastapi.testclient import TestClient
from app import app


client = TestClient(app)


# Test for /ping endpoint
def test_ping():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"message": "API is up and running"}


# Test for /showdata endpoint
def test_show_data():
    response = client.get("/showdata")
    assert response.status_code == 200


# todo: investigate for failures?
# Test for /showdata20 endpoint
def test_show_data20():
    response = client.get("/showdata20", auth=("alice", "wonderland"))
    assert response.status_code == 401


# Test for /predict endpoint
def test_predict():
    response = client.get(
        "/predict", params={"users_id": [1, 2, 3]}, auth=("alice", "wonderland")
    )
    assert response.status_code == 200


# Test for unauthorized access to /showdata20 endpoint
def test_unauthorized_access_show_data20():
    response = client.get("/showdata20", auth=("clementine", "mandarine"))
    assert response.status_code == 401
