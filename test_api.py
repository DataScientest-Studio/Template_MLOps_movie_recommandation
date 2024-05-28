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

# Test for /showdata20 endpoint
'''
 This endpoint returns the first twenty rows of some data as JSON, but it requires
 authentication. Users with the role "user" should not have access to this endpoint,
 while users with the role "administrator" should have access.
 This endpoint allows access to people who are authenticated and authorized as administrator. Otherwise, the person is unauthorized.
 '''
def test_show_data20():
    # Test with user role
    response = client.get("/showdata20", auth=("alice", "wonderland"))
    assert response.status_code == 401 # Access should be unauthorized for user role
   
    response = client.get("/showdata20", auth=("clementine", "mandarine"))
    assert response.status_code == 401
   
    # Test with administrator role
    response_admin = client.get("/showdata20", auth=("bob", "builder"))
    assert response_admin.status_code == 200
   
    # Test with random user
    response = client.get("/showdata20", auth=("harry", "potter"))
    assert response.status_code == 401      
     
# Test for /predict endpoint
'''
This endpoint makes some predictions based on user IDs provided in the request parameters.
It requires authentication; users with the role "user" should have access,
while users with the role "administrator" should not.
'''
def test_predict():
    # Test with user role
    response = client.get("/predict", params={"users_id": [1,2,3]}, auth=("alice", "wonderland"))
    assert response.status_code == 200
   
    response = client.get("/predict", params={"users_id":[1,2,3]}, auth=("clementine", "mandarine"))
    assert response.status_code == 200
   
    # Test with administrator role
    response_admin = client.get("/predict", params={"users_id":[1,2,3]}, auth=("bob", "builder"))
    assert response_admin.status_code == 401 # Access should be unauthorized for admin role
   
    # Test with random user
    response = client.get("/predict", params={"users_id":[1,2,3]}, auth=("harry", "potter"))
    assert response.status_code == 401

