from fastapi import FastAPI
from fastapi import Query
from typing import List
from src.models.predict_model import make_predictions
import joblib
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import pandas as pd

user_credentials = {
    "alice": ["wonderland", "user"],
    "bob": ["builder", "administrator"],
    "clementine": ["mandarine", "user"],
}

# Security scheme for HTTP Basic Authentication
security = HTTPBasic()

app = FastAPI()


@app.get("/")
async def index():
    return {"message": "Movie Recommendation"}


@app.get("/ping")
async def ping():
    return {"message": "API is up and running"}


# ---------------------------------------------


# df = pd.read_csv("app/src/data/data/processed/movie_matrix.csv")
df = pd.read_csv("data/processed/movie_matrix.csv")


@app.get("/showdata")
async def show_data():
    # Return the first five rows of the DataFrame as JSON
    return df.head().to_json(orient="records")


# ---------------------------------------------


def authenticate_user(credentials: HTTPBasicCredentials = Depends(security)):
    if (
        credentials is None
        or credentials.username is None
        or credentials.password is None
    ):
        # Credentials are not provided, return HTTP 401 with WWW-Authenticate header
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication for user failed(1): Please provide username and password",
            headers={"WWW-Authenticate": "Basic"},
        )
    # Credentials are provided, continue with authentication logic
    for username in user_credentials.keys():
        if credentials.username == username:
            if credentials.password == user_credentials[username][0]:
                print(credentials.username, credentials.password)
                return credentials
    # If provided credentials don't match, raise HTTP 401 with WWW-Authenticate header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication for user failed(2): Incorrect username or password",
        headers={"WWW-Authenticate": "Basic"},
    )


def authenticate_administrator(credentials: HTTPBasicCredentials = Depends(security)):
    if (
        credentials is None
        or credentials.username is None
        or credentials.password is None
    ):
        # Credentials are not provided, return HTTP 401 with WWW-Authenticate header
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication for administrator failed(1): Please provide username and password",
            headers={"WWW-Authenticate": "Basic"},
        )
    # Credentials are provided, continue with authentication logic
    else:
        for username in user_credentials.keys():
            if credentials.username == username:
                if credentials.password == user_credentials[username][0]:
                    print(credentials.username, credentials.password)
                    return credentials
        # If provided credentials don't match or role is not administrator, raise HTTP 401 with WWW-Authenticate header
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication for administrator failed(2): Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )


def authorize_user(credentials: HTTPBasicCredentials = Depends(security)):
    if (
        credentials is None
        or credentials.username is None
        or credentials.password is None
    ):
        # Credentials are not provided, return HTTP 401 with WWW-Authenticate header
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization for user failed(1): Please provide username and password",
            headers={"WWW-Authenticate": "Basic"},
        )
    # Credentials are provided, continue with authentication logic
    for username in user_credentials.keys():
        if credentials.username == username:
            if credentials.password == user_credentials[username][0]:
                print(credentials.username, credentials.password)
                if user_credentials[username][1] == "user":
                    # Return the role if it matches 'user'
                    return True
    # If provided credentials don't match or role is not user, raise HTTP 401 with WWW-Authenticate header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authorization for user failed(2): Incorrect username or password",
        headers={"WWW-Authenticate": "Basic"},
    )


def authorize_administrator(credentials: HTTPBasicCredentials = Depends(security)):
    print("authorization check started")
    print(credentials.username, credentials.password)
    if (
        credentials is None
        or credentials.username is None
        or credentials.password is None
    ):
        # Credentials are not provided, return HTTP 401 with WWW-Authenticate header
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication for administrator failed(1): Please provide username and password",
            headers={"WWW-Authenticate": "Basic"},
        )
    # Credentials are provided, continue with authentication logic
    else:
        print("authorization check 2 started")
        for username in user_credentials.keys():
            if credentials.username == username:
                if credentials.password == user_credentials[username][0]:
                    print(credentials.username, credentials.password)
                    if user_credentials[username][1] == "administrator":
                        # Return the role if it matches 'administrator'
                        return True
        # If provided credentials don't match or role is not administrator, raise HTTP 401 with WWW-Authenticate header
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication for administrator failed(2): Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )


@app.get("/showdata20")
async def show_data20(
    credentials: HTTPBasicCredentials = Depends(authenticate_administrator),
):
    authorize_administrator(credentials)
    # Return the first five rows of the DataFrame as JSON
    return df.head(20).to_json(orient="records")


@app.get("/predict")
async def predict(
    credentials: HTTPBasicCredentials = Depends(authenticate_user),
    users_id: List[int] = Query(..., description="List of user IDs")
):
    authorize_user(credentials)  # Ensure user is authorized
    predictions = make_predictions(
        users_id, "models/model.pkl", 
        "data/processed/user_matrix.csv"
    )
    return {"predictions": predictions.tolist()}

