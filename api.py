import hashlib
from typing import Annotated


from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import datetime


from src.models.predict_model import make_predictions


# API
api = FastAPI(
    title="Film reco",
    description="Todo.",
    version="0.0.1",
    openapi_tags=[
        {
            "name": "test",
            "description": "test API is running",
        },
        {"name": "inference", "description": "todo"},
    ],
)

# Security
security = HTTPBasic()

# User DB
# - username: (user_id, md5(mdp))
USER_DB: dict[str, tuple[int, str]] = {
    "alice": (1, "9dd4e461268c8034f5c8564e155c67a6"),
    "bob": (2, "9dd4e461268c8034f5c8564e155c67a6"),
}


class BadCredentialException(Exception):
    def __init__(self, name: str, date: str, message: str):
        self.name = name
        self.date = date
        self.message = message


@api.exception_handler(BadCredentialException)
def QCMExceptionHandler(request: Request, exception: BadCredentialException):
    return JSONResponse(
        status_code=418,
        content={
            "url": str(request.url),
            "name": exception.name,
            "message": exception.message,
            "date": exception.date,
        },
    )


# Utils
def manage_authentication(credentials) -> tuple:
    """Return user_id and user_name if the credentials are corrects, raise an error otherwise.

    Args:
        credentials (_type_): credentials

    Raises:
        BadCredentialException: bad cred error

    Returns:
        tuple: user_id and user_name
    """
    # Compute hash password from the API
    str_clean = credentials.password
    hash_password = hashlib.md5(str_clean.encode()).hexdigest()

    # Retrieve user in DB and compare API hash to user hash password in DB
    if (USER_DB.get(credentials.username) is None) or (
        USER_DB.get(credentials.username)[1] != hash_password
    ):
        raise BadCredentialException(
            name="bad_creds",
            message="username ou mot de passe incorrect.",
            date=str(datetime.datetime.now()),
        )
    return USER_DB.get(credentials.username)[0], credentials.username


# Routes
# - Hello
@api.get("/hello", name="Hello", tags=["test"])
def get_hello_secure(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
) -> dict:
    _, username = manage_authentication(credentials=credentials)
    return {"message": f"Hello {username}!"}


# - User recommendation
@api.get("/recommend", name="Recommend a list of movies", tags=["inference"])
def get_recommend_user(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
) -> dict:
    """Provides an user recommendation

    Returns:
        dict: recommendation the user
    """
    user_id, username = manage_authentication(credentials=credentials)
    predictions = make_predictions(
        users_ids=[
            user_id,
        ],
        model_filename="models/model.pkl",  # Best model version
        user_matrix_filename="data/processed/user_matrix.csv",
    )
    return {
        "recommend": predictions.tolist(),
        "user_name": username,
    }
