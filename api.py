from fastapi import FastAPI
import pandas as pd
import pickle
import numpy as np
import json

api = FastAPI(title="Movie Recommendation API")


def make_predictions(users_id, model_filename, user_matrix_filename):
    # Read user_matrix
    users = pd.read_csv(user_matrix_filename)

    # Filter with the list of users_id
    users = users[users["userId"].isin(users_id)]

    # Delete userId
    users = users.drop("userId", axis=1)

    # Open model
    filehandler = open(model_filename, "rb")
    model = pickle.load(filehandler)
    filehandler.close()

    # Calculate nearest neighbors
    _, indices = model.kneighbors(users)

    # Select 10 random numbers from each row
    selection = np.array(
        [np.random.choice(row, size=10, replace=False) for row in indices]
    )

    return selection


@api.get("/")
def get_index():
    return {"data": "The API is working fine."}


@api.get("/predict/{user_id}")
def get_predict(user_id):
    prediction = make_predictions([int(user_id)], "models/model.pkl", "data/processed/user_matrix.csv")
    return {"movies": json.dumps(prediction[0].tolist())}
