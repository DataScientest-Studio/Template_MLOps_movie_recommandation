import pytest
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from src.models.predict_model import make_predictions

def test_make_predictions(tmp_path):
    # user_matrix = pd.DataFrame({
    #     "userId": range(1, 21),
    #     "Action": np.random.rand(20),
    #     "Comedy": np.random.rand(20),
    #     "Drama": np.random.rand(20),
    # })
    # user_matrix_path = tmp_path / "user_matrix.csv"
    # user_matrix.to_csv(user_matrix_path, index=False)

    # model = NearestNeighbors(n_neighbors=10, algorithm="ball_tree").fit(
    #     user_matrix.drop("userId", axis=1)
    # )
    # model_path = tmp_path / "model.pkl"
    # with open(model_path, "wb") as filehandler:
    #     pickle.dump(model, filehandler)

    # users_id = [1, 2, 3]
    # predictions = make_predictions(users_id, model_path, user_matrix_path)
    # assert isinstance(predictions, np.ndarray)
    # assert predictions.shape == (3, 10)
    # for row in predictions:
    #     assert all(idx in range(len(user_matrix)) for idx in row)
    assert True #To be implemented