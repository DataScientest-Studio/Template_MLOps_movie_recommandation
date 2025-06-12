import pytest
import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors
from src.models.train_model import train_model

def test_train_model():
    movie_matrix = pd.DataFrame({
        "movieId": [1, 2, 3],
        "Action": [1, 0, 0],
        "Comedy": [0, 1, 0],
        "Drama": [0, 0, 1],
    })
    model = train_model(movie_matrix)
    assert isinstance(model, NearestNeighbors)
    assert hasattr(model, "n_samples_fit_")

def test_model_saving(tmp_path):
    movie_matrix = pd.DataFrame({
        "movieId": [1, 2, 3],
        "Action": [1, 0, 0],
        "Comedy": [0, 1, 0],
        "Drama": [0, 0, 1],
    })
    model = train_model(movie_matrix)
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as filehandler:
        pickle.dump(model, filehandler)
    assert model_path.exists()
    with open(model_path, "rb") as filehandler:
        loaded_model = pickle.load(filehandler)
    assert isinstance(loaded_model, NearestNeighbors)
    assert loaded_model.n_neighbors == model.n_neighbors