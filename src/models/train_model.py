import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle


def train_model(movie_matrix):
    nbrs = NearestNeighbors(n_neighbors=20, algorithm="ball_tree").fit(
        movie_matrix.drop("movieId", axis=1)
    )
    return nbrs


if __name__ == "__main__":
    movie_matrix = pd.read_csv("data/processed/movie_matrix.csv")
    model = train_model(movie_matrix)
    filehandler = open("models/model.pkl", "wb")
    pickle.dump(model, filehandler)
    filehandler.close()
