import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
import os
import json
import argparse

default_hyperparameters = {
    "n_neighbors": 20,
    "algorithm": "ball_tree"
}

def train_model(movie_matrix, hyperparams):
    nbrs = NearestNeighbors(**hyperparams).fit(
        movie_matrix.drop("movieId", axis=1)
    )
    return nbrs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hyperparams_dict",
        type=str,
        default=json.dumps(default_hyperparameters),
        help="Hyperparameters as JSON string, e.g., '{\"n_neighbors\": 15, \"algorithm\": \"kd_tree\"}'"
        )
    args = parser.parse_args()
    try:
        hyperparams = json.loads(args.hyperparams_dict)
    except json.JSONDecodeError:
        print("Error: hyperparams_dict must be a valid JSON string.")
        exit(1)

    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)

    # Load the movie matrix
    movie_matrix = pd.read_csv("data/processed/movie_matrix.csv")

    # Train the model
    model = train_model(movie_matrix, hyperparams)

    # Save the model to a file
    filehandler = open("models/model.pkl", "wb")
    pickle.dump(model, filehandler)
    filehandler.close()