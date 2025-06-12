import pandas as pd
import pickle
import numpy as np
import json
import os
import argparse
from typing import List, Dict, Union

def load_user_data(user_matrix: Union[str, pd.DataFrame], users_id: List[int] = None) -> pd.DataFrame:
    """
    Loads and filters user data from a file or DataFrame.

    Args:
        user_matrix: Path to a CSV file or a DataFrame containing user data.
        users_id: List of user IDs to filter. If None, returns all users.

    Returns:
        Filtered DataFrame containing only the specified users (or all users if users_id is None).
    """
    if isinstance(user_matrix, str):
        user_matrix = pd.read_csv(user_matrix)
    if users_id is not None:
        user_matrix = user_matrix[user_matrix["userId"].isin(users_id)]
    return user_matrix

def load_model(model_filename: str):
    """
    Loads a model from a pickle file.

    Args:
        model_filename: Path to the pickle file containing the model.

    Returns:
        Loaded model.
    """
    with open(model_filename, "rb") as filehandler:
        model = pickle.load(filehandler)
    return model

def make_predictions(model, user_data: pd.DataFrame, n_recos: int = 10) -> Dict[int, List[int]]:
    """
    Generates recommendations for users.

    Args:
        model: Trained model for generating recommendations.
        user_data: DataFrame containing user data (excluding the userId column).
        n_recos: Number of recommendations to generate per user.

    Returns:
        Dictionary of recommendations {user_id: [movie_index1, movie_index2, ...]}.
    """
    original_ids = user_data["userId"].values
    features = user_data.drop("userId", axis=1)
    _, indices = model.kneighbors(features)
    selection = indices[:, :n_recos]
    prediction_dict = {
        int(user_id): list(map(int, movie_indices))
        for user_id, movie_indices in zip(original_ids, selection)
    }
    return prediction_dict

def save_predictions_to_file(predictions: Dict[int, List[int]], output_path: str):
    """
    Saves predictions to a JSON file.

    Args:
        predictions: Dictionary of recommendations {user_id: [movie_index1, movie_index2, ...]}.
        output_path: Path to the output file where predictions will be saved.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"✅ Predictions saved to {output_path}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict movies for users.")
    parser.add_argument("--user_ids", type=str, help="Comma-separated list of user IDs (e.g., '1,2,3').")
    parser.add_argument("--n_recommendations", type=int, default=10, help="Number of recommendations to generate per user.")
    parser.add_argument("--no_save_to_file", action="store_true", help="Flag to disable saving predictions to a file.")
    args = parser.parse_args()

    # Load user data
    if args.user_ids:
        users_id = list(map(int, args.user_ids.split(",")))
    else:
        # Default behavior: load all users
        user_matrix_path = "data/processed/user_matrix.csv"
        print(f"⚙️ Loading all user IDs from {user_matrix_path}...")
        all_users = pd.read_csv(user_matrix_path)
        users_id = all_users["userId"].tolist()

    # Load user data and model
    user_data = load_user_data("data/processed/user_matrix.csv", users_id)
    model = load_model("models/model.pkl")

    # Generate predictions
    predictions = make_predictions(model, user_data, n_recos=args.n_recommendations)

    # Save predictions to file by default unless --no_save_to_file is provided
    if not args.no_save_to_file:
        save_predictions_to_file(predictions, "data/prediction/predictions.json")

    # Print predictions
    print(predictions)