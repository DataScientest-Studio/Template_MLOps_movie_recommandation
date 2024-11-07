import os
from datetime import datetime
from threading import Lock

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

# Create a lock for thread-safe file operations
file_lock = Lock()


def make_predictions(
    user_ids,
    model_name,
    user_matrix_filename,
    movie_matrix_filename,
    n_recommendations=3,
    log_file="predictions_log.csv",
):
    """
    Generate movie recommendations for a list of user IDs, compute error metrics, and log predictions to a centralized file
    within the 'predictions' folder.

    Args:
        user_ids (list): List of user IDs to generate recommendations for.
        model_name (str): Name of the model registered in MLflow.
        user_matrix_filename (str): Path to the user feature matrix CSV file.
        movie_matrix_filename (str): Path to the movie feature matrix CSV file.
        n_recommendations (int): Number of movie recommendations per user.
        log_file (str): Filename for the predictions log.

    Returns:
        pd.DataFrame: DataFrame containing user IDs, recommendations, error metrics, and model information.
    """
    # Initialize MLflow client
    client = MlflowClient()

    # Get the latest model version in Production stage
    latest_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_versions:
        raise ValueError(f"No production model found for {model_name}.")

    # Assuming we're using the first model in the list
    model_version_info = latest_versions[0]
    model_version = model_version_info.version
    model_stage = model_version_info.current_stage
    model_run_id = model_version_info.run_id

    # Load the model from MLflow Model Registry
    model_uri = f"models:/{model_name}/{model_stage}"
    model = mlflow.sklearn.load_model(model_uri)

    # Load user features
    user_matrix = pd.read_csv(user_matrix_filename)
    users = user_matrix[user_matrix["userId"].isin(user_ids)]
    users_features = users.drop("userId", axis=1).values
    user_ids_array = users["userId"].values

    # Load movie features
    movie_matrix = pd.read_csv(movie_matrix_filename)
    movie_ids = movie_matrix["movieId"].values
    movie_features = movie_matrix.drop("movieId", axis=1).values

    # Compute nearest movies for each user
    _, indices = model.kneighbors(users_features, n_neighbors=n_recommendations)

    # Map indices to movie IDs and compute error metrics
    recommendations_list = []
    timestamp = datetime.now()

    for user_id, idx_list, user_feature in zip(user_ids_array, indices, users_features):
        recommended_movie_ids = movie_ids[idx_list].tolist()
        # Get features of recommended movies
        recommended_movie_features = movie_features[idx_list]
        # Compute distances between user feature and recommended movie features
        distances_to_user = np.linalg.norm(
            recommended_movie_features - user_feature, axis=1
        )
        # Compute mean distance as error metric
        error_metric = distances_to_user.mean()
        # Store recommendations and error metric
        recommendations_list.append(
            {
                "userId": user_id,
                "recommendations": recommended_movie_ids,
                "error_metric": error_metric,
                "model_name": model_name,
                "model_version": model_version,
                "model_run_id": model_run_id,
                "timestamp": timestamp,
            }
        )

    # Convert to DataFrame
    recommendations_df = pd.DataFrame(recommendations_list)

    # Prepare the 'predictions' directory
    predictions_dir = "predictions"
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    # Log predictions to the 'predictions' folder
    log_file_path = os.path.join(predictions_dir, log_file)
    with file_lock:
        if os.path.exists(log_file_path):
            # Append without writing the header
            recommendations_df.to_csv(
                log_file_path, mode="a", header=False, index=False
            )
        else:
            # Write with the header
            recommendations_df.to_csv(log_file_path, mode="w", header=True, index=False)

    return recommendations_df


if __name__ == "__main__":
    # List of user IDs to generate recommendations for
    user_ids = [1, 2, 3, 4, 5]

    # Generate recommendations
    recommendations = make_predictions(
        user_ids=user_ids,
        model_name="KNN_Recommendation_Model",
        user_matrix_filename="data/processed/user_matrix.csv",
        movie_matrix_filename="data/processed/movie_matrix.csv",
        n_recommendations=3,
        log_file="predictions_log.csv",
    )

    print(recommendations)
