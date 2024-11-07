import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, ParameterGrid
import mlflow
from mlflow import MlflowClient

# The Mlflow need to be running on port 5000
# mlflow server --host 0.0.0.0 --port 5000
# according to the docker-compose 

def train_and_register_model():
    # Load data
    movie_matrix = pd.read_csv("./data/processed/movie_matrix.csv")
    user_matrix = pd.read_csv("./data/processed/user_matrix.csv")

    # Store IDs separately
    movie_ids = movie_matrix['movieId']
    user_ids = user_matrix['userId']

    # Drop IDs from feature matrices
    X_movies = movie_matrix.drop("movieId", axis=1)
    X_users = user_matrix.drop("userId", axis=1)

    # Split users into train and test sets
    X_train, X_test, user_train_ids, user_test_ids = train_test_split(
        X_users, user_ids, test_size=0.2, random_state=42
    )

    # Set MLflow experiment
    mlflow.set_experiment("Movie Recommendation Model")

    # Define hyperparameter grid
    param_grid = {
        'n_neighbors': [3, 5, 7, 10],
        'metric': ['cosine', 'euclidean']
    }

    # Initialize variables to track the best model
    best_mean_distance = float('inf')
    best_params = None
    best_model = None

    # Initialize MLflow client
    client = MlflowClient()

    # Start hyperparameter tuning
    for params in ParameterGrid(param_grid):
        with mlflow.start_run(run_name=f"KNN_n{params['n_neighbors']}_metric_{params['metric']}") as run:
            # Log parameters
            mlflow.log_params(params)

            # Define and train the KNN model
            knn_model = NearestNeighbors(
                n_neighbors=params['n_neighbors'],
                metric=params['metric'],
                algorithm='brute'
            )
            knn_model.fit(X_movies)

            # Evaluate the model
            distances, _ = knn_model.kneighbors(X_test, n_neighbors=params['n_neighbors'])
            mean_distance = np.mean(distances)
            mlflow.log_metric('mean_distance', mean_distance)

            # Log the model
            mlflow.sklearn.log_model(knn_model, artifact_path='model')

            # Update the best model if current is better
            if mean_distance < best_mean_distance:
                best_mean_distance = mean_distance
                best_params = params
                best_model = knn_model
                best_run_id = run.info.run_id

    # After tuning, register the best modele
    if best_model is not None:
        # Register the best modele
        model_name = "KNN_Recommendation_Model"
        model_uri = f"runs:/{best_run_id}/model"
        mlflow.register_model(model_uri=model_uri, name=model_name)

        # Transition the best modele to Production stage
        # Get the latest version of the registered modele
        latest_versions = client.get_latest_versions(model_name, stages=['None'])
        if latest_versions:
            version_to_stage = latest_versions[0].version
            client.transition_model_version_stage(
                name=model_name,
                version=version_to_stage,
                stage='Production',
                archive_existing_versions=True
            )
            print(f"Model version {version_to_stage} of '{model_name}' registered and moved to 'Production' stage.")
        else:
            print("No new model version found to transition to 'Production'.")

        print(f"Best Model Parameters: {best_params}")
        print(f"Best Mean Distance: {best_mean_distance}")
    else:
        print("No model was trained.")

if __name__ == "__main__":
    train_and_register_model()
