import pytest
import pandas as pd
import os
from src.models.predict_model import make_predictions
import numpy as np

def test_make_predictions():
    # Define the test log file
    test_log_file = "test_predictions_log.csv"
    
    try:
        # Generate predictions for user ID 1
        predictions_df = make_predictions(
            user_ids=[1],
            model_name="KNN_Recommendation_Model",
            user_matrix_filename="data/processed/user_matrix.csv",
            movie_matrix_filename="data/processed/movie_matrix.csv",
            n_recommendations=3,
            log_file=test_log_file  # Use the test log file
        )
        
        # Check that the returned DataFrame is not empty
        assert not predictions_df.empty, "The predictions DataFrame is empty."
        
        # Check that the DataFrame has the expected columns
        expected_columns = {
            'userId', 'recommendations', 'error_metric', 
            'model_name', 'model_version', 'model_run_id', 'timestamp'
        }
        assert expected_columns.issubset(predictions_df.columns), (
            f"The DataFrame columns are missing expected columns. "
            f"Found columns: {predictions_df.columns}"
        )
        
        # Check that the userId matches the input
        assert predictions_df['userId'].iloc[0] == 1, (
            "The userId in the predictions does not match the input."
        )
        
        # Check recommendations
        assert len(predictions_df['recommendations'].iloc[0]) > 0, (
            "No recommendations were made for the user."
        )
        
        # Check metric erreur
        error_metric = predictions_df['error_metric'].iloc[0]
        assert isinstance(error_metric, (int, float, np.float64, np.float32)), (
            "Error metric is not a number."
        )
        assert error_metric >= 0, "Error metric should be a non-negative number."
          
    finally:
        # Nettoyage
        predictions_dir = 'predictions'
        test_log_path = os.path.join(predictions_dir, test_log_file)
        if os.path.exists(test_log_path):
            os.remove(test_log_path)
            # On nettoie meme le dossier si il n'existait pas
            if not os.listdir(predictions_dir):
                os.rmdir(predictions_dir)
