import unittest


from src.models.predict_model import make_predictions


class TestPrediction(unittest.TestCase):
    def test_prediction(self):
        predictions = make_predictions(
            users_ids=[
                1,
            ],
            model_filename="models/model.pkl",
            user_matrix_filename="data/processed/user_matrix.csv",
        )
        self.assertTrue(len(list(predictions)), 1)
