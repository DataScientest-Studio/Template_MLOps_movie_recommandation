import os
import fire

from typing import Final

from src.utils.logger import logger
from src.data.import_raw_data import import_raw_data
from src.data.process_raw import process_raw
from src.models.predict_model import make_predictions

DATASET_DIR: Final[str] = os.path.join("data", "dataset")
DATA_RAW_FILENAMES: Final[list[str]] = [
    "genome-scores.csv",
    "genome-tags.csv",
    "links.csv",
    "movies.csv",
    "ratings.csv",
    "README.txt",
    "tags.csv",
]
DATA_BUCKET_URL: Final[str] = (
    "https://mlops-project-db.s3.eu-west-1.amazonaws.com/movie_recommandation/"
)
MODEL_PATH: str = "models/model.pkl"
USER_MATRIX_PATH: str = "data/dataset/processed/user_matrix.csv"


class RecoApp:
    def dataset(self):
        logger.info("Import raw data from s3...")
        raw_dir = import_raw_data(
            dataset_dir=DATASET_DIR,
            filenames=DATA_RAW_FILENAMES,
            bucket_object_url=DATA_BUCKET_URL,
        )
        logger.info("Process raw data ...")
        process_raw(raw_dir=raw_dir, dataset_dir=DATASET_DIR)

    def predict(self, users: list[int] = [1, 2, 3, 4, 5], model_path: str = MODEL_PATH):
        predictions = make_predictions(users, model_path, USER_MATRIX_PATH)
        logger.info(f"Predictions: {predictions}")

    def mdp(self, mdp: str = "x"):
        import hashlib

        result = hashlib.md5(mdp.encode())
        print("md5 for mdp", mdp, ":", result.hexdigest())


if __name__ == "__main__":
    fire.Fire(RecoApp)
