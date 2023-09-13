from src.features.MatrixFactorization import LightMF
from src.features.setup_project import read_ratings
import torch 


model = LightMF.load_from_checkpoint(checkpoint_path="lightning_logs/version_6/checkpoints/trained_model_75.ckpt") #path to the checkpoint

df = read_ratings("ratings.dat", data_dir = "data/processed")


print(df.head(15))

user = 0

item = 1104


scores = model(torch.tensor([user, item, 5]))

print(scores)