import pytorch_lightning as pl
from setup import LightDataModule, MovieLens






def main(args):
    data = LightDataModule(dataset = MovieLens())