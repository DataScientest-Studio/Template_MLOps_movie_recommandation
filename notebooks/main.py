from argparse import ArgumentParser

import pytorch_lightning as pl

from setup import LightDataModule, MovieLens





def main(args):
    data = LightDataModule(dataset = MovieLens())

    print(data)

if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)