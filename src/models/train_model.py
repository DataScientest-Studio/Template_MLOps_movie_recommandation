from src.features.setup_project import LightDataModule, MovieLens
from src.features.MatrixFactorization import LightMF, MatrixFactorization
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from torchmetrics import MeanSquaredError


METRICS = MeanSquaredError()


def main(args):
    data = LightDataModule(dataset = MovieLens("ratings.dat"),
                           batch_size = args.batch_size)
    #print("data is instanciated. Launching setup...")
    data.setup()
    #print(f"num_users = {data.num_users} \n num_movies = {data.num_movies} \n embedding_dim = {args.embedding_dim}")
    model = LightMF(MatrixFactorization, 
                    sparse = False,
                    lr = args.lr,
                    metrics = METRICS,
                    num_users = data.num_users,
                    num_movies = data.num_movies,
                    embedding_dim = args.embedding_dim)
    #print("Model is instanciated.")
    #logger = TensorBoardLogger("lightning_logs", name = f"MatrixFactorization_{args.embedding_dim}")
    trainer = Trainer.from_argparse_args(args) # logger = logger
    #print("Fitting the model...")
    trainer.fit(model, data)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-e", "--embedding_dim", type = int, default = 20)
    parser.add_argument("-b", "--batch_size", type = int, default = 32)
    parser.add_argument("-l", "--lr", type = float, default = 0.001)
    Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    #print("Launching main function")
    main(args)


# In case you're wondering, the resulting model is saved in the lightning_logs folder.
