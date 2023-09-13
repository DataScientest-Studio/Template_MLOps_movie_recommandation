import torch.nn as nn
from src.features.setup_project import LightModel

import torch.nn.functional as F


class MatrixFactorization(nn.Module):
    def __init__(self, embedding_dim, num_users, num_movies,
                 sparse = False, **kwargs):
        super().__init__()
        self.sparse = sparse

        self.user = nn.Embedding(num_embeddings = num_users, 
                                 embedding_dim = embedding_dim,
                                 sparse = sparse)
        self.user_bias = nn.Embedding(num_embeddings = num_users,
                                      embedding_dim = 1,
                                      sparse = sparse)
        
        self.movie = nn.Embedding(num_embeddings = num_movies, 
                                 embedding_dim = embedding_dim,
                                 sparse = sparse)
        self.movie_bias = nn.Embedding(num_embeddings = num_movies,
                                      embedding_dim = 1,
                                      sparse = sparse)
        
        # Initialize every parameter randomly.
        for param in self.parameters():
            nn.init.normal_(param, std = 0.01)

    def forward(self, user_id, movie_id):
        """
        The predicted rating user u gives to rating i is defined by:
        R_pred_ui = P.Qt + bu_u + bi_i

        Where R_pred is the complete predicted matrix,
        P is the user matrix, Q is the movie matrix, 
        bu and bi are the user and movie bias vectors respectively.

        To get the specific prediction, we compute the dot product between
        P and Qt (transpose of Q), to which we add the corresponding biases.


        """
        #print("\nTrying a forward pass...\n")
        P_u = self.user(user_id)
        #print(f"P_u shape: {P_u.shape}")
        bias_user_u = self.user_bias(user_id).flatten()

        Q_i = self.movie(movie_id)
        #print(f"Q_i shape: {Q_i.shape}")
        bias_movie_i = self.movie_bias(movie_id).flatten()

        return (P_u*Q_i).sum(-1) + bias_user_u + bias_movie_i

class LightMF(LightModel):
    def get_loss(self, model_outputs, batch):
        #print("Loss \n")
        return F.mse_loss(model_outputs, batch[-1])
    
    def update_metric(self, model_outputs, batch):
        #print("Update metric \n")
        _, _, true_rating = batch
        self.metrics.update(model_outputs, true_rating)
    
    def forward(self, batch):
        print(f"batch from forward call:\n{batch}")
        users, movies, _ = batch

        #print(f"Users: {users} \n Users shape: {users.shape}")
        #print(f"Movies: {movies} \n Movies shape: {movies.shape}")

        return self.model(users, movies)