import torch.nn as nn
from setup import LightModel

import torch.nn.functional as F


class MatrixFactorization(nn.Module):
    def __init__(self, embedding_dim, num_users, num_items,
                 sparse = False, **kwargs):
        super().__init__()
        self.sparse = sparse

        self.user = nn.Embedding(num_embeddings = num_users, 
                                 embedding_dim = embedding_dim,
                                 sparse = sparse)
        self.user_bias = nn.Embedding(num_embeddings = num_users,
                                      embedding_dim = 1,
                                      sparse = sparse)
        
        self.item = nn.Embedding(num_embeddings = num_items, 
                                 embedding_dim = embedding_dim,
                                 sparse = sparse)
        self.item_bias = nn.Embedding(num_embeddings = num_items,
                                      embedding_dim = 1,
                                      sparse = sparse)
        
        # Initialize every parameter randomly.
        for param in self.parameters():
            nn.init.normal_(param, std = 0.01)

    def forward(self, user_id, item_id):
        """
        The predicted rating user u gives to rating i is defined by:
        R_pred_ui = P.Qt + bu_u + bi_i

        Where R_pred is the complete predicted matrix,
        P is the user matrix, Q is the item matrix, 
        bu and bi are the user and item bias vectors respectively.

        To get the specific prediction, we compute the dot product between
        P and Qt (transpose of Q), to which we add the corresponding biases.


        """
        P_u = self.user(user_id)
        bias_user_u = self.user_bias(user_id).flatten()

        Q_i = self.item(item_id)
        bias_item_i = self.item_bias(item_id).flatten()

        return (P_u*Q_i).sum(-1) + bias_user_u + bias_item_i

class LightMF(LightModel):
    def get_loss(self, model_outputs, batch):
        return F.mse_loss(model_outputs, batch[-1])
    
    def update_metric(self, model_outputs, batch):
        _, _, true_rating = batch
        self.metrics.update(model_outputs, true_rating)
    
    def forward(self, batch):
        users, items, _ = batch
        return self.model(users, items)