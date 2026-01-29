import torch
import math
from torch import embedding, nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device = None, dtype = None):
        super().__init__()
        self.d_model = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))#只有浮点张量可以requires_grad
        std = math.sqrt(2.0/(num_embeddings + embedding_dim))
        nn.init.trunc_normal_(self.weight,0,std, -3*std, 3*std)

    def forward(self, token_ids):
        #token_ids = torch.LongTensor((self.d_model,self.vocab_size))
        token_embeddings = self.weight[token_ids]
        return token_embeddings        