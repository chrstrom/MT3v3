import torch
from torch import nn, Tensor
import math

class LearnedPositionEncoder(nn.Module):
    """
        Learned Position Encoder. Takes tensor of positional indicies and converts to learned embeddings 
    """

    def __init__(self, n_timesteps, d_model):
        super().__init__()
        self.embeddor = nn.Embedding(n_timesteps, d_model) # lookup table, each with vector of size d_model    
        nn.init.uniform_(self.embeddor.weight)

    def forward(self, pos_indicies):
        pos_indicies = pos_indicies.long()
        return self.embeddor(pos_indicies)

class Time2Vec(nn.Module):
    def __init__(self,params, omega=10000, scale=2*math.pi):
        super().__init__()
        self.params=params
        self.num_pos_feats = params.arch.d_model
        self.omega = omega
        self.scale = scale
        self.max_time = 100

    def forward(self, proposals):
        proposals = proposals + 1
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = self.omega ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        proposals = proposals / self.max_time * self.scale
        pos = proposals[:, :, None] / dim_t
        pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
        pos[:, :, 0] = proposals # Linear component

        return pos
