import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch import einsum
from .transformer_helper import Backbone, Transformer


class PointTransformer(nn.Module):
    def __init__(self, input_dim = 5, nblocks = 5, nneighbor = 16, transformer_dim = 128, n_p = 17):
        super().__init__()
        self.backbone = Backbone(
            nblocks = nblocks,
            nneighbor = nneighbor,
            input_dim = input_dim,
            transformer_dim = transformer_dim
        )
        
        self.nblocks = nblocks
        self.n_p = n_p

        dim, depth, heads, dim_head, mlp_dim, dropout = 512, 5, 4, 128, 256, 0.0
        self.joint_posembeds_vector = nn.Parameter(torch.tensor(self.get_positional_embeddings1(self.n_p, dim)).float())
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout)

        mid_dim = 32
        self.fc2 = nn.Sequential(
            nn.Linear(dim, 256), nn.Dropout(dropout),
            nn.ReLU(), nn.Linear(256, mid_dim),
        )
        self.fc3 = nn.Sequential(
            nn.ReLU(), nn.Linear(mid_dim, 64), nn.Dropout(dropout),
            nn.ReLU(), nn.Linear(64, 3),
        )
    
    def forward(self, x):
        if len(x.shape) == 4: 
            b, t, n, c = x.shape
            x = x.view(b, t*n, c)
        else:
            b, n, c = x.shape
        points, _ = self.backbone(x)
        joint_embedding = self.joint_posembeds_vector.expand(b, -1, -1)
        embedding = torch.cat([joint_embedding, points], dim=1)
        output = self.transformer(embedding)[:, :self.n_p, :]

        feat = self.fc2(output)
        pts = self.fc3(feat)
        
        return pts, feat
    
    def get_positional_embeddings1(self, sequence_length, d):
        result = np.ones([1, sequence_length, d])
        for i in range(sequence_length):
            for j in range(d):
                result[0][i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return result
