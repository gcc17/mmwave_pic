import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch import einsum
from .transformer_helper import Backbone, Transformer
from torch.utils.data import DataLoader
from tqdm import tqdm

class PointTransformer(nn.Module):
    def __init__(self, device, input_dim = 5, nblocks = 5, nneighbor = 16, transformer_dim = 128, n_p = 17):
        super().__init__()
        self.device = device
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

        self.feat_dim = 32
        self.fc2 = nn.Sequential(
            nn.Linear(dim, 256), nn.Dropout(dropout),
            nn.ReLU(), nn.Linear(256, self.feat_dim),
        )
        self.fc3 = nn.Sequential(
            nn.ReLU(), nn.Linear(self.feat_dim, 64), nn.Dropout(dropout),
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
        # print(embedding.shape, self.transformer(embedding).shape) # torch.Size([16, 77, 512]) torch.Size([16, 77, 512]) for mmfi
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

    def get_feature_embeddings(self, dataset, batch_size=128):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        feature_embeddings = []
        with torch.no_grad():
            for x, _ in tqdm(dataloader):
                x = x.type(torch.FloatTensor).to(self.device)
                _, feat = self.forward(x)
                # average along the -2 dimension (keypoints cnt)
                feat = feat.mean(dim=-2)
                feature_embeddings.append(feat)
        feature_embeddings = torch.cat(feature_embeddings, dim=0)
        return feature_embeddings