import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch import einsum
from .utils import *

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, kdim, qdim, vdim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        dim = qdim
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(kdim, inner_dim, bias = False)
        self.to_v = nn.Linear(vdim, inner_dim, bias = False)
        self.to_q = nn.Linear(qdim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k=None, v=None):
        if k == None or v == None:
            k,v = q,q
        b, n, _, h = *q.shape, self.heads
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        kqv = self.to_k(k), self.to_q(q), self.to_v(v)
        k, q, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), kqv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(kdim=dim, qdim=dim, vdim=dim, heads = heads, dim_head = dim_head, dropout = 0.))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k):
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        
    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz) # b x n x n (self correlation result)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k (k nearest neighbors)
        knn_xyz = index_points(xyz, knn_idx) # b x n x k x 3 (n points, each point with its k nearest neighbors)
        
        pre = features
        x = self.fc1(features) # b x n x f -> b x n x d_model (f = d_points)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
        # q: b x n x d_model, k: b x n x k x d_model, v: b x n x k x d_model
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x 3 -> b x n x k x d_model
        
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc) # b x n x k x d_model
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x d_model
        
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre # b x n x d_model -> b x n x f
        return res, attn


class TransitionDown(nn.Module):
    def __init__(self, in_channel, internal_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channel, internal_channel, 1),
            nn.BatchNorm1d(internal_channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(internal_channel, out_channel, 1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )
        
    def forward(self, xyz, features):
        """
        Input:
            xyz: input points position data, [B, N, 3]
            points: input points data, [B, N, D]
        Return:
            new_points_concat: sample points feature data, [B, N, 3+D]
        """
        new_features = torch.cat([xyz, features], dim=-1) # b x n x (3 + features)
        new_features = new_features.permute(0, 2, 1) # [B, 3+D, n]
        new_features = self.conv1(new_features) # [B, internal_channel, n]
        new_features = self.conv2(new_features) # [B, out_channel, n]
        new_features = new_features.permute(0, 2, 1) # [B, n, out_channel]
        return new_features


class Backbone(nn.Module):
    def __init__(self, nblocks, nneighbor, input_dim, transformer_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, transformer_dim, nneighbor) # input channels = 32
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        append_blocks = nblocks - 1
        for i in range(append_blocks):
            channel = 32 * 2 ** (i+1)
            self.transition_downs.append(TransitionDown(channel // 2 + 3, channel, channel))
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))
        self.append_blocks = append_blocks
    
    def forward(self, x):
        # x: (B, N, 5). 5 = (x, y, z, d, l)
        xyz = x[..., :3]
        # print(f"xyz: {xyz.shape}")
        # print(f"x: {x.shape}")
        points = self.transformer1(xyz, self.fc1(x))[0]
        # print(f"points: {points.shape}")
        xyz_and_feats = [(xyz, points)]
        for i in range(self.append_blocks):
            points = self.transition_downs[i](xyz, points)
            # print(f"points: {points.shape}")
            points = self.transformers[i](xyz, points)[0]
            # print(f"points: {points.shape}")
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats
