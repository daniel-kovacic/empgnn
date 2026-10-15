import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GraphNorm, MessagePassing

from .config import MergeMode
from .ops import BatchInfo


def activation(silu: bool) -> nn.Module:
    return nn.SiLU() if silu else nn.ReLU()


def mlp(in_dim: int, out_dim: int, silu: bool = False, hidden_dim: int | None = None) -> nn.Sequential:
    hidden = hidden_dim if hidden_dim is not None else max(in_dim, out_dim)
    return nn.Sequential(nn.Linear(in_dim, hidden), activation(silu), nn.Linear(hidden, out_dim))


class AffineFreeLayerNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, (self.dim,), eps=self.eps)


class Vnorm(nn.Module):

    def __init__(self, eps: float = 1.0):
        super().__init__()
        self.eps = eps

    def forward(self, z_dense: Tensor, C: int, d_e: int) -> Tensor:
        N = z_dense.shape[0]
        z_view = z_dense.reshape(N, C, d_e)
        norm = z_view.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(self.eps)
        return (z_view / norm).reshape(N, C * d_e)


class GINAgg(MessagePassing):

    def __init__(self, eps: float = 0.0):
        super().__init__(aggr="add")
        self.eps = nn.Parameter(torch.tensor(eps))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return (1.0 + self.eps) * x + self.propagate(edge_index, x=x)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j


class EGINLayer(nn.Module):
    def __init__(self, d_e: int, d_h: int, eps: float = 1.0, merge_mode: MergeMode = MergeMode.ADD):
        super().__init__()
        self.d_e = d_e
        self.d_h = d_h
        self.merge_mode = merge_mode

        self.gin = GINAgg(eps)
        self.norm = GraphNorm(d_e)
        self.phi = mlp(d_e, d_e)

        add = merge_mode == MergeMode.ADD
        self.mlp_equiv = mlp(d_e + (0 if add else d_h), d_e)
        self.linear_up = nn.Linear(d_h, d_e) if add else None
        self.mlp_inv = mlp(d_h + (0 if add else d_e), d_h)
        self.linear_down = nn.Linear(d_e, d_h) if add else None
        self.gin_h = GINAgg(eps)

    def forward(self, z: Tensor, h: Tensor, edge_index: Tensor, info: BatchInfo) -> tuple[Tensor, Tensor]:
        d_e, C, N = self.d_e, info.C, z.shape[0]

        agg_h = self.gin_h(h, edge_index)
        agg_z = self.gin(z, edge_index)

        agg_z = self.norm(agg_z.reshape(N * C, d_e), info.orig_graph).reshape(N, C * d_e)
        z_phi = self.phi(agg_z.reshape(N * C, d_e)).reshape(N, C * d_e)
        s_v = z_phi.reshape(N, C, d_e).mean(dim=1)

        agg_z_own = agg_z.reshape(N, C, d_e)
        if self.merge_mode == MergeMode.ADD:
            agg_z_own = agg_z_own + self.linear_up(h).unsqueeze(1)
            equiv_in = agg_z_own
        else:
            equiv_in = torch.cat([agg_z_own, h.unsqueeze(1).expand(-1, C, -1)], dim=-1)
        z_new = self.mlp_equiv(equiv_in.reshape(N * C, -1)).reshape(N, C * d_e)

        if self.merge_mode == MergeMode.ADD:
            inv_in = agg_h + self.linear_down(s_v)
        else:
            inv_in = torch.cat([agg_h, s_v], dim=-1)
        h_new = self.mlp_inv(inv_in)

        return z_new, h_new
