import math

import torch
import torch.nn as nn
from torch import Tensor

from .layers import AffineFreeLayerNorm, Vnorm, activation
from .ops import BatchInfo, scatter_sum


class PointwiseConv1d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 1))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.einsum("oi,nic->noc", self.weight.squeeze(-1), x)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1)
        return out


class PureENGNNConv(nn.Module):

    def __init__(self, d_h: int, d_e: int, mid_ratio: int, silu: bool = True):
        super().__init__()
        self.d_e = d_e
        middim = mid_ratio * d_h
        midL = mid_ratio * d_e
        act = activation(silu)

        self.convmlp1 = nn.Sequential(nn.Linear(d_h, middim), act)
        self.lincoeff = nn.Linear(d_h, midL)
        self.convlinZ = PointwiseConv1d(d_e, midL, bias=False)
        self.linXout = nn.Sequential(nn.Linear(middim, d_h))
        self.linZout = PointwiseConv1d(midL, d_e, bias=False)

    def forward(self, X: Tensor, Z: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        N = X.shape[0]
        src, dst = edge_index[0], edge_index[1]
        Zt = Z.transpose(1, 2)

        Xmsg = self.convmlp1(X).index_select(0, src)
        aggredX = scatter_sum(Xmsg, dst, N)

        coeffZ = self.lincoeff(X)
        convZ = self.convlinZ(Zt)
        Zmsg = coeffZ.index_select(0, src).unsqueeze(-1) * convZ.index_select(0, src)
        E, midL_, C_ = Zmsg.shape
        aggredZ = scatter_sum(Zmsg.reshape(E, midL_ * C_), dst, N).reshape(N, midL_, C_)

        retX = self.linXout(aggredX)
        retZt = self.linZout(aggredZ)
        retZ = retZt.transpose(1, 2)
        return retX, retZ


class ENGNNConv(nn.Module):

    def __init__(self, d_h: int, d_e: int, mid_ratio: int, silu: bool = True):
        super().__init__()
        self.d_e = d_e
        coordL = mid_ratio * d_e
        midL = mid_ratio * d_e
        middim = mid_ratio * d_h
        act = activation(silu)
        self.act = act

        self.coordlin1 = nn.Sequential(PointwiseConv1d(d_e, coordL, bias=True), act)
        self.coordlin2 = nn.Sequential(PointwiseConv1d(d_e, coordL, bias=True), act)
        self.convmlp1 = nn.Sequential(nn.Linear(d_h, middim))
        self.convmlp2 = nn.Sequential(nn.Linear(3 * coordL, middim, bias=False))
        self.noisegate = nn.Parameter(torch.zeros(middim), requires_grad=True)
        self.linXout = nn.Sequential(nn.Linear(middim, d_h), act)

        self.lincoeff1 = nn.Linear(middim, midL)
        self.lincoeff2 = nn.Linear(middim, midL, bias=False)
        self.lincoeff3 = nn.Linear(middim, midL, bias=False)
        self.lincoeff4 = nn.Sequential(act, nn.Linear(midL, midL))

        self.convlinZ = nn.Sequential(PointwiseConv1d(d_e, midL, bias=True), act)
        self.linZout = nn.Sequential(PointwiseConv1d(midL, d_e, bias=True))

    def forward(self, X: Tensor, Z: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        N = X.shape[0]
        src, dst = edge_index[0], edge_index[1]
        Zt = Z.transpose(1, 2)

        srcZ = self.coordlin1(Zt)
        dstZ = self.coordlin2(Zt)
        srcZ_src, dstZ_src, srcZ_dst, dstZ_dst = (
            srcZ.index_select(0, src), dstZ.index_select(0, src),
            srcZ.index_select(0, dst), dstZ.index_select(0, dst),
        )
        Zprod1 = (srcZ_src * dstZ_dst).sum(dim=-1)
        Zprod2 = (srcZ_src * dstZ_src).sum(dim=-1)
        Zprod3 = (srcZ_dst * dstZ_dst).sum(dim=-1)
        Zprod = torch.cat((Zprod1, Zprod2, Zprod3), dim=-1)

        Xmsg = self.act(self.convmlp1(X).index_select(0, src) + self.noisegate * self.convmlp2(Zprod))
        aggredX = scatter_sum(Xmsg, dst, N)

        coeffZ = self.lincoeff4(
            self.lincoeff1(aggredX).index_select(0, dst) + self.lincoeff2(Xmsg)
            + self.lincoeff3(aggredX).index_select(0, dst)
        )
        convZ = self.convlinZ(Zt)
        Zmsg = coeffZ.unsqueeze(-1) * convZ.index_select(0, src)
        E, midL_, C_ = Zmsg.shape
        aggredZ = scatter_sum(Zmsg.reshape(E, midL_ * C_), dst, N).reshape(N, midL_, C_)

        retX = self.linXout(aggredX)
        retZt = self.linZout(aggredZ)
        retZ = retZt.transpose(1, 2)
        return retX, retZ


class ENGNNComb(nn.Module):

    def __init__(self, d_h: int, d_e: int, mid_ratio: int, silu: bool = True):
        super().__init__()
        self.d_e = d_e
        coordL = mid_ratio * d_e
        midL = mid_ratio * d_e
        middim = mid_ratio * d_h
        act = activation(silu)
        self.act = act

        self.coordlin1 = nn.Sequential(PointwiseConv1d(2 * d_e, coordL, bias=True), act)
        self.coordlin2 = nn.Sequential(PointwiseConv1d(2 * d_e, coordL, bias=True), act)
        self.convmlp1 = nn.Sequential(nn.Linear(2 * d_h, middim))
        self.convmlp2 = nn.Sequential(nn.Linear(coordL, middim, bias=False))
        self.noisegate = nn.Parameter(torch.zeros(middim), requires_grad=True)
        self.linXout = nn.Sequential(nn.Linear(middim, d_h), act)

        self.lincoeff1 = nn.Linear(middim, midL)
        self.convlinZ = nn.Sequential(PointwiseConv1d(2 * d_e, midL, bias=True), act)
        self.linZout = nn.Sequential(PointwiseConv1d(midL, d_e, bias=True))

    def forward(self, X1: Tensor, X2: Tensor, Z1: Tensor, Z2: Tensor) -> tuple[Tensor, Tensor]:
        Z1t = Z1.transpose(1, 2)
        Z2t = Z2.transpose(1, 2)
        Zt = torch.cat((Z1t, Z2t), dim=-2)
        X = torch.cat((X1, X2), dim=-1)

        srcZ = self.coordlin1(Zt)
        dstZ = self.coordlin2(Zt)
        Zprod = (srcZ * dstZ).sum(dim=-1)

        Xmsg = self.act(self.convmlp1(X) + self.noisegate * self.convmlp2(Zprod))
        retX = self.linXout(Xmsg)

        coeffZ = self.lincoeff1(Xmsg)
        convZ = self.convlinZ(Zt)
        retZt = self.linZout(coeffZ.unsqueeze(-1) * convZ)
        retZ = retZt.transpose(1, 2)
        return retX, retZ


class ENGNNLayer(nn.Module):

    def __init__(self, d_h: int, d_e: int, mid_ratio: int, layer_idx: int, purelayers: int, silu: bool = True):
        super().__init__()
        self.d_h = d_h
        self.d_e = d_e
        self.layer_idx = layer_idx
        self.is_pure = layer_idx < purelayers

        self.conv = PureENGNNConv(d_h, d_e, mid_ratio, silu) if self.is_pure else ENGNNConv(d_h, d_e, mid_ratio, silu)
        self.comb = None if self.is_pure else ENGNNComb(d_h, d_e, mid_ratio, silu)

        self.pre_norm_h = AffineFreeLayerNorm(d_h) if layer_idx > 0 else None
        self.pre_norm_z = Vnorm() if layer_idx > 0 else None

    def forward(self, z: Tensor, h: Tensor, edge_index: Tensor, info: BatchInfo) -> tuple[Tensor, Tensor]:
        N = h.shape[0]
        C = info.C
        Z = z.reshape(N, C, self.d_e)

        if self.layer_idx > 0:
            x1 = self.pre_norm_h(h)
            Z1 = self.pre_norm_z(z, C, self.d_e).reshape(N, C, self.d_e) if C > 1 else Z
        else:
            x1, Z1 = h, Z

        Xn, Zn = self.conv(x1, Z1, edge_index)
        h_new, Z_new = h + Xn, Z + Zn

        if not self.is_pure:
            Xn2, Zn2 = self.comb(x1, Xn, Z1, Zn)
            h_new, Z_new = h_new + Xn2, Z_new + Zn2

        return Z_new.reshape(N, C * self.d_e), h_new
