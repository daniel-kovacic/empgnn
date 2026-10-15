import copy

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINConv, GraphNorm

from .assign import PaddedFilledAssigner
from .config import ModelCfg, NodeIdCfg
from .cost import gin_widths
from .engnn import ENGNNLayer
from .layers import EGINLayer, mlp
from .ops import BatchInfo
from .realize import build_realizer


class EquivNet(nn.Module):

    def __init__(self, cfg: ModelCfg, d_in: int, out_dim: int = 1):
        super().__init__()
        self.arch = cfg.arch
        self.d_e = cfg.d_e
        silu = cfg.arch == "engnn"
        head_hidden_dim = cfg.head_hidden_dim if cfg.head_hidden_dim is not None else 2 * cfg.d_e

        self.proj_x_inv = mlp(d_in, cfg.d_h, silu)
        if cfg.arch == "egin":
            self.layers = nn.ModuleList([
                EGINLayer(cfg.d_e, cfg.d_h, cfg.eps, cfg.merge_mode) for _ in range(cfg.n_layers)
            ])
            head_in = cfg.d_e + cfg.d_h
        else:
            self.layers = nn.ModuleList([
                ENGNNLayer(cfg.d_h, cfg.d_e, cfg.engnn_mid_ratio, layer_idx=i,
                           purelayers=cfg.engnn_purelayers, silu=silu)
                for i in range(cfg.n_layers)
            ])
            head_in = cfg.d_h
        self.head = mlp(head_in, out_dim, silu, hidden_dim=head_hidden_dim)

    def forward(self, batch: Batch, info: BatchInfo, id_feats: Tensor) -> Tensor:
        N, C, d_e = info.n_nodes, info.C, self.d_e
        z = id_feats.reshape(N, C * d_e)
        h = self.proj_x_inv(batch.x)
        for layer in self.layers:
            z, h = layer(z, h, batch.edge_index, info)
        if self.arch == "egin":
            h = torch.cat([z.reshape(N, C, d_e).mean(dim=1), h], dim=-1)
        return self.head(h)


class GINNet(nn.Module):

    def __init__(self, d_in: int, hidden_dim: int, n_layers: int, out_dim: int, d_id: int, readout_hidden_dim: int):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_dim = d_in + d_id
        for _ in range(n_layers):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)),
                                      train_eps=True))
            self.norms.append(GraphNorm(hidden_dim))
            in_dim = hidden_dim
        self.head = nn.Sequential(nn.Linear(hidden_dim, readout_hidden_dim), nn.ReLU(), nn.Linear(readout_hidden_dim, out_dim))

    def forward(self, batch: Batch, graph: Tensor, id_feats: Tensor | None) -> Tensor:
        x = torch.cat([batch.x, id_feats], dim=-1) if id_feats is not None else batch.x
        for conv, norm in zip(self.convs, self.norms):
            x = torch.relu(norm(conv(x, batch.edge_index), graph))
        return self.head(x)


def _uniform_batch(graphs: list[Data], ids_list: list[Tensor]) -> tuple[Batch, BatchInfo]:
    C = ids_list[0].shape[0]
    device = graphs[0].x.device
    data_list = []
    for g, ids in zip(graphs, ids_list):
        d = Data(x=g.x, edge_index=g.edge_index, ids=ids.T.to(device))
        d.num_nodes = g.x.shape[0]
        data_list.append(d)
    batch = Batch.from_data_list(data_list).to(device)
    return batch, _info(batch, C)


def _info(batch: Batch, C: int) -> BatchInfo:
    return BatchInfo(
        orig_graph=batch.batch.unsqueeze(1).expand(-1, C).reshape(-1),
        node_to_graph=batch.batch,
        n_graphs=batch.num_graphs,
        n_nodes=batch.x.shape[0],
        C=C,
    )


class EquivModel(nn.Module):

    def __init__(self, net: EquivNet, assigner, realizer, graph_copies: int = 1) -> None:
        super().__init__()
        self.net = net
        self.assigner = assigner
        self.realizer = realizer
        self.graph_copies = graph_copies
        if isinstance(realizer, nn.Module):
            self.add_module("_realizer_module", realizer)

    def expand_targets(self, y: Tensor, batch: Batch) -> Tensor:
        if not (self.training and self.graph_copies > 1):
            return y
        starts = batch.ptr[:-1].tolist()
        ends = batch.ptr[1:].tolist()
        reps = (self.graph_copies,) + (1,) * (y.dim() - 1)
        return torch.cat([y[s:e].repeat(*reps) for s, e in zip(starts, ends)])

    def _prepare(self, batch: Batch, rng: torch.Generator | None) -> tuple[Batch, BatchInfo]:
        if not (self.training and self.graph_copies > 1):
            node_counts = (batch.ptr[1:] - batch.ptr[:-1]).cpu()
            ids_all = self.assigner.assign_batch(node_counts, rng)
            inner = copy.copy(batch)
            inner.ids = ids_all.to(batch.x.device)
            return inner, _info(batch, ids_all.shape[1])

        cpu_batch = batch.cpu() if batch.x.device.type != "cpu" else batch
        graphs = cpu_batch.to_data_list()
        origin = torch.arange(len(graphs)).repeat_interleave(self.graph_copies)
        graphs = [graphs[i] for i in origin.tolist()]
        ids_list = [self.assigner.assign(int(g.num_nodes), rng) for g in graphs]
        return _uniform_batch(graphs, ids_list)

    def forward(self, batch: Batch, rng: torch.Generator | None = None) -> Tensor:
        inner, info = self._prepare(batch, rng)
        device = next(self.net.parameters()).device
        if device.type != "cpu":
            inner = inner.to(device)
            info = info.to(device)
        id_feats = self.realizer.realize(inner.ids)
        return self.net(inner, info, id_feats)


class GINModel(nn.Module):

    def __init__(self, net: GINNet, assigner=None, realizer=None) -> None:
        super().__init__()
        self.net = net
        self.assigner = assigner
        self.realizer = realizer
        if isinstance(realizer, nn.Module):
            self.add_module("_realizer_module", realizer)

    def forward(self, batch: Batch, rng: torch.Generator | None = None) -> Tensor:
        device = next(self.net.parameters()).device
        if batch.x.device != device:
            batch = batch.to(device)
        if self.assigner is None:
            return self.net(batch, batch.batch, None)
        node_counts = (batch.ptr[1:] - batch.ptr[:-1]).cpu()
        ids = self.assigner.assign_batch(node_counts, rng).to(device)
        id_feats = self.realizer.realize(ids)
        return self.net(batch, batch.batch, id_feats.reshape(id_feats.shape[0], -1))


def build_assigner(node_id: NodeIdCfg, n_max: int) -> PaddedFilledAssigner:
    if node_id.k > n_max and not node_id.uncapped_vocab:
        node_id.k = n_max
    return PaddedFilledAssigner(k=node_id.k, n_copies=node_id.n_copies, max_nodes=n_max, n_channels=node_id.n_channels,
                                uncapped_vocab=node_id.uncapped_vocab)


def build_model(model: ModelCfg, node_id: NodeIdCfg, graph_copies: int, d_in: int, n_max: int, e_avg: float,
                out_dim: int = 1) -> nn.Module:
    # this order fixes the random draws, and with them the initialisation
    assigner = build_assigner(node_id, n_max)
    realizer = build_realizer(node_id, model.d_e)
    if model.arch == "gin":
        head_hidden_dim = model.head_hidden_dim if model.head_hidden_dim is not None else 2 * model.d_e
        hidden, readout_hidden = gin_widths(model, node_id, d_in, out_dim, n_max, e_avg, head_hidden_dim)
        d_id = model.d_e * assigner.n_assignments if model.use_ids else 0
        net = GINNet(d_in, hidden, model.n_layers, out_dim, d_id, readout_hidden)
        return GINModel(net, assigner if model.use_ids else None, realizer if model.use_ids else None)
    if model.arch not in ("egin", "engnn"):
        raise ValueError(f"unknown arch {model.arch!r}")
    return EquivModel(EquivNet(model, d_in, out_dim), assigner, realizer, graph_copies)
