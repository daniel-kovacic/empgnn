import math

from .config import ModelCfg, NodeIdCfg


def _mlp_macs(in_dim: int, out_dim: int, rows: float, hidden_dim: int | None = None) -> float:
    hidden = hidden_dim if hidden_dim is not None else max(in_dim, out_dim)
    return rows * in_dim * hidden + rows * hidden * out_dim


def num_assignments(n_nodes: float, node_id: NodeIdCfg) -> int:
    return math.ceil(n_nodes / node_id.k) * node_id.n_copies


def _egin_per_layer_macs(model: ModelCfg, node_id: NodeIdCfg, n_nodes: float, n_edges: float) -> float:
    d_e, d_h, L = model.d_e, model.d_h, model.n_layers
    N, E, C = n_nodes, n_edges, num_assignments(n_nodes, node_id)
    equiv_in = d_e + d_h
    inv_in = d_h + d_e
    per_layer = (
        _mlp_macs(equiv_in, d_e, N * C) * L
        + _mlp_macs(d_e, d_e, N * C) * L
        + _mlp_macs(inv_in, d_h, N) * L
    )
    return per_layer


def _egin_readout_macs(model: ModelCfg, d_in: int, out_dim: int, n_nodes: float, node_id: NodeIdCfg,
                       head_hidden_dim: int) -> float:
    d_e, d_h = model.d_e, model.d_h
    N, C = n_nodes, num_assignments(n_nodes, node_id)
    readout = N * C * d_e * d_e
    readout += _mlp_macs(d_in, d_h, N)
    readout += _mlp_macs(d_e + d_h, out_dim, 1, hidden_dim=head_hidden_dim)
    return readout


def gin_widths(model: ModelCfg, node_id: NodeIdCfg, d_in: int, out_dim: int,
               n_nodes: float, n_edges: float, head_hidden_dim: int) -> tuple[int, int]:
    L = model.n_layers
    target_total = _egin_per_layer_macs(model, node_id, n_nodes, n_edges)
    C = num_assignments(n_nodes, node_id) if model.use_ids else 1
    d_id = model.d_e * C if model.use_ids else 0
    in_dim0 = d_in + d_id
    N = n_nodes

    a = N * (2 * L - 1)
    b = N * in_dim0
    c = -target_total
    if a == 0:
        h = -c / b if b != 0 else 1.0
    else:
        h = (-b + math.sqrt(max(b * b - 4 * a * c, 0.0))) / (2 * a)
    hidden = max(1, math.ceil(h))

    target = _egin_readout_macs(model, d_in, out_dim, n_nodes, node_id, head_hidden_dim)
    denom = 1 * (hidden + out_dim)
    R = target / denom if denom > 0 else 1.0
    return hidden, max(1, math.ceil(R))
