from dataclasses import dataclass

from torch import Tensor


@dataclass
class BatchInfo:
    orig_graph: Tensor
    node_to_graph: Tensor
    n_graphs: int
    n_nodes: int
    C: int

    def to(self, device) -> "BatchInfo":
        return BatchInfo(
            orig_graph=self.orig_graph.to(device),
            node_to_graph=self.node_to_graph.to(device),
            n_graphs=self.n_graphs,
            n_nodes=self.n_nodes,
            C=self.C,
        )


def scatter_sum(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    d = src.shape[-1]
    out = src.new_zeros(dim_size, d)
    out.scatter_add_(0, index.unsqueeze(1).expand(-1, d), src)
    return out
