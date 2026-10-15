import torch


class PaddedFilledAssigner:

    def __init__(self, k: int, n_copies: int, max_nodes: int, n_channels: int | None = None,
                 uncapped_vocab: bool = False):
        if max_nodes is None:
            raise ValueError("PaddedFilledAssigner requires max_nodes (the largest node count over all splits)")
        self.k = k
        self.n_copies = n_copies
        self.max_nodes = max_nodes
        self._n_channels = n_channels
        self.uncapped_vocab = uncapped_vocab
        if uncapped_vocab and k < max_nodes:
            raise ValueError(f"uncapped_vocab needs k >= max_nodes (got k={k}, max_nodes={max_nodes})")
        if n_channels is not None:
            need = -(-max_nodes // min(k, max_nodes))
            if n_channels < need:
                raise ValueError(f"n_channels={n_channels} < {need} channels needed for one partition of the largest graph")

    @property
    def n_assignments(self) -> int:
        if self._n_channels is not None:
            return self._n_channels
        k_max = min(self.k, self.max_nodes)
        return self.n_copies * (-(-self.max_nodes // k_max))

    def assign(self, n: int, rng: torch.Generator | None = None) -> torch.Tensor:
        return self.assign_batch(torch.tensor([n]), rng).t().contiguous()

    def assign_batch(self, node_counts: torch.Tensor, rng: torch.Generator | None = None) -> torch.Tensor:
        B = node_counts.shape[0]
        C_max = self.n_assignments
        if B == 0:
            return node_counts.new_zeros(0, C_max)
        if bool((node_counts > self.max_nodes).any()):
            raise ValueError(
                f"graph with {int(node_counts.max())} nodes exceeds max_nodes={self.max_nodes}; "
                f"max_nodes must be the largest node count over all splits"
            )

        node_counts = node_counts.long()
        k_eff = node_counts.clamp(min=1).clamp(max=self.k)
        c_g = (node_counts + k_eff - 1) // k_eff
        s_g = C_max // c_g
        if self.k == 1:
            s_g = torch.ones_like(s_g)
        slots = c_g * k_eff

        n_seg = int(s_g.sum().item())
        seg_graph = torch.arange(B).repeat_interleave(s_g)
        seg_first = torch.cumsum(s_g, 0) - s_g
        seg_part = torch.arange(n_seg) - seg_first[seg_graph]
        seg_sizes = slots[seg_graph]
        seg_offsets = torch.cumsum(seg_sizes, 0) - seg_sizes
        total = int(seg_sizes.sum().item())
        seg_of_slot = torch.arange(n_seg).repeat_interleave(seg_sizes)

        key = seg_of_slot.double() + torch.rand(total, generator=rng, dtype=torch.float64)
        order = torch.argsort(key)

        pos = torch.arange(total) - seg_offsets[seg_of_slot]
        graph = seg_graph[seg_of_slot]
        part = seg_part[seg_of_slot]
        keep = pos < node_counts[graph]

        slot = order - seg_offsets[seg_of_slot]
        k_g = k_eff[graph]
        channel = slot // k_g
        rank = slot % k_g + 1

        node_offsets = torch.cumsum(node_counts, 0) - node_counts
        node_row = node_offsets[graph] + pos
        col = part * c_g[graph] + channel

        ids = torch.zeros(int(node_counts.sum().item()), C_max, dtype=torch.long)
        ids[node_row[keep], col[keep]] = rank[keep]
        if self.uncapped_vocab:
            ids = self._remap_vocab(ids, node_counts, rng)
        return ids

    def _remap_vocab(self, ids: torch.Tensor, node_counts: torch.Tensor, rng: torch.Generator | None) -> torch.Tensor:
        B, C = node_counts.shape[0], ids.shape[1]
        perm = torch.argsort(torch.rand(B * C, self.k, generator=rng), dim=1) + 1
        seg = torch.arange(B).repeat_interleave(node_counts).unsqueeze(1) * C + torch.arange(C)
        mask = ids > 0
        out = torch.zeros_like(ids)
        out[mask] = perm[seg[mask], ids[mask] - 1]
        return out
