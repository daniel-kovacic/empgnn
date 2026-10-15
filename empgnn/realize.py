import math

import torch
import torch.nn as nn

from .config import NodeIdCfg, RealizerKind


class RandomFeatureRealizer:

    def __init__(self, d: int):
        self.d = d

    def realize(self, ids: torch.Tensor, rng: torch.Generator | None = None) -> torch.Tensor:
        gen_device = rng.device if rng is not None else ids.device
        features = torch.randn(*ids.shape, 1, generator=rng, device=gen_device)
        features = features.to(ids.device).expand(*ids.shape, self.d).clone()
        features[ids == 0] = 0.0
        return features


class FullRandomFeatureRealizer:

    def __init__(self, d: int):
        self.d = d

    def realize(self, ids: torch.Tensor, rng: torch.Generator | None = None) -> torch.Tensor:
        gen_device = rng.device if rng is not None else ids.device
        features = torch.randn(*ids.shape, self.d, generator=rng, device=gen_device)
        features = features.to(ids.device)
        features[ids == 0] = 0.0
        return features


class DiscreteFeatureRealizer(nn.Module):

    def __init__(self, vocab_size: int, d: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, d, padding_idx=0)

    def realize(self, ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.embedding(ids)


class FixedRandomFeatureRealizer(nn.Module):

    def __init__(self, vocab_size: int, d: int, rng: torch.Generator | None = None):
        super().__init__()
        table = torch.randn(vocab_size + 1, d, generator=rng)
        table[0] = 0.0
        self.register_buffer("table", table)

    def realize(self, ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.table[ids]


class FixedRandom1DFeatureRealizer(nn.Module):

    def __init__(self, vocab_size: int, d: int, rng: torch.Generator | None = None):
        super().__init__()
        scalars = torch.randn(vocab_size + 1, 1, generator=rng)
        table = scalars.expand(vocab_size + 1, d).clone()
        table[0] = 0.0
        self.register_buffer("table", table)

    def realize(self, ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.table[ids]


class OrthogonalFeatureRealizer(nn.Module):

    def __init__(self, vocab_size: int, d: int, rng: torch.Generator | None = None):
        super().__init__()
        if d < vocab_size:
            raise ValueError(
                f"OrthogonalFeatureRealizer requires d >= vocab_size (got d={d}, vocab_size={vocab_size})"
            )
        random_matrix = torch.randn(d, d, generator=rng)
        q, r = torch.linalg.qr(random_matrix)
        q = q * torch.diagonal(r).sign().unsqueeze(0)
        vectors = q[:vocab_size].contiguous()
        table = torch.zeros(vocab_size + 1, d)
        table[1:vocab_size + 1] = vectors
        self.register_buffer("table", table)

    def realize(self, ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.table[ids]


class OrdinalFeatureRealizer:

    def __init__(self, number_of_ids: int, d: int):
        self.number_of_ids = number_of_ids
        self.d = d

    def realize(self, ids: torch.Tensor, **kwargs) -> torch.Tensor:
        scalar = ids.float() / self.number_of_ids
        return scalar.unsqueeze(-1).expand(*ids.shape, self.d).clone()


class ScaledRealizer(nn.Module):
    def __init__(self, inner, scale: float) -> None:
        super().__init__()
        self.inner = inner
        self.scale = scale

    def realize(self, ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.inner.realize(ids, **kwargs) * self.scale


def renorm_id_table(realizer, mode: str) -> None:
    table = getattr(realizer, "table", None)
    if not isinstance(table, torch.Tensor):
        raise ValueError(f"id_norm={mode!r} needs a fixed-table realizer, got {type(realizer).__name__}")
    rows = table[1:]
    n_rows, d = rows.shape
    if mode == "equal":
        target = torch.full((n_rows,), math.sqrt(d))
    elif mode == "chi":
        target = torch.randn(n_rows, d).norm(dim=1)
    else:
        raise ValueError(f"unknown id_norm {mode!r}")
    table[1:] = rows / rows.norm(dim=1, keepdim=True) * target.unsqueeze(1)


def build_realizer(node_id: NodeIdCfg, d: int):
    realizer = _build_base_realizer(node_id, d)
    if node_id.id_norm != "native":
        renorm_id_table(realizer, node_id.id_norm)
    if node_id.id_scale != 1.0:
        return ScaledRealizer(realizer, node_id.id_scale)
    return realizer


def _build_base_realizer(node_id: NodeIdCfg, d: int):
    vocab_size = node_id.k + 1
    kind = node_id.realizer
    if kind == RealizerKind.RANDOM:
        return RandomFeatureRealizer(d=d)
    if kind == RealizerKind.FULL_RANDOM:
        return FullRandomFeatureRealizer(d=d)
    if kind == RealizerKind.ORDINAL:
        return OrdinalFeatureRealizer(number_of_ids=vocab_size, d=d)
    if kind == RealizerKind.FIXED_RANDOM:
        return FixedRandomFeatureRealizer(vocab_size=vocab_size, d=d)
    if kind == RealizerKind.FIXED_RANDOM_1D:
        return FixedRandom1DFeatureRealizer(vocab_size=vocab_size, d=d)
    if kind == RealizerKind.ORTHOGONAL:
        return OrthogonalFeatureRealizer(vocab_size=vocab_size, d=d)
    return DiscreteFeatureRealizer(vocab_size=vocab_size, d=d)
