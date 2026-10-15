from dataclasses import dataclass, field
from enum import Enum

import torch


class MergeMode(Enum):
    CONCAT = "concat"
    ADD = "add"


class RealizerKind(Enum):
    DISCRETE = "discrete"
    RANDOM = "random"
    FULL_RANDOM = "full_random"
    ORDINAL = "ordinal"
    FIXED_RANDOM = "fixed_random"
    FIXED_RANDOM_1D = "fixed_random_1d"
    ORTHOGONAL = "orthogonal"


@dataclass
class ModelCfg:
    arch: str = "egin"  # egin | engnn | gin
    d_e: int = 64
    d_h: int = 40
    n_layers: int = 8
    eps: float = 0.5
    merge_mode: MergeMode = MergeMode.ADD
    head_hidden_dim: int | None = None
    engnn_purelayers: int = 2
    engnn_mid_ratio: int = 2
    use_ids: bool = True


@dataclass
class NodeIdCfg:
    k: int = 2
    n_copies: int = 1
    n_channels: int | None = 30
    realizer: RealizerKind = RealizerKind.FIXED_RANDOM
    id_scale: float = 1.0
    id_norm: str = "native"  # native | equal | chi
    uncapped_vocab: bool = False


@dataclass
class TrainCfg:
    epochs: int = 750
    batch_size: int = 32
    graph_copies: int = 1
    lr: float = 2e-3
    weight_decay: float = 0.0
    loss: str = "smooth_l1"  # smooth_l1 | l1
    smooth_l1_beta: float = 1e-3
    grad_clip: float | None = 1.0
    warmup_epochs: int = 10
    const_epochs: int = 40
    initial_warmup_factor: float = 1e-3
    lr_min: float = 1e-6
    eval_every: int = 5
    eval_repeats: int = 5
    eval_seed: int = 0
    checkpoint_every: int = 10
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    verbose: bool = True


@dataclass
class DataCfg:
    task: str = "cycles6"
    data_dir: str = "data"


@dataclass
class Config:
    model: ModelCfg = field(default_factory=ModelCfg)
    node_id: NodeIdCfg = field(default_factory=NodeIdCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    data: DataCfg = field(default_factory=DataCfg)
    seed: int = 0
    out_dir: str = "runs"


TASKS: dict[str, int] = {
    "cycles3": 0, "cycles4": 1, "cycles5": 2, "cycles6": 3,
    "tailtri": 4, "chordcyc": 5, "4clique": 6, "4path": 7, "trirect": 8,
}


def config_from_dict(d: dict) -> Config:
    cfg = Config()
    for section in ("model", "node_id", "train", "data"):
        obj = getattr(cfg, section)
        for key, value in d[section].items():
            if isinstance(getattr(obj, key), Enum):
                value = type(getattr(obj, key))(value)
            setattr(obj, key, value)
    cfg.seed, cfg.out_dir = d["seed"], d["out_dir"]
    return cfg
