import copy
import math
from typing import Callable

from .config import TASKS, Config, MergeMode, RealizerKind

KS = (1, 2, 4, 8, 16, 30)
MAIN_TASKS = ("cycles6", "4path")
ARCHS = ("egin", "engnn")
N_MAX = 30
SEEDS = 3

CHANNEL_SWEEP = [
    (1, 1),
    (2, 1), (2, 2), (2, 3), (2, 4), (2, 8),
    (4, 1), (4, 2), (4, 3), (4, 4), (4, 6), (4, 8), (4, 16),
    (8, 1), (8, 2), (8, 3), (8, 4), (8, 8), (8, 11), (8, 15), (8, 32),
    (16, 1), (16, 2), (16, 3), (16, 4), (16, 8), (16, 15), (16, 22), (16, 30), (16, 64),
    (30, 1), (30, 2), (30, 3), (30, 4), (30, 8), (30, 16), (30, 30), (30, 45), (30, 60), (30, 128),
]

VOCAB_SIZES = (30, 45, 60, 90, 120)
REALIZERS = (
    RealizerKind.FIXED_RANDOM, RealizerKind.DISCRETE, RealizerKind.RANDOM, RealizerKind.ORDINAL,
    RealizerKind.FULL_RANDOM, RealizerKind.ORTHOGONAL, RealizerKind.FIXED_RANDOM_1D,
)


def make(task: str, arch: str, seed: int, node_id: dict | None = None, train: dict | None = None,
         model: dict | None = None, epochs: int | None = None) -> Config:
    cfg = Config()
    cfg.data.task = task
    cfg.model.arch = arch
    cfg.seed = seed
    for group, values in (("node_id", node_id), ("train", train), ("model", model)):
        for key, value in (values or {}).items():
            setattr(getattr(cfg, group), key, value)
    if epochs is not None:
        cfg.train.epochs = epochs
    return cfg


def _grid(tasks, archs, seeds):
    return [(t, a, s) for t in tasks for a in archs for s in range(seeds)]


def mode_ablation(tasks=MAIN_TASKS, archs=ARCHS, seeds=SEEDS) -> list[Config]:
    out = []
    for task, arch, seed in _grid(tasks, archs, seeds):
        for k in KS:
            out.append(make(task, arch, seed, dict(k=k, n_channels=N_MAX)))
            out.append(make(task, arch, seed, dict(k=k, n_channels=math.ceil(N_MAX / k))))
            out.append(make(task, arch, seed, dict(k=k, n_channels=math.ceil(N_MAX / k)), dict(graph_copies=k)))
    return out


def realizer_ablation(tasks=MAIN_TASKS, archs=ARCHS, seeds=SEEDS) -> list[Config]:
    return [
        make(task, arch, seed, dict(k=k, n_channels=N_MAX, realizer=realizer))
        for task, arch, seed in _grid(tasks, archs, seeds) for realizer in REALIZERS for k in KS
    ]


def channel_sweep(tasks=MAIN_TASKS, archs=ARCHS, seeds=SEEDS) -> list[Config]:
    return [
        make(task, arch, seed, dict(k=k, n_channels=parts * math.ceil(N_MAX / k)))
        for task, arch, seed in _grid(tasks, archs, seeds) for k, parts in CHANNEL_SWEEP
    ]


def vocab_ablation(tasks=MAIN_TASKS, archs=ARCHS, seeds=SEEDS) -> list[Config]:
    out = []
    for task, arch, seed in _grid(tasks, archs, seeds):
        # earlier experiments used concat for EGIN here, every other EGIN experiment uses add
        model = dict(merge_mode=MergeMode.CONCAT) if arch == "egin" else {}
        out.append(make(task, arch, seed, dict(k=N_MAX), model=model))
        for k in VOCAB_SIZES:
            out.append(make(task, arch, seed, dict(k=k, uncapped_vocab=True), model=model))
        out.append(make(task, arch, seed, dict(k=N_MAX, realizer=RealizerKind.FULL_RANDOM), model=model))
    return out


def gin_vs_egin(tasks=MAIN_TASKS, seeds=SEEDS) -> list[Config]:
    # the GIN width matching counts n_copies channels, so n_copies must equal n_channels
    ids = dict(k=N_MAX, n_copies=N_MAX, n_channels=N_MAX)
    out = []
    for task in tasks:
        for seed in range(seeds):
            out.append(make(task, "gin", seed, ids, model=dict(use_ids=False)))
            out.append(make(task, "gin", seed, ids))
            out.append(make(task, "egin", seed, ids))
    return out


def id_symmetry(tasks=MAIN_TASKS, seeds=SEEDS) -> list[Config]:
    rows = [
        dict(realizer=RealizerKind.FIXED_RANDOM),
        dict(realizer=RealizerKind.ORTHOGONAL),
        dict(realizer=RealizerKind.FIXED_RANDOM, id_norm="equal"),
        dict(realizer=RealizerKind.ORTHOGONAL, id_norm="chi"),
        dict(realizer=RealizerKind.FIXED_RANDOM, id_scale=0.125),
        dict(realizer=RealizerKind.ORTHOGONAL, id_scale=8.0, _tasks=("cycles6",)),
    ]
    out = []
    for task in tasks:
        for row in rows:
            row = dict(row)
            if task not in row.pop("_tasks", tasks):
                continue
            for k in (8, 30):
                for seed in range(seeds):
                    out.append(make(task, "engnn", seed, dict(row, k=k, n_channels=N_MAX)))
    return out


def final_table(tasks=tuple(TASKS), seeds=SEEDS, k: int = 2) -> list[Config]:
    return [
        make(task, arch, seed, dict(k=k, n_channels=N_MAX), epochs=1000)
        for task, arch, seed in _grid(tasks, ARCHS, seeds)
    ]


def acc_vs_mae(tasks=MAIN_TASKS, archs=ARCHS, seeds=SEEDS) -> list[Config]:
    return [
        make(task, arch, seed, dict(k=k, n_channels=N_MAX))
        for task, arch, seed in _grid(tasks, archs, seeds) for k in KS
    ]


FAMILIES: dict[str, Callable[..., list[Config]]] = {
    "mode_ablation": mode_ablation,
    "realizer_ablation": realizer_ablation,
    "channel_sweep": channel_sweep,
    "vocab_ablation": vocab_ablation,
    "gin_vs_egin": gin_vs_egin,
    "id_symmetry": id_symmetry,
    "final_table": final_table,
    "acc_vs_mae": acc_vs_mae,
}


def unique(configs: list[Config]) -> list[Config]:
    from .run import run_key
    seen, out = set(), []
    for cfg in configs:
        key = run_key(cfg)
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    return out
