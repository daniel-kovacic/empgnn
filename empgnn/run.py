import csv
import dataclasses
import enum
import hashlib
import json
import os
import time
import traceback

import torch

from .config import Config
from .data import Splits, load_splits, make_loader
from .model import build_model
from .train import CsvLogger, Trainer, build_loss_fn, make_rounded_accuracy

N_FINAL = 5


def config_to_dict(cfg) -> dict:
    if dataclasses.is_dataclass(cfg):
        return {f.name: config_to_dict(getattr(cfg, f.name)) for f in dataclasses.fields(cfg)}
    if isinstance(cfg, enum.Enum):
        return cfg.value
    return cfg


def run_key(cfg: Config) -> str:
    d = config_to_dict(cfg)
    d["train"].pop("device"), d["train"].pop("verbose"), d.pop("out_dir")
    digest = hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]
    n = cfg.node_id
    return f"{cfg.data.task}_{cfg.model.arch}_k{n.k}_c{n.n_channels or n.n_copies}_s{cfg.seed}_{digest}"


def build_for_data(cfg: Config, d_in: int, n_max: int, e_avg: float):
    torch.manual_seed(cfg.seed)
    return build_model(cfg.model, cfg.node_id, cfg.train.graph_copies, d_in, n_max, e_avg)


def build(cfg: Config, splits: Splits):
    n_max, e_avg = splits.size_stats()
    return build_for_data(cfg, splits.train.num_node_features, n_max, e_avg)


def _trainer(cfg: Config, model, splits: Splits, logger=None) -> Trainer:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    extra = {"rounded_accuracy": make_rounded_accuracy(splits.label_mean, splits.label_std)}
    return Trainer(model, optimizer, build_loss_fn(cfg.train), cfg.train, logger=logger, extra_metrics=extra)


def final_val_mae(csv_path: str, n: int = N_FINAL) -> float:
    last_by_step: dict[int, float] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["key"] == "val/mae":
                last_by_step[int(row["step"])] = float(row["value"])
    tail = [last_by_step[s] for s in sorted(last_by_step)][-n:]
    return sum(tail) / len(tail)


def train(cfg: Config, run_dir: str, splits: Splits | None = None) -> dict:
    splits = splits or load_splits(cfg.data)
    model = build(cfg, splits)
    train_loader = make_loader(splits.train, cfg.train.batch_size, shuffle=True)
    val_loader = make_loader(splits.val, cfg.train.batch_size, shuffle=False)

    os.makedirs(run_dir, exist_ok=True)
    logger = CsvLogger(run_dir, "epoch", verbose=cfg.train.verbose)
    trainer = _trainer(cfg, model, splits, logger)
    results = trainer.fit(
        train_loader, val_loader,
        checkpoint_path=os.path.join(run_dir, "best.pt"), resume_path=os.path.join(run_dir, "last.pt"),
    )
    logger.finish()
    results["final_val_mae"] = final_val_mae(logger.csv_path)
    return results


def evaluate_checkpoint(cfg: Config, checkpoint_path: str, split: str = "test", splits: Splits | None = None) -> dict:
    splits = splits or load_splits(cfg.data)
    model = build(cfg, splits)
    loader = make_loader(getattr(splits, split), cfg.train.batch_size, shuffle=False)
    trainer = _trainer(cfg, model, splits)
    trainer.load_checkpoint(checkpoint_path)
    return trainer.evaluate(loader)


def run(cfg: Config, name: str | None = None, splits: Splits | None = None, keep_checkpoints: bool = False) -> dict:
    name = name or run_key(cfg)
    run_dir = os.path.join(cfg.out_dir, name)
    record_path = os.path.join(run_dir, "record.json")
    if os.path.exists(record_path):
        print(f"[run] {record_path} exists, skipping")
        with open(record_path) as f:
            return json.load(f)

    record = {"name": name, "seed": cfg.seed, "config": config_to_dict(cfg), "metrics": {}, "error": None}
    start = time.time()
    try:
        splits = splits or load_splits(cfg.data)
        record["metrics"] = train(cfg, run_dir, splits)
        test = evaluate_checkpoint(cfg, os.path.join(run_dir, "best.pt"), "test", splits)
        record["metrics"].update({f"test_{k}": v for k, v in test.items()})
        if not keep_checkpoints:
            for ckpt in ("best.pt", "last.pt"):
                os.remove(os.path.join(run_dir, ckpt))
    except Exception:
        traceback.print_exc()
        record["error"] = traceback.format_exc()
    record["wall_seconds"] = round(time.time() - start, 2)

    os.makedirs(run_dir, exist_ok=True)
    with open(record_path if record["error"] is None else os.path.join(run_dir, "error.json"), "w") as f:
        json.dump(record, f, indent=2)
    return record
