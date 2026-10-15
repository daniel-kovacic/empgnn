import csv
import math
import os
import sys
import warnings
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from .config import TrainCfg


def mae(pred: Tensor, target: Tensor) -> float:
    return float((pred - target).abs().mean().item())


def make_rounded_accuracy(label_mean: float, label_std: float) -> Callable[[Tensor, Tensor], float]:
    def _rounded_accuracy(pred: Tensor, target: Tensor) -> float:
        pred_counts = (pred * label_std + label_mean).round()
        target_counts = (target * label_std + label_mean).round()
        return float((pred_counts == target_counts).float().mean().item())
    return _rounded_accuracy


def build_loss_fn(cfg: TrainCfg) -> Callable[[Tensor, Tensor], Tensor]:
    if cfg.loss == "smooth_l1":
        return nn.SmoothL1Loss(beta=cfg.smooth_l1_beta)
    if cfg.loss == "l1":
        return nn.L1Loss()
    raise ValueError(f"unknown loss {cfg.loss!r}")


class CsvLogger:

    def __init__(self, log_dir: str, run_name: str, verbose: bool = True) -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, f"{run_name}_metrics.csv")
        has_content = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0
        self._file = open(self.csv_path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=["step", "key", "value"])
        if not has_content:
            self._writer.writeheader()
        self._verbose = verbose

    def log_scalar(self, key: str, value: float, step: int) -> None:
        self._writer.writerow({"step": step, "key": key, "value": value})
        self._file.flush()
        if self._verbose:
            print(f"[step={step}] {key}={value:.6f}", file=sys.stdout)

    def finish(self) -> None:
        self._file.close()


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        cfg: TrainCfg,
        logger: CsvLogger | None = None,
        extra_metrics: dict[str, Callable[[Tensor, Tensor], float]] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.logger = logger
        self.extra_metrics = extra_metrics or {}
        self.device = torch.device(cfg.device)
        self.model.to(self.device)

        warmup_epochs = cfg.warmup_epochs
        const_epochs = cfg.const_epochs
        cosine_epochs = max(cfg.epochs - warmup_epochs - const_epochs, 1)
        warmup_factor = cfg.initial_warmup_factor
        min_ratio = cfg.lr_min / cfg.lr

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                t = epoch / max(warmup_epochs, 1)
                return warmup_factor + (1.0 - warmup_factor) * t
            if epoch < warmup_epochs + const_epochs:
                return 1.0
            t = min((epoch - warmup_epochs - const_epochs) / cosine_epochs, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * t))
            return min_ratio + (1.0 - min_ratio) * cosine

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def fit(self, train_loader, val_loader, checkpoint_path: str, resume_path: str | None = None) -> dict:
        best_metric = float("inf")
        best_epoch = 0
        best_val_metrics: dict = {}
        start_epoch = 0
        last_epoch = 0
        last_train_metrics: dict = {}
        last_val_metrics: dict = {}

        if resume_path is not None and os.path.exists(resume_path):
            resumed = self.load_resume_checkpoint(resume_path)
            start_epoch = resumed["epoch"]
            last_epoch = start_epoch
            best_metric = resumed["best_metric"]
            best_epoch = resumed["best_epoch"]
            best_val_metrics = resumed["best_val_metrics"]

        for epoch in range(start_epoch + 1, self.cfg.epochs + 1):
            train_metrics = self._train_epoch(train_loader, epoch)
            last_epoch = epoch
            last_train_metrics = train_metrics

            if (epoch % self.cfg.eval_every == 0) or (epoch == self.cfg.epochs):
                val_metrics = self.evaluate(val_loader)
                last_val_metrics = val_metrics
                current = val_metrics["mae"]
                if current < best_metric:
                    best_metric = current
                    best_epoch = epoch
                    best_val_metrics = dict(val_metrics)
                    self.save_checkpoint(checkpoint_path, epoch, val_metrics)
                if self.logger is not None:
                    for k, v in val_metrics.items():
                        self.logger.log_scalar(f"val/{k}", v, step=epoch)

            self.scheduler.step()

            if (epoch % self.cfg.checkpoint_every == 0) or (epoch == self.cfg.epochs):
                if resume_path is not None:
                    self.save_resume_checkpoint(resume_path, epoch, best_metric, best_epoch, best_val_metrics)

            if self.logger is not None:
                for k, v in train_metrics.items():
                    self.logger.log_scalar(f"train/{k}", v, step=epoch)

        return {
            "best_epoch": best_epoch,
            **{f"best_val_{k}": v for k, v in best_val_metrics.items()},
            "last_epoch": last_epoch,
            **{f"last_train_{k}": v for k, v in last_train_metrics.items()},
            **{f"last_val_{k}": v for k, v in last_val_metrics.items()},
        }

    def _metrics(self, loss: float, pred: Tensor, target: Tensor) -> dict:
        metrics = {"loss": loss, "mae": mae(pred.squeeze(-1), target.squeeze(-1))}
        for name, fn in self.extra_metrics.items():
            metrics[name] = fn(pred.squeeze(-1), target.squeeze(-1))
        return metrics

    def _train_epoch(self, loader, epoch: int) -> dict:
        self.model.train()
        copies = getattr(self.model, "graph_copies", 1)
        total_loss = 0.0
        all_pred, all_target = [], []
        n_skipped = 0

        for batch_idx, batch in enumerate(loader):
            y = batch.y.to(self.device)
            if copies > 1:
                y = self.model.expand_targets(y, batch)
            self.optimizer.zero_grad()
            pred = self.model(batch)
            loss = self.loss_fn(pred.squeeze(-1), y.squeeze(-1))
            loss.backward()

            grad_is_finite = all(
                torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None
            )
            if not grad_is_finite:
                n_skipped += 1
                warnings.warn(
                    f"non-finite gradient at epoch={epoch} batch={batch_idx} (loss={loss.item():.4g}), skipping step",
                    RuntimeWarning,
                )
                continue

            if self.cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            total_loss += float(loss.item()) * len(y)
            all_pred.append(pred.detach())
            all_target.append(y.detach())

        if n_skipped > 0:
            warnings.warn(
                f"epoch={epoch}: skipped {n_skipped} batch(es) with non-finite gradients", RuntimeWarning,
            )
        if not all_target:
            raise RuntimeError(f"epoch={epoch}: every batch had a non-finite gradient, cannot continue")

        n = sum(len(t) for t in all_target)
        return self._metrics(total_loss / n, torch.cat(all_pred), torch.cat(all_target))

    def evaluate(self, loader) -> dict:
        self.model.eval()
        n_repeats = max(1, self.cfg.eval_repeats)

        pred_per_repeat = []
        target_cat = None

        with torch.no_grad():
            for r in range(n_repeats):
                eval_rng = torch.Generator()
                eval_rng.manual_seed(self.cfg.eval_seed + r)

                preds, targets = [], []
                for batch in loader:
                    targets.append(batch.y.to(self.device))
                    preds.append(self.model(batch, rng=eval_rng))
                pred_per_repeat.append(torch.cat(preds))
                if target_cat is None:
                    target_cat = torch.cat(targets)

        pred_combined = torch.stack(pred_per_repeat, dim=0).mean(dim=0)
        loss = self.loss_fn(pred_combined.squeeze(-1), target_cat.squeeze(-1))
        return self._metrics(float(loss.item()), pred_combined, target_cat)

    def save_checkpoint(self, path: str, epoch: int, metrics: dict) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"epoch": epoch, "metrics": metrics, "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict()}, path)

    def load_checkpoint(self, path: str) -> dict:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        return ckpt

    def save_resume_checkpoint(
        self, path: str, epoch: int, best_metric: float, best_epoch: int, best_val_metrics: dict,
    ) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        rng_state: dict = {"torch": torch.get_rng_state()}
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()
        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "best_val_metrics": best_val_metrics,
            "rng_state": rng_state,
        }, path)

    def load_resume_checkpoint(self, path: str) -> dict:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        rng_state = ckpt["rng_state"]
        torch.set_rng_state(rng_state["torch"].cpu())
        if "cuda" in rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in rng_state["cuda"]])
        return {
            "epoch": ckpt["epoch"],
            "best_metric": ckpt["best_metric"],
            "best_epoch": ckpt["best_epoch"],
            "best_val_metrics": ckpt["best_val_metrics"],
        }
