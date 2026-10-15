import json
import math
import os

import numpy as np
import pandas as pd

from . import experiments, plots
from .config import TASKS, Config
from .run import run_key

TASK_ORDER = list(TASKS)
TASK_LABELS = {
    "cycles3": "3-cycle", "cycles4": "4-cycle", "cycles5": "5-cycle", "cycles6": "6-cycle", "tailtri": "tailed tri.",
    "chordcyc": "chordal cyc.", "4clique": "4-clique", "4path": "4-path", "trirect": "tri-rect",
}
ARCH_NAMES = {"egin": "EGIN", "engnn": "Modified ENGNN", "gin": "GIN"}


def load_frame(configs: list[Config], out_dir: str) -> tuple[pd.DataFrame, int]:
    rows, missing = [], 0
    for cfg in experiments.unique(configs):
        path = os.path.join(out_dir, run_key(cfg), "record.json")
        if not os.path.exists(path):
            missing += 1
            continue
        rec = json.load(open(path))
        if rec["error"] is not None:
            missing += 1
            continue
        c, m = rec["config"], rec["metrics"]
        n = c["node_id"]
        rows.append({
            "key": rec["name"], "task": c["data"]["task"], "arch": c["model"]["arch"], "seed": c["seed"],
            "k": n["k"], "n_channels": n["n_channels"], "n_copies": n["n_copies"],
            "graph_copies": c["train"]["graph_copies"], "realizer": n["realizer"], "id_norm": n["id_norm"],
            "id_scale": n["id_scale"], "uncapped": n["uncapped_vocab"],
            "use_ids": c["model"]["use_ids"], "merge_mode": c["model"]["merge_mode"],
            "final_val_mae": m["final_val_mae"], "best_val_mae": m["best_val_mae"], "last_val_mae": m["last_val_mae"],
            "last_val_acc_pct": 100 * m["last_val_rounded_accuracy"],
            "test_mae": m["test_mae"], "test_acc_pct": 100 * m["test_rounded_accuracy"],
        })
    return pd.DataFrame(rows), missing


def _std(v, ddof: int) -> float:
    v = np.asarray(v, dtype=float)
    return float(np.std(v, ddof=ddof)) if len(v) > ddof else float("nan")


def _pm(v, ddof: int = 0, digits: int = 4) -> str:
    v = list(v)
    if not v:
        return "missing"
    return f"{np.mean(v):.{digits}f} ± {_std(v, ddof):.{digits}f} (n={len(v)})"


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print(f"wrote {path}")


def _grid_table(df: pd.DataFrame, row_cols: list[str], col: str, value: str = "final_val_mae") -> str:
    cols = sorted(df[col].unique(), key=lambda x: (str(type(x)), x))
    lines = ["| " + " | ".join(row_cols + [f"{col}={c}" for c in cols]) + " |", "|" + "---|" * (len(row_cols) + len(cols))]
    for key, g in df.groupby(row_cols, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        cells = [_pm(g[g[col] == c][value]) for c in cols]
        lines.append("| " + " | ".join([str(x) for x in key] + cells) + " |")
    return "\n".join(lines) + "\n"


def _note(missing: int) -> str:
    return f"\n{missing} run(s) of this experiment have not finished and are not included.\n" if missing else ""


def mode_ablation(configs, out_dir, results_dir) -> None:
    df, missing = load_frame(configs, out_dir)
    df["condition"] = np.where(df["graph_copies"] > 1, "compensated", np.where(df["n_channels"] == experiments.N_MAX, "constant", "minimal"))
    k1 = df[df["k"] == 1]
    df = pd.concat([df] + [k1.assign(condition=c) for c in ("minimal", "compensated")], ignore_index=True)
    conditions = ["minimal", "constant", "compensated"]
    plots.plot_ablation_grid(df, conditions, list(experiments.MAIN_TASKS), list(experiments.ARCHS),
                             os.path.join(results_dir, "mode_ablation"), shared_x=1)
    table = _grid_table(df.drop_duplicates().query("condition in @conditions"), ["arch", "task", "condition"], "k")
    _write(os.path.join(results_dir, "mode_ablation.md"),
           "# Final validation MAE by channel regime (mean ± std over seeds)\n\n" + table + _note(missing))


REALIZER_FIGURES = {
    "realizer_random": ["fixed_random", "random", "full_random"],
    "realizer_highdim": ["fixed_random", "discrete", "orthogonal"],
    "realizer_1d": ["fixed_random", "ordinal", "fixed_random_1d"],
}


def realizer_ablation(configs, out_dir, results_dir) -> None:
    df, missing = load_frame(configs, out_dir)
    df["condition"] = df["realizer"]
    for name, conditions in REALIZER_FIGURES.items():
        plots.plot_ablation_grid(df, conditions, list(experiments.MAIN_TASKS), list(experiments.ARCHS),
                                 os.path.join(results_dir, name))
    table = _grid_table(df, ["arch", "task", "condition"], "k")
    _write(os.path.join(results_dir, "realizer_ablation.md"),
           "# Final validation MAE by realizer (mean ± std over seeds)\n\n" + table + _note(missing))


def channel_sweep(configs, out_dir, results_dir) -> None:
    df, missing = load_frame(configs, out_dir)
    plots.plot_channel_sweep(df, list(experiments.MAIN_TASKS), list(experiments.ARCHS),
                             os.path.join(results_dir, "channel_sweep"))
    lines = ["| arch | task | k | channels | final val MAE |", "|---|---|---|---|---|"]
    for (arch, task, k, c), g in df.groupby(["arch", "task", "k", "n_channels"]):
        lines.append(f"| {arch} | {task} | {k} | {c} | {_pm(g['final_val_mae'])} |")
    _write(os.path.join(results_dir, "channel_sweep.md"), "\n".join(lines) + "\n" + _note(missing))


def vocab_ablation(configs, out_dir, results_dir) -> None:
    df, missing = load_frame(configs, out_dir)
    df["series"] = np.where(df["realizer"] == "full_random", "random", np.where(df["uncapped"], "uncapped", "capped"))
    plots.plot_vocab_ablation(df, list(experiments.MAIN_TASKS), list(experiments.ARCHS),
                              os.path.join(results_dir, "vocab_ablation"))
    lines = ["| arch | task | series | k | final val MAE |", "|---|---|---|---|---|"]
    for (arch, task, series, k), g in df.groupby(["arch", "task", "series", "k"]):
        lines.append(f"| {arch} | {task} | {series} | {k} | {_pm(g['final_val_mae'])} |")
    note = "\nEGIN uses merge_mode=CONCAT in this experiment only (the vocabulary ablation predates the switch to ADD).\n"
    _write(os.path.join(results_dir, "vocab_ablation.md"), "\n".join(lines) + "\n" + note + _note(missing))


def count_params(cfg: Config) -> int:
    from .data import load_splits
    from .run import build
    model = build(cfg, load_splits(cfg.data))
    return sum(p.numel() for p in model.parameters())


def gin_vs_egin(configs, out_dir, results_dir) -> None:
    df, missing = load_frame(configs, out_dir)
    arms = [("GIN (no ids)", lambda r: r.arch == "gin" and not r.use_ids),
            ("GIN + RNI (sampled once)", lambda r: r.arch == "gin" and r.use_ids),
            ("EGIN", lambda r: r.arch == "egin")]
    tasks = list(experiments.MAIN_TASKS)
    params = {}
    for cfg in experiments.unique(configs):
        arm = "GIN (no ids)" if cfg.model.arch == "gin" and not cfg.model.use_ids else \
            "GIN + RNI (sampled once)" if cfg.model.arch == "gin" else "EGIN"
        if arm not in params:
            try:
                params[arm] = f"{count_params(cfg) / 1e6:.2f}M"
            except FileNotFoundError:
                params[arm] = "n/a"
    md = ["| model | params | " + " | ".join(tasks) + " |", "|---|---|" + "---|" * len(tasks)]
    tex = []
    for arm, match in arms:
        sel = df[df.apply(match, axis=1)] if not df.empty else df
        md.append(f"| {arm} | {params.get(arm, 'n/a')} | " + " | ".join(_pm(sel[sel["task"] == t]["final_val_mae"]) for t in tasks) + " |")
        cells = []
        for t in tasks:
            v = list(sel[sel["task"] == t]["final_val_mae"])
            cells.append("--" if not v else f"{np.mean(v):.4f}" + (f" $\\pm$ {_std(v, 0):.4f}" if len(v) > 1 else ""))
        tex.append(f"{arm} & {params.get(arm, 'n/a')} & " + " & ".join(cells) + r" \\")
    note = "\nGIN widths are MAC-matched to EGIN, not parameter-matched.\n"
    _write(os.path.join(results_dir, "gin_vs_egin.md"), "\n".join(md) + "\n" + note + _note(missing))
    _write(os.path.join(results_dir, "gin_vs_egin.tex"), "\n".join(tex) + "\n")


SYMMETRY_ROWS = [
    ("sampled_once", "sampled-once (reference)", ("fixed_random", "native", 1.0)),
    ("orthonormal", "orthonormal", ("orthogonal", "native", 1.0)),
    ("angle", "angle perturbation", ("fixed_random", "equal", 1.0)),
    ("magnitude", "magnitude perturbation", ("orthogonal", "chi", 1.0)),
    ("orth_x8", "orthonormal x8", ("orthogonal", "native", 8.0)),
    ("fixed_x0125", "sampled-once x1/8", ("fixed_random", "native", 0.125)),
]


def id_symmetry(configs, out_dir, results_dir) -> None:
    df, missing = load_frame(configs, out_dir)
    cols = [(t, k) for t in experiments.MAIN_TASKS for k in (8, 30)]

    def runs(task, k, spec):
        sel = df[(df["task"] == task) & (df["k"] == k) & (df["realizer"] == spec[0]) & (df["id_norm"] == spec[1])
                 & (df["id_scale"] == spec[2])]
        return dict(zip(sel["seed"], sel["final_val_mae"]))

    def diff(cond, ref):
        seeds = sorted(set(cond) & set(ref))
        if not seeds:
            return None
        d = [cond[s] - ref[s] for s in seeds]
        se = _std(d, 1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
        return float(np.mean(d)), se, len(d)

    def fmt(c):
        return "missing" if c is None else f"{c[0]:+.4f} ± {'n/a' if math.isnan(c[1]) else f'{c[1]:.4f}'} (n={c[2]})"

    md = ["| row | " + " | ".join(f"{TASK_LABELS.get(t, t)}, k={k}" for t, k in cols) + " |", "|---|" + "---|" * len(cols)]
    md.append("| sampled-once, final val MAE (mean ± std) | "
              + " | ".join(_pm(runs(t, k, SYMMETRY_ROWS[0][2]).values()) for t, k in cols) + " |")
    tex = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
           " & ".join(["Condition"] + [f"{TASK_LABELS.get(t, t)}, $k={k}$" for t, k in cols]) + r" \\", r"\midrule"]
    for key, label, spec in SYMMETRY_ROWS[1:]:
        cells = [diff(runs(t, k, spec), runs(t, k, SYMMETRY_ROWS[0][2])) for t, k in cols]
        md.append(f"| {label} | " + " | ".join(fmt(c) for c in cells) + " |")
        tex.append(" & ".join([label] + [
            r"\multicolumn{1}{c}{missing}" if c is None
            else f"${c[0]:+.4f} \\pm {'n/a' if math.isnan(c[1]) else f'{c[1]:.4f}'}$" for c in cells]) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    note = ("\nValues: final val MAE difference to sampled-once, mean ± SE over seeds (ddof=1), paired by seed. "
            "Negative = better than sampled-once.\n")
    _write(os.path.join(results_dir, "id_symmetry.md"), "\n".join(md) + "\n" + note + _note(missing))
    _write(os.path.join(results_dir, "id_symmetry.tex"), "\n".join(tex) + "\n")


def final_table(configs, out_dir, results_dir) -> None:
    df, missing = load_frame(configs, out_dir)
    k = configs[0].node_id.k if configs else "?"
    archs = ["egin", "engnn"]
    mean = {a: [np.mean(df[(df["arch"] == a) & (df["task"] == t)]["test_mae"]) if len(df) else float("nan") for t in TASK_ORDER] for a in archs}
    std = {a: [_std(df[(df["arch"] == a) & (df["task"] == t)]["test_mae"], 1) if len(df) else float("nan") for t in TASK_ORDER] for a in archs}

    md = ["| model | k | " + " | ".join(TASK_LABELS[t] for t in TASK_ORDER) + " |", "|---|---|" + "---|" * len(TASK_ORDER)]
    for a in archs:
        md.append(f"| {ARCH_NAMES[a]} | {k} | " + " | ".join(
            _pm(df[(df["arch"] == a) & (df["task"] == t)]["test_mae"], ddof=1) for t in TASK_ORDER) + " |")

    def tex_rows(with_std: bool) -> list[str]:
        rows = []
        for a in archs:
            other = [x for x in archs if x != a][0]
            cells = []
            for i, t in enumerate(TASK_ORDER):
                m, s = mean[a][i], std[a][i]
                if math.isnan(m):
                    cells.append("--")
                    continue
                body = f"{m:.4f}" + (f" $\\pm$ {s:.4f}" if with_std and not math.isnan(s) else "")
                better = round(m, 4) < round(mean[other][i], 4)
                cells.append(f"\\textbf{{{body}}}" if better else body)
            rows.append(f"{ARCH_NAMES[a]:<14} & " + " & ".join(cells) + r" \\")
        return rows

    note = ("\nNormalized test MAE at the best-validation checkpoint, mean ± sample std (ddof=1) over seeds. "
            "The published comparison also lists baselines from the literature and a shading of cells that is not "
            "generated here.\n")
    _write(os.path.join(results_dir, "final_table.md"), "\n".join(md) + "\n" + note + _note(missing))
    _write(os.path.join(results_dir, "final_table.tex"),
           "\n".join(tex_rows(False)) + "\n\n" + "\n".join(tex_rows(True)) + "\n")


def acc_vs_mae(configs, out_dir, results_dir) -> None:
    df, missing = load_frame(configs, out_dir)
    plots.plot_accuracy_vs_mae(df, list(experiments.MAIN_TASKS), list(experiments.ARCHS),
                               os.path.join(results_dir, "acc_vs_mae"))
    lines = ["| arch | task | k | final-epoch val MAE | final-epoch val accuracy (%) |", "|---|---|---|---|---|"]
    for (arch, task, k), g in df.groupby(["arch", "task", "k"]):
        lines.append(f"| {arch} | {task} | {k} | {_pm(g['last_val_mae'])} | {_pm(g['last_val_acc_pct'], digits=2)} |")
    _write(os.path.join(results_dir, "acc_vs_mae.md"), "\n".join(lines) + "\n" + _note(missing))


_REPORTS = {
    "mode_ablation": mode_ablation,
    "realizer_ablation": realizer_ablation,
    "channel_sweep": channel_sweep,
    "vocab_ablation": vocab_ablation,
    "gin_vs_egin": gin_vs_egin,
    "id_symmetry": id_symmetry,
    "final_table": final_table,
    "acc_vs_mae": acc_vs_mae,
}


def _with_results_dir(fn):
    def wrapped(configs, out_dir, results_dir) -> None:
        os.makedirs(results_dir, exist_ok=True)
        fn(configs, out_dir, results_dir)
    return wrapped


REPORTS = {name: _with_results_dir(fn) for name, fn in _REPORTS.items()}
