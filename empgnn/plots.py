import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

CONDITION_STYLES: dict[str, dict] = {
    "constant": dict(label="constant channels", color="#0072B2", marker="o", ls="-"),
    "minimal": dict(label="minimal channels", color="#D55E00", marker="s", ls="--"),
    "compensated": dict(label="compute-compensated", color="#009E73", marker="^", ls="-."),
    "discrete": dict(label="learned", color="#0072B2", marker="o", ls="-"),
    "ordinal": dict(label="evenly spaced", color="#D55E00", marker="s", ls="--"),
    "fixed_random": dict(label="sampled once", color="#009E73", marker="^", ls="-."),
    "orthogonal": dict(label="orthogonal", color="#CC79A7", marker="D", ls=":"),
    "random": dict(label="random", color="#E69F00", marker="v", ls="--"),
    "full_random": dict(label="full random", color="#56B4E9", marker="P", ls="-."),
    "fixed_random_1d": dict(label="random-1D, sampled once", color="#000000", marker="X", ls="-"),
}
K_STYLES: dict[int, dict] = {
    1: dict(color="#949494", marker="", ls="--"),
    2: dict(color="#0072B2", marker="o", ls="-"),
    4: dict(color="#E69F00", marker="s", ls="--"),
    8: dict(color="#009E73", marker="^", ls="-."),
    16: dict(color="#D55E00", marker="D", ls=":"),
    30: dict(color="#CC79A7", marker="v", ls="-"),
}
ARCH_STYLES: dict[str, dict] = {
    "egin": dict(color="#0072B2", marker="o", ls="-"),
    "engnn": dict(color="#D55E00", marker="s", ls="--"),
}
ARCH_LABELS = {"egin": "EGIN", "engnn": "ENGNN", "gin": "GIN"}
DATASET_TITLES = {"cycles6": "6-Cycle Count", "4path": "4-Path Count"}

THRESHOLD_MAE = 0.01
FIGWIDTH_IN = 6.3
PANEL_LETTERS = "abcdefghij"
LEGEND_KW = dict(handlelength=3, handletextpad=0.5, columnspacing=1.5)
YLABEL = "Validation MAE"

RC = {
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "axes.linewidth": 0.8, "savefig.dpi": 300,
    "pdf.fonttype": 42, "mathtext.fontset": "cm", "figure.labelsize": 9,
}


def _save(fig: plt.Figure, out_stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}")
    plt.close(fig)
    print(f"wrote {out_stem}.{{pdf,png}}")


def log_ticks(lo: float, hi: float) -> list[float]:
    exps = range(int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 1)
    for mantissas in ((1, 3), (1, 2, 5)):
        ticks = [m * 10.0**e for e in exps for m in mantissas]
        ticks = [t for t in ticks if lo <= t <= hi]
        if len(ticks) >= 2:
            return ticks
    return ticks


def style_log_axis(ax: plt.Axes, ticks: list[float]) -> None:
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def _row_title(ax: plt.Axes, text: str) -> None:
    ax.annotate(text, xy=(1.04, 0.5), xycoords="axes fraction", rotation=-90, ha="left", va="center",
                fontsize=plt.rcParams["axes.titlesize"])


def aggregate(df: pd.DataFrame, value: str, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by)[value]
    out = g.agg(mean="mean", std=lambda v: float(np.std(v.to_numpy()))).reset_index()
    out["n"] = g.size().to_numpy()
    return out


def plot_ablation_grid(
    df: pd.DataFrame, conditions: list[str], datasets: list[str], architectures: list[str], out_stem: str,
    threshold: float | None = THRESHOLD_MAE, shared_x: float | None = None, datasets_as_rows: bool = True,
    figwidth: float = FIGWIDTH_IN,
) -> None:
    plt.rcParams.update(RC)
    row_keys, col_keys = (datasets, architectures) if datasets_as_rows else (architectures, datasets)
    row_titles, col_titles = (DATASET_TITLES, ARCH_LABELS) if datasets_as_rows else (ARCH_LABELS, DATASET_TITLES)
    n_rows, n_cols = len(row_keys), len(col_keys)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figwidth, 2.1 * n_rows + 0.4 * (n_rows == 1)),
                             squeeze=False, sharey="row" if datasets_as_rows else "col", layout="constrained")

    all_x = sorted(df["k"].dropna().unique())
    step = min(0.1, 0.4 / max(len(conditions) - 1, 1))
    dodge = {c: step * (i - (len(conditions) - 1) / 2) for i, c in enumerate(conditions)}

    for i, row_key in enumerate(row_keys):
        for j, col_key in enumerate(col_keys):
            ax = axes[i][j]
            task, arch = (row_key, col_key) if datasets_as_rows else (col_key, row_key)
            cell = df[(df["arch"] == arch) & (df["task"] == task)]
            for cond in conditions:
                rows = cell[cell["condition"] == cond]
                if rows.empty:
                    continue
                agg = aggregate(rows, "final_val_mae", ["k"]).set_index("k").reindex(all_x)
                x = np.asarray(all_x, dtype=float)
                y, s = agg["mean"].to_numpy(), agg["std"].fillna(0.0).to_numpy()
                st = CONDITION_STYLES[cond]
                xd = x * 2.0 ** dodge[cond]
                own = np.ones(len(x), dtype=bool)
                if shared_x is not None:
                    at = x == shared_x
                    xd[at] = shared_x
                    own &= ~at
                ax.plot(xd, y, color=st["color"], linestyle=st["ls"], linewidth=1.4, zorder=3)
                ax.errorbar(xd[own], y[own], yerr=s[own], color=st["color"], marker=st["marker"], linestyle="none",
                            zorder=3, markersize=4, elinewidth=0.8, capsize=2, capthick=0.8)
            if shared_x is not None and not cell.empty:
                at = cell[cell["k"] == shared_x]
                if not at.empty:
                    agg = aggregate(at, "final_val_mae", ["k"])
                    ax.errorbar([shared_x], agg["mean"], yerr=agg["std"], color="0.2", marker="o", linestyle="none",
                                zorder=4, markersize=4, elinewidth=0.8, capsize=2, capthick=0.8)

            if threshold is not None:
                ax.axhline(threshold, color="0.5", linestyle="--", linewidth=0.8, zorder=1)
            ax.set_xscale("log", base=2)
            ax.set_xticks(all_x)
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            ax.set_yscale("log")
            ax.grid(True, which="major", linewidth=0.5, alpha=0.35)
            if i == 0:
                ax.set_title(col_titles.get(col_key, col_key))
            if j == n_cols - 1 and n_rows > 1:
                _row_title(ax, row_titles.get(row_key, row_key))
            ax.text(0.03, 0.95, f"({PANEL_LETTERS[i * n_cols + j]})", transform=ax.transAxes,
                    ha="left", va="top", fontsize=8, fontweight="bold")

    groups = [list(axes[i]) for i in range(n_rows)] if datasets_as_rows else [list(axes[:, j]) for j in range(n_cols)]
    for group in groups:
        lo, hi = group[0].get_ylim()
        for ax in group:
            style_log_axis(ax, log_ticks(lo, hi))
        group[0].set_ylim(lo, hi)

    fig.supxlabel("$k$")
    fig.supylabel(YLABEL)
    handles = [Line2D([], [], color=CONDITION_STYLES[c]["color"], marker=CONDITION_STYLES[c]["marker"],
                      ls=CONDITION_STYLES[c]["ls"], markersize=4, linewidth=1.0) for c in conditions]
    fig.legend(handles, [CONDITION_STYLES[c]["label"] for c in conditions], loc="outside upper center",
               ncols=len(conditions), frameon=False, **LEGEND_KW)
    _save(fig, out_stem)


def plot_channel_sweep(df: pd.DataFrame, datasets: list[str], architectures: list[str], out_stem: str) -> None:
    plt.rcParams.update(RC)
    ks = [k for k in K_STYLES if k != 1]
    cmap = plt.get_cmap("viridis")
    colors = {k: cmap(t) for k, t in zip(ks, np.linspace(0.75, 0.0, len(ks)))}
    dodge = {k: 0.07 * (i - (len(ks) - 1) / 2) for i, k in enumerate(ks)}
    ref = K_STYLES[1]

    n_rows, n_cols = len(datasets), len(architectures)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIGWIDTH_IN, 2.8 + 2.3 * (n_rows - 1)), sharex=True,
                             squeeze=False, layout="constrained")
    all_x = []
    for i, task in enumerate(datasets):
        for j, arch in enumerate(architectures):
            ax = axes[i][j]
            cell = df[(df["arch"] == arch) & (df["task"] == task)]
            lo, hi = 0.01, 0.01
            k1 = cell[cell["k"] == 1]
            if not k1.empty:
                m = float(k1["final_val_mae"].mean())
                ax.axhline(m, color=ref["color"], ls=ref["ls"], lw=0.9, zorder=1)
                lo, hi = min(lo, m), max(hi, m)
            for k in ks:
                agg = aggregate(cell[cell["k"] == k], "final_val_mae", ["n_channels"]).sort_values("n_channels") \
                    if (cell["k"] == k).any() else None
                if agg is None:
                    continue
                x = agg["n_channels"].to_numpy(dtype=float) * 2.0 ** dodge[k]
                y, s = agg["mean"].to_numpy(), agg["std"].to_numpy()
                all_x.extend(x)
                lo, hi = min(lo, (y - s)[y - s > 0].min(initial=y.min())), max(hi, (y + s).max())
                ax.errorbar(x, y, yerr=s, color=colors[k], marker=K_STYLES[k]["marker"], ls=K_STYLES[k]["ls"],
                            markersize=4, linewidth=1.3, elinewidth=0.5, capsize=1.2, capthick=0.5, zorder=3)
            if i == 0:
                ax.set_title(ARCH_LABELS[arch])
            if n_rows > 1 and j == n_cols - 1:
                _row_title(ax, DATASET_TITLES[task])
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_ylim(lo / 1.15, hi * 1.4)
            style_log_axis(ax, log_ticks(lo / 1.15, hi * 1.4))
            ax.grid(True, which="major", linewidth=0.5, alpha=0.35)

    ticks = [2**i for i in range(9) if min(all_x) / 1.2 <= 2**i <= max(all_x) * 1.2]
    for ax in axes.flat:
        ax.set_xlim(min(all_x) / 2**0.25, max(all_x) * 2**0.25)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    fig.supxlabel("Equivariant channels $c$")
    fig.supylabel(YLABEL)
    handles = [Line2D([], [], color=ref["color"], ls=ref["ls"], lw=0.9)]
    names = ["$k=1$"]
    handles += [Line2D([], [], color=colors[k], marker=K_STYLES[k]["marker"], ls=K_STYLES[k]["ls"], markersize=4,
                       linewidth=1.0) for k in ks]
    names += [f"$k={k}$" for k in ks]
    fig.legend(handles, names, loc="outside upper center", ncols=len(handles), frameon=False, **LEGEND_KW)
    _save(fig, out_stem)


def plot_vocab_ablation(df: pd.DataFrame, datasets: list[str], architectures: list[str], out_stem: str,
                        vocab_sizes=(30, 45, 60, 90, 120)) -> None:
    plt.rcParams.update(RC)
    main, capped, rand = CONDITION_STYLES["fixed_random"], K_STYLES[1], CONDITION_STYLES["full_random"]
    n_rows, n_cols = len(datasets), len(architectures)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIGWIDTH_IN if n_cols > 1 else 3.6, 2.6 + 2.1 * (n_rows - 1)),
                             sharey="row", squeeze=False, layout="constrained")
    for i, task in enumerate(datasets):
        for j, arch in enumerate(architectures):
            ax = axes[i][j]
            cell = df[(df["arch"] == arch) & (df["task"] == task)]
            unc = aggregate(cell[cell["series"] == "uncapped"], "final_val_mae", ["k"]).set_index("k").reindex(vocab_sizes)
            ax.errorbar(list(vocab_sizes), unc["mean"], yerr=unc["std"].fillna(0.0), color=main["color"],
                        marker=main["marker"], ls=main["ls"], markersize=4, linewidth=1.4, elinewidth=0.8,
                        capsize=2, capthick=0.8, zorder=3)
            cap, rnd = cell[cell["series"] == "capped"], cell[cell["series"] == "random"]
            if not cap.empty:
                ax.axhline(cap["final_val_mae"].mean(), color=capped["color"], ls=capped["ls"], lw=0.9, zorder=1)
            if not rnd.empty:
                ax.axhline(rnd["final_val_mae"].mean(), color=rand["color"], ls=rand["ls"], lw=1.1, zorder=1)
            if i == 0:
                ax.set_title(ARCH_LABELS[arch])
            if n_rows > 1 and j == n_cols - 1:
                _row_title(ax, DATASET_TITLES[task])
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(list(vocab_sizes))
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            ax.grid(True, which="major", linewidth=0.5, alpha=0.35)
    for row in axes:
        lo, hi = row[0].get_ylim()
        lo, hi = lo / 1.2, hi * 1.1
        for ax in row:
            style_log_axis(ax, log_ticks(lo, hi))
        row[0].set_ylim(lo, hi)

    fig.supxlabel("$k$ (vocabulary size)")
    fig.supylabel(YLABEL)
    fig.legend(
        [Line2D([], [], color=main["color"], marker=main["marker"], ls=main["ls"], markersize=4, linewidth=1.0),
         Line2D([], [], color=capped["color"], ls=capped["ls"], lw=0.9),
         Line2D([], [], color=rand["color"], ls=rand["ls"], lw=1.1)],
        ["uncapped vocabulary", "capped, $k=30$", "random, $k=30$"],
        loc="outside upper center", ncols=3 if n_cols > 1 else 2, frameon=False, **LEGEND_KW,
    )
    _save(fig, out_stem)


def plot_accuracy_vs_mae(df: pd.DataFrame, datasets: list[str], architectures: list[str], out_stem: str) -> None:
    plt.rcParams.update(RC)
    metrics = {
        "last_val_mae": ("Validation MAE", THRESHOLD_MAE, True),
        "last_val_acc_pct": ("Validation accuracy (%)", 100.0, False),
    }
    ks = sorted(df["k"].unique())
    dodge = {a: 0.05 * (2 * i - 1) for i, a in enumerate(architectures)}
    fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(FIGWIDTH_IN, 4.6), layout="constrained", squeeze=False)
    for i, task in enumerate(datasets):
        for j, (metric, (title, ref, y_log)) in enumerate(metrics.items()):
            ax = axes[i][j]
            for arch in architectures:
                sub = df[(df["task"] == task) & (df["arch"] == arch)]
                agg = aggregate(sub, metric, ["k"]).set_index("k").reindex(ks)
                x = np.asarray(ks, dtype=float) * 2.0 ** dodge[arch]
                st = ARCH_STYLES[arch]
                ax.plot(x, agg["mean"], color=st["color"], linestyle=st["ls"], linewidth=1.4, zorder=3)
                ax.errorbar(x, agg["mean"], yerr=agg["std"].fillna(0.0), color=st["color"], marker=st["marker"],
                            linestyle="none", zorder=3, markersize=4, elinewidth=0.8, capsize=2, capthick=0.8)
            ax.axhline(ref, color="0.5", linestyle="--", linewidth=0.8, zorder=1)
            ax.set_xscale("log", base=2)
            ax.set_xticks(ks)
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            if y_log:
                ax.set_yscale("log")
                lo, hi = ax.get_ylim()
                hi *= 1.3
                style_log_axis(ax, log_ticks(lo, hi))
                ax.set_ylim(lo, hi)
            ax.grid(True, which="major", linewidth=0.5, alpha=0.35)
            if i == 0:
                ax.set_title(title)
            if j == len(metrics) - 1:
                _row_title(ax, DATASET_TITLES[task])
            ax.text(0.03, 0.95, f"({PANEL_LETTERS[i * len(metrics) + j]})", transform=ax.transAxes,
                    ha="left", va="top", fontsize=8, fontweight="bold")
    fig.supxlabel("$k$")
    handles = [Line2D([], [], color=ARCH_STYLES[a]["color"], marker=ARCH_STYLES[a]["marker"], ls=ARCH_STYLES[a]["ls"],
                      markersize=4, linewidth=1.0) for a in architectures]
    fig.legend(handles, [ARCH_LABELS[a] for a in architectures], loc="outside upper center",
               ncols=len(architectures), frameon=False, **LEGEND_KW)
    _save(fig, out_stem)
