import argparse
import dataclasses
import enum
import inspect
import types
import typing

from . import experiments, report
from .config import Config
from .run import run, run_key

SECTIONS = ("model", "node_id", "train", "data")
ALIASES = {"--device": "train.device", "--task": "data.task", "--arch": "model.arch", "--k": "node_id.k",
           "--epochs": "train.epochs"}


def _parser_for(tp):
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        inner = [a for a in typing.get_args(tp) if a is not type(None)][0]
        parse = _parser_for(inner)
        return lambda s: None if s.lower() == "none" else parse(s)
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return lambda s: tp(s.lower())
    if tp is bool:
        return lambda s: s.lower() in ("1", "true", "yes")
    return tp


UNSET = object()


def add_config_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("config overrides (defaults: the thesis setup)")
    defaults = Config()
    group.add_argument("--seed", dest="seed", type=int, default=None, help=f"default {defaults.seed}")
    group.add_argument("--out-dir", dest="out_dir", type=str, default=None, help=f"default {defaults.out_dir}")
    for section in SECTIONS:
        section_hints = typing.get_type_hints(type(getattr(defaults, section)))
        for f in dataclasses.fields(type(getattr(defaults, section))):
            default = getattr(getattr(defaults, section), f.name)
            shown = default.value if isinstance(default, enum.Enum) else default
            flags = [f"--{section}.{f.name}"] + [a for a, dest in ALIASES.items() if dest == f"{section}.{f.name}"]
            group.add_argument(*flags, dest=f"{section}.{f.name}", type=_parser_for(section_hints[f.name]),
                               default=UNSET, help=f"default {shown}")


def apply_overrides(cfg: Config, ns: argparse.Namespace) -> Config:
    for section in SECTIONS:
        for f in dataclasses.fields(type(getattr(cfg, section))):
            value = getattr(ns, f"{section}.{f.name}", UNSET)
            if value is not UNSET:
                setattr(getattr(cfg, section), f.name, value)
    for name in ("seed", "out_dir"):
        if getattr(ns, name, None) is not None:
            setattr(cfg, name, getattr(ns, name))
    return cfg


def family_configs(args: argparse.Namespace) -> list[Config]:
    builder = experiments.FAMILIES[args.family]
    params = inspect.signature(builder).parameters
    kwargs = {}
    if args.tasks and "tasks" in params:
        kwargs["tasks"] = tuple(args.tasks.split(","))
    if args.archs and "archs" in params:
        kwargs["archs"] = tuple(args.archs.split(","))
    if args.seeds is not None:
        kwargs["seeds"] = args.seeds
    if args.final_k is not None and args.family == "final_table":
        kwargs["k"] = args.final_k
    configs = []
    for cfg in builder(**kwargs):
        configs.append(apply_overrides(cfg, args))
    return experiments.unique(configs)


def _add_family_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("family", choices=sorted(experiments.FAMILIES))
    p.add_argument("--tasks", type=str, default=None, help="comma-separated subset, e.g. cycles6,4path")
    p.add_argument("--archs", type=str, default=None, help="comma-separated subset of egin,engnn")
    p.add_argument("--seeds", type=int, default=None, help=f"seeds per config (default {experiments.SEEDS})")
    p.add_argument("--final-k", type=int, default=None, help="final_table only: k (default 2, as in the thesis table)")
    add_config_flags(p)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m empgnn", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="one run: train, then test the best-validation checkpoint")
    add_config_flags(p)
    p.add_argument("--name", type=str, default=None, help="run directory name (default: derived from the config)")
    p.add_argument("--keep-checkpoints", action="store_true")

    p = sub.add_parser("sweep", help="run all runs of an experiment family (resumable)")
    _add_family_args(p)
    p.add_argument("--shard", type=str, default=None, help="i/n: run only every n-th run, starting at i (to split over GPUs)")
    p.add_argument("--dry-run", action="store_true", help="list the runs without training")
    p.add_argument("--keep-checkpoints", action="store_true")

    p = sub.add_parser("report", help="tables and figures of an experiment family from its finished runs")
    _add_family_args(p)
    p.add_argument("--results-dir", type=str, default="results")

    sub.add_parser("list", help="list the experiment families and their number of runs")

    p = sub.add_parser("convert-data", help="build data/counting.npz from the two raw .mat files distributed with I2GNN")
    p.add_argument("cycles_mat", help="cycles_count/data.mat")
    p.add_argument("graphlet_mat", help="graphlet_count/data.mat")
    p.add_argument("--out", type=str, default="data/counting.npz")

    args = parser.parse_args(argv)

    if args.command == "convert-data":
        from .data import convert_mat
        convert_mat(args.cycles_mat, args.graphlet_mat, args.out)
        print("wrote", args.out)
        return
    if args.command == "train":
        cfg = apply_overrides(Config(), args)
        record = run(cfg, args.name, keep_checkpoints=args.keep_checkpoints)
        print(record["metrics"] if record["error"] is None else record["error"])
    elif args.command == "sweep":
        configs = family_configs(args)
        if args.shard:
            i, n = map(int, args.shard.split("/"))
            configs = configs[i::n]
        print(f"{len(configs)} runs")
        for j, cfg in enumerate(configs):
            print(f"[{j + 1}/{len(configs)}] {run_key(cfg)}")
            if not args.dry_run:
                run(cfg, keep_checkpoints=args.keep_checkpoints)
    elif args.command == "report":
        configs = family_configs(args)
        out_dir = configs[0].out_dir if configs else Config().out_dir
        report.REPORTS[args.family](configs, out_dir, args.results_dir)
    else:
        for name, builder in experiments.FAMILIES.items():
            print(f"{name:20s} {len(experiments.unique(builder())):5d} runs   {(builder.__doc__ or '').strip().splitlines()[0]}")


if __name__ == "__main__":
    main()
