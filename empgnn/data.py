import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .config import TASKS, DataCfg


class CountingDataset(Dataset):
    def __init__(self, graphs: list[Data]):
        self.graphs = graphs
        self.num_node_features = graphs[0].x.shape[1]

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]


@dataclass
class Splits:
    train: CountingDataset
    val: CountingDataset
    test: CountingDataset
    label_mean: float
    label_std: float

    def size_stats(self, include_test: bool = False) -> tuple[int, float]:
        datasets = [self.train, self.val] + ([self.test] if include_test else [])
        graphs = [g for ds in datasets for g in ds.graphs]
        return max(int(g.num_nodes) for g in graphs), sum(int(g.edge_index.shape[1]) for g in graphs) / len(graphs)


def data_file(cfg: DataCfg) -> str:
    return os.path.join(cfg.data_dir, "counting.npz")


def _graph(z, i: int, col: int, mean: float, std: float) -> Data:
    n0, n1 = z["node_ptr"][i], z["node_ptr"][i + 1]
    e0, e1 = z["edge_ptr"][i], z["edge_ptr"][i + 1]
    n = int(n1 - n0)
    edge_index = torch.from_numpy(z["edge_index"][:, e0:e1].astype(np.int64))
    x = torch.ones(n, 1, dtype=torch.float32)
    raw_y = z["F"][n0:n1, col]
    y = torch.tensor((raw_y - mean) / std, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=n)


def load_splits(cfg: DataCfg) -> Splits:
    if cfg.task not in TASKS:
        raise ValueError(f"unknown task {cfg.task!r}; choose from {sorted(TASKS)}")
    col = TASKS[cfg.task]
    path = data_file(cfg)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; see the Data section of the README")
    z = dict(np.load(path))
    idx = {name: z[f"{name}_idx"] for name in ("train", "val", "test")}

    labels = np.concatenate([z["F"][z["node_ptr"][i]:z["node_ptr"][i + 1], col] for i in np.concatenate([idx["train"], idx["val"]])])
    mean, std = float(labels.mean()), float(labels.std()) + 1e-8

    sets = {name: CountingDataset([_graph(z, i, col, mean, std) for i in ii]) for name, ii in idx.items()}
    return Splits(sets["train"], sets["val"], sets["test"], mean, std)


def convert_mat(cycles_mat: str, graphlet_mat: str, out: str) -> None:
    import scipy.io
    a, b = scipy.io.loadmat(cycles_mat), scipy.io.loadmat(graphlet_mat)
    A, Fa, Fb = a["A"][0], a["F"][0], b["F"][0]
    for name in ("train", "val", "test"):
        assert np.array_equal(a[f"{name}_idx"], b[f"{name}_idx"]), f"{name} split differs between the two files"
    same = lambda x, y: (x != y).nnz == 0 if hasattr(x, "nnz") else np.array_equal(x, y)  # noqa: E731  (sparse or dense)
    assert len(A) == len(b["A"][0]) and all(same(x, y) for x, y in zip(A, b["A"][0])), "graphs differ between the files"

    edges, n_nodes = [], []
    for A_i, F_i in zip(A, Fa):
        rows, cols = A_i.nonzero()
        edges.append(np.stack([rows, cols]))
        n_nodes.append(F_i.shape[0])
    edge_ptr = np.concatenate([[0], np.cumsum([e.shape[1] for e in edges])])
    node_ptr = np.concatenate([[0], np.cumsum(n_nodes)])
    F = np.concatenate([np.concatenate([fa, fb], axis=1) for fa, fb in zip(Fa, Fb)]).astype(np.float32)
    assert F.shape[1] == len(TASKS)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, edge_index=np.concatenate(edges, axis=1).astype(np.int16), edge_ptr=edge_ptr, node_ptr=node_ptr, F=F,
                        columns=np.array(list(TASKS)), **{f"{n}_idx": a[f"{n}_idx"][0].astype(np.int64) for n in ("train", "val", "test")})


def make_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available())
