# empgnn

Code accompanying my master's thesis, *Between Architectural and Learned Symmetry: Node Identifiers in Equivariant Graph Neural Networks*.

> **Note:** This is research code, published for transparency and reproducibility. It is not a maintained library
> and was developed and tested only with Python 3.12 (CPU and CUDA).

EGIN, Modified ENGNN and a GIN baseline with node-id assignments, for node-level substructure counting.
Defaults are the thesis setup and every config field can be overridden on the command line.

```
pip install torch torch_geometric scipy numpy matplotlib
```

## Usage

```
python -m empgnn train --task cycles6 --arch egin --k 2 --device cpu --epochs 5
python -m empgnn list
python -m empgnn sweep final_table --shard 0/4
python -m empgnn report final_table
```

`train` takes any config field as `--<section>.<field>` (e.g. `--model.d_e 32`). `sweep` runs a group of experiments
(use `--dry-run`, `--tasks`, `--archs`, `--seeds` to limit it) and skips finished runs,
`report` writes tables and figures to `results/`.

## Data

`data/counting.npz` holds the nine counting tasks (`cycles3..6`, `tailtri`, `chordcyc`, `4clique`, `4path`, `trirect`).
To rebuild it from the raw [I2GNN](https://github.com/GraphPKU/I2GNN) files:
`python -m empgnn convert-data cycles_count/data.mat graphlet_count/data.mat`.

## Notes

- The vocabulary ablation (`node_id.uncapped_vocab`) uses EGIN with concat merging, as in the thesis experiments.
- The thesis text describes the final table as k=4, but its numbers come from k=2 runs. `sweep final_table` reproduces
  k=2; `--final-k 4` runs k=4, where EGIN is clearly worse on tri-rect (0.121 vs 0.025) and Modified ENGNN is unchanged.

## License

MIT, see [LICENSE](LICENSE). The data derived from I2GNN and the referenced models remain under their original licenses.

## Credits

- Modified ENGNN follows [EquivNoiseGNN](https://github.com/MuLabPKU/EquivNoiseGNN): Wang & Zhang, *Expressive Graph Neural
  Networks via Equivariant Use of Noise*, ICML 2026.
- Data: Zhao, Jin, Akoglu & Shah, *From Stars to Subgraphs: Uplifting Any GNN with Local Structure Awareness*, ICLR 2022.
- GIN: Xu et al., *How Powerful are Graph Neural Networks?*, ICLR 2019.
