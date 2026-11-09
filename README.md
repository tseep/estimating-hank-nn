# Estimating Nonlinear Heterogeneous Agent Models with Neural Networks

Code and learning guide for [*Estimating Nonlinear Heterogeneous Agent Models with Neural Networks*](https://www.hannokase.com/research/deep-learning-zlb/KMR_NeuralNetwork_Hank.pdf)
by Hanno Kase, Leonardo Melosi, and Matthias Rottner.

The learning guide is published at <https://tseep.github.io/estimating-hank-nn/>.

The learning examples cover the **New Keynesian baseline and Extensions 1–10**, the
**Krusell–Smith model with one and two assets**, and the **one-asset HANK model**. Each New
Keynesian example contains its complete model and workflow in one percent-format Python
script, paired with a notebook. The heterogeneous agent examples keep the replication's
model, analysis, and plotting modules, driven by one main script and its notebook.
Quarto pages connect the equations to the implementation.

## Run the examples

Use Python 3.12. Run these setup commands once from the repository root. The editable
install (`pip install -e .`) makes the `estimating_hank_nn` package importable from this
checkout, so changes to its source take effect without reinstalling. Each example imports
its helper modules from its own folder.
The pinned environment below is tested locally on macOS CPU.

```bash
conda create -n estimating-hank-nn python=3.12
conda activate estimating-hank-nn
pip install -r requirements.txt
pip install -e .
```

Open `examples/rank/notebooks/rank_baseline.ipynb` and run the cells in order. Select the
environment above as the notebook kernel. The notebooks sit in each example's `notebooks`
folder, and their first cell moves to the example folder, where the scripts, helper modules,
and outputs are. Saved results load by default, and any missing stage is computed. Flags near
the top of each notebook select what to recompute: `load_model`, `load_pf_dataset`, and
`load_pf_network` in the New Keynesian examples, and `load_dss`, `load_full`, and
`load_analysis` in the heterogeneous agent examples. Set a flag to `False` to recompute that
stage. Rerunning a stage also refreshes its dependent results.

Every example ships its trained networks and saved results, so the notebooks reproduce the
figures without retraining. The heterogeneous agent examples also ship the SSJ reference
solution.

| Example | Notebook | Python source |
|---|---|---|
| Baseline | [Notebook](examples/rank/notebooks/rank_baseline.ipynb) | [Script](examples/rank/rank_baseline.py) |
| Baseline, fast version for class | [Notebook](examples/rank/notebooks/rank_baseline_fast.ipynb) | [Script](examples/rank/rank_baseline_fast.py) |
| Extension 1 | [Notebook](examples/rank/notebooks/rank_extension_01.ipynb) | [Script](examples/rank/rank_extension_01.py) |
| Extension 2 | [Notebook](examples/rank/notebooks/rank_extension_02.ipynb) | [Script](examples/rank/rank_extension_02.py) |
| Extension 3 | [Notebook](examples/rank/notebooks/rank_extension_03.ipynb) | [Script](examples/rank/rank_extension_03.py) |
| Extension 4 | [Notebook](examples/rank/notebooks/rank_extension_04.ipynb) | [Script](examples/rank/rank_extension_04.py) |
| Extension 5 | [Notebook](examples/rank/notebooks/rank_extension_05.ipynb) | [Script](examples/rank/rank_extension_05.py) |
| Extension 6 | [Notebook](examples/rank/notebooks/rank_extension_06.ipynb) | [Script](examples/rank/rank_extension_06.py) |
| Extension 7 | [Notebook](examples/rank/notebooks/rank_extension_07.ipynb) | [Script](examples/rank/rank_extension_07.py) |
| Extension 8 | [Notebook](examples/rank/notebooks/rank_extension_08.ipynb) | [Script](examples/rank/rank_extension_08.py) |
| Extension 9 | [Notebook](examples/rank/notebooks/rank_extension_09.ipynb) | [Script](examples/rank/rank_extension_09.py) |
| Extension 10 | [Notebook](examples/rank/notebooks/rank_extension_10.ipynb) | [Script](examples/rank/rank_extension_10.py) |
| Krusell–Smith, one asset | [Notebook](examples/ksone/notebooks/ksone.ipynb) | [Script](examples/ksone/ksone.py) |
| Krusell–Smith, two assets | [Notebook](examples/kstwo/notebooks/kstwo.ipynb) | [Script](examples/kstwo/kstwo.py) |
| One-asset HANK | [Notebook](examples/hankmp/notebooks/hankmp.ipynb) | [Script](examples/hankmp/hankmp.py) |

Use this same Python environment when selecting the notebook kernel or the interpreter
for `# %%` cells in your editor. Each New Keynesian extension is a complete script, so comparing
two consecutive scripts (for example with *Compare Selected* in VS Code) shows exactly what the
extension changes.

## Preview the documentation

Install [Quarto](https://quarto.org). On a fresh clone, copy the figures into the site once,
then preview or render the site from the repository root.

```bash
python3 docs/_prepare.py
quarto preview docs
```

`quarto render docs` writes a static site to `docs/_site/` instead. Quarto reruns
`docs/_prepare.py` before every render, so the site shows the figures the example scripts last
saved. The code shown on the pages is written into the `.qmd` files by hand. Neither step
executes notebooks or trains models. To view the site without internet, add
`--profile offline`, which loads the equation renderer from a local copy that is downloaded
the first time. Build output is ignored by Git.

## Edit the notebooks

Edit the percent-format `.py` scripts, then run `python scripts/sync_notebooks.py` to regenerate
the notebooks in the `notebooks` folders. The scripts carry no notebook metadata. The sync
script sets the kernel and adds the first cell that changes into the example folder.
`--execute` runs every notebook in fresh kernels and can start full training when checkpoints
are missing, so use it only when ready for those runs. It uses the active Python interpreter.
`--check` verifies that notebook cells match their sources.

Loss and likelihood plots use the shared package's replication plotting functions. Policy
plots in `examples/rank/rank_plots.py` adapt the analytical replication example to the
additional outputs and state. Saved checkpoints contain tensors and plain configuration data.
The raw likelihood datasets are not included, because the scripts regenerate them.

## License

The code is released under the MIT License. The outputs shipped with the examples, that is
the figures, trained networks, cached results, and SSJ reference files, are released under
the Creative Commons Attribution 4.0 International License. See `LICENSE`.

Please cite the main paper when using this code.
