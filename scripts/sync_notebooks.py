"""Generate the notebooks in each example's notebooks/ folder from the percent-format scripts.
Optionally execute them in fresh kernels."""

import argparse
import os
import sys
from pathlib import Path

import jupytext
import nbformat

# %% Settings
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (ROOT / "examples/rank/rank_baseline.py",
           ROOT / "examples/rank/rank_baseline_fast.py",
           *sorted((ROOT / "examples/rank").glob("rank_extension_*.py")),
           ROOT / "examples/ksone/ksone.py",
           ROOT / "examples/kstwo/kstwo.py",
           ROOT / "examples/hankmp/hankmp.py")
KERNEL = {"display_name": "Python 3", "language": "python", "name": "python3"}
# First cell of every notebook: the scripts, helper modules, and outputs live one folder up.
FIRST_CELL = """# The notebook lives in the notebooks folder. Work from the example folder, where the
# scripts, helper modules, and outputs are.
import os

if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")"""

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check", action="store_true", help="Check Python/notebook agreement without writing")
parser.add_argument("--execute", action="store_true", help="Run the notebooks using this Python interpreter")
parser.add_argument("--timeout", type=int, default=None, help="Seconds per cell when executing (default: no limit)")
parser.add_argument("names", nargs="*", help="Only these examples, by script name (default: all)")
args = parser.parse_args()
if args.check and args.execute:
    parser.error("--check and --execute cannot be combined")


def cell_sources(notebook):
    return [(cell.cell_type, cell.source) for cell in notebook.cells]


# %% Generate, check, or execute
for source in SCRIPTS:
    if args.names and source.stem not in args.names:
        continue
    path = source.parent / "notebooks" / source.with_suffix(".ipynb").name
    notebook = jupytext.read(source, fmt="py:percent")
    notebook.metadata = {"kernelspec": KERNEL}
    notebook.cells.insert(0, nbformat.v4.new_code_cell(FIRST_CELL))
    previous = nbformat.read(path, as_version=4) if path.exists() else None
    if args.check:
        if previous is None or cell_sources(notebook) != cell_sources(previous):
            raise SystemExit(f"Notebook is out of sync: {path.relative_to(ROOT)}")
        print("In sync:", path.relative_to(ROOT))
        continue
    if previous is not None:
        # Keep the ids of unchanged cells, so a notebook that did not change does not churn in git.
        for new, old in zip(notebook.cells, previous.cells):
            if (new.cell_type, new.source) == (old.cell_type, old.source) and "id" in old:
                new.id = old.id
    if args.execute:
        from jupyter_client import KernelManager
        from nbclient import NotebookClient

        os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
        os.environ.setdefault("IPYTHONDIR", str(ROOT / ".cache/ipython"))
        kernel = KernelManager(kernel_name="python3")
        kernel.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        NotebookClient(notebook, km=kernel, timeout=args.timeout,
                       resources={"metadata": {"path": str(path.parent)}}).execute(cleanup_kc=True)
    path.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, path)
    print("Executed:" if args.execute else "Generated:", path.relative_to(ROOT))
