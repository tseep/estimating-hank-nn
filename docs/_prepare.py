"""Copy the figures of the examples into the site. Quarto runs this before every render (the
pre-render hook in _quarto.yml), because it can only serve files inside the docs folder. With
the offline profile (quarto render docs --profile offline), it also fetches a local MathJax copy
the first time, so equations render without internet.
"""

import os
import shutil
import urllib.request
from pathlib import Path


# %% Settings
DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
# Figure folders, copied to assets/results/<example>/<run>/.
FIGURES = (*(("rank", lesson) for lesson in ("baseline", *(f"extension_{i:02}" for i in range(1, 11)))),
           ("ksone", "baseline"), ("kstwo", "baseline"), ("hankmp", "baseline"), ("hankmp", "robustness"))
MATHJAX = DOCS / "assets/mathjax/tex-svg-full.js"
MATHJAX_URL = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg-full.js"

# %% Copy the figures
for example, run in FIGURES:
    target = DOCS / "assets/results" / example / run
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    for source in (ROOT / f"examples/{example}/outputs/{run}/figures").glob("*.png"):
        shutil.copy2(source, target / source.name)

# %% MathJax for the offline profile
if "offline" in os.environ.get("QUARTO_PROFILE", "") and not MATHJAX.exists():
    print(f"Downloading MathJax to {MATHJAX.relative_to(ROOT)} (needs internet once)")
    MATHJAX.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MATHJAX_URL, MATHJAX)
