# Estimating Heterogeneous Agent Models With Neural Networks
Example code for the simple 3 equation New Keynesian model with analytical solution from "Estimating Heterogeneous Agent Models With Neural Networks" by H. Kase, L. Melosi, and M. Rottner

# Running the example

There are few options:

1. Clone the repository and install the package with `pip install .` in the root directory and run the analytical.py or analytical.ipynb files in the examples folder.
```
git clone https://github.com/tseep/estimating-hank-nn.git
cd estimating-hank-nn
pip install .
cd examples
python analytical.py
```
It is perhaps more instructive to open the analytical.py or analytical.ipynb files with vscode or some other python editor and run the code from there.

2. Run the colab_analytical.ipynb file in the examples folder directly on the Google Colab. https://colab.research.google.com/github/tseep/estimating-hank-nn/blob/main/examples/colab_analytical.ipynb

# Requirements
The code is written in Python 3.8 and requires the following packages:
```
numpy
scipy
matplotlib
torch
tqdm
```
Installing the helper package `estimating_hank_nn` should install these dependencies automatically.
```
pip install git+https://github.com/tseep/estimating-hank-nn.git
```
or alternatively
```
git clone https://github.com/tseep/estimating-hank-nn.git
cd estimating-hank-nn
pip install .
```