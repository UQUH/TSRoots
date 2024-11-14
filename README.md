<div align="center">
  <img src="docs/img/TSRoots_Logo.svg" alt="TSRoots_Logo" style="width: 25%;">
</div>

---
[![Test Suite](https://github.com/UQUH/TSRoots/actions/workflows/python-test.yml/badge.svg)](https://github.com/UQUH/TSRoots/actions/workflows/python-test.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/tsroots.svg)](https://pypi.org/project/tsroots/)

<div align="center">
  <img src="docs/img/bo_iterations.gif" alt="BO Iterations GIF" style="width: 100%;">
</div>

<p align="center">
<a href="https://openreview.net/forum?id=IpRLTVblaV">Paper</a>,
<a href="https://arxiv.org/abs/2410.22322">Paper</a>,
<a href="https://github.com/UQUH/TSRoots/tree/main/docs/tutorials">Tutorials</a>

&nbsp;\
**TSRoots**
> A Python package for efficient Gaussian process Thompson sampling in Bayesian optimization via rootfinding.

&nbsp;\
Bayesian optimization (BO) uses acquisition functions to guide the optimization of expensive objective functions.
Gaussian process Thompson sampling (GP-TS), a popular BO acquisition strategy, optimizes posterior samples to guide exploration and exploitation. 
However, these samples can be highly complex, which makes their global optimization computationally challenging.

TSRoots accelerates this process by leveraging the separability of multivariate Gaussian Process priors and
a decoupled representation of the posterior. Integrated with advanced root-finding techniques, TSRoots 
efficiently and adaptively selects starting points for gradient-based multistart optimization. 
This results in high-quality solutions for GP-TS, enabling robust performance in both low- and high-dimensional settings.


## Installation

### Requirements
- Python >= 3.7
- PyTorch
- chebpy 

#### Requirements Installation Dependencies
Some required dependencies, such as `torch` and `chebpy` are not installed by default. 
- To install PyTorch,
we recommend installing the appropriate version of PyTorch for your system by following the instructions here:
[PyTorch Installation Instructions](https://pytorch.org/get-started/locally/). Although least preferred, you can directly 
install for CPU version by running `pip install torch`. 
- To install chebpy, you can see installation instructions here: 
[Chepy Installation Instructions](https://github.com/chebpy/chebpy/blob/master/INSTALL.rst). You can also directly
install chebpy via `pip install git+https://github.com/chebpy/chebpy.git`

#### Once the above requirements have been satisfied, you can install the `TSRoots` package in various ways: using `pip`, or directly from Github.

### Lightweight Installation of TSRoots
Using pip:
```bash
pip install tsroots
```

[//]: # (Via conda:)

[//]: # (```bash)

[//]: # (conda install -c conda-forge tsroots)

[//]: # (```)

### Development Version

If you are contributing a pull request or for a full installation with examples, tests, and the latest updates, 
it is best to perform a manual installation:

```bash
git clone [https://github.com/your_username/TS-roots.git](https://github.com/UQUH/TSRoots.git)
cd TS-roots
pip install -e .[docs,pytorch,test]
pip install git+https://github.com/chebpy/chebpy.git  # Install Chebpy from git
````

To verify correct installation, you can run on the [test suite](tests/) on your terminal via:
```bash
python shell/run_all_tests.py
```

## Quick Start
This example demonstrates TSRoots' core functionality of generating new observation points for Bayesian optimization 
using normalized data and gradient-based rootfinding techniques.
For a more detailed overview of model fitting, rootfinding, decoupled GP representation, and BO implementation including
generating the animated plot above, check out the [Getting Started Notebook](docs/tutorials/notebook_getting_started.ipynb).

```python
from tsroots.optim import TSRoots
from tsroots.utils import generate_Xdata, generate_Ydata

import numpy as np
import matplotlib.pyplot as plt

# Define the objective function
def f_objective_example(x):
    return x * np.sin(x)

# Define bounds and generate sample data
lb_x_physical = np.array([-15])
ub_x_physical = np.array([15])
no_sample = 5
D = 1
seed = 42

# Generate initial samples and normalize them
X_physical_space, X_normalized = generate_Xdata(no_sample, D, seed, lb_x_physical, ub_x_physical)
Y_physical_space, Y_normalized = generate_Ydata(f_objective_example, X_physical_space)

# Instantiate and use TSRoots for optimization
TSRoots_BO = TSRoots(X_normalized, Y_normalized.flatten(), -np.ones(D), np.ones(D))
x_new_normalized, y_new_normalized, _ = TSRoots_BO.xnew_TSroots(plot=True)
# plot selected point
plt.scatter(x_new_normalized, y_new_normalized, color='blue', marker='x', linewidth=3.0, label='Selected Point')
plt.show()

print(f"New observation location: {x_new_normalized}")
print(f"New function value: {y_new_normalized}")
```


## Citation

If you found TSRoots helpful, please cite the [following
paper](https://openreview.net/forum?id=IpRLTVblaV):
```
@inproceedings{Adebiyi2024bdu,
title={Gaussian Process Thompson Sampling via Rootfinding},
author={Taiwo Adebiyi and Bach Do and Ruda Zhang},
booktitle={NeurIPS 2024 Workshop on Bayesian Decision-making and Uncertainty},
year={2024},
url={https://openreview.net/forum?id=IpRLTVblaV}
}
```

The full-length paper is available at:
```
@misc{Adebiyi2024tsroots,
title={Optimizing Posterior Samples for Bayesian Optimization via Rootfinding}, 
author={Taiwo A. Adebiyi and Bach Do and Ruda Zhang},
year={2024},
eprint={2410.22322},
archivePrefix={arXiv},
url={https://arxiv.org/abs/2410.22322}, 
}
```

## The Team

TSRoots is produced by the [Uncertainty Quantification Lab](https://uq.uh.edu/group-members) at the University of Houston.
The primary maintainers are:
- Taiwo A. Adebiyi
- Bach Do
- Ruda Zhang
