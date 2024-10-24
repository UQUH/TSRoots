
<div align="center">
  <img src="TSRoots_Logo.svg" alt="TSRoots_Logo" style="width: 25%;">
</div>
<div align="center">
  <img src="bo_iterations.gif" alt="BO Iterations GIF" style="width: 75%;">
</div>

<p align="center">
    <a href="https://uncertainty-toolbox.github.io/">Website</a>,
    <a href="https://uncertainty-toolbox.github.io/tutorial/">Tutorials</a>, and
    <a href="https://uncertainty-toolbox.github.io/docs/">Docs</a>

&nbsp;\
**TSRoots**
> A Python Package for Efficient Global Optimization of 
> Posterior-Based Acquisition Functions in Bayesian Optimization

&nbsp;\
Bayesian Optimization (BO) uses acquisition functions as surrogates for expensive objective functions.
Thompson Sampling, a popular BO strategy, optimizes posterior samples to guide exploration and exploitation. 
However, optimizing these samples can be complex and computationally challenging.

TSRoots streamlines this process by leveraging the separability of the multivariate Gaussian Process prior and
a decoupled representation of the posterior. Integrated with advanced root-finding techniques, TSRoots 
efficiently selects starting points for gradient-based multistart optimization. 
This results in higher-quality solutions for posterior sample-based acquisition functions, enabling robust 
performance in both low- and high-dimensional settings.


## Installation

### Requirements
- Python >= 3.7
- PyTorch
- chebpy 

#### Requirments Installation Dependencies
Some required dependencies, such as `torch` and `chebpy` are not installed by default. 
- To install PyTorch,
we recommend installing the appropriate version of PyTorch for your system by following the instructions here:
[PyTorch Installation Instructions](https://pytorch.org/get-started/locally/). Although least preferd, you can directly 
install for CPU version by running `pip install torch`. 
- To install ChebPy, you can see installation instructions here: 
[Chepy Installation Instructions](https://github.com/chebpy/chebpy/blob/master/INSTALL.rst)

#### You can install the `TSRoots` package in various ways: using `pip`,`conda`, or directly from Github.

### Lightweight Installation of TSRoots
Using pip:
```bash
pip install tsroots
```
Via conda:
```bash
conda install -c conda-forge tsroots
```


### Development Version

If you are contributing a pull request or for a full installation with examples, tests, and the latest updates, 
it is best to perform a manual installation:

```bash
git clone https://github.com/your_username/TS-roots.git
cd TS-roots
pip install -e .[docs,chebpy_git,pytorch,test]
````

To verify correct installation, you can run on the [test suite](tests/) on your terminal via:
```bash
python shell/run_all_tests.py
```

## Quick Start
This example demonstrates the core functionality of TS_roots to generate new points for Bayesian Optimization 
using normalized data and gradient-based rootfinding techniques.
For a more detailed overview of model fitting, rootfinding, decoupled GP representation, and BO implementation including
generating the dynamic plot above, check out the [Getting Started Notebook](notebook_getting_started.ipynb).

```python
import numpy as np
from tsroots.optim import TSRoots
from tsroots.utils import generate_Xdata, generate_Ydata

# Define the objective function
def f_objective_example(x):
    return x * np.sin(x)

# Define bounds and generate sample data
lb_x_physical = np.array([-15])
ub_x_physical = np.array([15])
no_sample = 5
D = 1
seed = 10

# Generate initial samples and normalize them
X_physical_space, X_normalized = generate_Xdata(no_sample, D, seed, lb_x_physical, ub_x_physical)
Y_physical_space, Y_normalized = generate_Ydata(f_objective_example, X_physical_space)

# Instantiate and use TSRoots for optimization
TSRoots_BO = TSRoots(X_normalized, Y_normalized.flatten(), -np.ones(D), np.ones(D))
x_new_normalized, y_new_normalized, _ = TSRoots_BO.xnew_TSroots()

print(f"New observation location: {x_new_normalized}")
print(f"New function value: {y_new_normalized}")
```


## Citation

If you found TSRoots helpful, please cite the [following
paper](https://openreview.net/forum?id=IpRLTVblaV):
```
@inproceedings{
adebiyi2024gaussian,
title={Gaussian Process Thompson Sampling via Rootfinding},
author={Taiwo Adebiyi and Bach Do and Ruda Zhang},
booktitle={NeurIPS 2024 Workshop on Bayesian Decision-making and Uncertainty},
year={2024},
url={https://openreview.net/forum?id=IpRLTVblaV}
}
```

## The Team

TSRoots is produced by the [Uncertainty Quantification Lab](https://uq.uh.edu/) at the University of Houston; the primary maintainers are:
- [Taiwo A. Adebiyi](https://www.linkedin.com/in/taiwo-adebiyi-055750174/) 
- [Bach Do](https://scholar.google.com/citations?user=O6vKWWYAAAAJ&hl=en) 
- [Ruda Zhang](https://scholar.google.com/citations?user=ttmax_wAAAAJ&hl=en)
