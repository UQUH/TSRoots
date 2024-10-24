import sys
import os

# Add the src directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import pytest
from tsroots.decoupled_GP import Decoupled_GP


@pytest.fixture
def sample_data():
    x_data = np.linspace(-1, 1, 10).reshape(-1, 1)
    y_data = np.sin(x_data).flatten()
    return x_data, y_data


def test_decoupled_gp_initialization(sample_data):
    x_data, y_data = sample_data
    model = Decoupled_GP(x_data, y_data, sigma=1.0, noise_level=1e-3)

    assert model.x_data.shape == (10, 1), "x_data shape is incorrect"
    assert model.y_data.shape == (10, ), "y_data shape is incorrect"
    assert model.sigma == 1.0
    assert model.noise_level == 1e-3


def test_uni_GP_path(sample_data):
    x_data, y_data = sample_data
    model = Decoupled_GP(x_data, y_data)
    w = np.random.randn(10)

    gp_path = model.uni_GP_path(10, x_data.flatten(), w)

    assert gp_path.shape == (10,)
    assert not np.isnan(gp_path).any()


def test_multi_GP_path(sample_data):
    x_data, y_data = sample_data
    model = Decoupled_GP(x_data, y_data)

    f, df = model.multi_GP_path(x_data)

    assert f.shape == (10,)
    assert df.shape == (10, 1)
    assert not np.isnan(f).any()
    assert not np.isnan(df).any()


def test_ard_kernel(sample_data):
    x_data, y_data = sample_data
    model = Decoupled_GP(x_data, y_data)

    kernel_matrix = model.ard_square_exponential_kernel(x_data, x_data)

    assert kernel_matrix.shape == (10, 10), "Kernel matrix shape is incorrect"
    assert np.allclose(kernel_matrix, kernel_matrix.T), "Kernel matrix is not symmetric"
    assert kernel_matrix.diagonal().sum() > 0, "Diagonal elements should be positive"


def test_mix_posterior(sample_data):
    x_data, y_data = sample_data
    model = Decoupled_GP(x_data, y_data)

    # Precompute values (these can come from prior training)
    v_vec = model.v_vec()  # Compute the v vector
    W = model.W  # Feature coefficients
    length_scale_vec = model.lengthscales
    n_eigen_vec = model.n_eigen_vec

    # Random test inputs
    X_new = np.random.randn(10, 1)

    # Compute GP posterior
    f_posterior, df_posterior = model.mixPosterior(
        X_new, v_vec, X_data=x_data, Y_data=y_data, W=W,
        length_scale_vec=length_scale_vec, n_eigen_vec=n_eigen_vec,
        sigma=model.sigma, sigmaf=model.sigmaf, sigman=model.sigman, diff=True
    )

    # Check the shape and presence of NaN values
    assert f_posterior.shape == (10,), "Posterior function values shape is incorrect"
    assert df_posterior.shape == (10, 1), "Posterior derivatives shape is incorrect"
    assert not np.isnan(f_posterior).any(), "Posterior function values contain NaN"
    assert not np.isnan(df_posterior).any(), "Posterior derivatives contain NaN"
