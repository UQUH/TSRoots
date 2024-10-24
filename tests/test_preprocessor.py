import numpy as np
import pytest
from tsroots.preprocessor import SE_Mercer, Hyperlearn

@pytest.fixture
def sample_data():
    x_data = np.linspace(-1, 1, 20).reshape(-1, 1)
    y_data = np.sin(x_data).flatten()
    return x_data, y_data

def test_SE_Mercer_initialization(sample_data):
    x_data, y_data = sample_data
    se_mercer = SE_Mercer(x_data, y_data, noise_level=1e-3, learning_rate=0.1)
    assert se_mercer.x_data.shape == (20, 1)
    assert se_mercer.y_data.shape == (20, )
    assert se_mercer.noise_level == 1e-3
    assert se_mercer.learning_rate == 0.1

def test_SE_Mercer_eigen_parameters(sample_data):
    x_data, y_data = sample_data
    se_mercer = SE_Mercer(x_data, y_data)
    a, b, c = se_mercer.eigen_parameters(1.0, 1.0)
    assert a > 0 and b > 0 and c > 0, "Eigen parameters should be positive"

def test_Hyperlearn_initialization(sample_data):
    x_data, y_data = sample_data
    hyperlearn = Hyperlearn(x_data, y_data, noise_level=1e-3)
    assert hyperlearn.x_data.shape == (20, 1)
    assert hyperlearn.y_data.shape == (20, )

def test_Hyperlearn_hyperparameters(sample_data):
    x_data, y_data = sample_data
    hyperlearn = Hyperlearn(x_data, y_data, noise_level=1e-3)
    lengthscales, sigmaf, sigman = hyperlearn.get_hyperparameters()
    assert len(lengthscales) > 0
    assert sigmaf > 0 and sigman > 0, "Hyperparameters should be positive"


def test_phi_function(sample_data):
    x_data, y_data = sample_data
    se_mercer = SE_Mercer(x_data, y_data)

    # Parameters for phi function
    n = se_mercer.n_terms_SE()[0]

    # Compute phi function
    phi_values = se_mercer.phi(n, x_data.flatten())

    # Assert phi_values shape
    assert phi_values.shape == (x_data.shape[0], n)


def test_diff_phi_function(sample_data):
    x_data, y_data = sample_data
    se_mercer = SE_Mercer(x_data, y_data)

    # Parameters for phi function
    n = se_mercer.n_terms_SE()[0]

    # Compute phi function
    diff_phi_values = se_mercer.diff_phi(n, x_data.flatten())

    # Assert phi_values shape
    assert diff_phi_values.shape == (x_data.shape[0], n)
