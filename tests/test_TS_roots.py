import sys
import os

# Add the src directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import pytest
from tsroots.optim import TSRoots
from tsroots.utils import scale_Xn, unscale_Xn, scale_Y, unscale_Y


@pytest.fixture
def sample_data():
    # Dynamically generate x_data with shape (5, 1) in the physical space [-15, 15]
    lb_x_physical = np.repeat(-15, 1)
    ub_x_physical = np.repeat(15, 1)

    # Generate 5 samples between -15 and 15
    x_data = np.linspace(-15, 15, 5).reshape(-1, 1)

    # Generate corresponding y_data based on the function y = x * sin(x)
    def f_objective_example(x):
        return x * np.sin(x)

    y_data = f_objective_example(x_data).flatten()

    # Normalize the x_data and y_data using your scaling functions
    x_data_normalized = scale_Xn(x_data, lb_x_physical, ub_x_physical)

    mean_Y = np.mean(y_data)
    std_Y = np.std(y_data)
    y_data_normalized = scale_Y(y_data)

    lbS = -np.ones(1)  # Lower bound in normalized space
    ubS = np.ones(1)  # Upper bound in normalized space

    return x_data_normalized, y_data_normalized, lbS, ubS, lb_x_physical, ub_x_physical, mean_Y, std_Y


def test_TSRoots_initialization(sample_data):
    # Test initialization of TSRoots object with normalized data
    x_data_normalized, y_data_normalized, lbS, ubS, lb_x_physical, ub_x_physical, mean_Y, std_Y = sample_data
    ts_roots = TSRoots(x_data_normalized, y_data_normalized.flatten(), lbS, ubS)

    assert ts_roots.x_data.shape == (5, 1), "Incorrect shape for x_data"
    assert ts_roots.y_data.shape == (5,), "Incorrect shape for y_data"
    assert ts_roots.lb.shape == lbS.shape, "Incorrect shape for lower bounds"
    assert ts_roots.ub.shape == ubS.shape, "Incorrect shape for upper bounds"

def test_multi_func_roots_cheb(sample_data):
    # Test the Chebyshev root-finding method
    x_data_normalized, y_data_normalized, lbS, ubS, lb_x_physical, ub_x_physical, mean_Y, std_Y = sample_data
    ts_roots = TSRoots(x_data_normalized, y_data_normalized.flatten(), lbS, ubS)

    # Test multi_func_roots_cheb method
    x_critical, func_x_critical, dfunc_x_critical, d2func_x_critical, no_combi = \
        ts_roots.multi_func_roots_cheb(lbS, ubS)

    # Validate the output of the method
    assert len(x_critical) > 0, "No critical points found"
    assert len(func_x_critical) == len(x_critical), "Mismatch between critical points and function values"
    assert len(dfunc_x_critical) == len(x_critical), "Mismatch between critical points and first derivatives"
    assert len(d2func_x_critical) == len(x_critical), "Mismatch between critical points and second derivatives"




def test_xnew_TSroots_with_scaling(sample_data):
    # Test the xnew_TSroots method for generating a new solution point, using scaling and unscaling
    x_data_normalized, y_data_normalized, lbS, ubS, lb_x_physical, ub_x_physical, mean_Y, std_Y = sample_data
    noise_level = 1e-3  # Example noise level
    learning_rate = 0.07  # Example learning rate
    seed = 42  # Seed for reproducibility

    # Initialize the TSRoots object with normalized data
    TSRoots_BO = TSRoots(x_data_normalized, y_data_normalized.flatten(), lbS, ubS,
                           noise_level=noise_level, learning_rate=learning_rate, seed=seed)

    # Call xnew_TSroots to get a new solution point
    x_new_normalized, y_new_normalized, _ = TSRoots_BO.xnew_TSroots()

    # Check that the new scaled x values are within [-1, 1] (scaled bounds)
    assert -1 <= x_new_normalized <= 1, "x_new_normalized is out of scaled bounds before unscaling"

    # Unscale the newly generated x and y values
    x_new_unscaled = unscale_Xn(x_new_normalized.reshape(1, -1), lb_x_physical, ub_x_physical)
    y_new_unscaled = unscale_Y(y_new_normalized, mean_Y, std_Y)

    # Assertions to validate the output
    assert x_new_normalized is not None, "x_new_normalized is None"
    assert y_new_normalized is not None, "y_new_normalized is None"

    # Check that x_new_normalized is an array with shape (1,)
    assert x_new_normalized.shape == (1,), "x_new_normalized shape is incorrect"

    # Since y_new_normalized is a float, we check if it's a scalar
    assert isinstance(y_new_normalized, (int, float)), "y_new_normalized should be a scalar value"
    print(f"x_new_normalized: {x_new_normalized}, shape: {x_new_normalized.shape}")
    print(f"y_new_normalized: {y_new_normalized}, type: {type(y_new_normalized)}")


    # Check that the new points are within the scaled bounds
    assert lbS <= x_new_normalized <= ubS, "x_new_normalized is out of bounds"
    assert np.isfinite(x_new_normalized).all(), "x_new_scaled contains NaN or infinite values"
    assert np.isfinite(y_new_normalized), "y_new_scaled contains NaN or infinite values"

    # Check that the unscaled values are within the original physical bounds
    assert lb_x_physical <= x_new_unscaled <= ub_x_physical, "x_new_unscaled is out of bounds after unscaling"
    assert np.isfinite(x_new_unscaled).all(), "x_new_unscaled contains NaN or infinite values"
    assert np.isfinite(y_new_unscaled), "y_new_unscaled contains NaN or infinite values"


