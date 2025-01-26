import os
import h5py
import numpy as np
import time
from datetime import datetime
from tsroots.optim import TSRoots
from tsroots.utils import *

# Define the ackley function
def ackley(X, a=20, b=0.2, c=2 * np.pi):
    d = X.shape[1]  # number of dimensions
    sum1 = np.sum(X**2, axis=1)
    sum2 = np.sum(np.cos(c * X), axis=1)
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)


# Problem setup
function_name = "ackley"
d = 16  # number of variables
lbX = -10 * np.ones(d)  # lower bounds
ubX = 10 * np.ones(d)  # upper bounds

# Number of samples, dimensions, and random seed
no_sample = 10 * d  # number of initial samples
D = d  # dimensions
seed_list = [20, 44, 1, 42, 1000, 13, 25, 18, 37, 10, 2000, 113, 308, 99, 4, 7, 3000, 78, 418, 65]  # Fixed seed value
count = 1
Seed = 20
print(f'Using seed 20')
noise_level = 1e-3

# Lower and upper bounds in standardized space
lbS = -np.ones(D)
ubS = np.ones(D)

# HDF5 File Setup
hdf5_filename = f"ackley_TSRoots_solution_seed_20.h5"

# Initialize or resume from a previous run
completed_iterations = 0  # Default value for new runs
if os.path.exists(hdf5_filename):
    try:
        with h5py.File(hdf5_filename, "r") as f:
            if "X_r" in f and "Y_r" in f and "solutions" in f:
                # Determine the last completed iteration
                completed_iterations = len(f["solutions"]) - 1
                print(f"Resuming from iteration {completed_iterations + 1}")

                # Load existing data
                X_r = np.array(f["X_r"])
                Y_r = np.array(f["Y_r"]).flatten()
                X_s = scale_Xn(X_r, lbX, ubX)
                print(f"Loaded X_r with shape {X_r.shape} and Y_r with shape {Y_r.shape}")
            else:
                print("HDF5 file exists but is incomplete. Initializing a new run.")
    except Exception as e:
        print(f"Error reading HDF5 file: {e}. Initializing a new run.")
else:
    print("HDF5 file does not exist. Starting a new run.")

# If this is a new run, initialize the data
if completed_iterations == 0:
    # Generate initial data
    X_r, X_s = generate_Xdata(no_sample, D, Seed, lbX, ubX)
    #print(f"Initial location points: {X_r}")
    #print(f"Starting initial simulation...")
    Y_r, _ = generate_Ydata(ackley, X_r)
    Y_r += np.random.normal(0, noise_level, Y_r.shape)

    # Ensure Y_r is 1D
    Y_r = Y_r.flatten()

    # Create the HDF5 file and save initial datasets
    with h5py.File(hdf5_filename, "w") as f:
        f.create_dataset("X_r", data=X_r, maxshape=(None, D))  # Allow resizing for 2D array
        f.create_dataset("Y_r", data=Y_r, maxshape=(None,))   # Allow resizing for 1D array
        solutions_grp = f.create_group("solutions")

# Extract the initial best solution
X_r_best, Y_r_best = TSRoots.extract_min(X_r, Y_r)
print(f"Initial X_r_best: {X_r_best}, Initial Y_r_best: {Y_r_best}")

# Save the initial best solution in the HDF5 file
if completed_iterations == 0:
    with h5py.File(hdf5_filename, "a") as f:
        grp = f["solutions"].create_group("iteration_0")
        grp.create_dataset("X_r_best", data=X_r_best)
        grp.create_dataset("Y_r_best", data=Y_r_best)

# BO loop parameters
k = 800  # Total number of iterations
start = time.time()

# BO loop
for i in range(completed_iterations + 1, k + 1):  # Start from the next iteration
    # Scale data for TS-roots optimization
    Y_s = scale_Y(Y_r)
    TSRoots_BO = TSRoots(X_s, Y_s.flatten(), lbS, ubS, noise_level=noise_level, learning_rate=0.07)

    # Get new sample point using TS-roots
    X_s_new, Y_s_new, _ = TSRoots_BO.xnew_TSroots()

    # Convert new point back to the physical space
    X_r_new = unscale_Xn(X_s_new.reshape(-1, D), lbX, ubX)

    # Compute new observation point in the physical space
    print(f"Running simulation for iteration {i}")
    Y_r_new, _ = generate_Ydata(ackley, X_r_new)

    # Append new raw and standardized data points
    X_r = np.vstack((X_r, X_r_new))
    Y_r = np.append(Y_r, Y_r_new.flatten())  # Ensure Y_r remains 1D
    X_s = scale_Xn(X_r, lbX, ubX)  # Update standardized space data

    # Save updated datasets to HDF5
    with h5py.File(hdf5_filename, "a") as f:
        f["X_r"].resize(X_r.shape)
        f["X_r"][:] = X_r
        f["Y_r"].resize(Y_r.shape)
        f["Y_r"][:] = Y_r

    # Extract the best-found solution
    X_r_best, Y_r_best = TSRoots.extract_min(X_r, Y_r)

    # Save best solutions incrementally
    with h5py.File(hdf5_filename, "a") as f:
        grp = f["solutions"].create_group(f"iteration_{i}")
        grp.create_dataset("X_r_best", data=X_r_best)
        grp.create_dataset("Y_r_best", data=Y_r_best)

    # Report the best-found solution at each iteration
    print(f"# BO iter = {i}; y_best = {Y_r_best}...")

# Record and report total runtime
end = time.time()
print(f"Total time for all iterations: {end - start} seconds")
print(f"Data and solutions stored in '{hdf5_filename}'.")
