from tsroots.optim import TSRoots
from tsroots.utils import *

import numpy as np
import scipy
from numpy import log
import scipy
import time
import matplotlib
from matplotlib import gridspec
matplotlib.use('TkAgg')  # Or 'Qt5Agg'

# --------------------------------------------------------------
# Test TSRoots on xSin(x)
# --------------------------------------------------------------

# defining the objective function
def f_objective_example(x):  # this can be any objective function
    return (x * np.sin(x))

lb_x_physical = np.repeat(-15,1)
ub_x_physical = np.repeat(15, 1)

no_sample = 5
D = 1
seed = 42
noise_level = 1e-3

# Define f_true in the normalized space, and ensure consistent scaling
X_normalized_plot = np.linspace(-1, 1, 400).reshape(-1, D)

# Unscale X_normalized to physical space for f_true evaluation
X_true_plot = unscale_Xn(X_normalized_plot, lb_x_physical, ub_x_physical)

# Compute true values in physical space
Y_true_plot = f_objective_example(X_true_plot)

mean_Y_true = np.mean(Y_true_plot)
std_Y_true = np.std(Y_true_plot)

Y_normalized_plot = (Y_true_plot - mean_Y_true) / std_Y_true


X_physical_space, X_normalized = generate_Xdata(no_sample, D, seed, lb_x_physical, ub_x_physical)

Y_physical_space, _ = generate_Ydata(f_objective_example, X_physical_space)
Y_physical_space = Y_physical_space + np.random.normal(0, noise_level)

Y_normalized = ((Y_physical_space - mean_Y_true) / std_Y_true)

print(f"X_physical_space: {X_physical_space}")
print(f"Y_physical_space: {Y_physical_space}")
print(f"X_normalized: {X_normalized}")
print(f"Y_normalized: {Y_normalized}")

# Lower and upper bounds in normalized space
lb_normalized = -np.ones(D)
ub_normalized = np.ones(D)

# Setting up parameters and initial conditions
k = 50
bo_iterMax = 100
xr_best, yr_best = TSRoots.extract_min(X_physical_space, Y_physical_space)
print(f"initial minimum point: {(xr_best.item(), yr_best.item())}")

# TSRoots_BO = TSRoots(X_normalized, Y_normalized.flatten(), lb_normalized, ub_normalized, noise_level=noise_level)

# -----------
# BO loop
# -----------
plt.ion()  # Turn on interactive mode

start = time.time()
for i in range(k):
    # Clear the plot
    plt.clf()

    plt.ylim(-6, 3)
    #plt.axis('off')

    # Apply Bayesian Optimization policy using TS-roots
    TSRoots_BO = TSRoots(X_normalized, Y_normalized.flatten(), lb_normalized, ub_normalized,
                           noise_level=noise_level, learning_rate=0.07, seed=seed)

    x_new_normalized, y_new_normalized, _ = TSRoots_BO.xnew_TSroots()

    # length_scale_vec = TSRoots_BO.decoupled_gp.lengthscales
    # sigmaf = TSRoots_BO.decoupled_gp.sigmaf
    # sigma_n = TSRoots_BO.decoupled_gp.sigman
    #
    # print(f"length_scale_vec: {length_scale_vec}; sigmaf: {sigmaf}; sigma_n: {sigma_n}")

    plt.scatter(x_new_normalized, y_new_normalized, color='blue', marker='x', linewidth=3.0, label='Selected Point')

    # Convert new point back to the physical space
    x_new_physical_space = unscale_Xn(x_new_normalized.reshape(-1, D), lb_x_physical, ub_x_physical)

    # Compute new observation point in the physical space
    y_new_physical_space, _ = generate_Ydata(f_objective_example, x_new_physical_space)

    # Append new raw data points
    X_physical_space = np.append(X_physical_space, x_new_physical_space).reshape(-1, D)
    Y_physical_space = np.append(Y_physical_space, y_new_physical_space)

    # Update scaled data
    X_normalized = np.append(X_normalized, x_new_normalized).reshape(-1, D)
    Y_normalized = ((Y_physical_space - mean_Y_true) / std_Y_true) #+ np.random.normal(0, TSRoots_BO.decoupled_gp.sigman)
    #Y_normalized = scale_Y(Y_physical_space) + np.random.normal(0, TSRoots_BO.decoupled_gp.sigman)

    # Extract the best-found solution so far
    xr_best, yr_best = TSRoots.extract_min(X_physical_space, Y_physical_space)

    plt.plot(X_normalized_plot, Y_normalized_plot, linestyle="-", linewidth=1.5, color="#FF0000",
             alpha=.7, label=f'$f_{{true}}$')

    # Optionally, add legend and show the plot
    plt.legend(loc='upper left')
    # Add title in latex format
    plt.title(f'Plot of $y = x \sin(x)$; Iteration {i+1}')

    # Save the plot with new points
    #plt.savefig(f'TS_acquisition_{i + 1}_seed_{seed}_path.pdf', format="pdf", bbox_inches="tight")

    plt.show()

    # Report the best-found solution at each iteration
    if i < bo_iterMax:
        print(f"# BO iter = {i + 1}; y_best = {yr_best}...")
    else:
        print(f"# BO iter = {i + 1}; y_best = {yr_best}##.")

    # Show the plot dynamically
    plt.draw()

    # Pause to allow for plot display at each iteration
    plt.pause(0.25)
# Record and report total runtime
end = time.time()
print(f"Total time: {end - start} seconds")