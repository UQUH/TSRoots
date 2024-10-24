import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from matplotlib.animation import FuncAnimation
import imageio

# Import your TSRoots package
from tsroots.optim import TSRoots
from tsroots.utils import *

# Define the objective function
def f_objective_example(x):  # this can be any objective function
    return (x * np.sin(x))

# Physical space bounds
lb_x_physical = np.repeat(-15, 1)
ub_x_physical = np.repeat(15, 1)

no_sample = 5
D = 1
seed = 42
noise_level = 1e-3

# Define f_true in the normalized space, and ensure consistent scaling
X_normalized_plot = np.linspace(-1, 1, 400).reshape(-1, D)
X_true_plot = unscale_Xn(X_normalized_plot, lb_x_physical, ub_x_physical)
Y_true_plot = f_objective_example(X_true_plot)

mean_Y_true = np.mean(Y_true_plot)
std_Y_true = np.std(Y_true_plot)
Y_normalized_plot = (Y_true_plot - mean_Y_true) / std_Y_true

X_physical_space, X_normalized = generate_Xdata(no_sample, D, seed, lb_x_physical, ub_x_physical)
Y_physical_space, _ = generate_Ydata(f_objective_example, X_physical_space)

# Update Y_normalized
Y_physical_space += np.random.normal(0, noise_level)
Y_normalized = ((Y_physical_space - mean_Y_true) / std_Y_true)

# Lower and upper bounds in normalized space
lb_normalized = -np.ones(D)
ub_normalized = np.ones(D)

# Parameters
k = 30
bo_iterMax = 100
xr_best, yr_best = TSRoots.extract_min(X_physical_space, Y_physical_space)
print(f"initial minimum point: {(xr_best.item(), yr_best.item())}")

# Create the plot
fig, ax = plt.subplots(figsize=(8, 4))  # Adjusted size for better fit
fig.patch.set_facecolor('#2E2E2E')  # Dark gray background

# Remove axes to create a cleaner plot
ax.set_facecolor('#2E2E2E')  # Set the background of the axes to match
ax.set_axis_off

# Initialize the scatter plot for new points
sc = ax.scatter([], [], color='blue', marker='x')

# Initialize the line plot for the true function
line, = ax.plot(X_normalized_plot, Y_normalized_plot, linestyle="-", linewidth=1.5, color="#FF0000", alpha=.7)

def update(frame):
    """Update function for the animation"""
    global X_normalized, Y_normalized, X_physical_space, Y_physical_space

    # Clear the plot
    ax.clear()

    # Apply Bayesian Optimization policy using TS-roots
    TSRoots_BO = TSRoots(X_normalized, Y_normalized.flatten(), lb_normalized, ub_normalized, noise_level=noise_level, learning_rate=0.07, seed=seed)

    x_new_normalized, y_new_normalized, _ = TSRoots_BO.xnew_TSroots()

    # Convert new point back to the physical space
    x_new_physical_space = unscale_Xn(x_new_normalized.reshape(-1, D), lb_x_physical, ub_x_physical)
    y_new_physical_space, _ = generate_Ydata(f_objective_example, x_new_physical_space)

    # Append new raw data points
    X_physical_space = np.append(X_physical_space, x_new_physical_space).reshape(-1, D)
    Y_physical_space = np.append(Y_physical_space, y_new_physical_space)

    # Update scaled data
    X_normalized = np.append(X_normalized, x_new_normalized).reshape(-1, D)
    Y_normalized = ((Y_physical_space - mean_Y_true) / std_Y_true)


    # Update the scatter plot with the new point
    sc = ax.scatter(x_new_normalized, y_new_normalized, color='blue', marker='x', linewidth=3.0)

    # Update the true function line
    line, = ax.plot(X_normalized_plot, Y_normalized_plot, linestyle="-", linewidth=1.5, color="#FF0000", alpha=.8)

    # Set y-axis limit to avoid distraction
    #ax.set_ylim(-5, 4)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove space around the figure

    # Extract the best-found solution so far
    xr_best, yr_best = TSRoots.extract_min(X_physical_space, Y_physical_space)

    # Report the best-found solution at each iteration
    if frame < bo_iterMax:
        print(f"# BO iter = {frame + 1}; y_best = {yr_best}...")
    else:
        print(f"# BO iter = {frame + 1}; y_best = {yr_best}##.")

    # Set the title for each frame (Removed)
    # ax.set_title(f'BO Iteration {frame + 1}')  # This has been removed

    return sc, line

# Create the animation
ani = FuncAnimation(fig, update, frames=range(k), blit=True)

ani.save('bo_iterations.gif', writer='imagemagick', fps=1, savefig_kwargs={'transparent': True})  # Adjust fps to control speed
print("GIF has been created and saved as 'bo_iterations.gif' in the current directory.")
