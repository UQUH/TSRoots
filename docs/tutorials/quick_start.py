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
seed = 10

# Generate initial samples and normalize them
X_physical_space, X_normalized = generate_Xdata(no_sample, D, seed, lb_x_physical, ub_x_physical)
Y_physical_space, Y_normalized = generate_Ydata(f_objective_example, X_physical_space)

# Instantiate and use TSRoots for optimization
TSRoots_BO = TSRoots(X_normalized, Y_normalized.flatten(), -np.ones(D), np.ones(D))
x_new_normalized, y_new_normalized, _ = TSRoots_BO.xnew_TSroots(plot=True)

print(f"New observation location: {x_new_normalized}")
print(f"New function value: {y_new_normalized}")

# plot selected point
plt.figure(figsize=(10, 6))
plt.scatter(x_new_normalized, y_new_normalized, color='blue', marker='x', linewidth=3.0, label='Selected Point')
plt.show()
