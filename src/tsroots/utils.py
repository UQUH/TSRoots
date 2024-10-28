from tsroots.preprocessor import SE_Mercer
from tsroots.decoupled_GP import Decoupled_GP

from chebpy import chebfun
import numpy as np
import matplotlib.pyplot as plt
from pyDOE3 import lhs
import matplotlib.lines as mlines

def lhs_with_bounds(D, no_sample):
    """
    Generates a Latin Hypercube Sampling plan with bounds included in the samples.

    Parameters:
    D (int): Number of input variables (dimensions).
    no_sample (int): Number of samples.

    Returns:
    X_s1 (ndarray): LHS samples in [0, 1]^D, with bounds included.
    """
    # Latin hypercube sampling plan in [0,1]^D
    X_s1 = lhs(D, samples=no_sample - 2, criterion='centermaximin')  #centermaximin

    # Manually add lower and upper bounds
    lower_bound = np.zeros((1, D))
    upper_bound = np.ones((1, D))

    # Combine LHS samples with the lower and upper bounds
    X_s1 = np.vstack((lower_bound, X_s1, upper_bound))

    return X_s1


def unscale_X(X_scaled, lbX, ubX):
    X_data = np.zeros_like(X_scaled)

    for i in range(X_scaled.shape[1]):
        X_data[:, i] = (X_scaled[:, i]) * (ubX[i] - lbX[i]) + lbX[i]
    return X_data


def scale_Xn(Xdata, lbX, ubX):
    """
    Scale the input matrix Xdata to the range [-1, 1].

    Parameters:
    Xdata (numpy.ndarray): N-by-D matrix of input variables
    lbX (numpy.ndarray): 1-by-D array of lower bounds of input variables
    ubX (numpy.ndarray): 1-by-D array of upper bounds of input variables

    Returns:
    numpy.ndarray: Scaled input matrix
    """
    X_scaled = np.zeros_like(Xdata)
    for i in range(Xdata.shape[1]):
        X_scaled[:, i] = 2 * (Xdata[:, i] - lbX[i]) / (ubX[i] - lbX[i]) - 1
    return X_scaled


# Convert scaled input variables back to their physical values.

def unscale_Xn(X_scaled, lbX, ubX):
    """
    Convert scaled input variables back to their physical values.

    Parameters:
    X_scaled : ndarray
        N-by-D matrix of scaled input variables.
    lbX : ndarray
        1D array of lower bounds of input variables.
    ubX : ndarray
        1D array of upper bounds of input variables.

    Returns:
    X_data : ndarray
        Matrix of physical input variables.
    """
    X_data = np.zeros_like(X_scaled)

    for i in range(X_scaled.shape[1]):
        X_data[:, i] = ((X_scaled[:, i] + 1) / 2) * (ubX[i] - lbX[i]) + lbX[i]

    return X_data

def scale_Y(Ydata):
    """
    Scale the output data Ydata to have zero mean and unit variance.

    Parameters:
    Ydata (numpy.ndarray): Array of output data

    Returns:
    numpy.ndarray: Scaled output data
    """
    mean_Y = np.mean(Ydata)
    std_Y = np.std(Ydata)
    Y_scaled = (Ydata - mean_Y) / std_Y
    return Y_scaled

def unscale_Y(Y_scaled, mean_Y, std_Y):
    """
    Unscale the scaled output data to its original scale using the provided mean and standard deviation.

    Parameters:
    Y_scaled (numpy.ndarray): Array of scaled output data
    mean_Y (float): The mean of the original Ydata
    std_Y (float): The standard deviation of the original Ydata

    Returns:
    numpy.ndarray: Unscaled output data
    """
    Y_unscaled = (Y_scaled * std_Y) + mean_Y
    return Y_unscaled

def generate_Xdata(no_sample, D, Seed, lbX, ubX):
    """
    Generate samples of input variables in both physical and standardized space.

    Parameters:
    no_sample (int): Number of initial samples.
    D (int): Number of input variables.
    Seed (int): Random seed for reproducibility.
    lbX (array-like): Lower bounds of input variables (1 by D).
    ubX (array-like): Upper bounds of input variables (1 by D).

    Returns:
    X_r (ndarray): Samples of input variables in physical space (no_sample by D).
    X_s (ndarray): Samples of input variables in standardized space (no_sample by D).
    """
    np.random.seed(Seed)

    # Latin hypercube sampling plan in [0,1]^D
    X_s1 = lhs_with_bounds(D, no_sample)
    #X_s1 = rlh(no_sample, D, Seed, 1)

    #np.random.seed()
    # Convert standardized data to physical data
    X_r = unscale_X(X_s1, lbX, ubX)

    # Convert physical data to standardized space [-1, 1]^D
    X_s = scale_Xn(X_r, lbX, ubX)

    return X_r, X_s

# Generate Y data based on the objective function and input samples
def generate_Ydata(f_objective, X_r):
    Y_r = f_objective(X_r)  # Directly pass the matrix X_r to the function
    Y_s = scale_Y(Y_r)
    return Y_r, Y_s

def generate_Ydata_ustd(f_objective, X):
    Y = f_objective(X)  # Directly pass the matrix X_r to the function
    return Y


def plot_prior_sample(Decoupled_GP_instance, lb, ub, x_critical_points=None, f_critical_points=None):
    """
    Plot the prior sample path and optionally its critical points.

    Args:
        W (list): List of weight vectors for the GP paths.
        length_scale_vec (list or numpy.ndarray): Length scales for each dimension.
        n_eigen_vec (list or numpy.ndarray): Number of leading eigenfunctions for each dimension.
        sigma (float): Standard deviation parameter.
        sigmaf (float): Signal variance.
        x_critical_points (list of numpy.ndarray): List of critical points in the input space for each dimension.
        f_critical_points (list of numpy.ndarray): List of function values at the critical points.
        lb (list or numpy.ndarray): Lower bounds for the input space in each dimension.
        ub (list or numpy.ndarray): Upper bounds for the input space in each dimension.
    """

    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes()
    plt.style.use('fivethirtyeight')
    prior_F = []

    for i in range(len(Decoupled_GP_instance.lengthscales)):
        prior_fun = lambda x_test: Decoupled_GP_instance.uni_GP_path(Decoupled_GP_instance.n_eigen_vec[i], x_test,
                                                                     Decoupled_GP_instance.W[i], Decoupled_GP_instance.sigma,
                                                                     Decoupled_GP_instance.lengthscales[i],
                                                                     Decoupled_GP_instance.sigmaf)
        prior_cheb = chebfun(prior_fun, [lb[i], ub[i]])
        ax = prior_cheb.plot(linewidth=2, label=f'len_scale: {Decoupled_GP_instance.lengthscales[i]:.3f}')
        if x_critical_points is not None:
            prior_F.append(prior_cheb(x_critical_points[i]))
            ax.plot(x_critical_points[i], f_critical_points[i], ".b", markersize=10)
            ax.set_title("Critical Points of a Prior Sample Path")
        ax.set_title("Prior Sample Path")
        ax.legend(loc="upper left")
    plt.show()


def plot_posterior_sample(Decoupled_GP_instance, X_data, Y_data):
    fig = plt.figure(figsize=(8, 6))
    plt.style.use('fivethirtyeight')
    X_pred = np.linspace(-1, 1, 400).reshape(-1, 1)
    y_post = Decoupled_GP_instance.mixPosterior(X_pred, Decoupled_GP_instance.v_vec(), X_data, Y_data, Decoupled_GP_instance.W,
                                            Decoupled_GP_instance.lengthscales, Decoupled_GP_instance.n_eigen_vec,
                                            Decoupled_GP_instance.sigma, Decoupled_GP_instance.sigmaf,
                                            Decoupled_GP_instance.sigman, diff=False)
    plt.plot(X_pred, y_post, linewidth=1, color='#e8b923', label=f'len_scale: {Decoupled_GP_instance.lengthscales[0]:.3f}')
    plt.scatter(X_data, Y_data, color='black', label=r'Data')
    plt.title("Posterior Sample Path")
    plt.legend(loc="upper left")
    plt.show()


def plot_posterior_TS(Decoupled_GP_instance, X_data, Y_data, length_scale_vec, sigma, sigmaf, sigma_n,
                      W=None, v_vec=None, n_eigen_vec=None, x_new=None, y_new=None):
    # f_post = lambda p: mixPosterior(np.asarray(p), X_data, W, length_scale_vec, n_eigen_vec, sigma, sigmaf, v_vec)[0]
    # plt.figure(figsize=(10, 6))

    # Prediction points for plotting
    X_pred = np.linspace(-1, 1, 400).reshape(-1, 1)

    if W is not None:
        y_post = Decoupled_GP_instance.mixPosterior(X_pred, v_vec, X_data, Y_data, W,
                                                    length_scale_vec, n_eigen_vec, sigma, sigmaf, sigma_n, diff=False)

        plt.plot(X_pred, y_post, linewidth=1, color='#e8b923', label='Selected sample path')

    Cnn = Decoupled_GP_instance.ard_square_exponential_kernel(X_data, X_data, length_scale_vec, sigmaf,
                                        sigma_n)  # recompute covariance matrix
    Cnn_inv = Decoupled_GP_instance.cholesky_inverse(Cnn)  # recompute the inverse of covariance matrix

    # Plot mean and confidence interval

    K_s = Decoupled_GP_instance.cross_covariance_kernel(X_data, X_pred, length_scale_vec, sigmaf)  # (N by n)
    K_ss = Decoupled_GP_instance.cross_covariance_kernel(X_pred, X_pred, length_scale_vec, sigmaf)  # (n by n)

    mu_pred = K_s.T @ Cnn_inv @ Y_data.flatten()
    cov_pred = K_ss - K_s.T @ Cnn_inv @ K_s
    std_pred = np.sqrt(np.diag(cov_pred))

    plt.plot(X_pred, mu_pred, linewidth=1.5, color='#0066b2', alpha=1.0, label='Posterior mean')
    plt.fill_between(X_pred.ravel(),
                     mu_pred - 1.96 * std_pred,
                     mu_pred + 1.96 * std_pred,
                     color='#a6cce3', alpha=1)

    plt.scatter(X_data, Y_data, color='black')  # label=r'Data')

    if x_new != None and y_new != None:
        # plot new point
        plt.scatter(x_new, y_new, color='blue', linewidth=3, label='post_glob_min')
    #plt.xlabel('x')
    #plt.ylabel('y')
    #plt.legend()
    #plt.show()

def plot_posterior_TS_2D(TS_roots_instance, X_data, Y_data, length_scale_vec, sigma, sigmaf, sigma_n, X_new=None, y_new=None):
    # Create the figure for plotting
    #fig = plt.figure(figsize=(14, 7))
    # 1. 2D Contour Plot in normalized input space and standardized output space
    #ax1 = fig.add_subplot(1, 2, 1)
    grid_vals = np.linspace(-1, 1, 100)
    X1_grid, X2_grid = np.meshgrid(grid_vals, grid_vals)
    X_grid = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

    # Recompute covariance matrix
    Cnn = TS_roots_instance.decoupled_gp.ard_square_exponential_kernel(X_data, X_data, length_scale_vec, sigmaf, sigma_n)
    Cnn_inv = TS_roots_instance.decoupled_gp.cholesky_inverse(Cnn)

    # Compute predictive mean and covariance
    K_s = TS_roots_instance.decoupled_gp.cross_covariance_kernel(X_data, X_grid, length_scale_vec, sigmaf)  # (N by n)
    K_ss = TS_roots_instance.decoupled_gp.cross_covariance_kernel(X_grid, X_grid, length_scale_vec, sigmaf)  # (n by n)

    mu_pred = K_s.T @ Cnn_inv @ Y_data.flatten()
    cov_pred = K_ss - K_s.T @ Cnn_inv @ K_s
    std_pred = np.sqrt(np.diag(cov_pred))

    # Reshape mu_pred for plotting
    mu_pred = mu_pred.reshape(X1_grid.shape)

    # 2D Contour plot of the mean prediction
    contour = plt.contour(X1_grid, X2_grid, mu_pred, levels=50, cmap='viridis', linewidths=0.7)
    #ax1.set_title(f"Iteration i", fontsize=14, pad=15)
    plt.title(f"Himmelblau's function", fontsize=14, pad =15)
    #ax1.set_xlabel('x1 (normalized)', fontsize=12, labelpad=10)
    plt.xlabel('x1 (normalized)', fontsize=12, labelpad=10)
    #ax1.set_ylabel('x2 (normalized)', fontsize=12, labelpad=10)
    plt.xlabel('x1 (normalized)', fontsize=12, labelpad=10)
    #fig.colorbar(contour, ax=ax1)
    plt.colorbar(contour)

    # Create a proxy artist for the contour plot to add to the legend
    contour_proxy = mlines.Line2D([], [], color='k', linestyle='-', label='GP Mean')

    # ---------------------------------------------------
    # Plot true minimum points for Himmelblau's function
    # ---------------------------------------------------
    X_true_scaled = np.array([[0.5, 0.33333333],
                              [-0.46751967, 0.52188533],
                              [-0.629885, -0.54719767],
                              [0.59740467, -0.308021]])
    y_true_scaled = np.array([0.0, 0.0, 0.0, 0.0])

    # Plot true minima points
    plt.scatter(X_true_scaled[:, 0], X_true_scaled[:, 1], c='red', marker='*', s=150, label="True minima", zorder=5)

    # Plot the observations (X_data points) as black dots
    plt.scatter(X_data[:, 0], X_data[:, 1], c='black', s=50, label="Observations", zorder=5)

    # Plot the points (x_new, y_new) on the 2D contour plot
    if X_new is not None and y_new is not None:
        plt.scatter(X_new[0], X_new[1], c='blue', s=40, label="New Point", zorder=5)

    # Add legend showing GP Mean, True minima, Observations, and New Points
    plt.legend(handles=[
        contour_proxy,
        mlines.Line2D([], [], color='red', marker='*', linestyle='None', markersize=10, label='True minima'),
        mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=6, label='Observations'),
        mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=6, label='New Points')
    ])

    # # 2. 3D Surface Plot of the mean prediction
    # ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # surf = ax2.plot_surface(X1_grid, X2_grid, mu_pred, cmap='viridis', edgecolor='none', alpha=0.8)
    # fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5)
    # ax2.set_title("Gaussian Process Posterior Mean (3D Surface)", fontsize=14, pad=15)
    # ax2.set_xlabel('x1 (normalized)', fontsize=12, labelpad=10)
    # ax2.set_ylabel('x2 (normalized)', fontsize=12, labelpad=10)
    # ax2.set_zlabel('GP Mean', fontsize=12, labelpad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    #plt.show()

if __name__ == "__main__":
    # Example usage:
    def f_objective_example(x): # this can be any objective function
        return x * np.sin(x)

    X_r = np.linspace(-15, 15, 400).reshape(-1,1)
    print(X_r.shape)
    Y_r, Y_s = generate_Ydata(f_objective_example, X_r)
    print("Y_r_shape:", Y_r.shape)
    print("Y_s_shape:", Y_s.shape)

    plt.figure(figsize=(10, 6))
    plt.plot(X_r, Y_r, label=r'$y = x \sin(x)$')
    plt.title(r'Plot of $y = x \sin(x)$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()