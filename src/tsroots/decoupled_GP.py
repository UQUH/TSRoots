from tsroots.preprocessor import Hyperlearn
from tsroots.preprocessor import SE_Mercer

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import solve_triangular
from scipy.linalg import cholesky, cho_solve

import time


class Decoupled_GP:
    def __init__(self, x_data, y_data, sigma=1.0, noise_level=1e-3, learning_rate=0.07, seed=None):
        self.x_data = x_data
        self.y_data = y_data
        self.sigma = sigma
        self.noise_level = noise_level
        self.learning_rate = learning_rate
        self.seed = seed
        self.get_preprocessor()
        # Dictionary to store precomputed phi and lambda values
        self.precomputed_values = {}

    def get_preprocessor(self):
        self.SE_Mercer_instance = SE_Mercer(self.x_data, self.y_data, noise_level=self.noise_level,
                                            learning_rate=self.learning_rate)
        self.lengthscales, self.sigmaf, self.sigman = self.SE_Mercer_instance.get_hyperparameters()
        self.n_eigen_vec = self.SE_Mercer_instance.n_terms_SE()
        self.W = self.SE_Mercer_instance.W_array(self.n_eigen_vec)
        param_dict = {
            'lengthscales_vec': self.lengthscales,
            'sigmaf': self.sigmaf,
            'sigman': self.sigman,
            'n_eigen_vec': self.n_eigen_vec,
            'W_shape': [k.shape for k in self.W]
        }
        return param_dict

    def uni_GP_path(self, n, x, w, sigma=None, length_scale=None, sigmaf=None):
        if sigma is None:
            sigma = self.sigma
        if length_scale is None:
            length_scale = self.lengthscales.item()
        if sigmaf is None:
            sigmaf = self.sigmaf

        # Extract hyperparameters and compute phi and lambda_n
        phi_values = self.SE_Mercer_instance.phi(n, x, sigma, length_scale)
        lambda_values = self.SE_Mercer_instance.lambda_n(n, sigma, length_scale)

        # Store the computed phi and lambda for reuse
        self.precomputed_values['phi'] = phi_values
        self.precomputed_values['lambda'] = lambda_values

        # BGLM feature matrix
        HN = phi_values @ np.diag(lambda_values ** 0.5)
        f = sigmaf * (HN @ w)

        return f

    def diff_uni_GP_path(self, n, x, w, sigma=None, length_scale=None, sigmaf=None):
        if sigma is None:
            sigma = self.sigma
        if length_scale is None:
            length_scale = self.lengthscales.item()
        if sigmaf is None:
            sigmaf = self.sigmaf

        # Retrieve precomputed phi and lambda if available
        phi_values = self.precomputed_values.get('phi')
        lambda_values = self.precomputed_values.get('lambda')

        if phi_values is None or lambda_values is None:
            phi_values = self.SE_Mercer_instance.phi(n, x, sigma, length_scale)
            lambda_values = self.SE_Mercer_instance.lambda_n(n, sigma, length_scale)

        print(f'phi values is not none')

        # Compute diff_phi using the precomputed phi and lambda
        diff_phi_values = self.SE_Mercer_instance.diff_phi(n, x, sigma, length_scale, precomputed_phi=phi_values)

        DHN = diff_phi_values @ np.diag(lambda_values ** 0.5)
        diff_f = sigmaf * (DHN @ w)

        return diff_f

    def multi_GP_path(self, X, n_eigen_vec=None, W=None, sigma=None, sigmaf=None, length_scale_vec=None, diff=True):
        if n_eigen_vec is None:
            n_eigen_vec = self.n_eigen_vec
        if W is None:
            W = self.W
        if sigma is None:
            sigma = self.sigma
        if sigmaf is None:
            sigmaf = self.sigmaf
        if length_scale_vec is None:
            length_scale_vec = self.lengthscales

        X = np.asarray(X)
        d = len(length_scale_vec)

        if X.ndim == 1:
            X = X.reshape(-1, d)

        if X.shape[1] != len(length_scale_vec):
            raise ValueError("Lengthscales must have the same number of dimensions as the input points")

        separable_f = np.zeros_like(X, dtype=float)
        df = np.zeros_like(X, dtype=float)

        for i in range(d):
            phi_values = self.SE_Mercer_instance.phi(n_eigen_vec[i], X[:, i], sigma, length_scale_vec[i])
            lambda_values = self.SE_Mercer_instance.lambda_n(n_eigen_vec[i], sigma, length_scale_vec[i])

            # Store phi and lambda for reuse in the current dimension
            self.precomputed_values[f'phi_{i}'] = phi_values
            self.precomputed_values[f'lambda_{i}'] = lambda_values

            separable_f[:, i] = (phi_values @ (np.diag(lambda_values ** 0.5))) @ W[i]

        f = sigmaf * np.prod(separable_f, axis=1)

        if not diff:
            return f
        else:
            for i in range(d):
                diff_phi_values = self.SE_Mercer_instance.diff_phi(
                    n_eigen_vec[i], X[:, i], sigma, length_scale_vec[i],
                    precomputed_phi=self.precomputed_values[f'phi_{i}']
                )
                df[:, i] = np.multiply(
                    (sigmaf * (diff_phi_values @ (np.diag(self.precomputed_values[f'lambda_{i}'] ** 0.5))) @ W[i]),
                    np.prod(np.delete(separable_f, i, 1), axis=1)
                )
            return f, df


    def ard_square_exponential_kernel(self, X1, X2, lengthscales=None, sigma_f=None, sigma_n=None):
        """
        Computes the ARD (Automatic Relevance Determination) square exponential kernel between two sets of inputs.

        Args:
            X1 (numpy.ndarray): First set of input points of shape (n1, d).
            X2 (numpy.ndarray): Second set of input points of shape (n2, d).
            lengthscales (numpy.ndarray, optional): Lengthscales for each dimension.
                                                    Defaults to instance attribute if not provided.
            sigma_f (float, optional): Marginal variance. Defaults to instance attribute if not provided.
            sigma_n (float, optional): Noise variance (nugget). Defaults to instance attribute if not provided.

        Returns:
            numpy.ndarray: The computed kernel matrix of shape (n1, n2).
        """

        # Use instance attributes if parameters are not provided
        if lengthscales is None:
            lengthscales = self.lengthscales
        if sigma_f is None:
            sigma_f = self.sigmaf
        if sigma_n is None:
            sigma_n = self.sigman

        # Ensure inputs are numpy arrays
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        lengthscales = np.asarray(lengthscales)
        d = len(lengthscales)

        # Make (1,) to (1,d) ndarray shape
        if X1.ndim == 1:
            X1 = X1.reshape(-1, d)
        if X2.ndim == 1:
            X2 = X2.reshape(-1, d)

        # Validate dimensions
        if X1.shape[1] != X2.shape[1]:
            raise ValueError("X1 and X2 must have the same number of dimensions (features)")
        if X1.shape[1] != lengthscales.shape[0]:
            raise ValueError("Lengthscales must have the same number of dimensions as the input points")

        # Scale the inputs by the lengthscales
        X1_scaled = X1 / lengthscales
        X2_scaled = X2 / lengthscales

        # Compute the squared distances using cdist for efficiency
        sqdist = cdist(X1_scaled, X2_scaled, metric='sqeuclidean')

        # Compute the kernel matrix
        K = sigma_f ** 2 * np.exp(-0.5 * sqdist)


        # Add the noise term (nugget) to the diagonal elements
        K += (sigma_n ** 2 + sigma_f ** 2 * 1e-12) * np.eye(X1.shape[0])

        return K

    def cross_covariance_kernel(self, X1, X, lengthscales=None, sigmaf=None):
        """
        Computes the cross-covariance kernel between training inputs X1 and unseen inputs X.

        Args:
            X1 (numpy.ndarray): Training set input points of shape (n1, d).
            X (numpy.ndarray): Unseen set of input points of shape (n, d).
            lengthscales (numpy.ndarray, optional): Lengthscales for each dimension.
                                                    Defaults to instance attribute if not provided.
            sigmaf (float, optional): Marginal variance.. Defaults to instance attribute if not provided.

        Returns:
            numpy.ndarray: The computed cross-covariance kernel matrix of shape (n1, n).
        """

        # Use instance attributes if parameters are not provided
        if lengthscales is None:
            lengthscales = self.lengthscales
        if sigmaf is None:
            sigmaf = self.sigmaf

        # Ensure inputs are numpy arrays
        X1 = np.asarray(X1)
        X = np.asarray(X)
        lengthscales = np.asarray(lengthscales)
        d = len(lengthscales)

        # Make (1,) to (1,d) ndarray shape
        if X1.ndim == 1:
            X1 = X1.reshape(-1, d)
        if X.ndim == 1:
            X = X.reshape(-1, d)

        # Validate dimensions
        if X1.shape[1] != X.shape[1]:
            raise ValueError("X1 and X must have the same number of dimensions (features)")
        if X1.shape[1] != lengthscales.shape[0]:
            raise ValueError("Lengthscales must have the same number of dimensions as the input points")

        # Scale the inputs by the lengthscales
        X1_scaled = X1 / lengthscales
        X_scaled = X / lengthscales

        # Compute the squared distances using cdist for efficiency
        sqdist = cdist(X1_scaled, X_scaled, metric='sqeuclidean')

        # Compute the cross-covariance kernel matrix
        K_cross = sigmaf ** 2 * np.exp(-0.5 * sqdist)

        return K_cross

    def derivative_ard_cross_covariance_kernel(self, X1, X, lengthscales=None, sigmaf=None):
        """
        Computes the derivative of the cross-covariance kernel with respect to the input points X.

        Args:
            X1 (numpy.ndarray): Training set input points of shape (n1, d).
            X (numpy.ndarray): Unseen set of input points of shape (n, d).
            lengthscales (numpy.ndarray, optional): Lengthscales for each dimension.
                                                    Defaults to instance attribute if not provided.
            sigmaf (float, optional): Marginal variance. Defaults to instance attribute if not provided.

        Returns:
            numpy.ndarray: The derivative of the cross-covariance kernel matrix, of shape (n, n1*d).
        """

        # Use instance attributes if parameters are not provided
        if lengthscales is None:
            lengthscales = self.lengthscales
        if sigmaf is None:
            sigmaf = self.sigmaf

        # Ensure inputs are numpy arrays
        X1 = np.asarray(X1)
        X = np.asarray(X)
        lengthscales = np.asarray(lengthscales)
        d = len(lengthscales)

        # Make (1,) to (1,d) ndarray shape
        if X1.ndim == 1:
            X1 = X1.reshape(-1, d)
        if X.ndim == 1:
            X = X.reshape(-1, d)

        # Validate dimensions
        if X1.shape[1] != X.shape[1]:
            raise ValueError("X1 and X must have the same number of dimensions (features)")
        if X1.shape[1] != lengthscales.shape[0]:
            raise ValueError("Lengthscales must have the same number of dimensions as the input points")

        # Compute the cross-covariance kernel matrix
        K_cross = self.cross_covariance_kernel(X1, X, lengthscales, sigmaf)

        # Vectorized computation of derivative
        diff = (X1[:, np.newaxis, :] - X[np.newaxis, :, :]) / (lengthscales ** 2)
        dK_cross = diff * K_cross[..., np.newaxis]  # Broadcasting to multiply K_cross with diff

        # Flatten the derivative matrix
        dK_cross_flattened = dK_cross.reshape(X1.shape[0], -1)

        return dK_cross_flattened

    @staticmethod
    def cholesky_inverse(A):
        """
        Computes the inverse of a positive-definite matrix using Cholesky decomposition.

        Args:
            A (numpy.ndarray): A positive-definite matrix.

        Returns:
            numpy.ndarray: The inverse of the matrix A.
        """

        # Suppose A is your positive definite matrix
        L = cholesky(A, lower=True)

        # Create an identity matrix of the same size as A
        identity_matrix = np.eye(A.shape[0])

        # Solve for A_inv using cho_solve, which avoids explicit inversion of L
        A_inv = cho_solve((L, True), identity_matrix)

        return A_inv

    def v_vec(self, X_data=None, Y_data=None, W=None, length_scale_vec=None, n_eigen_vec=None, sigma=None, sigmaf=None, sigman=None):
        """
        Computes the v vector used in Gaussian Process regression.

        Args:
            W (list): List of weight vectors.
            length_scale_vec (numpy.ndarray, optional): Length scales for each dimension. Defaults to instance attribute if not provided.
            n_eigen_vec (list or numpy.ndarray, optional): Number of leading eigenfunctions for each dimension. Defaults to instance attribute if not provided.
            sigma (float, optional): Standard deviation parameter. Defaults to instance attribute if not provided.
            sigmaf (float, optional): Marginal variance. Defaults to instance attribute if not provided.
            sigman (float, optional): Noise variance (nugget). Defaults to instance attribute if not provided.

        Returns:
            numpy.ndarray: The computed v vector.
        """

        # Use instance attributes if parameters are not provided
        if X_data is None:
            X_data = self.x_data
        if Y_data is None:
            Y_data = self.y_data
        if length_scale_vec is None:
            length_scale_vec = self.lengthscales
        if sigma is None:
            sigma = self.sigma
        if sigmaf is None:
            sigmaf = self.sigmaf
        if sigman is None:
            sigman = self.sigman
        if n_eigen_vec is None:
            n_eigen_vec = self.SE_Mercer_instance.n_terms_SE(sigma, length_scale_vec)
        if W is None:
            W = self.W

        # Compute the covariance matrix
        self.Cnn = self.ard_square_exponential_kernel(X_data, X_data, length_scale_vec, sigmaf, sigman)

        # Inverse of the covariance matrix using Cholesky decomposition
        self.Cnn_inv = self.cholesky_inverse(self.Cnn)

        # Compute prior sample values at X_data
        F_prior_data = self.multi_GP_path(X_data, n_eigen_vec, W, sigma, sigmaf, length_scale_vec, diff=False)

        v_vec = (self.Cnn_inv @ (Y_data.reshape(-1, 1) - F_prior_data.reshape(-1, 1) -
                            np.random.normal(0, sigman, np.size(Y_data)).reshape(-1, 1))).flatten()

        return v_vec

    def mixPosterior(self, X, v_vec, X_data=None, Y_data=None, W=None, length_scale_vec=None, n_eigen_vec=None, sigma=None,
                         sigmaf=None, sigman=None, diff=True):
        """
        Compute GP posterior function (multivariate) and its derivative evaluated at x
        parameterized by the spectral features of the squared exponential kernel.

        Args:
            X (numpy.ndarray): Evaluation locations of shape (N, d).
            X_data (numpy.ndarray, optional): Input data for computing the posterior.
                                                Defaults to instance attribute if not provided.
            Y_data (numpy.ndarray, optional): Output data corresponding to X_data.
                                                Defaults to instance attribute if not provided.
            W (numpy.ndarray, optional): Feature coefficient matrix. Defaults to instance attribute if not provided.
            length_scale_vec (numpy.ndarray, optional): Length scales for each dimension.
                                                        Defaults to instance attribute if not provided.
            n_eigen_vec (list or numpy.ndarray, optional): Number of leading eigenfunctions for each dimension.
                                                            Defaults to instance attribute if not provided.
            sigma (float, optional): Standard deviation parameter. Defaults to instance attribute if not provided.
            sigmaf (float, optional): Signal variance. Defaults to instance attribute if not provided.
            sigman (float, optional): Noise variance (nugget). Defaults to instance attribute if not provided.
            diff (bool, optional): Whether to compute derivatives. Defaults to True.

        Returns:
            tuple: GP path function evaluated at x locations and optionally its derivatives.
        """

        # Use instance attributes if parameters are not provided
        if X_data is None:
            X_data = self.x_data
        if Y_data is None:
            Y_data = self.y_data
        if length_scale_vec is None:
            length_scale_vec = self.lengthscales
        if sigma is None:
            sigma = self.sigma
        if sigmaf is None:
            sigmaf = self.sigmaf
        if sigman is None:
            sigman = self.sigman
        if n_eigen_vec is None:
            n_eigen_vec = self.SE_Mercer_instance.n_terms_SE(sigma, length_scale_vec)
        if W is None:
            W = self.W

        d = len(length_scale_vec)

        if X.ndim == 1:
            X = X.reshape(-1, d)

        N = X.shape[0]

        # Sample function values
        f1, df1 = self.multi_GP_path(X, n_eigen_vec, W, sigma, sigmaf, length_scale_vec, diff=True)

        # Cross-covariance matrix between X and X_data
        k_star = self.cross_covariance_kernel(X_data, X, length_scale_vec, sigmaf)  # (N by n)

        # Check if any row of X is in X_data
        f2 = k_star.T @ v_vec
        if d == 1:
            f = f1 + f2
        else:
            f = f1 + f2

        if diff == False:
            return f
        else:
            df2 = np.zeros_like(X.T, dtype=float)
            # Derivatives of covariance matrix between X and X_data
            dk_star = self.derivative_ard_cross_covariance_kernel(X_data, X, length_scale_vec, sigmaf)  # (n by Nd)
            df2_stack = dk_star.T @ v_vec  # (Nd by 1)
            df2 = df2_stack.reshape(N, d)
            df = df1 + df2
            return f, df


if __name__ == "__main__":
    # Input data
    xData = np.array([[-1.],
                          [-0.59899749],
                          [-0.19799499],
                          [0.20300752],
                          [0.60401003]])
    yData = np.array([[1.4012621],
                          [0.47086259],
                          [-0.04986313],
                          [-0.08344665],
                          [0.37753832]]).flatten()


    # ------------------------------------------
    # Test Decoupled_GP class
    # ------------------------------------------

    # Instantiating the Decoupled_GP class
    Decoupled_GP_instance = Decoupled_GP(xData, yData)

    # Test Decoupled_GP.get_proprcessor()
    print(f"get_preprocessor\n: {Decoupled_GP_instance.get_preprocessor()}")

    # Test Decoupled_GP.uni_GP_path()
    w = SE_Mercer.W_array([10])
    print(f"uni_GP_path\n: {Decoupled_GP_instance.uni_GP_path(10, xData.flatten(), w[0])}")

    # Test Decoupled_GP.diff_uni_GP_path()
    print(f"diff_uni_GP_path\n: {Decoupled_GP_instance.diff_uni_GP_path(10, xData.flatten(), w[0])}")

    # Test Decoupled_GP_instance.multi_GP_path()
    print(f"multi_GP_path without inputs\n: {Decoupled_GP_instance.multi_GP_path(xData)}")

    # Test Decoupled_GP_instance.ard_square_exponential_kernel()
    print(f"ard_square_exponential_kernel without inputs\n: {Decoupled_GP_instance.ard_square_exponential_kernel(xData, xData)}")

    X = np.array([[0.6]])
    # Test Decoupled_GP_instance.cross_covariance_kernel()
    print(f"cross_covariance_kernel without inputs\n: {Decoupled_GP_instance.cross_covariance_kernel(xData, X)}")

    # Test Decoupled_GP_instance.derivative_ard_cross_covariance_kernel()
    print(f"derivative_ard_cross_covariance_kernel without inputs\n: {Decoupled_GP_instance.derivative_ard_cross_covariance_kernel(xData, X)}")

    # Test Decoupled_GP_instance.ard_square_exponential_kernel()
    print(f"v_vec without inputs\n: {Decoupled_GP_instance.v_vec()}")

    # Test Decoupled_GP_instance.mixPosterior_rev()
    v_vec = Decoupled_GP_instance.v_vec()
    print(f'mixPosterior without inputs: {Decoupled_GP_instance.mixPosterior(X, v_vec)}')

