import numpy as np
import scipy
from scipy.optimize import minimize
from chebpy import chebfun
import matplotlib.pyplot as plt
import math
from math import ceil, log, sqrt, exp
import random
import time
from pylab import *
from scipy.stats import norm
#import seaborn as sns
import heapq

import torch
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import Adam
from torch.optim import SGD
from torch.optim import LBFGS
from torch.optim.lr_scheduler import StepLR



class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        #self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=train_x.size(1)))  # Define the ARD kernel here
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class Hyperlearn:

    """
    ===============================================
    Fit GP model for given datasets using GPyTorch
    ===============================================
    Setting the hyperparameters (length scale, marginal standard deviation, and observation noise).

    By calling the Hyperlearn.train() method of an instantiated class--
    hyperlearn_instance = Hyperlearn(xData, yData)--,
    a GP model is fit to the given dataset using the Automatic Relevance Determination (ARD) kernel.

    To obtain the optimal hyperparameters from the trained model,
    call hyperlearn_instance.get_hyperparameters() which returns a tuple of the
    length scale, marginal standard deviation, and observation noise.
    Note that the observation noise is kept fixed and hence not trained as a parameter.

    """

    def __init__(self, x_data, y_data, noise_level):
        """
        Initialize the Hyperlearn class.

        Args:
            xData (numpy.ndarray): Input data of shape (n_samples, n_features).
            yData (numpy.ndarray): Target data of shape (n_samples,).
            noise_level (float): Fixed observation noise (i.e., untrained)
            [PS: code can easily extended to trained observation noise]
        """
        # Convert data to PyTorch tensors
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.float32)
        self.noise_level = noise_level

        # Define likelihood
        self.likelihood = GaussianLikelihood()
        self.likelihood.noise = noise_level  # set noise value
        self.likelihood.noise_covar.raw_noise.requires_grad_(False)  # Mark that we don't want to train the noise.
        #self.likelihood.noise_covar.raw_noise.requires_grad_(True)

        # Define model
        self.model = ExactGPModel(self.x_data, self.y_data, self.likelihood)

    def train(self, num_iterations=200, learning_rate=0.1):
        """
        Train the Gaussian Process model using the selected optimizer.

        Args:
            num_iterations (int): Number of optimization iterations.
            learning_rate (float): Learning rate for the optimizer.
        """

        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        #scheduler = StepLR(optimizer, step_size=50, gamma=0.3)  # Reduce learning rate by .3 every 50 iterations

        # Set up the marginal log likelihood
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Training
        self.model.train()
        self.likelihood.train()


        for i in range(num_iterations):
            # For Adam and SGD, use the standard training loop
            optimizer.zero_grad()  # Zero gradients from previous iteration
            output = self.model(self.x_data)  # Output from model
            loss = -mll(output, self.y_data)  # Calculate loss and backward gradients
            loss.backward()
            optimizer.step()
            #scheduler.step()

            # Print the loss at every 10th iteration
            # if (i + 1) % 10 == 0:
            #     print(f'Iteration {i + 1}/{num_iterations} - Loss: {loss.item()}')

    def get_hyperparameters(self):
        """
        Get the learned hyperparameters after training.

        Returns:
            tuple: length scale, marginal standard deviation (sigmaf), and observation noise (sigman).
        """
        self.model.eval()
        self.likelihood.eval()

        # Lengthscale (ARD kernel) and signal variance (scale_kernel)
        lengthscales = self.model.covar_module.base_kernel.lengthscale.detach().numpy().flatten()
        sigmaf = self.model.covar_module.outputscale.item() ** 0.5 # Take the square root of the output scale

        # Noise variance (Gaussian likelihood)
        sigman = self.model.likelihood.noise.item()

        return lengthscales, sigmaf, sigman,


class SE_Mercer:
    def __init__(self, x_data, y_data, sigma=1.0, noise_level=1e-3, num_iterations=150, learning_rate=0.08):
        """
        Initialize the SE_Mercer class, use Hyperlearn to Gaussian process model's hyperparameters.

        Args:
            x_data (numpy.ndarray): Input data of shape (n_samples, n_features).
            train_y (numpy.ndarray): Target data of shape (n_samples,).
            sigma (float): standard deviation of the Gaussian measure on the real line
            initial_noise (float): Initial observation noise for Hyperlearn.
            num_iterations (int): Number of optimization iterations for Hyperlearn.
            learning_rate (float): Learning rate for Hyperlearn.
        """
        self.x_data = x_data
        self.y_data = y_data.flatten()
        self.sigma = sigma
        self.noise_level = noise_level
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self._get_hyperparameters()

    def _get_hyperparameters(self):
        """
        Use the Hyperlearn class to get sigma and lengthscales.

        Returns:
            tuple: Lengthscales, sigmaf, sigman.
        """
        # Initialize Hyperlearn and train to get hyperparameters
        hyperlearn_instance = Hyperlearn(self.x_data, self.y_data, self.noise_level)
        hyperlearn_instance.train(num_iterations=self.num_iterations, learning_rate=self.learning_rate)

        self.lengthscales, self.sigmaf, self.sigman = hyperlearn_instance.get_hyperparameters()
        #print(f'lengthscales, sigmaf, sigman: {self.lengthscales}, {self.sigmaf}, {self.sigman}')
        return self.lengthscales, self.sigmaf, self.sigman

    def get_hyperparameters(self):
        return self.lengthscales, self.sigmaf, self.sigman

    def eigen_parameters(self, sigma=None, length_scale=None):
        """
        Compute parameters a, b, c which are required for eigenvalue and eigenfunctions for SE
        spectral representations.

        Args:
            sigma (float, optional): Standard deviation of the Gaussian measure on the real line.
            Defaults to instance attribute if not provided.

            length_scale (float, optional): Length scale for one dimension.
            Defaults to instance attribute if not provided.

        Returns:
            tuple: Parameters a, b, c.
        """
        if sigma == None and length_scale == None:
            length_scale_vec, _, _ = self.get_hyperparameters()
            sigma = self.sigma
            length_scale = length_scale_vec.item()

        sigma = float(sigma)
        a = (2 * sigma ** 2) ** (-1)
        b = (2 * length_scale ** 2) ** (-1)
        c = sqrt(a ** 2 + 4 * a * b)
        return a, b, c

    def n_terms_SE(self, sigma=None, length_scale_vec=None, residual=1e-6):
        """
        Calculate the number of terms for the SE kernel approximation.

        Args:
            sigma (float, optional): Standard deviation of the Gaussian measure on the real line.
            Defaults to instance attribute if not provided.

            length_scale (float, optional): Length scale for one dimension.
            Defaults to instance attribute if not provided.

            residual (float): residual sum of eigenvalues.

        Returns:
            numpy.ndarray: Number of terms for each lengthscale.
        """

        if sigma==None and length_scale_vec==None:
            sigma = self.sigma
            length_scale_vec, _, _ = self.get_hyperparameters()
        n_terms = np.zeros(len(length_scale_vec), dtype=int)
        for i in range(len(length_scale_vec)):
            a, b, c = self.eigen_parameters(sigma, length_scale_vec[i])
            A = (1/2) * a + b + (1/2) * c
            n_terms[i] = int(ceil(log(residual) / log(b / A)))
            if n_terms[i] >= 800:
                n_terms[i] = 800
        return n_terms


    @staticmethod
    def W_array(n_eigen_vec, seed=None):
        """
        Generate a W array where each element is a random normal vector
        corresponding to the length of n_eigen_vec.

        Args:
            n_eigen_vec (list or numpy.ndarray): A list of eigenvector lengths.

        Returns:
            list: A list of numpy arrays where each array is filled with
                  random normal values of the corresponding length from n_eigen_vec.
        """

        # if seed != None:
        #     np.random.seed(seed)

        d = len(n_eigen_vec)
        W = []
        for i in range(d):
            W.append(np.random.normal(0, 1, int(n_eigen_vec[i])))
        return W

    def lambda_n(self, n, sigma=None, length_scale=None):
        """
        Compute the first n eigenvalues.

        Args:
            n (int): Number of terms.

            sigma (float, optional): Standard deviation of the Gaussian measure on the real line.
            Defaults to instance attribute if not provided.

            length_scale (float, optional): Length scale for one dimension.
            Defaults to instance attribute if not provided.

        Returns:
            numpy.ndarray: first n eigenvalues
        """

        if sigma == None and length_scale == None:
            length_scale_vec, _, _ = self.get_hyperparameters()
            sigma = self.sigma
            length_scale = length_scale_vec.item()
        a, b, c = self.eigen_parameters(sigma, length_scale)
        A = (1/2) * a + b + (1/2) * c
        iEigen = np.arange(n)  # Vectorized range from 0 to n-1
        return np.sqrt(a / A) * (b / A) ** iEigen


    def phi(self, n, x, sigma=None, length_scale=None):
        """
        Computes the first n eigenfunctions at x locations using a recurrence relation

        Args:
            n (int): number of leading eigenfunctions to return

            x (numpy.ndarray): N input locations (N,)

            sigma (float, optional): Standard deviation of the Gaussian measure on the real line.
            Defaults to instance attribute if not provided.

            length_scale (float, optional): Length scale for one dimension.
            Defaults to instance attribute if not provided.

        Returns:
            numpy.ndarray: first n eigenfunctions at N locations x
        """

        x = np.asarray(x)
        N = x.shape[0]
        if sigma == None and length_scale == None:
            length_scale_vec, _, _ = self.get_hyperparameters()
            sigma = self.sigma
            length_scale = length_scale_vec.item()
        a, b, c = self.eigen_parameters(sigma, length_scale)
        phi = np.zeros((N, n))
        if n == 1:
            phi[:, 0] = (c / a) ** (0.25) * exp(-0.5 * (c - a) * x ** 2)
        if n == 2:
            phi[:, 0] = (c / a) ** (0.25) * exp(-0.5 * (c - a) * x ** 2)
            phi[:, 1] = sqrt(2 * c) * x * phi[:, 0]
        if n > 2:
            phi[:, 0] = (c / a) ** (0.25) * exp(-0.5 * (c - a) * x ** 2)
            phi[:, 1] = sqrt(2 * c) * x * phi[:, 0]
            for i in range(2, n):
                phi[:, i] = sqrt(2 * c) * x * phi[:, i - 1] / sqrt(i) - sqrt((i - 1) / (i)) * phi[:, i - 2]
        return phi


    def diff_phi(self, n, x, sigma=None, length_scale=None):
        """
        Computes the first n derivatives of eigenfunctions at N locations x using the recurrence relations

        Args:
            n (int): number of leading eigenfunctions to return

            x (numpy.ndarray): N input locations (N,)

            sigma (float, optional): Standard deviation of the Gaussian measure on the real line.
            Defaults to instance attribute if not provided.

            length_scale (float, optional): Length scale for one dimension.
            Defaults to instance attribute if not provided.

        Returns:
            numpy.ndarray: Derivative of phi values.
        """

        x = np.asarray(x)
        N = x.shape[0]
        if sigma == None and length_scale == None:
            length_scale_vec, _, _ = self.get_hyperparameters()
            sigma = self.sigma
            length_scale = length_scale_vec.item()
        a, b, c = self.eigen_parameters(sigma, length_scale)
        phi = self.phi(n, x, sigma, length_scale)
        phi_diff = np.zeros((N, n))
        if n == 0:
            phi_diff[:, 0] = -(c - a) * x * phi[:, 0]
        if n == 1:
            phi_diff[:, 0] = -(c - a) * x * phi[:, 0]
            phi_diff[:, 1] = sqrt(2 * c) * (phi[:, 0] + x * phi_diff[:, 0])
        if n > 1:
            phi_diff[:, 0] = -(c - a) * x * phi[:, 0]
            phi_diff[:, 1] = sqrt(2 * c) * (phi[:, 0] + x * phi_diff[:, 0])
            for i in range(2, n):
                j = i - 1
                phi_diff[:, i] = sqrt(2 * c / i) * (phi[:, i - 1] + x
                                                        * phi_diff[:, i - 1]) - sqrt((i - 1) / i) * phi_diff[:, i - 2]
        return phi_diff

# Testing the functions

if __name__ == "__main__":

    # Test Hyperlearn class ----------------------------------------------> update with generateX and Y func

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

    # Instantiating the Hyperlearn class
    hyperlearn_instance = Hyperlearn(xData, yData, noise_level=1e-3)

    # Test Hyperlearn.train() and Hyperlearn.get_hyperparemeters()
    hyperlearn_instance.train()
    lengthscales, sigmaf, sigman = hyperlearn_instance.get_hyperparameters()
    print(f'lengthscales, sigmaf, sigman = {lengthscales, sigmaf, sigman}')
    print('\n')

    # ------------------------------------------
    # Test SE_Mercer class
    # ------------------------------------------

    # Instantiating the SE_Mercer class
    SE_Mercer_instance = SE_Mercer(xData, yData)

    # Test SE_Mercer._get_hyperparemeters()
    print(f"get_hyperparameters\n: {SE_Mercer_instance.get_hyperparameters()}")

    # Test SE_Mercer.eigen_parameters()
    print(f'eigen_parameters\n: {SE_Mercer_instance.eigen_parameters(1.0, 1.0)}')


    # Test SE_Mercer.n_terms_SE()
    print(f'n_terms\n: {SE_Mercer_instance.n_terms_SE(1, [1.0, 2.0])}')

    # Test SE_Mercer.W_array()
    print(f'W_array:\n {SE_Mercer.W_array([5, 10])}')

    # Test SE_Mercer.lambda_n()
    print(f'lambda_n:\n {SE_Mercer_instance.lambda_n(10, 1.0, 1.0)}')

    #Test SE_Mercer.phi()
    x = np.array([1, 2, 3])
    print(f'phi no lenscale & sigma:\n {SE_Mercer_instance.phi(10, x)}')
    #print(f'phi:\n {SE_Mercer_instance.phi(10, x, 1.0, 1.0)}')

    #Test SE_Mercer.diff_phi()
    print(f'diff_phi no lenscale & sigma:\n {SE_Mercer_instance.diff_phi(10, x)}')
    #print(f'diff_phi:\n {SE_Mercer_instance.diff_phi(10, x, 1.0, 1.0)}')







