from .preprocessor import Hyperlearn
from .preprocessor import SE_Mercer
from .decoupled_GP import Decoupled_GP
from .utils import *
from .max_k_sum import *

import numpy as np
from numpy import log
import scipy
from scipy.optimize import minimize, Bounds
from scipy.stats import norm
from pylab import *
import time
from chebpy import chebfun
import matplotlib.pyplot as plt

class TSRoots:
    def __init__(self, x_data, y_data, lb, ub, sigma=1.0, noise_level=1e-3, learning_rate=0.05, seed=None):
        self.x_data = x_data
        self.y_data = y_data
        self.lb = lb
        self.ub = ub
        self.sigma = sigma
        self.noise_level = noise_level
        self.learning_rate = learning_rate
        self.seed = seed

        # Initialize an instance of Decoupled_GP inside TSRoots
        self.decoupled_gp = Decoupled_GP(x_data, y_data, sigma=self.sigma, noise_level=self.noise_level,
                                         learning_rate=self.learning_rate, seed=self.seed)


    def multi_func_roots_cheb(self, lb, ub, W=None, length_scale_vec=None, n_eigen_vec=None, sigma=None, sigmaf=None):
        """
        Find critical points and second derivatives of the GP function using Chebyshev approximation.

        Args:
            lb (list): Lower bounds for each dimension.
            ub (list): Upper bounds for each dimension.
            W (list, optional): List of weight vectors for each dimension. Defaults to precomputed values if not provided.
            length_scale_vec (numpy.ndarray, optional): Length scales for each dimension. Defaults to precomputed values if not provided.
            n_eigen_vec (list, optional): Number of leading eigenfunctions for each dimension. Defaults to precomputed values if not provided.
            sigma (float, optional): Standard deviation. Defaults to precomputed value if not provided.
            sigmaf (float, optional): Scaling factor for GP function. Defaults to precomputed value if not provided.

        Returns:
            tuple: x_critical (list), func_x_critical (list), dfunc_x_critical(list), d2func_x_critical (list), num_combi (int)
        """

        # Use precomputed values if optional arguments are not provided
        if lb is None:
            lb = self.lb
        if ub is None:
            ub = self.ub
        if W is None:
            W = self.decoupled_gp.W
        if length_scale_vec is None:
            length_scale_vec = self.decoupled_gp.lengthscales
        if n_eigen_vec is None:
            n_eigen_vec = self.decoupled_gp.n_eigen_vec
        if sigma is None:
            sigma = self.sigma
        if sigmaf is None:
            sigmaf = self.decoupled_gp.sigmaf

        d = len(length_scale_vec)  # Dimensionality
        x_critical = [None] * d
        func_x_critical = [None] * d
        dfunc_x_critical = [None] * d
        d2func_x_critical = [None] * d
        num_combi = 1

        for i in range(d):
            # Define the GP path function for this dimension
            f = lambda x_test: self.decoupled_gp.uni_GP_path(n_eigen_vec[i], x_test, W[i], sigma, length_scale_vec[i],
                                                             sigmaf)

            # Approximate the function using Chebyshev polynomial
            f_cheb = chebfun(f, [lb[i], ub[i]])

            # First and second derivatives using Chebyshev approximation
            df_cheb = f_cheb.diff()
            d2f_cheb = df_cheb.diff()

            # Get critical points and corresponding function values
            critical_points = df_cheb.roots()

            # Add lower and upper bounds to the critical points
            x_critical[i] = np.hstack((critical_points, lb[i], ub[i]))  # Ensure bounds are the last two elements
            func_x_critical[i] = f_cheb(x_critical[i])
            dfunc_x_critical[i] = df_cheb(x_critical[i])
            d2func_x_critical[i] = d2f_cheb(x_critical[i])

            # Update combination count
            num_combi *= x_critical[i].size

        return x_critical, func_x_critical, dfunc_x_critical, d2func_x_critical, num_combi

    def sort_mixed_mono(self, x_critical, func_x_critical, dfunc_x_critical, d2func_x_critical):
        """
        Sort the critical points into mono and mix candidates based on function values,
        first and second derivatives.

        Args:
            x_critical (list of arrays): Critical points for each dimension.
            func_x_critical (list of arrays): Function values at each critical point.
            dfunc_x_critical (list of arrays): First derivatives at each critical point.
            d2func_x_critical (list of arrays): Second derivatives at each critical point.

        Returns:
            tuple:
                x_critical_mono (list): Mono critical points.
                x_critical_mixed (list): Mix critical points.
                func_x_critical_mono (list): Function values at mono critical points.
                func_x_critical_mixed (list): Function values at mix critical points.
                no_combi_mono (int): Number of mono combinations.
                no_combi_mixed (int): Number of mix combinations.
        """
        d = len(x_critical)  # Dimensionality

        # Initialize lists to store results for each dimension
        x_critical_mono = [None] * d
        x_critical_mixed = [None] * d
        func_x_critical_mono = [None] * d
        func_x_critical_mixed = [None] * d
        no_mono = np.zeros(d, dtype=int)
        no_mixed = np.zeros(d, dtype=int)

        # Loop through each dimension
        for i in range(d):
            func = func_x_critical[i]
            dfunc = dfunc_x_critical[i]
            d2func = d2func_x_critical[i]

            # Calculate h and handle the last two bounds separately
            h = func * d2func  # Interior points calculation
            #h[-2:] = func[-2:] * dfunc[-2:]  # Lower and upper bounds (end-1 and end) before RDZ adjustment
            h[-2] = func[-2] * dfunc[-2]
            h[-1] = func[-1] * (-dfunc[-1])


            # Find indices for mono and mix candidates in one pass
            monoidx = (h > 0)
            mixedidx = (h < 0)

            # Store mono candidates
            if np.any(monoidx):
                x_critical_mono[i] = x_critical[i][monoidx]
                func_x_critical_mono[i] = func[monoidx]
                no_mono[i] = x_critical_mono[i].shape[0]
            else:
                x_critical_mono[i] = np.array([])
                func_x_critical_mono[i] = np.array([])
                no_mono[i] = 0

            # Store mix candidates
            if np.any(mixedidx):
                x_critical_mixed[i] = x_critical[i][mixedidx]
                func_x_critical_mixed[i] = func[mixedidx]
                no_mixed[i] = x_critical_mixed[i].shape[0]
            else:
                x_critical_mixed[i] = np.array([])
                func_x_critical_mixed[i] = np.array([])
                no_mixed[i] = 0

        # Calculate the number of possible combinations
        no_combi_mono = np.prod(no_mono)  # Monotonic combinations
        no_combi_mixed = np.prod(no_mixed)  # Mixed combinations

        return x_critical_mono, x_critical_mixed, func_x_critical_mono, func_x_critical_mixed, no_combi_mono, no_combi_mixed

    @staticmethod
    def fullfact_design(levels):  # remember to copyright
        """
        Generate a full factorial design matrix based on the given levels.

        Code extracted and modified from:
        https://github.com/tirthajyoti/Design-of-experiment-Python/blob/master/pyDOE_corrected.py
        Copyright (c) 2019 Tirthajyoti Sarkar. Licensed under the MIT License.

        Args:
            levels (list): A list of integers where each entry represents the number of levels for that factor.

        Returns:
            numpy.ndarray: A full factorial design matrix.
        """
        n = len(levels)  # number of factors
        nb_lines = np.prod(levels)  # number of trial conditions
        H = np.zeros((nb_lines, n), dtype=int)  # Initialize the design matrix
        level_repeat = 1
        range_repeat = np.prod(levels)

        for i in range(n):  # Loop through factors
            range_repeat //= levels[i]  # Reduce the repeat range
            lvl = []  # Temporary list for the current factor's levels
            for j in range(levels[i]):  # Repeat each level 'level_repeat' times
                lvl += [j] * level_repeat
            rng = lvl * range_repeat  # Repeat the pattern 'range_repeat' times
            level_repeat *= levels[i]  # Update 'level_repeat' for the next factor
            H[:, i] = rng  # Assign the pattern to the design matrix column

        return H

    def root_combinations(self, multi_roots, func_multi_roots):
        """
            Generate all possible combinations of roots across multiple dimensions and compute the corresponding
            function values at these combinations.

            Args:
                multi_roots (list of numpy.ndarray): Roots (critical points) for each dimension.
                func_multi_roots (list of numpy.ndarray): Function values at the roots for each dimension.

            Returns:
                tuple:
                    - roots_combi (numpy.ndarray): Combinations of roots across all dimensions.
                    - func_multi_dim (numpy.ndarray): Function values corresponding to each root combination.
        """
        # Dimension size
        d = len(multi_roots)

        # Get number of roots for each dimension
        num_roots = [len(multi_roots[i]) for i in range(d)]

        # Check if all dimensions have roots
        if np.all(np.array(num_roots) > 0):
            # Get indices for multidimensional root combinations using full factorial design
            idx = self.fullfact_design(num_roots)

            # Initialize arrays for root combinations and function values
            roots_combi = np.zeros((idx.shape[0], d), dtype=float)
            func_roots_combi = np.zeros_like(roots_combi, dtype=float)

            # Populate roots_combi and func_roots_combi using the indices
            for i in range(d):
                roots_combi[:, i] = multi_roots[i][idx[:, i]]
                func_roots_combi[:, i] = func_multi_roots[i][idx[:, i]]

            # Calculate the product of function values across dimensions for each combination
            func_multi_dim = np.prod(func_roots_combi, axis=1)
        else:
            # If any dimension has no roots, return empty arrays
            roots_combi = np.array([])
            func_multi_dim = np.array([])

        return roots_combi, func_multi_dim


    def ordering_summax_mixed(self, multi_x_cri_mixed, multi_f_mixed, multi_f, k):
        """
        Select a subset of the set of all possible mixed combinations of roots when the number of possible combinations
        exceeds a threshold (k), including rows with negative function values.

        Args:
            multi_x_cri_mixed (list of arrays): Mixed critical points (roots) for each dimension.
            multi_f_mixed (list of arrays): Mixed function values for each dimension.
            multi_f (list of arrays): Function values for each dimension.
            k (int): Number of top combinations to select.

        Returns:
            tuple:
                - x_matrix_max (numpy.ndarray): Selected subset of mixed roots (size k x d).
                - combi_f (numpy.ndarray): Product of function values for each selected combination (size k).
                - negaidx (numpy.ndarray): Indices of combinations with negative function values.
        """
        d = len(multi_x_cri_mixed)

        # Compute relative function values for each dimension
        rela_multi_f_mixed = []
        for i in range(d):
            rela_multi_f_mixed.append(np.log(np.abs(multi_f_mixed[i]) / np.max(np.abs(multi_f[i]))))

        # Use the find_max_k_sum_without_dp function to select top k combinations
        ORD_max = find_max_k_sum_without_dp(rela_multi_f_mixed, k)

        # Preallocate matrices for storing the results
        x_matrix_max = np.zeros((k, d))
        f_matrix = np.zeros((k, d))

        # Populate the matrices using advanced indexing for faster execution
        for j in range(k):
            x_matrix_max[j, :] = np.array([multi_x_cri_mixed[i][ORD_max[j][i]] for i in range(d)])
            f_matrix[j, :] = np.array([multi_f_mixed[i][ORD_max[j][i]] for i in range(d)])

        # Compute the product of function values across dimensions for each combination
        combi_f = np.prod(f_matrix, axis=1)

        # Find rows with negative function values
        negaidx = np.where(combi_f < 0)[0]

        return x_matrix_max, combi_f, negaidx

    @staticmethod
    def create_objective_and_derivative_wrapper(func, *args):
        """
        Create an objective function and its derivative, with caching.

        Args:
            func (callable): The function that returns both the objective value and the derivative.
            *args: Additional arguments to pass to func.

        Returns:
            tuple: The objective function and its derivative.
        """
        cache = {}

        def objective(x):
            x_tuple = tuple(x)
            if x_tuple not in cache:
                cache[x_tuple] = func(x, *args)
            return cache[x_tuple][0]  # Return the objective value

        def derivative(x):
            x_tuple = tuple(x)
            if x_tuple not in cache:
                cache[x_tuple] = func(x, *args)
            return cache[x_tuple][1]  # Return the derivative

        return objective, derivative

    @staticmethod
    def multistart_optimization(objective_func, jac_func, initial_guesses, method='SLSQP', **kwargs):
        """
        Perform multistart optimization using different initial guesses.

        Args:
            objective_func (callable): The objective function.
            jac_func (callable): The Jacobian (derivative) function.
            initial_guesses (list of arrays): A list of initial guesses to start optimization from.
            method (str): Optimization method (default is 'SLSQP').
            **kwargs: Additional keyword arguments for scipy.optimize.minimize.

        Returns:
            result (OptimizeResult): The best optimization result.
        """
        best_result = None
        for x0 in initial_guesses:
            # Perform optimization with the current initial guess
            result = minimize(fun=objective_func, x0=x0, jac=jac_func, method=method, **kwargs)
            # Keep track of the best result
            if best_result is None or result.fun < best_result.fun:
                best_result = result
        return best_result

    @staticmethod
    def confun(x, X_data):
        """
        Nonlinear inequality constraints for optimization.

        Args:
            x (numpy.ndarray): Current point in the optimization.
            X_data (numpy.ndarray): The data points to compare against.

        Returns:
            tuple: c (numpy.ndarray), ceq (numpy.ndarray)
                - c: The nonlinear inequality constraints.
                - ceq: The nonlinear equality constraints (empty in this case).
        """
        N, d = X_data.shape
        x_m = np.tile(x, (N, 1))
        min_norm = np.min(np.linalg.norm(x_m - X_data, axis=1))

        # Compute the nonlinear inequalities at x
        c = min_norm - 1e-6  # Inequality constraint: c >= 0
        ceq = np.array([])  # No equality constraints

        return c, ceq

    @staticmethod
    def constraint_wrapper(x, X_data):
        """
        Wrapper to make the constraints compatible with scipy.optimize.minimize.

        Args:
            x (numpy.ndarray): Current point in the optimization.
            X_data (numpy.ndarray): The data points to compare against.

        Returns:
            numpy.ndarray: The concatenation of inequality and equality constraints.
        """
        c, ceq = TSRoots.confun(x, X_data)
        return np.hstack((c, ceq))


    def xnew_TSroots(self, X_data=None, y_data=None, sigma=None, sigmaf=None, sigman=None, length_scale_vec=None,
                     lb=None, ub=None, residual=None, plot=True):
        """
        Selects a new solution point using TSroots.

        Args:
            X_data (ndarray, optional): Input data of shape (n, d). Defaults to precomputed values if not provided.
            y_data (ndarray, optional): Output data of shape (n, 1). Defaults to precomputed values if not provided.
            sigmaf (float, optional): Marginal standard deviation. Defaults to precomputed value if not provided.
            length_scale_vec (ndarray, optional): Vector of length scales of the ARD SE kernel of shape (1, d).
                                                    Defaults to precomputed values if not provided.
            lb (ndarray, optional): Lower bound vector of shape (1, d). Defaults to instance's lb if not provided.
            ub (ndarray, optional): Upper bound vector of shape (1, d). Defaults to instance's ub if not provided.

        Returns:
            tuple: x_new (ndarray), y_new (float), no_iniPoints (int)
        """

        # Use precomputed values if optional arguments are not provided
        if X_data is None:
            X_data = self.x_data
        if y_data is None:
            y_data = self.y_data
        if sigma is None:
            sigma = self.sigma
        if sigmaf is None:
            sigmaf = self.decoupled_gp.sigmaf
        if sigman is None:
            sigman = self.decoupled_gp.sigman
        if length_scale_vec is None:
            length_scale_vec = self.decoupled_gp.lengthscales
        if lb is None:
            lb = self.lb
        if ub is None:
            ub = self.ub
        if residual is None:
            residual = 10 ** (-16)

        # Get n_eigen_vec and W_array from precomputed values
        n_eigen_vec = self.decoupled_gp.SE_Mercer_instance.n_terms_SE(sigma=sigma, length_scale_vec=length_scale_vec,
                                                                      residual=residual)
        W = self.decoupled_gp.SE_Mercer_instance.W_array(n_eigen_vec)

        # Compute the v vector
        v_vec = self.decoupled_gp.v_vec(X_data, y_data, W, length_scale_vec, n_eigen_vec, sigma, sigmaf, sigman)

        # Compute local minima using multi_func_roots_cheb
        twoNe = 500
        multi_x_cri, multi_f, multi_df, multi_d2f, _ = self.multi_func_roots_cheb(lb=lb, ub=ub, W=W,
                                                         length_scale_vec=length_scale_vec, n_eigen_vec=n_eigen_vec,
                                                          sigma=sigma, sigmaf=sigmaf)

        # Sort mono and mixed candidates
        multi_x_cri_mono, multi_x_cri_mixed, multi_f_mono, multi_f_mixed, no_combi_mono, no_combi_mixed = \
                                                self.sort_mixed_mono(multi_x_cri, multi_f, multi_df, multi_d2f, )

        if no_combi_mixed <= twoNe:
            print("# We enumerate all possible combinations...")
            # Handle mono type candidates
            combiroots_mono, combif_mono = self.root_combinations(multi_x_cri_mono, multi_f_mono)
            posi_fmonoidx = np.where(combif_mono > 0)[0]
            xmin_mono = combiroots_mono[posi_fmonoidx, :] if len(posi_fmonoidx) > 0 else []

            # Handle mixed type candidates
            combiroots_mixed, combif_mixed = self.root_combinations(multi_x_cri_mixed, multi_f_mixed)
            nega_fmixedidx = np.where(combif_mixed < 0)[0]
            xmin_mixed = combiroots_mixed[nega_fmixedidx, :] if len(nega_fmixedidx) > 0 else []


            # Filter out any empty arrays before concatenation
            x_min_candidates = [xmin_mono, xmin_mixed]
            x_min_candidates = [x for x in x_min_candidates if len(x) > 0]

            # If there are any valid candidates, concatenate them; otherwise, create an empty array
            if x_min_candidates:
                x_min = np.vstack(x_min_candidates)
            else:
                x_min = np.empty((0, X_data.shape[1]))

        else:
            print("# We select a subset of the set of all possible combinations...")
            twoN0 = 400
            combiroots, _, negaidx = self.ordering_summax_mixed(multi_x_cri_mixed, multi_f_mixed, multi_f, twoN0)
            x_min = combiroots[negaidx, :]  # Keep only candidates with negative function values

        no_xmin = x_min.shape[0]

        # Exploration set
        if len(x_min) != 0:
            n_epr = len(x_min)  # Exploration set size
            fp_c = self.decoupled_gp.mixPosterior(np.array(x_min), v_vec, X_data, y_data, W, length_scale_vec,
                                                  n_eigen_vec, sigma, sigmaf, diff=False)
            idc = np.argsort(fp_c, axis=0).flatten()[:n_epr]  # Indices of smallest elements
            x_start_1 = np.array(x_min)[idc]
        else:
            x_start_1 = np.empty((0, X_data.shape[1]))

        # Exploitation set
        k2 = 200
        n_epl = min(X_data.shape[0], k2)
        idx = np.argsort(y_data.flatten())[:n_epl]
        x_start_2 = X_data[idx]

        # Combine exploration and exploitation sets
        x_start = np.vstack((x_start_1, x_start_2)) if x_start_1.size > 0 else x_start_2
        no_iniPoints = x_start.shape[0]

        # Create the objective and derivative functions using the wrapper
        objective_value, objective_derivative = self.create_objective_and_derivative_wrapper(
            self.decoupled_gp.mixPosterior, v_vec, X_data, y_data, W, length_scale_vec, n_eigen_vec, sigma, sigmaf, sigman)

        # Define optimization bounds and constraints
        bounds = Bounds(lb, ub)
        constraints = {'type': 'ineq', 'fun': self.constraint_wrapper, 'args': (X_data,)}

        # Set additional optimizer options
        options = {
            'maxiter': 1000,
            'disp': False,
            'ftol': 1e-12,
        }

        # Perform multistart optimization
        start = time.time()
        best_result = self.multistart_optimization(objective_value, objective_derivative, x_start, bounds=bounds,
                                                   options=options)
        #best_result = self.multistart_optimization(objective_value, objective_derivative, x_start, bounds=bounds,
                                                   #options=options, constraints=constraints)

        end = time.time()
        #print(f"Optimization time: {end - start} seconds")

        x_new = best_result.x
        y_new = best_result.fun

        # Plotting
        if plot == True:
            if np.shape(X_data)[1] == 1:
                # Plot the posterior mean and CI only (useful for multiple samples points plot):
                #plot_posterior_TS(self.decoupled_gp, X_data, y_data, length_scale_vec, sigma, sigmaf, sigman)

                # Provide x_new, y,_new to plot the posterior mean, CI, and the newly selected point without sample path:
                #plot_posterior_TS(self.decoupled_gp, X_data, y_data, length_scale_vec, sigma, sigmaf, sigman,
                #                  x_new=x_new, y_new=y_new)

                # Provide arguments W, v_vec, n_eigen_vec to plot posterior mean, CI and sample path (without chosen point):
                plot_posterior_TS(self.decoupled_gp, X_data, y_data, length_scale_vec, sigma, sigmaf, sigman, W, v_vec, n_eigen_vec)
                #plot_posterior_TS(self.decoupled_gp, X_data, y_data, plot_sample_path=True)


            elif np.shape(X_data)[1] == 2:
                plot_posterior_TS_2D(self, X_data, y_data, length_scale_vec, sigma, sigmaf, sigman, x_new, y_new)

        return x_new, y_new, no_iniPoints

    @staticmethod
    def extract_min(X_r, Y_r):
        """
            Extract the minimum function value and its corresponding input vector.

            Args:
                X_r (ndarray): 2D array where each row represents an input vector.
                Y_r (ndarray): 1D array of function values corresponding to the rows in X_r.

            Returns:
                x (ndarray): The input vector corresponding to the minimum function value in Y_r.
                f (float): The minimum function value in Y_r.
            """
        idx = np.argmin(Y_r)
        f = Y_r[idx]
        x = X_r[idx, :]

        return x, f


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

    lbS = -np.ones(1)
    ubS = np.ones(1)


    # ------------------------------------------
    # Test TSRoots class
    # ------------------------------------------

    # Instantiating the TSRoots.multi_func_roots_cheb() class
    TSRoots_instance = TSRoots(xData, yData, lbS, ubS)

    # Accesing some parameters from TSRoots
    W = TTSRoots_instance.decoupled_gp.W
    lengthscales = TSRoots_instance.decoupled_gp.lengthscales
    n_terms = TSRoots_instance.decoupled_gp.n_eigen_vec
    sigmaf = TTSRoots_instance.decoupled_gp.sigmaf
    sigman = TSRoots_instance.decoupled_gp.sigman

    # Test TSRoots.multi_func_roots_cheb()
    print(f"multi_func_roots_cheb without inputs\n: {TSRoots_instance.multi_func_roots_cheb(lbS, ubS)}")

    x_critical, func_x_critical, dfunc_x_critical, d2func_x_critical, no_combi =\
        TSRoots_instance.multi_func_roots_cheb(lbS, ubS)

    # Test TSRoots.sort_mixed_mono()
    print(f"sort_mixed_mono\n: {TSRoots_instance.sort_mixed_mono(x_critical, func_x_critical, dfunc_x_critical, d2func_x_critical)}")

    # Test TSRoots.root_combinations()
    multi_roots, func_multi_roots, _, _, _ = TSRoots_instance.multi_func_roots_cheb(lbS, ubS)
    print(f"root_combinations:\n {TSRoots_instance.root_combinations(multi_roots, func_multi_roots)}")

    x_critical_mono, x_critical_mixed, func_x_critical_mono, func_x_critical_mixed, no_combi_mono, no_combi_mixed = \
        TSRoots_instance.sort_mixed_mono(x_critical, func_x_critical, dfunc_x_critical, d2func_x_critical)

    # Test TSRoots.ordering_summax_mixed()
    print(f"ordering_summax_mixed\n: "
          f"{TSRoots_instance.ordering_summax_mixed(x_critical_mixed, func_x_critical_mixed, func_multi_roots, 1)}")

    # Test TSRoots.xnew_TSroots()
    print(f"xnew_TSroots:\n {TSRoots_instance.xnew_TSroots()}")

    # ------------------------------------------------------
    # See TSRoots extension for BO in 1D_xSinx_function.py
    # -----------------------------------

