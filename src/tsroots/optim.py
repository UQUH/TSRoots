from tsroots.utils import *
from tsroots.max_k_sum import *

from scipy.optimize import minimize, Bounds
from pylab import *
import time
from chebpy import chebfun

class TSRoots:
    def __init__(self, x_data, y_data, lb, ub, sigma=1.0, noise_level=1e-3, learning_rate=0.07, seed=None):
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
            W (list, optional): List of weight vectors for each dimension.
                                Defaults to precomputed values if not provided.
            length_scale_vec (numpy.ndarray, optional): Length scales for each dimension.
                                                        Defaults to precomputed values if not provided.
            n_eigen_vec (list, optional): Number of leading eigenfunctions for each dimension.
                                            Defaults to precomputed values if not provided.
            sigma (float, optional): Standard deviation. Defaults to precomputed value if not provided.
            sigmaf (float, optional): Scaling factor for GP function. Defaults to precomputed value if not provided.

        Returns:
            tuple: x_critical (list), func_x_critical (list), dfunc_x_critical(list), d2func_x_critical (list),
                   num_combi (int)
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


    def sort_mixed_mono_final(self, x_critical, func_x_critical, dfunc_x_critical, d2func_x_critical):
        """
        Sort the critical points into mono and mixed candidates based on function values,
        first and second derivatives, and compute various metrics.

        Args:
            x_critical (list of arrays): Critical points for each dimension.
            func_x_critical (list of arrays): Function values at each critical point.
            dfunc_x_critical (list of arrays): First derivatives at each critical point.
            d2func_x_critical (list of arrays): Second derivatives at each critical point.

        Returns:
            tuple:
                x_critical_mono (list): Mono critical points.
                x_critical_mixed (list): Mixed critical points.
                func_x_critical_mono (list): Function values at mono critical points.
                func_x_critical_mixed (list): Function values at mixed critical points.
                N_zero (int): Product of counts of zero indicators (J_i).
                N_one (int): Product of counts of one indicators (neg_J_i).
                S_zero (int): Product of S_zero values across dimensions.
                S_one (int): Product of S_one values across dimensions.
        """
        d = len(x_critical)  # Dimensionality

        # Initialize lists to store results for each dimension
        x_critical_mono = [None] * d
        x_critical_mixed = [None] * d
        func_x_critical_mono = [None] * d
        func_x_critical_mixed = [None] * d

        no_mixed = np.zeros(d, dtype=int)
        no_mono = np.zeros(d, dtype=int)

        n_zero = np.zeros(d, dtype=int)
        n_one = np.zeros(d, dtype=int)
        s_zero = np.zeros(d, dtype=int)
        s_one = np.zeros(d, dtype=int)

        for i in range(d):
            # Compute h values
            h = d2func_x_critical[i]  # Interior points
            h[-2] = dfunc_x_critical[i][-2]  # Lower bound
            h[-1] = -dfunc_x_critical[i][-1]  # Upper bound

            # Calculate fh values
            fh = func_x_critical[i] * h

            # Find indices for mono and mixed candidates
            monoidx = np.where(fh > 0)[0]  # Mono candidates
            mixedidx = np.where(fh < 0)[0]  # Mixed candidates

            # Process mono candidates
            if len(monoidx) == 0:
                x_critical_mono[i] = np.array([])
                func_x_critical_mono[i] = np.array([])
            else:
                x_critical_mono[i] = x_critical[i][monoidx]
                func_x_critical_mono[i] = func_x_critical[i][monoidx]

            # Process mixed candidates
            if len(mixedidx) == 0:
                x_critical_mixed[i] = np.array([])
                func_x_critical_mixed[i] = np.array([])
            else:
                x_critical_mixed[i] = x_critical[i][mixedidx]
                func_x_critical_mixed[i] = func_x_critical[i][mixedidx]

            no_mixed[i] = x_critical_mixed[i].shape[0]
            no_mono[i] = x_critical_mono[i].shape[0]

            # Calculate metrics
            J_i = (fh > 0).astype(int)
            neg_J_i = (fh < 0).astype(int)
            P_i = (func_x_critical[i] > 0).astype(int)
            neg_P_i = (func_x_critical[i] < 0).astype(int)
            PiJi = P_i & J_i
            neg_PiJi = neg_P_i & J_i
            Pineg_Ji = P_i & neg_J_i
            neg_Pineg_Ji = neg_P_i & neg_J_i

            # Sum metrics for current dimension
            n_zero[i] = np.sum(J_i)
            n_one[i] = np.sum(neg_J_i)
            n_plus_zero = np.sum(PiJi)
            n_minus_zero = np.sum(neg_PiJi)
            n_plus_one = np.sum(Pineg_Ji)
            n_minus_one = np.sum(neg_Pineg_Ji)
            s_zero[i] = n_plus_zero - n_minus_zero
            s_one[i] = n_plus_one - n_minus_one

        # Calculate final metrics across dimensions
        N_zero = np.prod(n_zero)
        N_one = np.prod(n_one)
        S_zero = np.prod(s_zero)
        S_one = np.prod(s_one)
        # Calculate the number of possible combinations
        no_combi_mixed = np.prod(no_mixed) if np.prod(no_mixed) > 0 else 0

        no_combi_mono = np.prod(no_mono) if np.prod(no_mono) > 0 else 0

        return x_critical_mono, x_critical_mixed, func_x_critical_mono, func_x_critical_mixed, N_zero, N_one, S_zero, S_one

    @staticmethod
    def fullfact_design(levels):
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

    def ordering_summax_mono(self, multi_x_cri_mono, multi_f_mono, multi_f, k):
        """
        Select a subset of the set of all possible mono combinations of roots when the number of possible combinations
        exceeds a threshold (k), including rows with negative function values.

        Args:
            multi_x_cri_mono (list of arrays): Mono critical points (roots) for each dimension.
            multi_f_mono (list of arrays): Mono function values for each dimension.
            multi_f (list of arrays): Function values for each dimension.
            k (int): Number of top combinations to select.

        Returns:
            tuple:
                - x_matrix_max (numpy.ndarray): Selected subset of mono roots (size k x d).
                - combi_f (numpy.ndarray): Product of function values for each selected combination (size k).
                - posiidx (numpy.ndarray): Indices of combinations with positive function values.
        """
        d = len(multi_x_cri_mono)

        # Compute relative function values for each dimension
        rela_multi_f_mono = []
        for i in range(d):
            rela_multi_f_mono.append(np.log(np.abs(multi_f_mono[i]) / np.max(np.abs(multi_f[i]))))

        # Use the find_max_k_sum_without_dp function to select top k combinations
        ORD_max = find_max_k_sum_without_dp(rela_multi_f_mono, k)

        # Preallocate matrices for storing the results
        x_matrix_max = np.zeros((k, d))
        f_matrix = np.zeros((k, d))

        # Populate the matrices using advanced indexing for faster execution
        for j in range(k):
            x_matrix_max[j, :] = np.array([multi_x_cri_mono[i][ORD_max[j][i]] for i in range(d)])
            f_matrix[j, :] = np.array([multi_f_mono[i][ORD_max[j][i]] for i in range(d)])

        # Compute the product of function values across dimensions for each combination
        combi_f = np.prod(f_matrix, axis=1)

        # Find rows with negative function values
        posiidx = np.where(combi_f > 0)[0]


        return x_matrix_max, combi_f, posiidx

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
            return np.array(cache[x_tuple][1]).flatten()  # Ensure the gradient is 1D

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

    def xnew_TSroots(self, X_data=None, y_data=None, sigma=None, sigmaf=None, sigman=None, length_scale_vec=None,
                            lb=None, ub=None, residual=None, n_o=None, n_e=None, n_x=None, plot=False):
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
        if n_o is None:
            n_o = 5000
        if n_e is None:
            n_e = 1000
        if n_x is None:
            n_x = 1000


        # Get n_eigen_vec and W_array from precomputed values
        n_eigen_vec = self.decoupled_gp.SE_Mercer_instance.n_terms_SE(sigma=sigma, length_scale_vec=length_scale_vec,
                                                                      residual=residual)
        W = self.decoupled_gp.SE_Mercer_instance.W_array(n_eigen_vec)

        # Compute the v vector
        v_vec = self.decoupled_gp.v_vec(X_data, y_data, W, length_scale_vec, n_eigen_vec, sigma, sigmaf, sigman)

        # Compute local minima using multi_func_roots_cheb
        n_o_limit = 500  # set to 5000
        multi_x_cri, multi_f, multi_df, multi_d2f, _ = self.multi_func_roots_cheb(lb=lb, ub=ub, W=W,
                                                                                  length_scale_vec=length_scale_vec,
                                                                                  n_eigen_vec=n_eigen_vec,
                                                                                  sigma=sigma, sigmaf=sigmaf)

        # Sort mono and mixed candidates
        multi_x_cri_mono, multi_x_cri_mixed, multi_f_mono, multi_f_mixed, N_zero, N_one, S_zero, S_one = \
            self.sort_mixed_mono_final(multi_x_cri, multi_f, multi_df, multi_d2f)

        alpha = 3
        num_local_min = int(round(0.5 * (N_one - S_one + N_zero + S_zero), 0))
        num_neg_local_min = int(round(0.5 * (N_one - S_one), 0))


        if n_o <= num_neg_local_min:
            print("# We select a subset of the set of all possible combinations...")

            combiroots_mixed, _, negaidx = self.ordering_summax_mixed(multi_x_cri_mixed, multi_f_mixed, multi_f,
                                                                      int(round(alpha * n_o,  0)))
            x_min = combiroots_mixed[negaidx, :]  # Keep only candidates with negative function values
            if len(x_min) >= n_o:
                x_min = x_min[:n_o]  # The n_o smallest f in mixed local min

        else:
            print("# We enumerate all possible combinations...")
            # Get possible combinations of mixed type candidates and their corresponding prior function values
            combiroots_mixed, combif_mixed = self.root_combinations(multi_x_cri_mixed, multi_f_mixed)

            # Implement sorted multi-indices of negative local minima based on prior function values:

            # Identify the indices where combif_mixed is less than 0
            nega_fmixedidx = np.where(combif_mixed < 0)[0]

            # Extract the negative values from combif_mixed
            combif_mixed_negatives = combif_mixed[nega_fmixedidx]

            # Sort the extracted negative values in ascending order
            sorted_indices_within_negatives = np.argsort(combif_mixed_negatives)
            nega_fmixedidx_sorted = nega_fmixedidx[sorted_indices_within_negatives]

            xmin_mixed = combiroots_mixed[nega_fmixedidx_sorted, :] if len(nega_fmixedidx_sorted) > 0 else []

            if n_o <= num_local_min:
                print("# We select a subset of the set of all possible combinations...")
                # The alpha * n_o largest |f| in mixed type candidates coordinates
                combiroots_mono, _, posiidx = self.ordering_summax_mono(multi_x_cri_mono, multi_f_mono, multi_f,
                                                                        int(round(alpha * (n_o - num_neg_local_min),
                                                                                  0)))
                x_min_mono = combiroots_mono[posiidx, :]  # Keep only candidates with negative function values
                if len(x_min_mono) >= int(round(n_o - num_neg_local_min)):
                    x_min_mono = x_min_mono[
                                 :int(round(n_o - num_neg_local_min))]  # The n_o smallest f in mixed local min

            else:
                # Get possible combinations of mixed type candidates and their corresponding prior function values
                combiroots_mono, combif_mono = self.root_combinations(multi_x_cri_mono, multi_f_mono)

                # Identify the indices where combif_mono is greater than 0
                posi_fmonoidx = np.where(combif_mono > 0)[0]

                # Extract the positive values from combif_mono
                combif_mono_positives = combif_mono[posi_fmonoidx]

                # Sort the extracted positive values in ascending order
                sorted_indices_within_positives = np.argsort(combif_mono_positives)
                posi_fmonoidx_sorted = posi_fmonoidx[sorted_indices_within_positives]

                # Filter and sort xmin_mono based on the indices
                xmin_mono = combiroots_mono[posi_fmonoidx_sorted, :] if len(posi_fmonoidx_sorted) > 0 else []
            x_min = [xmin_mono, xmin_mixed]

            # Filter out any empty arrays before concatenation
            x_min = [x for x in x_min if len(x) > 0]

            # If there are any valid candidates, concatenate them; otherwise, create an empty array
            if x_min:
                x_min = np.vstack(x_min)
            else:
                x_min = np.empty((0, X_data.shape[1]))

        # Exploration set
        if len(x_min) != 0:
            n_epr = min(len(x_min), n_e)
            fp_c = self.decoupled_gp.mixPosterior(np.array(x_min), v_vec, X_data, y_data, W, length_scale_vec,
                                                  n_eigen_vec, sigma, sigmaf, diff=False)
            idc = np.argsort(fp_c, axis=0).flatten()[:n_epr]  # Indices of the smallest elements
            self.x_start_1 = np.array(x_min)[idc]
        else:
            self.x_start_1 = np.empty((0, X_data.shape[1]))

        # Exploitation set
        n_epl = min(X_data.shape[0], n_x)

        K_s = self.decoupled_gp.cross_covariance_kernel(X_data, X_data, length_scale_vec, sigmaf)  # (N by n)

        mu = K_s.T @ self.decoupled_gp.Cnn_inv @ y_data.flatten()

        idx = np.argsort(mu.flatten())[:n_epl]

        self.x_start_2 = X_data[idx]

        # Combine exploration and exploitation sets
        x_start = np.vstack((self.x_start_1, self.x_start_2)) if self.x_start_1.size > 0 else self.x_start_2
        no_iniPoints = x_start.shape[0]

        # Create the objective and derivative functions using the wrapper
        objective_value, objective_derivative = self.create_objective_and_derivative_wrapper(
            self.decoupled_gp.mixPosterior, v_vec, X_data, y_data, W, length_scale_vec, n_eigen_vec, sigma, sigmaf,
            sigman)

        # Define optimization bounds and constraints
        bounds = Bounds(lb, ub)

        # Set additional optimizer options
        options = {
            'ftol': 1e-10,
            'disp': False,
            'maxiter': 500
        }

        # Perform multistart optimization
        start = time.time()
        # Perform multistart optimization and get indices of initial guesses that converged to the best local minimum
        best_result = self.multistart_optimization(
            objective_value, objective_derivative, x_start, bounds=bounds, options=options
        )

        x_new = best_result.x
        y_new = best_result.fun

        # Plotting
        if plot == True:
            if np.shape(X_data)[1] == 1:

                plot_posterior_TS(self.decoupled_gp, X_data, y_data, length_scale_vec, sigma, sigmaf, sigman, W, v_vec,
                                  n_eigen_vec)

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
    W = TSRoots_instance.decoupled_gp.W
    lengthscales = TSRoots_instance.decoupled_gp.lengthscales
    n_terms = TSRoots_instance.decoupled_gp.n_eigen_vec
    sigmaf = TSRoots_instance.decoupled_gp.sigmaf
    sigman = TSRoots_instance.decoupled_gp.sigman

    # Test TSRoots.multi_func_roots_cheb()
    print(f"multi_func_roots_cheb without inputs\n: {TSRoots_instance.multi_func_roots_cheb(lbS, ubS)}")

    x_critical, func_x_critical, dfunc_x_critical, d2func_x_critical, no_combi =\
        TSRoots_instance.multi_func_roots_cheb(lbS, ubS)

    # Test TSRoots.sort_mixed_mono()
    print(f"sort_mixed_mono\n: {TSRoots_instance.sort_mixed_mono_final(x_critical, func_x_critical, dfunc_x_critical, d2func_x_critical)}")

    # Test TSRoots.root_combinations()
    multi_roots, func_multi_roots, _, _, _ = TSRoots_instance.multi_func_roots_cheb(lbS, ubS)
    print(f"root_combinations:\n {TSRoots_instance.root_combinations(multi_roots, func_multi_roots)}")

    x_critical_mono, x_critical_mixed, func_x_critical_mono, func_x_critical_mixed, N_zero, N_one, S_zero, S_one = \
        TSRoots_instance.sort_mixed_mono_final(x_critical, func_x_critical, dfunc_x_critical, d2func_x_critical)

    # Test TSRoots.ordering_summax_mixed()
    print(f"ordering_summax_mixed\n: "
          f"{TSRoots_instance.ordering_summax_mixed(x_critical_mixed, func_x_critical_mixed, func_multi_roots, 1)}")

    # Test TSRoots.xnew_TSroots()
    print(f"xnew_TSroots:\n {TSRoots_instance.xnew_TSroots()}")

    # ------------------------------------------------------
    # See TSRoots extension for BO in 1D_xSinx_function.py
    # ------------------------------------------------------

