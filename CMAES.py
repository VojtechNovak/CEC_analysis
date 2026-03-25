import cma
import numpy as np
from typing import Callable, Optional, Tuple
from qiskit_algorithms.optimizers import OptimizerResult

class CMAESOptimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer for continuous optimization problems.

    The CMA-ES algorithm is a state-of-the-art method for solving non-linear, non-convex optimization problems, 
    particularly useful for complex, high-dimensional, and noisy objective functions.

    References:
    - Hansen, N. (2006). The CMA Evolution Strategy: A Comparing Review. In J.A. Lozano, P. Larranaga, 
      I. Inza, & E. Bengoetxea (Eds.), Towards a New Evolutionary Computation (pp. 75-102). Springer. 
      https://doi.org/10.1007/3-540-32494-1_4
    - Hansen, N., Akimoto, Y., & Baudis, P. (2019). CMA-ES/pycma on Github. Zenodo. 
      DOI:10.5281/zenodo.2559634
    - The `cma` Python package: https://github.com/CMA-ES/pycma
    """

    def __init__(
        self,
        sigma0: float = 0.5,
        maxiter: int = 17500,
        popsize: int = 50,
        tolx: float = 1e-6,
        tolfun: float = 1e-6,
        verbose: bool = True,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None
    ):
        """
        Initialize the CMA-ES optimizer with specific parameters.

        :param sigma0: Initial step-size for the CMA-ES algorithm.
        :param maxiter: Maximum number of iterations for the optimization process.
        :param popsize: Population size for the CMA-ES algorithm.
        :param tolx: Tolerance for termination based on changes in the solution vector.
        :param tolfun: Tolerance for termination based on changes in the function value.
        :param verbose: Flag to control verbosity of the optimization process.
        :param bounds: A tuple of two arrays specifying the lower and upper bounds for each parameter.
        :param callback: Optional callback function to track optimization progress.
        """
        self.sigma0 = sigma0
        self.maxiter = maxiter
        self.popsize = popsize
        self.tolx = tolx
        self.tolfun = tolfun
        self.verbose = verbose
        self.bounds = bounds
        self.callback = callback  # Save the callback function

    def minimize(self, fun: Callable, x0: np.ndarray, jac=None, bounds=None, options=None) -> OptimizerResult:
        """
        Minimize an objective function using the CMA-ES algorithm.

        :param fun: The objective function to minimize.
        :param x0: Initial guess for the solution.

        :return: OptimizerResult object containing the results of the optimization.
        """
        cma_options = {
            'maxiter': self.maxiter,
            'popsize': self.popsize,
            'tolx': self.tolx,
            'tolfun': self.tolfun,
            'verb_disp': 1 if self.verbose else 0,
            'bounds': self.bounds,
        }

        if options is not None:
            cma_options.update(options)

        # Initialize function evaluation counter
        self._num_function_evaluations = 0

        # Wrapped function to include the callback
        def wrapped_fun(x):
            value = fun(x)
            self._num_function_evaluations += 1  # Increment function evaluation count
            
            # Call the callback function if it's defined
            if self.callback:
                self.callback(self._num_function_evaluations, x, value, 0.0)
            
            return value

        # Perform the optimization using CMA-ES
        result = cma.fmin(wrapped_fun, x0, self.sigma0, cma_options)

        best_x = result[0]
        best_y = result[1]
        func_evals = result[4]

        opt_result = OptimizerResult()
        opt_result.x = best_x
        opt_result.fun = best_y
        opt_result.nfev = func_evals
        opt_result.message = 'Optimization finished'

        return opt_result
