import cvxpy as cp
import numpy.typing as npt
import numpy as np
from cvxpy.constraints.constraint import Constraint
from typing import List, Callable, Optional, Tuple


def frank_wolfe(
        n: int,
        x0: npt.NDArray[np.float64],
        jac: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        build_constraints: Callable[[cp.Variable], List[Constraint]],
        max_iter: int = 1000,
        rtol: float = 1.e-5,
        callback: Optional[Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64], int ], bool]] = None,
        solver: Optional[str] = None,
        verbose: bool = False
        ) -> Tuple[npt.NDArray[np.float64], float, int]:
    """Run the Frank Wolfe algorithm to solve a convex constrained problem
    
    See also https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm

    Args:
        n (int): number of parameters
        x0 (npt.NDArray[np.float64]): initial point
        jac (Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]): Callback to compute the gradient. The callback should be
            of the form fun(x) -> List[float]
        build_constraints (Callable[[cp.Variable], List[Constraint]]): Callback to build the constraints. The callback
            should accept an input that is the CVXPY variable to optimize, and should return a list of CVXPY constraints
        max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
        rtol (float, optional): Relative tolerance for stopping the algorithm. Defaults to 1.e-5.
        callback (Optional[Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64], int ], bool]], optional): 
            Optional callback of the form fun(x, xnext, k) -> bool, where x is the current point, xnext is the next point
            and k is the current iteration. If the callback returns true, then the FW algorithm stops. Defaults to None.
        solver (Optional[str], optional): Specify a CVXPY solver to use. Defaults to None.
        verbose (bool, optional): enable verbosity of the solver. Defaults to False.

    Returns:
        Tuple[npt.NDArray[np.float64], float, int]: _description_
    """    
    # Construct the Frank-Wolfe parametric problem
    x = cp.Variable(n)
    grad_f = cp.Parameter(n)
    objective = x @ grad_f
    
    constraints = build_constraints(x)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    assert problem.is_dcp(dpp=True), 'Problem is not convex!'

    # Initialize iterations
    next_x = x0
    for k in range(max_iter):
        current_x = next_x
        
        # Compute gradient
        x.value = current_x
        grad_f.value = jac(current_x)
        
        # Solve problem
        res = problem.solve(warm_start=True, verbose=verbose, solver=solver)
        
        # Compute next iterate
        alpha = 2 / (k + 2)
        next_x = current_x + alpha * (x.value - current_x)

        if callback is not None:
            if callback(current_x, next_x, k):
                break
        
        # If solution is close to the previous one, stop iterating
        if np.isclose(np.linalg.norm(next_x - current_x), 0, rtol=rtol, atol=0):
            break

    return next_x, res, k
        
        
        
    
    
    