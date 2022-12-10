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
        
        
        
    
    
    