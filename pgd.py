import cvxpy as cp
import numpy.typing as npt
import numpy as np
from cvxpy.constraints.constraint import Constraint
from typing import List, Callable, Optional, Tuple


def pgd(
        n: int,
        x0: npt.NDArray[np.float64],
        eval_fn: Callable[[npt.NDArray[np.float64]], float],        
        jac: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        build_constraints: Callable[[cp.Variable], List[Constraint]],
        lr: float,
        max_iter: int = 1000,
        rtol: float = 1.e-5,
        callback: Optional[Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64], int ], bool]] = None,
        solver: Optional[str] = None,
        verbose: bool = False
        ) -> Tuple[npt.NDArray[np.float64], float, int]:
    """Run the Projected gradient descent method
    
    See also https://angms.science/doc/CVX/CVX_PGD.pdf

    Args:
        n (int): number of parameters
        x0 (npt.NDArray[np.float64]): initial point
        jac (Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]): Callback to compute the gradient. The callback should be
            of the form fun(x) -> List[float]
        build_constraints (Callable[[cp.Variable], List[Constraint]]): Callback to build the constraints. The callback
            should accept an input that is the CVXPY variable to optimize, and should return a list of CVXPY constraints
        lr (float): learning rate
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
    # Construct the projection
    x = cp.Variable(n)
    y = cp.Parameter(n)
    objective = 0.5 * (cp.norm(x - y, p = 2) ** 2)
    
    constraints = build_constraints(x)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    assert problem.is_dcp(dpp=True), 'Problem is not convex!'

    # Initialize iterations
    next_x = x0
    prev_res = None
    
    # Golden ratio to decrease the learning rate
    golden = (1 + 5 ** 0.5) / 2
    for k in range(max_iter):
        current_x = next_x
        
        # Compute gradient
        grad = jac(current_x)
        

        # Solve problem
        MAX_NUMBER_INNER_ITERATIONS = max_iter // 10
        for _i in range(MAX_NUMBER_INNER_ITERATIONS):
            x.value = current_x
            y.value = current_x - lr * grad#np.clip(grad, -1e15, 1e15)
            try:
                res = problem.solve(warm_start=True, verbose=verbose, solver=solver)

                if prev_res is not None and res > prev_res:                    
                    lr = lr / golden
                else: break
            except Exception as e:
                lr = lr / golden

        if _i == MAX_NUMBER_INNER_ITERATIONS-1:
            break
        if prev_res is not None:
            if np.isclose(np.abs(res-prev_res), 0, atol=0, rtol=rtol):
                break

        # Compute next iterate
        prev_res = res
        next_x = x.value

        if callback is not None:
            if callback(current_x, next_x, k):
                break

        # If solution is close to the previous one, stop iterating
        if np.isclose(np.linalg.norm(next_x - current_x, ord=2), 0, rtol=rtol, atol=0):
            break

    return next_x, eval_fn(next_x), k
        
        
        
    
    
    