import numpy as np
import cvxpy as cp
from cvxpy.constraints.constraint import Constraint
from numpy.typing import NDArray
from typing import Optional, Tuple, List, Callable
from scipy.linalg._fblas import dger, dgemm


def policy_evaluation(
        gamma: float,
        P: NDArray[np.float64],
        R: NDArray[np.float64],
        pi: NDArray[np.int64],
        V0: Optional[NDArray[np.float64]] = None,
        atol: float = 1e-6) -> NDArray[np.float64]:
    """Policy evaluation

    Args:
        gamma (float): Discount factor
        P (NDArray[np.float64]): Transition function of shape (num_states, num_actions, num_states)
        R (NDArray[np.float64]): Reward function of shape (num_states, num_actions)
        pi (Optional[NDArray[np.int64]], optional): policy
        V0 (Optional[NDArray[np.float64]], optional): Initial value function. Defaults to None.
        atol (float): Absolute tolerance

    Returns:
        NDArray[np.float64]: Value function
    """
    
    NS, NA = P.shape[:2]
    # Initialize values
    if V0 is None:
        V0 = np.zeros(NS)
    
    V = V0.copy()
    while True:
        Delta = 0
        V_next = np.array([P[s, pi[s]] @ (R[s, pi[s]] + gamma * V) for s in range(NS)])
        Delta = np.max([Delta, np.abs(V_next - V).max()])
        V = V_next
        
        if Delta < atol:
            break
    return V
        

def policy_iteration(
        gamma: float,
        P: NDArray[np.float64],
        R: NDArray[np.float64],
        pi0: Optional[NDArray[np.int64]] = None,
        V0: Optional[NDArray[np.float64]] = None,
        atol: float = 1e-6) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Policy iteration

    Args:
        gamma (float): Discount factor
        P (NDArray[np.float64]): Transition function of shape (num_states, num_actions, num_states)
        R (NDArray[np.float64]): Reward function of shape (num_states, num_actions)
        pi0 (Optional[NDArray[np.int64]], optional): Initial policy. Defaults to None.
        V0 (Optional[NDArray[np.float64]], optional): Initial value function. Defaults to None.
        atol (float): Absolute tolerance

    Returns:
        NDArray[np.float64]: Optimal value function
        NDArray[np.float64]: Optimal policy
        NDArray[np.float64]: Optimal Q function
    """
    
    NS, NA = P.shape[:2]

    # Initialize values    
    V = V0 if V0 is not None else np.zeros(NS)
    pi = pi0 if pi0 is not None else np.random.binomial(1, p=0.5, size=(NS))
    next_pi = np.zeros_like(pi)
    policy_stable = False
    while not policy_stable:
        policy_stable = True
        V = policy_evaluation(gamma, P, R, pi, V, atol)
        Q = [[P[s,a] @ (R[s,a] + gamma * V) for a in range(NA)] for s in range(NS)]
        next_pi = np.argmax(Q, axis=1)
        
        if np.any(next_pi != pi):
            policy_stable = False
        pi = next_pi

    return V, pi, Q

def project_omega(
        x: NDArray[np.float64],
        P: NDArray[np.float64],
        tol: float=1e-2) -> NDArray[np.float64]:
    """Project omega using navigation constraints

    Parameters
    ----------
    x : NDArray[np.float64]
        Allocation vector to project
    P : NDArray[np.float64]
        Transition matrix (S,A,S)

    Returns
    -------
    NDArray[np.float64]
        The projected allocation vector
    """
    assert tol < 1 and tol >0, 'Tolerance needs to be in (0,1)'
    ns, na = P.shape[:2]
    omega = cp.Variable((ns, na), nonneg=True)
    constraints = [cp.sum(omega) == 1, omega >= tol]
    constraints.extend([cp.sum(omega[s]) == cp.sum(cp.multiply(P[:,:,s], omega)) for s in range(ns)])
    problem = cp.Problem(cp.Minimize(cp.norm(x - omega)), constraints)
    res = problem.solve(verbose=False)
    #print(f'res: {res}')
    return omega.value

def is_positive_definite(x: np.ndarray, atol: float = 1e-9) -> bool:
    """Check if a matrix is positive definite
    Args:
        x (np.ndarray): matrix
        atol (float, optional): absolute tolerance. Defaults to 1e-9.
    Returns:
        bool: Returns True if the matrix is positive definite
    """    
    return np.all(np.linalg.eigvals(x) > atol)

def is_symmetric(a: np.ndarray, rtol: float = 1e-05, atol: float = 0) -> bool:
    """Check if a matrix is symmetric
    Args:
        a (np.ndarray): matrix to check
        rtol (float, optional): relative tolerance. Defaults to 1e-05.
        atol (float, optional): absolute tolerance. Defaults to 1e-08.
    Returns:
        bool: returns True if the matrix is symmetric
    """    
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def mean_cov(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns mean and covariance of a matrix
    See https://groups.google.com/g/scipy-user/c/FpOU4pY8W2Y
    Args:
        X (np.ndarray): _description_
    Returns:
        Tuple[np.ndarray, np.ndarray]: (Mean,Covariance) tuple
    """   
    n, p = X.shape
    m = X.mean(axis=0)
    # covariance matrix with correction for rounding error
    # S = (cx'*cx - (scx'*scx/n))/(n-1)
    # Am Stat 1983, vol 37: 242-247.
    cx = X - m
    scx = cx.sum(axis=0)
    scx_op = dger(-1.0/n,scx,scx)
    S = dgemm(1.0, cx.T, cx.T, beta=1.0,
    c = scx_op, trans_a=0, trans_b=1, overwrite_c=1)
    S[:] *= 1.0/(n-1)
    return m, S.T
