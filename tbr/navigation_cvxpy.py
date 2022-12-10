import numpy as np
import numpy.typing as npt
import scipy.optimize as sciopt
import torch
import jax.numpy as jnp
import jax
import nlopt
import cvxpy as cp
from utils import policy_iteration
from typing import NamedTuple
from generative_optimal_allocation import Allocation, compute_allocation as generative_compute_allocation



def _differential_evolution(
    discount_factor: float,
    P: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
    abs_tol: float = 1e-6) -> Allocation:
    ns, na = P.shape[:2]
    _, pi, _ = policy_iteration(discount_factor, P, R, tol=abs_tol)
    
    x0 = np.ones((ns * na)) / (ns * na)
    bounds = list(zip(np.zeros(ns * na), np.ones(ns * na)))
    
    gen_allocation = generative_compute_allocation(discount_factor, P, R, abs_tol)
    idxs = np.array([[False if pi[s] == a else True for a in range(na)] for s in range(ns)])

    def objective_function(x: npt.NDArray[np.float64]):
        w = np.reshape(x, (ns, na))
        w_pi_min = np.min([w[s, pi[s]] for s in range(ns)])
        H = gen_allocation.H[idxs]

        wgood = w[idxs]
        idx_good = np.argwhere(~np.isclose(wgood, 0))
        objective  =  wgood[idx_good]/H[idx_good]# + gen_allocation.Hstar/ (ns * w_pi_min)
        return -np.min(objective)

    def navigation_constraints(x: npt.NDArray[np.float64]):
        w = np.reshape(x, (ns, na))
        res = np.array([w[s].sum() - P[:,:,s].flatten() @ x for s in range(ns)])
        return res

    constraints = [
        sciopt.LinearConstraint(np.ones(ns * na), lb=1, ub=1),
        sciopt.NonlinearConstraint(navigation_constraints, lb=np.zeros(ns), ub=np.zeros(ns))
    ]
    
    res = sciopt.differential_evolution(objective_function, bounds=bounds, constraints=constraints, popsize=100, workers=1, x0=x0, maxiter=1000)
    print(res['fun'])
    return Allocation(gen_allocation.T1, gen_allocation.T2, gen_allocation.T3, gen_allocation.T4, gen_allocation.H, gen_allocation.Hstar, res['x'].reshape(ns,na))

    
    

def compute_allocation(
    discount_factor: float,
    P: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
    abs_tol: float = 1e-6) -> Allocation:
    """
    Computes the optimal allocation $omega*$ in eq. (7) in 
    https://arxiv.org/pdf/2106.02847.pdf     

    Args:
        discount_factor (float): discount factor
        P (npt.NDArray[np.float64]): Transition function of shape SxAxS
        R (npt.NDArray[np.float64]): Reward function of shape SxAxS
        abs_tol (float, optional): absolute tolerance. Defaults to 1e-6.

    Returns:
        Allocation: an object of type Allocation that contains the results
    """
    ns, na = P.shape[:2]
    _, pi, _ = policy_iteration(discount_factor, P, R, tol=abs_tol)
    
    x0 = np.ones((ns * na)) / (ns * na)
    bounds = list(zip(np.zeros(ns * na), np.ones(ns * na)))
    
    gen_allocation = generative_compute_allocation(discount_factor, P, R, abs_tol)
    idxs = np.array([[False if pi[s] == a else True for a in range(na)] for s in range(ns)])
    idxs_pi = ~idxs
    H = gen_allocation.H[idxs]
    
    x = cp.Variable(ns*na, pos=True)
    w = cp.reshape(x, (ns, na))
    constraints = [cp.sum(x) == 1, x>=0, x<=1, cp.sum(w, axis=1) == P.reshape(ns*na, ns).T  @ x]
  
    import pdb
    pdb.set_trace()
    objective = cp.max(gen_allocation.Hstar /(ns *w[~idxs])) + cp.max(cp.multiply(H, cp.inv_pos(w[idxs])))#+ gen_allocation.Hstar * cp.inv_pos(ns * cp.min(w[~idxs]))

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem = problem.solve(solver=cp.MOSEK, verbose=False, gp=True)

    print(problem)

    return Allocation(gen_allocation.T1, gen_allocation.T2, gen_allocation.T3, gen_allocation.T4, gen_allocation.H, gen_allocation.Hstar, x.value.reshape(ns,na))



if __name__ == '__main__':
    ns, na = 2, 2
    np.random.seed(0)
    P = np.random.dirichlet(np.ones(ns), size=(ns, na))
    R = np.random.dirichlet(np.ones(ns), size=(ns, na))
    print(P)
    
    discount_factor = 0.99
    allocation = compute_allocation(discount_factor, P, R)
    print(allocation.omega / allocation.omega.sum(1)[:, np.newaxis])
    
    allocation = generative_compute_allocation(discount_factor, P, R)
    print(allocation.omega / allocation.omega.sum(1)[:, np.newaxis])