import numpy as np
import numpy.typing as npt
import scipy.optimize as sciopt
import torch
import jax.numpy as jnp
import jax
import nlopt
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
        objective  =  np.max(H[idx_good]/wgood[idx_good]) + gen_allocation.Hstar/ (ns * w_pi_min)
        return -1/objective

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

def _jax_optimizer(
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

    idxs_pi = ~idxs

    def objective_function(x: npt.NDArray[np.float64]):
        w = np.reshape(x, (ns, na))
        w_pi_min = np.min(w[idxs_pi])
        H = gen_allocation.H[idxs]

        wgood = w[idxs]
        idx_good = np.argwhere(~np.isclose(wgood, 0))
        objective  = np.max(H[idx_good]/wgood[idx_good]) + gen_allocation.Hstar/ (ns * w_pi_min)
        return -1/objective

    def navigation_constraints(x: npt.NDArray[np.float64]):
        w = np.reshape(x, (ns, na))
        res = np.array([w[s].sum() - P[:,:,s].flatten() @ x for s in range(ns)])
        return res

    constraints = [
        sciopt.LinearConstraint(np.ones(ns * na), lb=1, ub=1),
        sciopt.NonlinearConstraint(navigation_constraints, lb=np.zeros(ns), ub=np.zeros(ns))
    ]
    
    res = sciopt.minimize(objective_function, x0=x0, bounds=bounds, constraints=constraints, method='trust-constr')#, popsize=200, workers=1, x0=x0, maxiter=5000)
    print(res['fun'])
    return Allocation(gen_allocation.T1, gen_allocation.T2, gen_allocation.T3, gen_allocation.T4, gen_allocation.H, gen_allocation.Hstar, res['x'].reshape(ns,na))


def _nlopt(
    discount_factor: float,
    P: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
    abs_tol: float = 1e-6) -> Allocation:
    ns, na = P.shape[:2]
    _, pi, _ = policy_iteration(discount_factor, P, R, tol=abs_tol)
    
    
    gen_allocation = generative_compute_allocation(discount_factor, P, R, abs_tol)
    idxs = np.array([[False if pi[s] == a else True for a in range(na)] for s in range(ns)])
    idxs_pi = ~idxs
    

    def objective_function(x: npt.NDArray[np.float64]):
        w = jnp.reshape(x, (ns, na))
        
        w_pi_min = jnp.min(w[idxs_pi])
        H = gen_allocation.H[idxs]

        wgood = w[idxs]
        idx_good = jnp.argwhere(~jnp.isclose(wgood, 0))
        objective  = jnp.max(H[idx_good]/wgood[idx_good]) + gen_allocation.Hstar/ (ns * w_pi_min)
        return -1/objective

    def navigation_constraints(x: npt.NDArray[np.float64]):
        w = jnp.reshape(x, (ns, na))
        res = jnp.array([w[s].sum() - P[:,:,s].flatten() @ x for s in range(ns)])
        return res
    
    def simplex_constraint(x: npt.NDArray[np.float64]):
        return jnp.sum(x) - 1

    derivative_obj_fn = jax.grad(objective_function)
    derivative_smplx_cn = jax.grad(simplex_constraint)
    derivative_nvgt_cn = jax.jacfwd(navigation_constraints)

    def _opt_callback(x: npt.NDArray[np.float64], grad: npt.NDArray[np.float64], main_fn, grad_fn):
        if grad.size > 0:
            grad[:] = grad_fn(x)
        res =  main_fn(x)
        # print(x)
        # print(res)
        return res.item()
    
    def _opt_v_callback(res: npt.NDArray[np.float64], x: npt.NDArray[np.float64], grad: npt.NDArray[np.float64], main_fn, grad_fn):
        if grad.size > 0:
            grad[:] = grad_fn(x)
        res[:] =  main_fn(x)
        # print(np.reshape(x, (ns, na)))
        # print(res)
    
    
    opt = nlopt.opt(nlopt.LD_AUGLAG, ns*na)
    opt.set_max_objective(lambda x, grad: _opt_callback(x, grad, objective_function, derivative_obj_fn))
    opt.set_lower_bounds(np.zeros(ns*na))
    opt.set_upper_bounds(np.ones(ns*na))
    opt.add_equality_constraint(lambda x, grad: _opt_callback(x, grad, simplex_constraint, derivative_smplx_cn), 1e-1)
    opt.add_equality_mconstraint(lambda res, x, grad: _opt_v_callback(res, x, grad, navigation_constraints, derivative_nvgt_cn), [1e-1] * ns)

    res = opt.optimize(np.ones((ns * na)) / (ns * na))
    minf = opt.last_optimum_value()
    print(minf)
    return Allocation(gen_allocation.T1, gen_allocation.T2, gen_allocation.T3, gen_allocation.T4, gen_allocation.H, gen_allocation.Hstar, res.reshape(ns,na))

    
    

def compute_allocation(
    discount_factor: float,
    P: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
    type: int, 
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
    if type == 0:
        return _jax_optimizer(discount_factor, P, R, abs_tol)
    elif type == 1:
        return _nlopt(discount_factor, P, R, abs_tol)
    else:
        return _differential_evolution(discount_factor, P, R, abs_tol)


if __name__ == '__main__':
    ns, na = 2, 2
    np.random.seed(0)
    P = np.random.dirichlet(np.ones(ns), size=(ns, na))
    R = np.random.dirichlet(np.ones(ns), size=(ns, na))
    print(P)
    
    discount_factor = 0.99
    allocation = compute_allocation(discount_factor, P, R, 0)
    print(allocation.omega / allocation.omega.sum(1)[:, np.newaxis])
    # allocation = compute_allocation(discount_factor, P, R, 1)
    # print(allocation.omega)
    allocation = compute_allocation(discount_factor, P, R, 2)
    print(allocation.omega / allocation.omega.sum(1)[:, np.newaxis])
    print(generative_compute_allocation(discount_factor, P, R).omega)