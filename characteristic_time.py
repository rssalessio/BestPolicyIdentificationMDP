import numpy as np
import numpy.typing as npt
import torch
import cvxpy as cp
import math
from utils import policy_iteration
from typing import NamedTuple, List
from frank_wolfe import frank_wolfe
from cvxpy.constraints.constraint import Constraint


class CharacteristicTime(NamedTuple):
    """CharacteristicTime results

    Check the variables in Thm.1 in
    http://proceedings.mlr.press/v139/marjani21a/marjani21a.pdf

    Args:
        T1 (npt.NDArray[np.float64]): Matrix T1
        T2 (npt.NDArray[np.float64]): Matrix T2
        T3 (float): Value T3
        T4 (float): Value T4
        H (npt.NDArray[np.float64]): Matrix H
        Hstar (float): Value Hstar
        omega (npt.NDArray[np.float64]): optimal allocation
        U (float): Opper bound on sample complexity factor
    """    
    T1: npt.NDArray[np.float64]
    T2: npt.NDArray[np.float64]
    T3: float
    T4: float
    H: npt.NDArray[np.float64]
    Hstar: float
    omega: npt.NDArray[np.float64]
    U: float

def compute_generative_characteristic_time(
    discount_factor: float,
    P: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
    abs_tol: float = 1e-6) -> CharacteristicTime:
    """
    Computes the characteristic time and the optimal allocation $omega*$ according
    to Thm. 1 in http://proceedings.mlr.press/v139/marjani21a/marjani21a.pdf
    for generative models

    Args:
        discount_factor (float): discount factor
        P (npt.NDArray[np.float64]): Transition function of shape SxAxS
        R (npt.NDArray[np.float64]): Reward function of shape SxAxS
        abs_tol (float, optional): absolute tolerance. Defaults to 1e-6.

    Returns:
        Allocation: an object of type Allocation that contains the results
    """   

    ns, na = P.shape[:2]
    Rmax, Rmin = np.max(R), np.min(R)
    
    # Normalize rewards
    if Rmax > 1 or Rmin < 0:
        R = (R - Rmin) / (Rmax - Rmin)

    # Policy iteration
    V, pi, Q = policy_iteration(discount_factor, P, R, atol=abs_tol)
    idxs_subopt_actions = np.array([[False if pi[s] == a else True for a in range(na)] for s in range(ns)])

    # Compute Delta
    delta_sq = (V[:, np.newaxis] - Q) ** 2
    delta_sq_subopt = delta_sq[idxs_subopt_actions]
    delta_sq_min = max(1e-16, delta_sq_subopt.min())
    
    # Compute variance of V, VarMax and Span
    avg_V = P @ V
    span_V = np.max(V) - np.min(V)
    var_V =  P @ (V ** 2) - (avg_V) ** 2
    var_max_V = np.max(var_V[~idxs_subopt_actions])
    
    #span_V = np.maximum(np.max(V) - avg_V.mean(), avg_V.mean() - np.min(V))

    # Compute T terms
    T1 = np.zeros((ns, na))
    T2_1 = np.zeros_like(T1)
    T2_2 = np.zeros_like(T1)
    T1[idxs_subopt_actions] = 2 / delta_sq_subopt
    T2_1[idxs_subopt_actions] = 16 * var_V[idxs_subopt_actions] / delta_sq_min
    T2_2[idxs_subopt_actions] = 6 * span_V ** (4/3) / delta_sq_subopt ** 2/3
    T2 = np.maximum(T2_1, T2_2)
    
    T3 = 2 / (delta_sq_min * ((1 -  discount_factor) ** 2))
    
    T4 = min(
        27 / (delta_sq_min * (1 -  discount_factor) ** 3),
        max(
            16 * var_max_V /  (delta_sq_min * (1 - discount_factor)**2),
            6 * (span_V / discount_factor) ** (4/3)  * (1/delta_sq_min ** 2/3)
        )
    )
    
    # Compute H and Hstar
    H = T1 + T2
    Hstar = ns * (T3 + T4)

    # Compute allocation vector
    omega = np.copy(H)
    for s in range(ns):
        _temp =  (H[s].sum() - H[s,pi[s]])
        omega[s, pi[s]] = np.sqrt(Hstar * _temp) / ns

    omega = omega / omega.sum()
    
    # In eq (10) in http://proceedings.mlr.press/v139/marjani21a/marjani21a.pdf
    # the authors claim that is 2*(sum(H) + Hstar), however, from results it seems like
    # it's just (sum(H) + Hstar)
    U = (np.sum(H) + Hstar)

    return CharacteristicTime(T1, T2, T3, T4, H, Hstar, omega, U)

def compute_characteristic_time_fw(
    discount_factor: float,
    P: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
    with_navigation_constraints: bool = False,
    atol: float = 1e-6) -> CharacteristicTime:
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
    _, pi, _ = policy_iteration(discount_factor, P, R, atol=atol)
    
    x0 = np.ones((ns * na)) / (ns * na)
    gen_allocation = compute_generative_characteristic_time(discount_factor, P, R, atol)
    idxs = torch.tensor([[False if pi[s] == a else True for a in range(na)] for s in range(ns)]).bool()
    idxs_pi = ~idxs
    H = torch.tensor(gen_allocation.H)[idxs]

    def objective_function(x: torch.Tensor):
        w = torch.reshape(x, (ns, na))
        objective  = torch.max(H/w[idxs] + torch.max(gen_allocation.Hstar/ (ns * w[idxs_pi])))
        objective = 1/objective
        return -objective

    def build_constraints(x: cp.Variable) -> List[Constraint]:
        w = cp.reshape(x, (ns, na))
        constraints = [cp.sum(x) == 1, x>=0, x<=1]
        constraints.extend(
            [] if with_navigation_constraints is False else [cp.sum(w, axis=1) == P.reshape(ns*na, ns).T  @ x] )
        return constraints

    def derivative_obj_fn(x: npt.NDArray[np.float64]):
        x = torch.tensor(x, requires_grad=True, dtype=torch.double)
        obj = objective_function(x)
        obj.backward()
        return x.grad.detach().numpy()

    
    x, res, k = frank_wolfe(ns * na, x0=x0, jac=derivative_obj_fn, build_constraints=build_constraints)
    
    print(f'Stopped at iteration {k}')
    return CharacteristicTime(
        gen_allocation.T1, gen_allocation.T2, gen_allocation.T3, gen_allocation.T4, gen_allocation.H, gen_allocation.Hstar,
        x.reshape(ns,na), -1/res)



if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    ns, na = 5,3
    np.random.seed(2)
    P = np.random.dirichlet([0.9, 0.5, 0.3, 0.1, 0.05], size=(ns, na))
    R = np.random.dirichlet([0.9, 0.5, 0.3, 0.1, 0.05], size=(ns, na))
    
    discount_factor = 0.99
    allocation = compute_generative_characteristic_time(discount_factor, P, R)
    print(allocation.omega)
    print(allocation.U)
    
    allocation = compute_characteristic_time_fw(discount_factor, P, R)
    print(allocation.omega)
    print(allocation.U)
    
    allocation = compute_characteristic_time_fw(discount_factor, P, R, with_navigation_constraints=True)
    print(allocation.omega)
    print(allocation.U)