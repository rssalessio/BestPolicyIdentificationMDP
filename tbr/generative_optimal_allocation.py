import numpy as np
import numpy.typing as npt
import torch
import cvxpy as cp
from utils import policy_iteration
from typing import NamedTuple, List
from frank_wolfe import frank_wolfe
from cvxpy.constraints.constraint import Constraint


class Allocation(NamedTuple):
    """Allocation results

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

def compute_allocation(
    discount_factor: float,
    P: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
    abs_tol: float = 1e-6) -> Allocation:
    """
    Computes the optimal allocation $omega*$ according to Thm. 1 in
    http://proceedings.mlr.press/v139/marjani21a/marjani21a.pdf
    

    Args:
        discount_factor (float): discount factor
        P (npt.NDArray[np.float64]): Transition function of shape SxAxS
        R (npt.NDArray[np.float64]): Reward function of shape SxAxS
        abs_tol (float, optional): absolute tolerance. Defaults to 1e-6.

    Returns:
        Allocation: an object of type Allocation that contains the results
    """   

    ns, na = P.shape[:2]
    V, pi, Q = policy_iteration(discount_factor, P, R, tol=abs_tol)

    # Compute Delta
    delta = V[:, np.newaxis] - Q
    delta_sq = delta ** 2
    idx_good_delta = ~np.isclose(delta_sq, 0)
    delta_sq_min = delta_sq[idx_good_delta].min()
    
    # Compute variance of V, VarMax and Span
    var_V = np.array([[P[s, a] @ ((V - P[s, a] @ V ) ** 2) for a in range(na)] for s in range(ns)])
    var_max_V = np.max([var_V[s,pi[s]] for s in range(ns)])
    span_V = np.max(V) - np.min(V)
    
    
    # Compute T terms
    T1 = np.zeros_like(delta)
    T1[idx_good_delta] = 2 / delta_sq[idx_good_delta]
    
    T2_1 = np.zeros_like(delta)
    T2_2 = np.zeros_like(delta)
    T2_1[idx_good_delta] = 16 * var_V[idx_good_delta] / delta_sq_min
    T2_2[idx_good_delta] = 6 * span_V ** (4/3) /  delta_sq[idx_good_delta] ** 2/3
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
    omega = np.zeros((ns, na))
    omega = H
    U = Hstar
    for s in range(ns):
        _temp =  (H[s].sum() - H[s,pi[s]])
        omega[s, pi[s]] = np.sqrt(Hstar * _temp) / ns
        U += _temp
        
    omega = omega / omega.sum()#1)[:, np.newaxis]
    U *= 2
    
    idxs = torch.tensor([[False if pi[s] == a else True for a in range(na)] for s in range(ns)]).bool()
    idxs_pi = ~idxs
    w_pi_min = np.min(omega[idxs_pi])

    wgood = omega[idxs]
    #idx_good = ~torch.isclose(wgood, zero_tensor)
    U2  = np.max(H[idxs]/(wgood)) + Hstar/ (ns * w_pi_min)
    import pdb
    pdb.set_trace()
    return Allocation(T1, T2, T3, T4, H, Hstar, omega, U)

def alternative_compute_allocation(
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
    
    gen_allocation = compute_allocation(discount_factor, P, R, abs_tol)
    idxs = torch.tensor([[False if pi[s] == a else True for a in range(na)] for s in range(ns)]).bool()
    idxs_pi = ~idxs
    H = torch.tensor(gen_allocation.H)[idxs]
    zero_tensor = torch.tensor(0., requires_grad=False, dtype=torch.double)
    def objective_function(x: torch.Tensor):
        w = torch.reshape(x, (ns, na))
        w_pi_min = torch.min(w[idxs_pi])

        wgood = w[idxs]
        #idx_good = ~torch.isclose(wgood, zero_tensor)
        objective  = torch.max(H/(wgood)) + gen_allocation.Hstar/ (ns * w_pi_min)
        # import pdb
        # pdb.set_trace()
        objective = 1/objective
        #print(objective.item())
        return -objective #* 1e-10

    def build_constraints(x: cp.Variable) -> List[Constraint]:
        w = cp.reshape(x, (ns, na))
        return [cp.sum(x) == 1, x>=0, x<=1]

    #derivative_obj_fn = jax.grad(objective_function)
    def derivative_obj_fn(x: npt.NDArray[np.float64]):
        x = torch.tensor(x, requires_grad=True, dtype=torch.double)
        obj = objective_function(x)
        obj.backward()
        return x.grad.detach().numpy()

    
    x, res = frank_wolfe(ns * na, x0=x0, jac=derivative_obj_fn, build_constraints=build_constraints)
    import pdb
    pdb.set_trace()
    print(res)
    print(np.exp(res))
    
    return Allocation(
        gen_allocation.T1, gen_allocation.T2, gen_allocation.T3, gen_allocation.T4, gen_allocation.H, gen_allocation.Hstar,
        x.reshape(ns,na), -1/res)


if __name__ == '__main__':
    ns, na = 2, 2
    np.random.seed(0)
    P = np.random.dirichlet(np.ones(ns), size=(ns, na))
    R = np.random.dirichlet(np.ones(ns), size=(ns, na))
    print(P)
    
    discount_factor = 0.99
    allocation = compute_allocation(discount_factor, P, R)
    print(allocation.omega / allocation.omega.sum(1)[:, np.newaxis])
    print(allocation.U)
    allocation =alternative_compute_allocation(discount_factor, P, R)
    print(allocation.omega / allocation.omega.sum(1)[:, np.newaxis])
    print(allocation.U)
    import pdb
    pdb.set_trace()
    #66750715.103464514
    #62345018.928198844