import numpy as np
import numpy.typing as npt
import scipy.optimize as sciopt
import torch
import jax.numpy as jnp
import jax
import nlopt
import cvxpy as cp
from utils import policy_iteration
from typing import NamedTuple, List
from characteristic_time import CharacteristicTime, compute_generative_characteristic_time
from frank_wolfe import frank_wolfe
from cvxpy.constraints.constraint import Constraint

def compute_allocation(
    discount_factor: float,
    P: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
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
        return [cp.sum(x) == 1, x>=0, x<=1, cp.sum(w, axis=1) == P.reshape(ns*na, ns).T  @ x]

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
    ns, na = 4, 3
    np.random.seed(0)
    P = np.random.dirichlet(np.ones(ns), size=(ns, na))
    R = np.random.dirichlet(np.ones(ns), size=(ns, na))
    
    discount_factor = 0.99
    
    allocation = compute_generative_characteristic_time(discount_factor, P, R)
    print(allocation.omega / allocation.omega.sum(1)[:, np.newaxis])
    print(allocation.U)

    allocation = compute_allocation(discount_factor, P, R)
    print(allocation.omega / allocation.omega.sum(1)[:, np.newaxis])
    print(allocation.U)
    
    