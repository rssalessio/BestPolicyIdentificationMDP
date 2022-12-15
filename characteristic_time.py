import numpy as np
import numpy.typing as npt
import cvxpy as cp
import jax.numpy as jnp
from jax import grad, jit
from .utils import policy_iteration
from typing import NamedTuple, List, Union, Literal, Optional, Dict
from .frank_wolfe import frank_wolfe
from cvxpy.constraints.constraint import Constraint
from .pgd import pgd
from .cem import DirichletPopulation, optimize

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
    var_V =  P @ (V ** 2) - (avg_V) ** 2
    var_max_V = np.max(var_V[~idxs_subopt_actions])
    
    span_V = np.maximum(np.max(V) - avg_V, avg_V- np.min(V))
    span_max_V = np.max(span_V[~idxs_subopt_actions])

    # Compute T terms
    T1 = np.zeros((ns, na))
    T2_1 = np.zeros_like(T1)
    T2_2 = np.zeros_like(T1)
    T1[idxs_subopt_actions] = np.nan_to_num(2 / delta_sq_subopt, nan=0, posinf=0, neginf=0)
    T2_1[idxs_subopt_actions] = np.nan_to_num(16 * var_V[idxs_subopt_actions] / delta_sq_subopt, nan=0, posinf=0, neginf=0)
    T2_2[idxs_subopt_actions] = np.nan_to_num(6 * span_V[idxs_subopt_actions] ** (4/3) / delta_sq_subopt ** 2/3, nan=0, posinf=0, neginf=0)
    T2 = np.maximum(T2_1, T2_2)
    
    T3 = np.nan_to_num(2 / (delta_sq_min * ((1 -  discount_factor) ** 2)), nan=0, posinf=0, neginf=0)
    
    T4 = np.nan_to_num(min(
        max(
            27 / (delta_sq_min * (1 -  discount_factor) ** 3),
             8 / (delta_sq_min * ((1-discount_factor) **2.5 )),
             14 * (span_max_V/((delta_sq_min ** 2/3) * ((1 - discount_factor)**(4/3))))
        ),
        max(
            16 * var_max_V /  (delta_sq_min * (1 - discount_factor)**2),
            6 * (span_max_V/ ((delta_sq_min ** 2/3) * ((1-discount_factor) ** (4/3))))
        )
    ), nan=0, posinf=0, neginf=0)
    
    # Compute H and Hstar
    H = T1 + T2
    Hstar = ns * (T3 + T4)

    # Compute allocation vector
    omega = np.copy(H)
    if np.isclose(H.sum(), 0, atol=0):
        omega = np.ones((ns, na)) / (ns * na)
    else:
        omega[~idxs_subopt_actions] = np.sqrt(H.sum() * Hstar) / ns
        omega = omega / omega.sum()
   

    # In eq (10) in http://proceedings.mlr.press/v139/marjani21a/marjani21a.pdf
    # the authors claim that is 2*(sum(H) + Hstar), however, from results it seems like
    # it's just (sum(H) + Hstar)
    _U1 = 2 * (np.sum(H) + Hstar)
    
    # This comes from the code of the original paper, even though corollary 1 has not the following form
    _U2 = (H.sum() + Hstar + 2*np.sqrt(H.sum() * Hstar) )
    
    # Exact result
    U = np.max(H[idxs_subopt_actions]/omega[idxs_subopt_actions] + np.max(Hstar/ (ns * omega[~idxs_subopt_actions])))

    return CharacteristicTime(T1, T2, T3, T4, H, Hstar, omega, U)

def compute_characteristic_time_fw(
    discount_factor: float,
    P: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
    x0: Optional[npt.NDArray[np.float64]] = None,
    with_navigation_constraints: bool = False,
    atol: float = 1e-6,
    max_iter: int = 1000,
    backend: Union[Literal['cpu'], Literal['gpu']] = 'cpu',
    use_pgd: bool = False,
    **solver_kwargs) -> CharacteristicTime:
    """
    Computes the optimal allocation $omega*$ in eq. (7) in 
    https://arxiv.org/pdf/2106.02847.pdf     

    Args:
        discount_factor (float): discount factor
        P (npt.NDArray[np.float64]): Transition function of shape SxAxS
        R (npt.NDArray[np.float64]): Reward function of shape SxAxS
        abs_tol (float, optional): absolute tolerance. Defaults to 1e-6.
        backend (str, 'cpu' or 'gpu'): which backend to use to compute the gradients. Defaults to 'cpu'.
        solver_kwargs (dict, optional): additional args to be passed to the Frank Wolfe algorithm.

    Returns:
        Allocation: an object of type Allocation that contains the results
    """
    ns, na = P.shape[:2]
    
    _, pi, _ = policy_iteration(discount_factor, P, R, atol=atol)
    
    x0 = x0 if x0 is not None else np.ones((ns * na)) / (ns * na)
    gen_allocation = compute_generative_characteristic_time(discount_factor, P, R, atol)
    idxs = jnp.array([[False if pi[s] == a else True for a in range(na)] for s in range(ns)])
    idxs_pi = ~idxs
    H = jnp.array(gen_allocation.H)[idxs]

    def objective_function(x: jnp.ndarray):
        w = jnp.reshape(x, (ns, na))
        objective  = jnp.max(H/w[idxs] + jnp.max(gen_allocation.Hstar/ (ns * w[idxs_pi])))
        #objective = 1/objective
        return jnp.log(objective + 1)

    def build_constraints(x: cp.Variable) -> List[Constraint]:
        w = cp.reshape(x, (ns, na))
        constraints = [cp.sum(x) == 1, x>=0, x<=1]
        constraints.extend(
            [] if with_navigation_constraints is False else [cp.sum(w, axis=1) == P.reshape(ns*na, ns).T  @ x] )
        return constraints

    _derivative_obj_fn = jit(grad(objective_function), backend=backend)
    derivative_obj_fn = lambda x: np.asarray(_derivative_obj_fn(x))
    eval_fn = lambda x: np.asarray(objective_function(x))

    if not use_pgd:
        x, res, k = frank_wolfe(ns * na, x0=x0, jac=derivative_obj_fn, build_constraints=build_constraints, max_iter=max_iter, **solver_kwargs)
    else:
        x, res, k = pgd(ns * na, x0=x0, eval_fn=eval_fn, jac=derivative_obj_fn, build_constraints=build_constraints, lr=1e-2, max_iter=max_iter, **solver_kwargs)
    
    x = x.reshape(ns,na)
    res = np.max(gen_allocation.H[idxs]/x[idxs] + np.max(gen_allocation.Hstar/ (ns * x[~idxs])))

    return CharacteristicTime(
        gen_allocation.T1, gen_allocation.T2, gen_allocation.T3, gen_allocation.T4, gen_allocation.H, gen_allocation.Hstar,
        x, res)


def cem_method(
    discount_factor: float,
    P: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
    atol: float = 1e-6) -> CharacteristicTime:
    ns, na = P.shape[:2]
    _, pi, _ = policy_iteration(discount_factor, P, R, atol=atol)
    
    x0 = np.ones((ns * na)) / (ns * na)
    gen_allocation = compute_generative_characteristic_time(discount_factor, P, R, atol)
    idxs = np.array([[False if pi[s] == a else True for a in range(na)] for s in range(ns)])
    idxs_pi = ~idxs
    H = np.array(gen_allocation.H)[idxs]

    def objective_function(x: np.ndarray):
        w = np.reshape(x, (ns, na))
        objective  = np.max(H/w[idxs] + np.max(gen_allocation.Hstar/ (ns * w[idxs_pi])))
        return 1/objective

    population = DirichletPopulation(np.ones(ns * na)/200)
    
    res, x, last_epoch = optimize(objective_function, population, 1000, threshold=3e-2, elite_fraction=0.1)

    return CharacteristicTime(
        gen_allocation.T1, gen_allocation.T2, gen_allocation.T3, gen_allocation.T4, gen_allocation.H, gen_allocation.Hstar,
        x.reshape(ns,na), 1/res)


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    ns, na = 5,3
    np.random.seed(2)
    P = np.random.dirichlet(np.ones(ns), size=(ns, na))
    R = np.random.dirichlet(np.ones(ns), size=(ns, na))
    
    discount_factor = 0.99
    allocation = compute_generative_characteristic_time(discount_factor, P, R)
    print(allocation.omega)
    print(allocation.U)
    
    discount_factor = 0.99
    allocation = cem_method(discount_factor, P, R)
    print(allocation.omega)
    print(allocation.U)
    
    allocation = compute_characteristic_time_fw(discount_factor, P, R, use_pgd=False, max_iter=100)
    print(allocation.omega)
    print(allocation.U)
    
    allocation = compute_characteristic_time_fw(discount_factor, P, R, use_pgd=True)
    print(allocation.omega)
    print(allocation.U)
    
    allocation = compute_characteristic_time_fw(discount_factor, P, R, with_navigation_constraints=True, use_pgd=False, max_iter=100)
    print(allocation.omega)
    print(allocation.U)
    
    allocation = compute_characteristic_time_fw(discount_factor, P, R, with_navigation_constraints=True, use_pgd=True)
    print(allocation.omega)
    print(allocation.U)