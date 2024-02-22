import numpy as np
import math
import itertools as it
from scipy import linalg
from sklearn.model_selection import train_test_split

# Generate perturbation matrix of the form (nodes, conditions)
def pert_mat(n_nodes, combos, n_conds_lst=None, use_seed=True):
    """Generate perturbation matrix of the form (nodes, conditions)

    Args:
        n_nodes (int): number of nodes
        combos (list): perturbation acting on combination of k nodes, 
            e.g. [1,2] specifies that perturbation is to be applied to single node and node pairs
        n_conds_lst (list): number of conditions to sample per combination, of same length as `combos`, default (NONE)
            is to use all combinations
    """
    if n_conds_lst and len(combos) != len(n_conds_lst):
        raise ValueError(
            "Argument combos and n_conds_lst must have same length, see help(pert_mat)."
        )

    from math import comb

    def build_mat(list_conds_gen):
        ncols = sum(n_conds_lst) if n_conds_lst else sum(
            comb(n_nodes, k) for k in combos)
        pmat = np.zeros((n_nodes, ncols), dtype=bool)
        for j, cond in enumerate(it.chain(*list_conds_gen)):
            pmat[list(cond), j] = 1
        return pmat

    if n_conds_lst:  # random sample n_conds_lst[i] perturbation conditions for each node size k = combos[i]
        lst_conds = []
        for i, k in enumerate(combos):
            n_total_conds = comb(n_nodes, k)

            seed = 0 + i if use_seed else None
            rng = np.random.default_rng(seed)
            selector = np.zeros(n_total_conds, dtype=np.intp)
            idx = rng.choice(n_total_conds,
                             size=min(n_conds_lst[i], n_total_conds),
                             replace=False)
            selector[idx] = 1

            conds = it.compress(it.combinations(range(n_nodes), k), selector)
            lst_conds.append(conds)

        out = build_mat(lst_conds)
    else:  # all combinations for each perturbation node set whose size is specified in combos
        lst_conds = [it.combinations(range(n_nodes), k) for k in combos]
        out = build_mat(lst_conds)

    return out


# Define GLV dynamics
def glvp2(t, x, r, A, eps, P, T, integrate_end=None):
    """Define generalized lotka-volterra dynamic system with time-dependent
        perturbations

        x --- (n_species,) Species (dimensionless) absolute abundances
        r --- (n_species,) Growth rate
        A --- (n_species, n_species) Species interaction matrix
        eps --- (n_species, K) eps_{ij}: Species i's susceptibility to perturbation j
        P --- (T+1, K) Time-dependent perturbation matrix: P_{dp} = 1 if pert p is applied at day d
        T --- duration of the observation in days
        integrate_end --- end point of numerical integration, used to scale T
    """
    if integrate_end is None:
        integrate_end = T

    out = x * (r + A @ x + eps @ P[int(T * t / integrate_end)])
    return out


def get_ode_params(n_species, p=None, perts=0, seed=None):
    """Get ODE parameters suited for simulation. 
       
    Args:
        n_species (int): Number of species
        p (nparray): Perturbation matrix of shape (n_species, n_conds) returned from pert_mat().
                    If None, will assume time series data and perts will be used.
        perts (int): kinds of perturbations in time series data, perts=1 if only a single type of 
                     perturbation was applied (maybe at multiple time points)
    
    Returns:
        If p provided:
            (growth rate r, interaction matrix A, susceptibility vector eps,
            steady state solutions across all pert conditions of shape (n_species, n_conds))
        Otherwise:
            (growth rate r, interaction matrix A, susceptibility matrix eps of shape (n_species, perts),
            steady state solutions of shape (n_species,))
    """
    if p is not None and perts > 0:
        raise ValueError("Provide one of pert matrix (n_species, n_conds) or "
                         "number of perturbations for time sereis data, but not both.")
    if p is not None and n_species != p.shape[0]:
        raise ValueError(
            "Number of species does not match first dimension of pert matrix.")

    rng = np.random.default_rng(seed)
    i = 0
    while True:
        i += 1
        # Diagonal of A: a_{ii} = -1, off-diag a_{ij} ~ N(0,1/(4n))
        A = rng.normal(0, 1 / (2 * n_species**(0.5)), (n_species, n_species))
        np.fill_diagonal(A, -1)

        # r: Unif(0, 1)
        r = rng.random((n_species, ))

        if p is not None:
            # eps: Unif(-0.2,1)
            eps = rng.uniform(-0.2, 1, (n_species, ))
            # Steady state solution across all pert conditions
            X_ss = -linalg.inv(A) @ (r[:, np.newaxis] + eps[:, np.newaxis] * p)
        else:
            eps = rng.uniform(-0.2, 1, (n_species, perts))
            X_ss = -linalg.inv(A) @ r[:, np.newaxis]

        # Check all solutions are positive
        if np.all(X_ss > 0):
            break

        if i >= 500:
            print(
                f'''Failed to find an all positive steady-state solutions across all 
                perturbation conditions after {i} attempts. Return a steady state 
                solution with negative entries.''')
            break

    return (r, A, eps, X_ss)  # namedtuple?


def add_noise(params, stdev, seed=None):
    """ Add (Gaussian) noise to the ODE parameters

    Args: 
        params (dict):  dictionary containing parameters for the ODE, keys must be "A", "r" or "eps"
        stdev (dict): dictionary containing standard deviations for each parameter in params
    
    Returns:
        Dictionary containing noisy parameters
    """

    rng = np.random.default_rng(seed)

    newparams = {}
    for key in params:
        dkey = rng.normal(size=params[key].shape, scale=stdev[key])
        newparams[key] = params[key] + dkey
        if key == "A": # not updating the diagnal
            np.fill_diagonal(newparams[key], np.diag(params[key]))
    
    # Make sure the new growth rates are all positive
    if 'r' in newparams:
        for i, ri in enumerate(newparams['r']):
            if ri < 0:
                while ri < 0:
                    ri = params['r'][i] + rng.normal(scale=stdev['r'])
                newparams['r'][i] = ri

    return newparams

    
def mbpert_split(x0, X_ss, P, **kwargs):
    """ Data split wrapper: train-test split
    
    Args:
        x0: flattened initial state vector containing contiguous blocks where
            each block corresponds to one condition
        X_ss: array of shape (n_species, n_conds), steady states
        P: array of shape (n_species, n_conds), perturbation matrix
        kwargs: keyward arguments `traiin_test_split` knows
    """
    if x0.size != X_ss.size:
        raise ValueError(
            "Incompatible shape between initial and steady states.")

    X0 = x0.reshape(X_ss.shape, order='F')
    X0_train, X0_test, X_ss_train, X_ss_test, P_train, P_test = train_test_split(
        X0.T, X_ss.T, P.T, **kwargs)

    # Flatten X0 and X_ss
    x0_train, x0_test, x_ss_train, x_ss_test = [
        np.ravel(x) for x in (X0_train, X0_test, X_ss_train, X_ss_test)
    ]

    # Transpose P_train and P_test
    P_train, P_test = P_train.T, P_test.T

    return x0_train, x0_test, x_ss_train, x_ss_test, P_train, P_test


def mbpert_split_by_cond(x0, X_ss, P, n_conds_lst, keep_as_train=2):
    """ Data split by perturbation condition, for the scenario where, say, 
        all mono-species and pairwise-species perturbations form training set, 
        higher order combinations form the test set
    
    Args:
        x0: flattened initial state vector containing contiguous blocks where
            each block corresponds to one condition
        X_ss: array of shape (n_species, n_conds), steady states
        P: array of shape (n_species, n_conds), perturbation matrix
        n_conds_lst: the same argument as in pert_mat()
        keep_as_train: n_conds_lst[:keep_as_train] will form the training set
    """
    if x0.size != X_ss.size:
        raise ValueError(
            "Incompatible shape between initial and steady states.")
    
    if sum(n_conds_lst) != P.shape[1]:
        raise ValueError(
            "Total number of pert conditions does not match P."
        )

    X0 = x0.reshape(X_ss.shape, order='F')
    split_idx = sum(n_conds_lst[:keep_as_train])
    X0_train, X0_test = X0.T[:split_idx], X0.T[split_idx:]
    X_ss_train, X_ss_test = X_ss.T[:split_idx], X_ss.T[split_idx:]
    P_train, P_test = P.T[:split_idx], P.T[split_idx:]

    # Flatten X0 and X_ss
    x0_train, x0_test, x_ss_train, x_ss_test = [
        np.ravel(x) for x in (X0_train, X0_test, X_ss_train, X_ss_test)
    ]

    # Transpose P_train and P_test
    P_train, P_test = P_train.T, P_test.T

    return x0_train, x0_test, x_ss_train, x_ss_test, P_train, P_test


def mbpert_writer(spliter_output, ode_params=None):
    """Data writer: write output of the split helper 
    
    Args:
        ode_params: if not None, should be a dictionary of ODE params
                    to save.
    """
    x0_train, x0_test, x_ss_train, x_ss_test, P_train, P_test = spliter_output

    np.savetxt("x0_train.txt", x0_train)
    np.savetxt("x0_test.txt", x0_test)
    np.savetxt("x_ss_train.txt", x_ss_train)
    np.savetxt("x_ss_test.txt", x_ss_test)
    np.savetxt("P_train.txt", P_train, fmt='%i')
    np.savetxt("P_test.txt", P_test, fmt='%i')

    if ode_params:
        np.savetxt("A.txt", ode_params['A'])
        np.savetxt("r.txt", ode_params['r'])
        np.savetxt("eps.txt", ode_params['eps'])