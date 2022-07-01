import numpy as np
import math
import itertools as it
from scipy import linalg
from sklearn.model_selection import train_test_split

# Generate perturbation matrix of the form (nodes, conditions)
def pert_mat(n_nodes, combos, n_conds=None):
    """Generate perturbation matrix of the form (nodes, conditions)

    Args:
        n_nodes (int): number of nodes
        combos (list): perturbation acting on combination of k nodes, 
            e.g. [1,2] specifies that perturbation is to be applied to single node and node pairs
        n_conds (list): number of conditions to sample per combination, of same length as `combos`, default (NONE)
            is to use all combinations
    """
    if n_conds and len(combos) != len(n_conds):
        raise ValueError(
            "Argument combos and n_conds must have same length, see help(pert_mat)."
        )

    from math import comb

    def build_mat(list_conds_gen):
        ncols = sum(n_conds) if n_conds else sum(
            comb(n_nodes, k) for k in combos)
        pmat = np.zeros((n_nodes, ncols), dtype=bool)
        for j, cond in enumerate(it.chain(*list_conds_gen)):
            pmat[list(cond), j] = 1
        return pmat

    if n_conds:  # random sample n_conds[i] perturbation conditions for each node size k = combos[i]
        lst_conds = []
        for i, k in enumerate(combos):
            n_total_conds = comb(n_nodes, k)

            rng = np.random.default_rng(0 + i)
            selector = np.zeros(n_total_conds, dtype=np.intp)
            idx = rng.choice(n_total_conds,
                             size=min(n_conds[i], n_total_conds),
                             replace=False)
            selector[idx] = 1

            conds = it.compress(it.combinations(range(n_nodes), k), selector)
            lst_conds.append(conds)

        out = build_mat(lst_conds)
    else:  # all combinations for each perturbation node set whose size is specified in combos
        lst_conds = [it.combinations(range(n_nodes), k) for k in combos]
        out = build_mat(lst_conds)

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
    if p and perts > 0:
        raise ValueError("Provide one of pert matrix (n_species, n_conds) or "
                         "number of perturbations for time sereis data, but not both.")
    if p and n_species != p.shape[0]:
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

        if p:
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


def mbpert_writer(spliter_output, ode_params=None):
    """Data writer: write output of the split helper 
    
    Args:
        ode_params: if not None, should be a dictionary of ODE params
                    to save.
    """
    x0_train, x0_test, x_ss_train, x_ss_test, P_train, P_test = split_outputs

    np.savetxt("x0_train.txt", x0_train)
    np.savetxt("x0_test.txt", x0_test)
    np.savetxt("x_ss_train.txt", x_ss_train)
    np.savetxt("x_ss_test.txt", x_ss_test)
    np.savetxt("P_train.txt", P_train, fmt='%i')
    np.savetxt("P_test.txt", P_test, fmt='%i')

    if ode_params:
        np.savetxt("A.txt", ode_params['A'])
        np.savetxt("r.txt", ode_params['r'])
        np.savetxt("eps", ode_params['eps'])