# Simulate microbiome time series data: 
# 10 species, sampling times are at days 0---90 with an interval of 5
# One perturbation applied at day 30 and day 50
# Data points from the first 60 days used for training, those from the remaining 30 days used for testing

import os
import numpy as np
from scipy.integrate import solve_ivp
from mbpert.simhelper import get_ode_params
from mbpert.mbpertTS import glvp2

INTEGRATE_END = 30 # define end point of numerical integration

if __name__ == '__main__':
    n_species = 10
    r, A, eps, _ = get_ode_params(n_species, perts=1, seed=0)
    x0 = 0.1 * np.ones(n_species)  # initial state chosen arbitrary
    T = 90
    P = np.zeros(T+1).reshape(-1, 1)
    P[np.array([30, 50])] = 1
    tobs = np.arange(0, T+1, 5)
    t = INTEGRATE_END * tobs / T

    sol = solve_ivp(glvp2, [0, INTEGRATE_END], x0, args = (r, A, eps, P, T), dense_output=True)
    z = sol.sol(t)

    train_stop_idx = int(60/5 + 1)

    if not os.path.exists('data/ts'):
        os.makedirs('data/ts')
        
    np.savetxt("data/ts/X_train.txt", z[:,:train_stop_idx])
    np.savetxt("data/ts/tobs_train.txt", tobs[:train_stop_idx])

    # For test data need to prepend the initial state
    np.savetxt("data/ts/X_test.txt", np.concatenate((z[:,0].reshape(-1,1), 
                                                     z[:, train_stop_idx:]), axis=1))
    np.savetxt("data/ts/tobs_test.txt", np.concatenate(([0], tobs[train_stop_idx:])))

    np.savetxt("data/ts/P.txt", P, fmt='%i')

    # Save ODE parameters
    np.savetxt("data/ts/r.txt", r)
    np.savetxt("data/ts/A.txt", A)
    np.savetxt("data/ts/eps.txt", eps)
    
