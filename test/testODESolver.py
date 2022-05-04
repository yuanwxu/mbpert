import torch
import numpy as np
from scipy import linalg
from scipy.integrate import solve_ivp
from mbpert.simhelper import get_ode_params
from mbpert.odesolver import RK45
from mbpert.mbpert import reshape_fortran, glvp

# numpy version
def glvp2(t, x, r, A, eps, p):
    """To vectorize over conditions, create a long state vector of shape (n_specis*n_conds,)
       of contiguous blocks of species abundance, where each block corresponds to a condition
    """
    x = x.reshape(p.shape, order='F')
    out = x * (r[:, np.newaxis] + A @ x + eps[:, np.newaxis] * p)
    return np.ravel(out, order='F')

if __name__ == '__main__':
    x0 = reshape_fortran(0.2 * torch.ones((3,3)), (-1,)) # 3 species (rows), 3 conditions (columns)
    p = torch.eye(3) # 3 single-species perturbations

    r, A, eps, X_ss = get_ode_params(3, p.numpy(), seed=0)
    solver = RK45(glvp, [0,20], args=(torch.from_numpy(r.astype('float32')), torch.from_numpy(A.astype('float32')), torch.from_numpy(eps.astype('float32')), p))

    x_pred = solver.solve(x0)

    print(f'My RK45 solver: {x_pred}')

    # Now use scipy's solve_ivp
    sol = solve_ivp(glvp2, [0, 20], x0.numpy(), args=(r, A, eps, p.numpy()))

    print(f'scipy solve_ivp: {sol.y[:,-1]}') # should be similar to x_pred