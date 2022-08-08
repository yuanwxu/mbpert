# -*- coding: utf-8 -*-
# Extention to handeling time series data

import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from mbpert.odesolver import RK45


# End point of integration, assume integration starts at t = 0. Used to rescale
# the actual observation time point in days. A larger value should be used if
# the observed time series are steady state abundances
INTEGRATE_END = 30 

def glvp2(t, x, r, A, eps, P, T):
    """Define generalized lotka-volterra dynamic system with time-dependent
       perturbations

       x --- (n_species,) Species (dimensionless) absolute abundances 
       r --- (n_species,) Growth rate
       A --- (n_species, n_species) Species interaction matrix
       eps --- (n_species, perts) eps_{ij}: Species i's susceptibility to perturbation j
       P --- (T+1, perts) Time-dependent perturbation matrix: P_{dp} = 1 if pert p is applied at day d 
       T --- duration of the observation in days, used to scale t
    """
    assert t <= INTEGRATE_END

    out = x * (r + A @ x + eps @ P[int(T * t / INTEGRATE_END)])
    return out


# Custom PyTorch module

class MBPertTS(nn.Module):
  def __init__(self, n_species, P):
    super().__init__()
    self.r = nn.Parameter(torch.rand((n_species, )))
    self.eps = nn.Parameter(torch.randn(n_species, P.shape[1]))

    # Proper initialization of interaction matrix for stability
    self.A = 1 / (2 * n_species**(0.5)) * torch.randn(n_species, n_species)
    self.A = nn.Parameter(self.A.fill_diagonal_(-1))
    # mask = ~torch.eye(n_species, dtype=torch.bool)
    # self.A = -torch.eye(n_species) # making diag elements -1
    # self.A[mask] = 1 / (2 * n_species**(0.5)) * torch.randn(n_species**2 - n_species, requires_grad=True)
    # self.A = nn.Parameter(self.A)

    self.P = P # time-dependent perturbation matrix
    self.T = P.shape[0] - 1 # max days of the experiment

  def forward(self, x, t):
    self.solver = RK45(glvp2, [0,t], args=(self.r, self.A, self.eps, self.P, self.T))
    return self.solver.solve(x)

# Custom Dataset

# Custome Dataset to handle each data unit. Here each data unit corresponds to
# one time slice: initial state at t = 0 and output state at t = t
class MBPertTSDataset(Dataset):
  def __init__(self, X, P, tobs, transform=None, target_transform=None):
    """X --- (n_species, n_t) Microbiome time series data X_{i} giving species i's abundance trajectory.
                              The first column is the initial state.
       P --- (T+1, perts) Time-dependent perturbation matrix: P_{dp} = 1 if pert p is applied at day d
                           where d = 0, 1, ..., T 
       tobs --- (n_t,) Actual time units (days) at which the data was observed. Should be in increasing
                       order and correspond to columns of X. The first entry should be 0, i.e. initial 
                       observation is always at day 0. 
    """
    self.X = np.loadtxt(X, dtype=np.float32) if isinstance(X, str) else X.astype(np.float32)
    self.P = np.loadtxt(P, dtype=bool) if isinstance(P, str) else P.astype(bool)

    # If there is only one column/perturbation, np.loadtxt ignores the column dimension,
    # so here make P a column vector
    if self.P.ndim == 1:
      self.P = self.P.reshape(-1, 1)
    self.P = torch.from_numpy(self.P).float()
      
    self.tobs = np.loadtxt(tobs, dtype=np.float32) if isinstance(tobs, str) else tobs.astype(np.float32)
    self.n_species = self.X.shape[0]
    self.T = self.P.shape[0] - 1

    self.transform = transform
    self.target_transform = target_transform

    if len(self.tobs) != self.X.shape[1] or len(self.tobs) < 2 or self.tobs[0] != 0:
      raise ValueError("Incorrect input data size.")

  def __len__(self):
    return len(self.tobs) - 1

  def __getitem__(self, idx):
    x0 = torch.from_numpy(self.X[:, 0])
    t = self.tobs[idx + 1] * INTEGRATE_END / self.T
    xt = torch.from_numpy(self.X[:, idx + 1])

    if self.transform:
      x0 = self.transform(x0)
    if self.target_transform:
      xt = self.target_transform(xt)

    return (x0, t), xt











