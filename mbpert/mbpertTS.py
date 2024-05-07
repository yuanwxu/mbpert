# -*- coding: utf-8 -*-
# Extention to handeling time series data

import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from mbpert.odesolver import RK45


# Custom PyTorch module

class MBPertTS(nn.Module):
  def __init__(self, n_species, P):
    super().__init__()
    self.r = nn.Parameter(torch.rand((n_species, )))

    if torch.any(P):
      self.eps = nn.Parameter(torch.randn(n_species, P.shape[1]))
    else: # no perturbations, so eps not a Parameter any more
      self.eps = torch.zeros(n_species, P.shape[1])

    # Proper initialization of interaction matrix for stability
    self.A = 1 / (2 * n_species**(0.5)) * torch.randn(n_species, n_species)
    self.A = nn.Parameter(self.A.fill_diagonal_(-1))
    # mask = ~torch.eye(n_species, dtype=torch.bool)
    # self.A = -torch.eye(n_species) # making diag elements -1
    # self.A[mask] = 1 / (2 * n_species**(0.5)) * torch.randn(n_species**2 - n_species, requires_grad=True)
    # self.A = nn.Parameter(self.A)

    self.P = P.to('cuda') if torch.cuda.is_available() else P # time-dependent perturbation matrix
    self.T = P.shape[0] - 1 # end of actual time unit (e.g. days) of the experiment, 
                            # assume the starting time is at zero

  def forward(self, x, t1, t2):
    # with profiler.record_function("RK45"):
    self.solver = RK45(MBPertTS.glvp2, [t1,t2], args=(self.r, self.A, self.eps, self.P, self.T))
    return self.solver.solve(x)

  @staticmethod
  def glvp2(t, x, r, A, eps, P, T):
    """Define generalized lotka-volterra dynamic system with time-dependent
       perturbations

       x --- (n_species,) Species (dimensionless) absolute abundances
       r --- (n_species,) Growth rate
       A --- (n_species, n_species) Species interaction matrix
       eps --- (n_species, K) eps_{ij}: Species i's susceptibility to perturbation j
       P --- (T+1, K) Time-dependent perturbation matrix: P_{dp} = 1 if pert p is applied at day d
       T --- End of actual time unit
    """
    out = x * (r + A @ x + eps @ P[int(T * t / MBPertTSDataset.INTEGRATE_END)])
    return out

# Custom Dataset

# Custome Dataset to handle each data unit. Here each data unit corresponds to
# one time slice: initial state at t = t' and output state at t = t'+1
class MBPertTSDataset(Dataset):

  INTEGRATE_END = None # End point of numerical integration

  def __init__(self, X, P, meta, scale_integration_time=False, transform=None, target_transform=None):
    """X --- (n_species, n_t) Microbiome time series data X_{i} giving species i's abundance trajectory.
                              The first column is the initial state. For multiple groups, X is a columnwise
                              concatenation of species trajectories of all groups.
       P --- (T+1, perts) Time-dependent perturbation matrix: P_{dp} = 1 if pert p is applied at time unit d
                           where d = 0, 1, ..., T
       meta --- (n_t, 2) Metadata with two columns, group id and measurement time.
                         The first column is group id in natural order (if all data are from one individual 
                         then this will be a column of 1s), the second column contains actual time units (days)
                         at which the data was observed, and should be the same size as number of columns in X. 
    """
    self.X = np.loadtxt(X, dtype=np.float32) if isinstance(X, str) else X.astype(np.float32)
    self.P = np.loadtxt(P, dtype=bool) if isinstance(P, str) else P.astype(bool)

    # If there is only one column/perturbation, np.loadtxt ignores the column dimension,
    # so here make P a column vector
    if self.P.ndim == 1:
      self.P = self.P.reshape(-1, 1)
    self.P = torch.from_numpy(self.P).float()

    self.meta = np.loadtxt(meta, dtype=np.float32) if isinstance(meta, str) else meta.astype(np.float32)
    self.tobs = self.meta[:, 1]
    self.n_species = self.X.shape[0]
    self.T = self.P.shape[0] - 1
    self.gids = self.meta[:, 0]
    self.n_groups = int(max(self.gids))

    MBPertTSDataset.INTEGRATE_END = 30 if scale_integration_time else self.T

    self.transform = transform
    self.target_transform = target_transform

    if len(self.tobs) != self.X.shape[1] or len(self.tobs) < 2:
      raise ValueError("Incorrect input data size.")

  def __len__(self):
    return len(self.tobs) - 1

  def __getitem__(self, idx):
    gid = self.gids[idx] # which group does 'idx' correspond to

    # If next idx corresponds to a different group, then we are at the end time point, 
    # no data is available for the current group, return the first data unit of next group.
    # Note that this will produce a duplicate data unit for the first data unit of each group
    # from the second to the last group. But this allows us to continue using len(self.tobs) - 1
    # instead of len(self.tobs) - self.n_groups in defining the number of instances in the Dataset, 
    # which works for single group case. 
    start = idx + 1 if self.gids[idx + 1] != gid else idx
    
    xtp = torch.from_numpy(self.X[:, start])  # state at t'
    xtpp1 = torch.from_numpy(self.X[:, start + 1]) # state at t' + 1

    t1 = self.tobs[start] * MBPertTSDataset.INTEGRATE_END / self.T # integration time at t'
    t2 = self.tobs[start + 1] * MBPertTSDataset.INTEGRATE_END / self.T # integration time at t' + 1
    
    if self.transform:
      xtp = self.transform(xtp)
    if self.target_transform:
      xtpp1 = self.target_transform(xtpp1)

    return (xtp, t1, t2), xtpp1











