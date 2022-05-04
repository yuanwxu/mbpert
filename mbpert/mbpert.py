# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from mbpert.odesolver import RK45


# A helper to reshape a tensor in Fortran-like order
# Reference: https://stackoverflow.com/questions/63960352/reshaping-order-in-pytorch-fortran-like-index-ordering
def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

def glvp(t, x, r, A, eps, p):
    """Define generalized lotka-volterra dynamic system with perturbations
       To vectorized over conditions, create a long state vector holding 
       species abundances at time t across all conditions

       x --- (n_species*n_conds,) Species (dimensionless) absolute abundances 
                                  under all pert conditions. Formed by contiguous
                                  blocks of species abundances, where each block
                                  corresponds to a condition
       r --- (n_species) Growth rate
       A --- (n_species, n_species) Species interaction matrix
       eps --- (n_species,) Species susceptibility to perturbation
    """
    x = reshape_fortran(x, p.shape)
    out = x * (r[:, None] + A @ x + eps[:, None] * p)
    return reshape_fortran(out, (-1,))


# Custom PyTorch module

class MBPert(nn.Module):
  def __init__(self, n_species):
    """U (torch.Tensor): perturbation matrix of shape (n_species, n_conds)"""
    super().__init__()
    self.r = nn.Parameter(torch.rand((n_species, )))
    self.eps = nn.Parameter(torch.randn(n_species, ))

    # Proper initialization of interaction matrix for stability
    self.A = 1 / (2 * n_species**(0.5)) * torch.randn(n_species, n_species)
    self.A = nn.Parameter(self.A.fill_diagonal_(-1))
    # mask = ~torch.eye(n_species, dtype=torch.bool)
    # self.A = -torch.eye(n_species) # making diag elements -1
    # self.A[mask] = 1 / (2 * n_species**(0.5)) * torch.randn(n_species**2 - n_species, requires_grad=True)
    # self.A = nn.Parameter(self.A)

  def forward(self, x, p):
    self.solver = RK45(glvp, [0,20], args=(self.r, self.A, self.eps, p))
    return self.solver.solve(x)

# Custom Dataset

# Custome Dataset to handle each data unit. Here each data unit corresponds to
# one perturbation condition. Input initial and steady-states are vectors of 
# length n_species * n_conds, of `n_conds` blocks with `n_species` elements per 
# block.
class MBPertDataset(Dataset):
  def __init__(self, x0_file, xss_file, p_file, transform=None, target_transform=None):
    self.x0 = np.loadtxt(x0_file, dtype=np.float32)
    self.xss = np.loadtxt(xss_file, dtype=np.float32)

    self.p = np.loadtxt(p_file, dtype=bool)
    self.n_species = self.p.shape[0]
    self.n_conds = self.p.shape[1]
    
    self.transform = transform
    self.target_transform = target_transform

    if len(self.x0) != len(self.xss) or len(self.x0) != self.p.size:
      raise ValueError("Incorrect input data size.")

  def __len__(self):
    return self.n_conds

  def __getitem__(self, idx):
    unit_idx = np.s_[(idx * self.n_species):(idx * self.n_species + self.n_species)]
    du_x0 = torch.from_numpy(self.x0[unit_idx]) 
    du_xss = torch.from_numpy(self.xss[unit_idx])
    du_p = torch.from_numpy(self.p[:,idx])

    if self.transform:
      du_x0 = self.transform(du_x0)
    if self.target_transform:
      du_xss = self.target_transform(du_xss)

    return (du_x0, du_p), du_xss











