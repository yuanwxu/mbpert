import torch


def reg_loss_interaction(A, reg_lambda = 0.001, order=2):
  """ Regularization loss for the off-diag elements of interaction matrix A """
  mask = ~torch.eye(A.shape[0], dtype=torch.bool)
  return reg_lambda * torch.linalg.norm(A[mask], order)

def reg_loss_r(r, reg_lambda = 0.001, order=2):
  """ Regularization loss for the growth rate r """
  return reg_lambda * torch.linalg.norm(r, order)

def reg_loss_eps(eps, reg_lambda = 0.001, order=2):
  """ Regularization loss for the susceptibility eps """
  return reg_lambda * torch.linalg.norm(eps, order)

def reg_loss_eps_mat(eps, reg_lambda=0.001, order='fro'):
  """ Regularization loss for the susceptibility matrix eps """
  return reg_lambda * torch.linalg.norm(eps, order)