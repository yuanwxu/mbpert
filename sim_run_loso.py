# Leave-one-species-out: 10 speices, all conditions involving one species left out
# the rest will be used for training

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from mbpert.main import MBP
from mbpert.loss import reg_loss_interaction, reg_loss_r, reg_loss_eps
from mbpert.mbpert import MBPertDataset, MBPert
from mbpert.simhelper import pert_mat, get_ode_params
from mbpert.plot import plot_loss_folds, plot_ss_folds

if __name__ == '__main__':
    # Data generation
    n_species = 10
    P = pert_mat(n_species, list(range(1, n_species+1)))
    r, A, eps, X_ss = get_ode_params(n_species, P, seed=0)
    n_conds = P.shape[1]
    X_0 = 0.1 * np.ones((n_species, n_conds))  # initial state chosen arbitrary

    # Define loss function (MSE + reg)
    def loss_fn(y_hat, y, mbpert):
        criterion = torch.nn.MSELoss()
        loss = criterion(y_hat, y)
        loss = loss + reg_loss_interaction(mbpert.A) + \
                      reg_loss_r(mbpert.r) + \
                      reg_loss_eps(mbpert.eps)
        return loss

    loss_folds = [] # to store loss over epochs across all folds
    x_folds = [] # to store steady state prediction on the withheld set across all folds
    for i in range(n_species):
        P_train_foldi = P[:, ~P[i]]
        P_test_foldi = P[:, P[i]]
        xss_train_foldi = np.ravel(X_ss[:, ~P[i]], order='F')
        xss_test_foldi = np.ravel(X_ss[:, P[i]], order='F')
        x0_train_foldi = np.ravel(X_0[:, ~P[i]], order='F')
        x0_test_foldi = np.ravel(X_0[:, P[i]], order='F')

        # Build Dataset and Dataloader for Fold i (i = 1, ... ,n_species)
        trainset = MBPertDataset(x0_train_foldi, xss_train_foldi, P_train_foldi)  
        testset = MBPertDataset(x0_test_foldi, xss_test_foldi, P_test_foldi)
        trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
        testloader = DataLoader(testset, batch_size=32) 

        # Model configuration (loss function defined once outside the loop)
        torch.manual_seed(i + 1)
        mbpert = MBPert(n_species)

        optimizer = torch.optim.Adam(mbpert.parameters())

        # Model training for Fold i
        mbp = MBP(mbpert, loss_fn, optimizer)
        mbp.set_loaders(trainloader, testloader)

        print(f'Training Fold {i} (leavinig species {i} out)...')
        mbp.train(n_epochs=200, verbose=True) # 500
        print(f'Done Fold {i}.\n')

        # Record loss and prediction on the left-out data for Fold i
        loss_df = pd.DataFrame({'epoch': range(mbp.total_epochs),
                                'loss_train': mbp.losses, 
                                'loss_val': mbp.val_losses})
        loss_df['fold'] = i 
        loss_folds.append(loss_df)

        x_df = mbp.predict_val()
        x_df['fold'] = i
        x_folds.append(x_df)

    # Combine results across folds into one data frame
    df_loss = pd.concat(loss_folds)
    df_ss_pred = pd.concat(x_folds)
    #df_loss.to_csv('data/loss_loso.csv', index=False)   
    #df_ss_pred.to_csv('data/ss_loso.csv', index=False)
 
    # Make plots 
    # Train and test loss across folds
    plot_loss_folds(df_loss)

    # Predicted and true steady states for each left-out test set
    plot_ss_folds(df_ss_pred)

    







