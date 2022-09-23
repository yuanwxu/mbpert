# Leave-one-group-out: for microbiome time-series data with multiple groups, train with all but the 
# left-out group

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from mbpert.main import MBP
from mbpert.loss import reg_loss_interaction, reg_loss_r, reg_loss_eps_mat
from mbpert.mbpertTS import MBPertTSDataset, MBPertTS
from mbpert.plot import plot_loss_folds, plot_traj_groups

if __name__ == '__main__':
    # Read data
    # --- X: Species trajectory across groups of shape (n_species, n_samples) where
    #     n_samples = sum_g (n_g), n_g is the number of time points in group g
    # --- P: Time dependent perturbation matrix common to all groups, specifying
    #     which perturbation is applied to which day in {0,1,...,T}
    # --- Metadata with columns group id and measurement times corresponding to
    #     the samples in the column of X

    # Load C.diff infection experiment data from MDSINE paper
    X = np.loadtxt("data/ts/mdsine/X.txt", dtype=np.float32)
    P = np.loadtxt("data/ts/mdsine/P.txt", dtype=bool)
    meta = np.loadtxt("data/ts/mdsine/meta.txt", dtype = np.float32)

    n_groups = int(max(meta[:,0]))

    def loss_fn_ts(y_hat, y, mbpertTS):
        # Compute loss (MSE + reg)
        criterion = torch.nn.MSELoss()
        loss = criterion(y_hat, y)
        loss = loss + reg_loss_interaction(mbpertTS.A) + \
                        reg_loss_r(mbpertTS.r) + \
                        reg_loss_eps_mat(mbpertTS.eps)
        return loss

    loss_folds = [] # to store loss over epochs across all folds
    val_groups = [] # to store prediction on the hold-out groups/folds
    for i in range(1, n_groups + 1):
        logivec_i = meta[:,0].astype(int) == i
        X_train_foldi = X[:, ~logivec_i]
        meta_train_foldi = meta[~logivec_i, :]
        X_test_foldi = X[:, logivec_i]
        meta_test_foldi = meta[logivec_i, :]

        # Build Dataset and Dataloader for fold/group i
        trainset = MBPertTSDataset(X_train_foldi, P, meta_train_foldi)
        testset = MBPertTSDataset(X_test_foldi, P, meta_test_foldi)
        trainloader = DataLoader(trainset, batch_size=4, shuffle=False)
        testloader = DataLoader(testset, batch_size=4, shuffle=False)

        # Model configuration (loss function defined once outside the loop)
        torch.manual_seed(i + 1)
        mbpertTS = MBPertTS(trainset.n_species, trainset.P)

        optimizer = torch.optim.Adam(mbpertTS.parameters())

        # Model training
        mbp = MBP(mbpertTS, loss_fn_ts, optimizer, ts_mode=True)
        mbp.set_loaders(trainloader, testloader)

        print(f'Training Fold {i} (leavinig group {i} out)...')
        mbp.train(n_epochs=100, verbose=True) 
        print(f'Done Fold {i}.\n')

        # Record loss and prediction on the hold-out group i
        loss_df = pd.DataFrame({'epoch': range(mbp.total_epochs),
                                'loss_train': mbp.losses, 
                                'loss_val': mbp.val_losses})
        loss_df['fold'] = i 
        loss_folds.append(loss_df)

        val_df = mbp.predict_val_ts()
        val_df['group'] = i
        val_groups.append(val_df)

    # Combine results across folds into one data frame
    df_loss = pd.concat(loss_folds)
    df_val = pd.concat(val_groups)

    # Make plots 
    # Train and test loss across folds
    plot_loss_folds(df_loss)

    # Predicted and true dynamics for each hold-out group
    plot_traj_groups(df_val)


    







