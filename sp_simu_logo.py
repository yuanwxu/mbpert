import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from mbpert.simhelper import get_ode_params
from mbpert.main import MBP
from mbpert.loss import reg_loss_interaction, reg_loss_r, reg_loss_eps_mat
from mbpert.mbpertTS import MBPertTSDataset, MBPertTS, glvp2
from mbpert.plot import plot_loss_folds, plot_ts_error, plot_traj_groups

DATA_DIR = "data/sp_simu_logo/"
OUT_DIR = "output/sp_simu_logo/"

INTEGRATE_END = 30 # define end point of numerical integration

if __name__ == '__main__':
    # Data Generation: Leave-one-group-out
    # For microbiome time-series data with multiple groups/individuals, 
    # train with all but the left-out group.
    # Simulate microbiome time series data from 5 groups: 
    # 10 species, sampling times span 180 days with an interval of 5 days
    # One type of perturbation applied at day 25, 55, 90 and 125
    # Initial states randomly sampled for each group
    n_species = 10
    n_groups = 5
    r, A, eps, _ = get_ode_params(n_species, perts=1, seed=0)
    T = 180
    P = np.zeros((T+1, 1))
    P[np.array([25, 55, 90, 125])] = 1
    samp_days = np.arange(0, T+1, 5)
    t = INTEGRATE_END * samp_days / T
    # Metadata with two columns: group id and measurement time in days, 
    gids = np.repeat(range(1, n_groups+1), len(samp_days))
    meta = np.column_stack((gids, np.tile(samp_days, n_groups)))   

    # Simulate trajectory for each group
    rng = np.random.default_rng(123)    
    X = []
    for i in range(n_groups):
        x0 = rng.lognormal(size=n_species)
        sol = solve_ivp(glvp2, [0, INTEGRATE_END], x0, args = (r, A, eps, P, T), dense_output=True)
        z = sol.sol(t)
        X.append(z)
    X = np.concatenate(X, axis=1)

    # Save exact parameters and simulated trajectories
    np.savetxt(DATA_DIR + "r.txt", r)
    np.savetxt(DATA_DIR + "A.txt", A)
    np.savetxt(DATA_DIR + "eps.txt", eps)

    # X --- Species trajectory across groups of shape (n_species, n_samples) 
    # where n_samples = sum_g (n_g), n_g is the number of time points in group g
    # P ---Time dependent perturbation matrix common to all groups, specifying
    # which perturbation is applied to which day in {0,1,...,T}
    np.savetxt(DATA_DIR + "X.txt", X)
    np.savetxt(DATA_DIR + "meta.txt", meta)
    np.savetxt(DATA_DIR + "P.txt", P, fmt='%i')

    # Number of epochs
    N_EPOCHS = 100
    
    loss_all_folds = [] # to store loss over epochs across all folds
    val_pred_all_folds = [] # to store prediction on the hold-out groups/folds
    for i in range(1, n_groups + 1):
        print(f'Training fold {i} (leavinig group {i} out)...')
        logivec_i = meta[:,0].astype(int) == i
        X_train_foldi = X[:, ~logivec_i]
        meta_train_foldi = meta[~logivec_i, :]
        X_test_foldi = X[:, logivec_i]
        meta_test_foldi = meta[logivec_i, :]

        # Build Dataset and Dataloader for fold/group i
        trainset = MBPertTSDataset(X_train_foldi, P, meta_train_foldi)
        testset = MBPertTSDataset(X_test_foldi, P, meta_test_foldi)
        trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
        testloader = DataLoader(testset, batch_size=16, shuffle=False)

        # Model configuration 
        mbpertTS = MBPertTS(n_species, trainset.P)

        def loss_fn_ts(y_hat, y, mbpertTS):
            # Compute loss (MSE + reg)
            criterion = torch.nn.MSELoss()
            loss = criterion(y_hat, y)
            loss = loss + reg_loss_interaction(mbpertTS.A) + \
                            reg_loss_r(mbpertTS.r) + \
                            reg_loss_eps_mat(mbpertTS.eps)
            return loss

        optimizer = torch.optim.Adam(mbpertTS.parameters())

        # Model training
        mbp = MBP(mbpertTS, loss_fn_ts, optimizer, ts_mode=True)
        mbp.set_loaders(trainloader, testloader)
        mbp.train(n_epochs=N_EPOCHS, verbose=False, seed=i*51) 

        # Record loss and prediction on the hold-out group i
        loss_df = pd.DataFrame({'epoch': range(mbp.total_epochs),
                                'loss_train': mbp.losses, 
                                'loss_val': mbp.val_losses})
        loss_df['leftout_group'] = i 
        loss_all_folds.append(loss_df)

        vp = mbp.predict_val_ts()
        vp['leftout_group'] = i
        val_pred_all_folds.append(vp)

    # Combine results across folds into one data frame
    df_loss = pd.concat(loss_all_folds)
    val_pred = pd.concat(val_pred_all_folds)

    # Save results
    df_loss.to_csv(OUT_DIR + "loss.csv")
    val_pred.to_csv(OUT_DIR + "val_pred.csv")

    # Plot train and validation loss across folds
    plot_loss_folds(df_loss, facet_by="leftout_group", file_out=OUT_DIR + "loss.png")

    # Plot predicted and true states for each hold-out group, merging all time points
    plot_ts_error(val_pred, facet_by="leftout_group", file_out=OUT_DIR + "ts_error_logo.png")

    # Predicted and true species dynamics for each hold-out group
    # plot_traj_groups(val_pred)




