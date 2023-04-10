import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from mbpert.main import MBP
from mbpert.loss import reg_loss_interaction, reg_loss_r, reg_loss_eps
from mbpert.mbpert import MBPertDataset, MBPert
from mbpert.simhelper import pert_mat, get_ode_params
from mbpert.plot import plot_loss_folds, plot_ss_folds, barplot_pcorr_folds

DATA_DIR = "data/pp_simu_loso/"
OUT_DIR = "output/pp_simu_loso/"

if __name__ == '__main__':
    # Data generation
    # Leave-one-species-out: 10 speices, all conditions involving one species left out
    # the rest will be used for training
    n_species = 10
    P = pert_mat(n_species, list(range(1, n_species+1)))
    r, A, eps, X_ss = get_ode_params(n_species, P, seed=0)

    # Save the exact parameters
    np.savetxt(DATA_DIR + "r.txt", r)
    np.savetxt(DATA_DIR + "A.txt", A)
    np.savetxt(DATA_DIR + "eps.txt", eps)

    n_conds = P.shape[1]
    X_0 = 0.1 * np.ones((n_species, n_conds))  # initial state chosen arbitrary

    # Number of epochs
    N_EPOCHS = 300

    loss_all_folds = [] # to store loss over epochs across all folds
    val_pred_all_folds = [] # to store steady state prediction on the withheld set across all folds
    for i in range(n_species):
        print(f'Training model {i} (leavinig species {i} out)...')
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

        # Model configuration 
        mbpert = MBPert(n_species)

        def loss_fn(y_hat, y, mbpert):
            # Compute loss (MSE + reg)
            criterion = torch.nn.MSELoss()
            loss = criterion(y_hat, y)
            loss = loss + reg_loss_interaction(mbpert.A) + \
                        reg_loss_r(mbpert.r) + \
                        reg_loss_eps(mbpert.eps)
            return loss

        optimizer = torch.optim.Adam(mbpert.parameters())

        # Model training for Fold i
        mbp = MBP(mbpert, loss_fn, optimizer)
        mbp.set_loaders(trainloader, testloader)
        mbp.train(n_epochs=N_EPOCHS, verbose=False, seed=i*50)

        # Record loss history and prediction on the left-out species
        loss_df = pd.DataFrame({'epoch': range(mbp.total_epochs),
                                'loss_train': mbp.losses, 
                                'loss_val': mbp.val_losses})
        loss_df['leftout_species'] = i 
        loss_all_folds.append(loss_df)

        vp = mbp.predict_val()
        vp['leftout_species'] = i
        val_pred_all_folds.append(vp)

    # Combine results across folds into one data frame
    df_loss = pd.concat(loss_all_folds)
    val_pred = pd.concat(val_pred_all_folds)

    # Save results
    df_loss.to_csv(OUT_DIR + "loss.csv")
    val_pred.to_csv(OUT_DIR + "val_pred.csv")
  
    # Plot train and validation loss across folds
    plot_loss_folds(df_loss, facet_by="leftout_species", file_out=OUT_DIR + "loss.png")

    # Plot predicted and true steady states for each left-out test set
    plot_ss_folds(val_pred, file_out=OUT_DIR + "ss_error_all_folds.png")

    # Show a barplot of correlations between predicted and true steady states 
    # for each left-out test set 
    barplot_pcorr_folds(val_pred, file_out=OUT_DIR + "pcorr_all_folds.png")

    







