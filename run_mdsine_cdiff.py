import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from mbpert.main import MBP
from mbpert.loss import reg_loss_interaction, reg_loss_r, reg_loss_eps_mat
from mbpert.mbpertTS import MBPertTSDataset, MBPertTS
from mbpert.plot import plot_loss_folds, plot_ts_error, plot_traj_groups

DATA_DIR = "data/mdsine/cdiff/"
OUT_DIR = "output/mdsine/cdiff/"


if __name__ == '__main__':

    # Load C.diff infection experiment data from MDSINE paper
    # X (n_species, n_samples) --- Species trajectory of all groups concatenated 
    # together, where n_samples = sum_g (n_g), n_g is the number of time points 
    # in group g
    X = np.loadtxt(DATA_DIR + "X.txt", dtype=np.float32)

    # P (T+1, K) --- Time dependent perturbation matrix common to all groups,
    # boolean type, specifying which perturbation is applied to which time unit
    # in {0,1,...,T}, where T is the total duration of experiment.
    # K is the number of types of perturbation, for example, if both diet and 
    # inoculation were used throughout the course of the experiment, then K=2
    # If only one kind of perturbation was used, then K=1.
    P = np.loadtxt(DATA_DIR + "P.txt", dtype=bool)

    # Meta (n_samples, 2) --- Metadata with two columns: An ID column of integers
    # specifying the group the measurement was taken from, and a Day column
    # specifying the day the measurement was taken on, assuming the start of the 
    # experiment was at Day 0. Must align with the columns of X. For example,
    # if there were 5 groups each had 10 observations, then 
    # Meta[:,0] = np.repeat([1,2,3,4,5],10)
    Meta = np.loadtxt(DATA_DIR + "meta.txt", dtype=np.float32)
    n_groups = int(max(Meta[:,0]))
    n_species = X.shape[0]

    # Number of epochs
    N_EPOCHS = 100

    # Store ODE parameter estimates
    A_est = np.empty((n_groups, n_species, n_species))
    r_est = np.empty((n_groups, n_species))
    eps_est = np.empty((n_groups, n_species, P.shape[1]))
    
    loss_all_folds = [] # to store loss over epochs across all folds
    val_pred_all_folds = [] # to store prediction on the hold-out groups/folds

    torch.manual_seed(100)
    for i in range(1, n_groups + 1):
        print(f'Training fold {i} (leavinig group {i} out)...')
        logivec_i = Meta[:,0].astype(int) == i
        X_train_foldi = X[:, ~logivec_i]
        meta_train_foldi = Meta[~logivec_i, :]
        X_test_foldi = X[:, logivec_i]
        meta_test_foldi = Meta[logivec_i, :]

        # Build Dataset and Dataloader for fold/group i
        trainset = MBPertTSDataset(X_train_foldi, P, meta_train_foldi)
        testset = MBPertTSDataset(X_test_foldi, P, meta_test_foldi)
        trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
        testloader = DataLoader(testset, batch_size=16, shuffle=False)

        # Model configuration 
        mbpertTS = MBPertTS(trainset.n_species, trainset.P)

        def loss_fn_ts(y_hat, y, mbpertTS):
            criterion = torch.nn.MSELoss()
            loss = criterion(y_hat, y)
            return loss
        
        optimizer = torch.optim.AdamW(mbpertTS.parameters(), lr=0.001, weight_decay=0.01)

        # Model training
        mbp = MBP(mbpertTS, loss_fn_ts, optimizer, ts_mode=True)
        mbp.set_loaders(trainloader, testloader)
        mbp.train(n_epochs=N_EPOCHS, verbose=True, seed=None) 

        # Record estmated ODE parameters for each fold
        A_est[i-1] = mbp.model.state_dict()['A'].numpy()
        r_est[i-1] = mbp.model.state_dict()['r'].numpy()
        eps_est[i-1] = mbp.model.state_dict()['eps'].numpy()

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
    np.save(OUT_DIR + "A_est", A_est)
    np.save(OUT_DIR + "r_est", r_est)
    np.save(OUT_DIR + "eps_est", eps_est)

    df_loss.to_csv(OUT_DIR + "loss.csv")
    val_pred.to_csv(OUT_DIR + "val_pred.csv")

    # Plot train and validation loss across folds
    plot_loss_folds(df_loss, facet_by="leftout_group", file_out=OUT_DIR + "loss.png")

    # Plot predicted and true states for each hold-out group, merging all time points
    plot_ts_error(val_pred, facet_by="leftout_group", file_out=OUT_DIR + "ts_error_logo.png")

    # Predicted and true species dynamics for each hold-out group
    # plot_traj_groups(val_pred)

    # Extract interaction weights
    # A_est = np.load(OUT_DIR + "A_est.npy")
    A_est = A_est.mean(axis=0)
    species_names = np.loadtxt(DATA_DIR + "species_names.txt", dtype=str)

    species_mapping = dict(zip(range(len(species_names)), species_names))
    edges = {"Source": [], "Target":[], "Weight":[], "Type":[]}
    for i in range(n_species):
        for j in range(n_species):
            if abs(A_est[i, j]) > 0.01 and i != j:  
                interaction_type = "promotion" if A_est[i, j] > 0 else "inhibition"
                edges['Source'].append(species_mapping[i])
                edges['Target'].append(species_mapping[j])
                edges['Weight'].append(A_est[i,j])
                edges['Type'].append(interaction_type)

    edges = pd.DataFrame(edges)

    # Filter all edges with abs weight (interaction coeff) greater than 0.25
    edges.sort_values('Weight', ascending=False, key=abs)\
        .query('abs(Weight) > 0.25')\
        .to_csv(OUT_DIR + "interactions.csv", index=False)