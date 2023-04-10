import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import torch
from torch.utils.data import DataLoader
from mbpert.simhelper import get_ode_params
from mbpert.main import MBP
from mbpert.loss import reg_loss_interaction, reg_loss_r, reg_loss_eps_mat
from mbpert.mbpertTS import MBPertTSDataset, MBPertTS, glvp2
from mbpert.plot import plot_ts_error, plot_error_A, plot_r_eps

DATA_DIR = "data/sp_simu_multi_starts/"
OUT_DIR = "output/sp_simu_multi_starts/"

INTEGRATE_END = 30 # define end point of numerical integration

if __name__ == '__main__':
    # Data Generation:
    # 10 species, sampling times span 180 days with an interval of 5 days
    # One type of perturbation applied at day 25, 55, 90 and 125
    # Data points from the first 120 days used for training, so prediction task includes
    # predicting the dynamics after the last perturbation at day 125
    # Repeat for multiple random initial states

    n_species = 10
    r, A, eps, _ = get_ode_params(n_species, perts=1, seed=0)
    T = 180
    P = np.zeros((T+1, 1))
    P[np.array([25, 55, 90, 125])] = 1
    samp_days = np.arange(0, T+1, 5)
    train_stop_idx = int(120/5 + 1)
    t = INTEGRATE_END * samp_days / T
    # Metadata with two columns: group id and measurement time in days,
    # here we have only one group 
    meta = np.column_stack((np.ones(len(samp_days), dtype=int), samp_days))   

    # Save the exact parameters
    np.savetxt(DATA_DIR + "r.txt", r)
    np.savetxt(DATA_DIR + "A.txt", A)
    np.savetxt(DATA_DIR + "eps.txt", eps)

    # Train the model N times with different initial species concentrations
    N = 100

    # Number of epochs per training
    N_EPOCHS = 200

    # Store ODE parameter estimates
    A_est = np.empty((N, n_species, n_species))
    r_est = np.empty((N, n_species))
    eps_est = np.empty((N, *eps.shape))

    rng = np.random.default_rng(123)
    for i in range(N):
        print(f'\nTraining model {i} with initial states {i}...')
        x0 = rng.lognormal(size=n_species)
        sol = solve_ivp(glvp2, [0, INTEGRATE_END], x0, args = (r, A, eps, P, T), dense_output=True)
        z = sol.sol(t)

        # Build Dataset and Dataloader
        X_train, meta_train = z[:,:train_stop_idx], meta[:train_stop_idx]
          
        # For test data need to prepend the initial state
        X_test = np.column_stack((z[:, 0], z[:, train_stop_idx:]))     
        meta_test = np.row_stack((meta[0], meta[train_stop_idx:]))

        trainset = MBPertTSDataset(X_train, P, meta_train) 
        testset = MBPertTSDataset(X_test, P, meta_test)
        trainloader = DataLoader(trainset, batch_size=16, shuffle=False)
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
        # mbp.set_tensorboard('sim_sequential_pert')
        mbp.train(n_epochs=N_EPOCHS, verbose=False, seed=i*50)

        A_est[i] = mbp.model.state_dict()['A'].numpy()
        r_est[i] = mbp.model.state_dict()['r'].numpy()
        eps_est[i] = mbp.model.state_dict()['eps'].numpy()

        # Store prediction on validation
        if i == 0:
            val_pred = np.empty((N, X_test.shape[0]*(X_test.shape[1]-1), 4))
        val_pred[i] = mbp.predict_val_ts().to_numpy()

    # Save results
    np.save(OUT_DIR + "A_est", A_est)
    np.save(OUT_DIR + "r_est", r_est)
    np.save(OUT_DIR + "eps_est", eps_est)
    np.save(OUT_DIR + "val_pred", val_pred)

    # Visualize results
    # Plot exact and estimated parameters averaged over all training rounds
    plot_error_A(A_est.mean(axis=0), A, file_out=OUT_DIR + "A_error_heatmap.png")
    plot_r_eps(r_est.mean(axis=0), eps_est.mean(axis=0), r, eps,\
               file_out=OUT_DIR + "r_eps_error.png")

    # Plot predicted vs true states at test time points, for one set of initial 
    # species concentrations
    plot_ts_error(pd.DataFrame(val_pred[0], columns=['species_id', 't', 'pred', 'true']),\
                  facet_by="t", file_out=OUT_DIR + "ts_error_test_for_one.png")
    
    # Save one of the models checkpoint (last model from for loop)
    mbp.save_checkpoint(OUT_DIR + "sp_simu_multi_starts.pth")

    # Load checkpoint
    # n_species = 10
    # mbpertTS = MBPertTS(n_species, P)
    # mbp = MBP(mbpertTS, loss_fn_ts, optimizer, ts_mode=True)
    # mbp.load_checkpoint(OUT_DIR + "sp_simu_multi_starts.pth")
    # mbp.plot_losses()