import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from mbpert.main import MBP
from mbpert.loss import reg_loss_interaction, reg_loss_r, reg_loss_eps
from mbpert.mbpert import MBPertDataset, MBPert
from mbpert.plot import plot_error_A, plot_r_eps, plot_ss_test
from mbpert.simhelper import pert_mat, get_ode_params, mbpert_split


DATA_DIR = "data/pp_simu_random_split/"
OUT_DIR = "output/pp_simu_random_split/"

if __name__ == '__main__':
    # Data Generation
    # 10 speices, all single-node perturbations + 50% of each of the k-node combo for k = 2,...,5
    n_species = 10
    n_conds_list = [n_species] + [
        round(0.5 * math.comb(n_species, k)) for k in range(2, 6)
    ]
    p = pert_mat(n_species, list(range(1, 6)), n_conds_list)
    r, A, eps, X_ss = get_ode_params(n_species, p, seed=0)

    # Save the exact parameters
    np.savetxt(DATA_DIR + "r.txt", r)
    np.savetxt(DATA_DIR + "A.txt", A)
    np.savetxt(DATA_DIR + "eps.txt", eps)

    n_conds = sum(n_conds_list)
    # Initial state log-normal distributed and replicated across all perturbations
    rng = np.random.default_rng(100)
    x0 = np.tile(rng.lognormal(sigma=0.2, size=n_species), n_conds)

    # Train the model N times with independent random partitions (70-30 split)
    # We want to preserve the proportions of k-species perturbations for train
    # and test sets, so we create a dummy class label vector to stratify by in
    # train_test_split
    N = 200
    stratify_by = np.repeat(np.arange(len(n_conds_list)), n_conds_list)

    # Number of epochs per training
    N_EPOCHS = 400

    # Store ODE parameter estimates
    A_est = np.empty((N, n_species, n_species))
    r_est = np.empty((N, n_species))
    eps_est = np.empty((N, n_species))

    torch.manual_seed(100)
    for i in range(N):
        print(f'\nTraining model {i} with random partition...')
        split_outputs = mbpert_split(x0, X_ss, p, test_size=0.3, stratify=stratify_by)
        x0_train, x0_test, x_ss_train, x_ss_test, P_train, P_test = split_outputs
        trainset = MBPertDataset(x0_train, x_ss_train, P_train)
        testset = MBPertDataset(x0_test, x_ss_test, P_test)

        trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
        testloader = DataLoader(testset, batch_size=32)  

        # Model configuration
        mbpert = MBPert(n_species)

        def loss_fn(y_hat, y, mbpert):
            # Compute loss (MSE + reg)
            criterion = torch.nn.MSELoss()
            loss = criterion(y_hat, y)
            loss = loss + reg_loss_interaction(mbpert.A, reg_lambda=1e-4) + \
                        reg_loss_r(mbpert.r, reg_lambda=1e-5) + \
                        reg_loss_eps(mbpert.eps, reg_lambda=1e-5)
            return loss

        optimizer = torch.optim.Adam(mbpert.parameters())

        # Model training
        mbp = MBP(mbpert, loss_fn, optimizer)
        mbp.set_loaders(trainloader, testloader)
        # mbp.set_tensorboard('sim_parallel_pert')
        mbp.train(n_epochs=N_EPOCHS, verbose=False, seed=None)
        
        A_est[i] = mbp.model.state_dict()['A'].numpy()
        r_est[i] = mbp.model.state_dict()['r'].numpy()
        eps_est[i] = mbp.model.state_dict()['eps'].numpy()

        # Store prediction on validation
        if i == 0:
            val_pred = np.empty((N, P_test.size, 2))
        val_pred[i] = mbp.predict_val().to_numpy()

    # Save results
    np.save(OUT_DIR + "A_est", A_est)
    np.save(OUT_DIR + "r_est", r_est)
    np.save(OUT_DIR + "eps_est", eps_est)
    np.save(OUT_DIR + "val_pred", val_pred)

    # Visualize results
    # Plot exact and estimated parameters averaged over all training rounds
    plot_error_A(A_est.mean(axis=0), A, relative=False,\
                 file_out=OUT_DIR + "A_error_heatmap.png")
    plot_r_eps(r_est.mean(axis=0), eps_est.mean(axis=0), r, eps,\
               file_out=OUT_DIR + "r_eps_error.png")

    # Plot predicted vs true steady states in test set for one training round
    plot_ss_test(pd.DataFrame(val_pred[0], columns=['pred', 'true']),\
                 file_out=OUT_DIR + "ss_error_for_one_testset.png")

    # Save one of the models checkpoint (the last model from the for loop)
    mbp.save_checkpoint(OUT_DIR + "pp_simu_random_split.pth")

    # Load checkpoint
    # n_species = 10
    # mbpert = MBPert(n_species)
    # mbp = MBP(mbpert, loss_fn, optimizer)
    # mbp.load_checkpoint(OUT_DIR + "pp_simu_random_split.pth")
    # mbp.plot_losses()




