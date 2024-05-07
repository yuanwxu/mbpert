import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from mbpert.main import MBP
from mbpert.loss import reg_loss_interaction, reg_loss_r, reg_loss_eps
from mbpert.mbpert import MBPertDataset, MBPert
from mbpert.plot import plot_ss_test
from mbpert.simhelper import pert_mat, get_ode_params, mbpert_split_by_cond


DATA_DIR = "data/pp_simu_split_by_cond/"
OUT_DIR = "output/pp_simu_split_by_cond/"

if __name__ == '__main__':
    # Data Generation
    # 10 speices, all monospecies perturbations + all multi-species perturbations
    # up to `max_comb` species
    n_species = 10
    max_comb = 5
    n_conds_list = [math.comb(n_species, k) for k in range(1, max_comb+1)]
    p = pert_mat(n_species, list(range(1, max_comb + 1)))
    r, A, eps, X_ss = get_ode_params(n_species, p, seed=0)

    # Save the exact parameters
    np.savetxt(DATA_DIR + "r.txt", r)
    np.savetxt(DATA_DIR + "A.txt", A)
    np.savetxt(DATA_DIR + "eps.txt", eps)

    n_conds = p.shape[1]
    # Initial state log-normal distributed and replicated across all perturbations
    rng = np.random.default_rng(100)
    x0 = np.tile(rng.lognormal(sigma=0.2, size=n_species), n_conds)

    # Train the model `max_comb - 1` times. First on all monospecies, then on
    # mono- and pairwise-species, and so on.
    N = max_comb - 1

    # Number of epochs per training
    N_EPOCHS = 800

    # Store ODE parameter estimates
    A_est = np.empty((N, n_species, n_species))
    r_est = np.empty((N, n_species))
    eps_est = np.empty((N, n_species))

    # Store prediction on validation. Use a list to append results to as
    # validation sets sizes are different
    val_pred = [] 

    torch.manual_seed(100)
    for i in range(N):
        print(f'\nTraining model {i} with up to {i+1} species combinations...')
        split_outputs = mbpert_split_by_cond(x0, X_ss, p, n_conds_lst=n_conds_list, keep_as_train=i+1)
        x0_train, x0_test, x_ss_train, x_ss_test, P_train, P_test = split_outputs

        print(f'Model {i} number of conditions in validation: {P_test.shape[1]}')

        trainset = MBPertDataset(x0_train, x_ss_train, P_train)
        testset = MBPertDataset(x0_test, x_ss_test, P_test)

        trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
        testloader = DataLoader(testset, batch_size=32)  

        # Model configuration
        n_species = trainset.n_species
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

        # Store prediction on validation, add a column to label
        # species combinations used for the current training
        vp = mbp.predict_val()
        vp['train_combo'] = i + 1 
        val_pred.append(vp)

        # Save model checkpoint
        mbp.save_checkpoint(OUT_DIR + "pp_simu_split_by_cond_" + str(i+1) + ".pth")

    # Save results
    np.save(OUT_DIR + "A_est", A_est)
    np.save(OUT_DIR + "r_est", r_est)
    np.save(OUT_DIR + "eps_est", eps_est)
    pd.concat(val_pred).to_csv(OUT_DIR + "val_pred.csv")

    # Plot predicted vs true steady states in each test set 
    for i in range(N):
        plot_ss_test(pd.concat(val_pred).query('train_combo == @i+1'),\
                    file_out=OUT_DIR + "ss_error_" + str(i+1) + "species.png")          
    
    
    # Load checkpoint
    # n_species = 10
    # mbpert = MBPert(n_species)
    # mbp = MBP(mbpert, loss_fn, optimizer)
    # mbp.load_checkpoint(OUT_DIR + "pp_simu_split_by_cond_1.pth")
    # mbp.plot_losses()




