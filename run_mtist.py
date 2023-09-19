import pickle
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from mbpert.main import MBP
from mbpert.loss import reg_loss_interaction, reg_loss_r
from mbpert.mbpertTS import MBPertTSDataset, MBPertTS, glvp2
from mtist.infer_mtist import calculate_es_score as mtist_es_score

DATA_DIR = "mtist/mtist1.0/"
OUT_DIR = "output/mtist/"


def load_mtist_dataset(did):
    df = pd.read_csv(DATA_DIR + f"mtist_datasets/dataset_{did}.csv", index_col=0)

    # Species temporal dynamics array of shape (n_species, n_groups * samples_per_group)
    X = df.loc[:, df.columns.str.contains("species_")].to_numpy().T

    # Treat 'time' column in mtist dataset as equivalent to the number of days into the
    # experiment
    samp_days = df['time']

    # Meta-data required for MBPert
    n_groups, samp_per_group = df['n_timeseries'][0], df['n_timepoints'][0]
    gids = np.repeat(range(1, n_groups + 1), samp_per_group)
    meta = np.column_stack((gids, samp_days))

    return X, meta


def infer_from_did_mbpert(did, set_seed=True, save_model=False):
    X, meta = load_mtist_dataset(did)
    P = np.zeros((30+1, 1)) # at least as large as the max time in the 'time' column in mtist master dataset
    
    # Inferenc with MBPert
    n_species = X.shape[0]
    dataset = MBPertTSDataset(X, P, meta)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True,\
                            generator=torch.Generator(device='cuda') if torch.cuda.is_available() else None) 
    if set_seed:
        torch.manual_seed(did*50)

    mbpertTS = MBPertTS(n_species, dataset.P)

    # Notice no `eps` term here in the loss since no perturbation used
    def loss_fn_ts(y_hat, y, mbpertTS):
        # Compute loss (MSE + reg)
        criterion = torch.nn.MSELoss()
        loss = criterion(y_hat, y)
        loss = loss + reg_loss_interaction(mbpertTS.A, reg_lambda=1e-4) + \
                        reg_loss_r(mbpertTS.r, reg_lambda=1e-5)
        return loss

    optimizer = torch.optim.Adam(mbpertTS.parameters(), lr=0.01)
   
    # Model training
    N_EPOCHS = 400 # this will be max number of epochs if `stop_training_early` is set
                   # in mbp.train
    mbp = MBP(mbpertTS, loss_fn_ts, optimizer, ts_mode=True)
    mbp.set_loaders(dataloader)
    mbp.train(n_epochs=N_EPOCHS, verbose=False, seed=did*50+1 if set_seed else None,\
              stop_training_early={'epochs':10, 'eps':0.01}) 

    if mbp.total_epochs < N_EPOCHS:
        print(f"Model for MTIST dataset {did} stopped early at epoch {mbp.total_epochs}\n"
              f"dut to stopping criterion set in mbp.train")

    A_est = mbp.model.state_dict()['A'].cpu().numpy()
    r_est = mbp.model.state_dict()['r'].cpu().numpy()

    if save_model:
        mbp.save_checkpoint(OUT_DIR + f"mbp_mtist_dataset_{did}.pth")

    return A_est, r_est
    

if __name__ == '__main__':
    # Get job array index for slurm job array
    TASK_ID = int(sys.argv[1])
    # Get max job array index set in slurm script: --array=1-TASK_MAX
    # to be used to partition the datasets
    TASK_MAX = int(sys.argv[2])

    # Test MBPert on MTIST datasets (a suite of realisticly simulated human gut
    # microbiome time series datasets for testing new inference algorithms)

    # Get ground truth ids corresponding to the datasets
    df_gt = pd.read_csv(DATA_DIR + f"mtist_datasets/mtist_metadata.csv")
    gts = dict(zip(df_gt['did'], df_gt['ground_truth'])) # dict mapping dataset id to ground truth id

    es_score_all = {} # dict mapping dataset id to the calculated ES score
    inferred_aij_all = {} # dict mapping dataset id to the inferred A matrix

    num_datasets = max(gts.keys()) + 1 # 648

    if num_datasets % TASK_MAX != 0:
        raise ValueError(f"Number of datasets ({num_datasets}) is not divisible by"
                         f" number of tasks ({TASK_MAX}) in array job.")

    par = list(range(0, num_datasets, int(num_datasets/TASK_MAX)))
    par.append(num_datasets)

    for did in range(*par[(TASK_ID-1):(TASK_ID+1)]):  # loop over partition defined by current TASK_ID
        gt = gts[did].replace('gt', 'aij')
        true_aij = np.loadtxt(DATA_DIR + f"ground_truths/interaction_coefficients/{gt}.csv", delimiter=',')
        inferred_aij, _ = infer_from_did_mbpert(did)
        es_score_all[did] = mtist_es_score(true_aij, inferred_aij)
        inferred_aij_all[did] = inferred_aij

    # Save ES scores
    es_score_df = pd.DataFrame(es_score_all, index=['ES_score']).T #.reset_index(names='did')
    es_score_df.reset_index().rename(columns={'index':'did'}).to_csv(OUT_DIR + f"es_score_task{TASK_ID}.csv", index=False)

    # Save inferred A matrix
    with open(OUT_DIR + f'inferred_aij_dict_task{TASK_ID}.pkl', 'wb') as f:
        pickle.dump(inferred_aij_all, f)
            
    # with open(OUT_DIR + 'inferred_aij_dict.pkl', 'rb') as f:
    #     inferred_aij_all = pickle.load(f)

   
    # Load checkpoint
    # n_species = 10
    # mbpertTS = MBPertTS(n_species, P)
    # mbp = MBP(mbpertTS, loss_fn_ts, optimizer, ts_mode=True)
    # mbp.load_checkpoint(OUT_DIR + "mbp_mtist_dataset_643.pth")
    # mbp.plot_losses()
