import pickle
import asyncio # for parallelizing loop over datasets
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

    # Convert 'time' column in mtist dataset to the unit of days assuming 25 days duration
    # as stated in the mtist paper
    samp_days = 25 * df['time']/max(df['time'])

    # Meta-data required for MBPert
    n_groups, samp_per_group = df['n_timeseries'][0], df['n_timepoints'][0]
    gids = np.repeat(range(1, n_groups + 1), samp_per_group)
    meta = np.column_stack((gids, samp_days))

    return X, meta


def infer_from_did_mbpert(did, set_seed=True, save_model=False):
    X, meta = load_mtist_dataset(did)
    P = np.zeros((25+1, 1)) 
    
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
        # Set reg loss for the interaction matrix to be 1-norm if >= 10 species
        od = 1 if mbpertTS.A.shape[0] >= 10 else 2

        # Compute loss (MSE + reg)
        criterion = torch.nn.MSELoss()
        loss = criterion(y_hat, y)
        loss = loss + reg_loss_interaction(mbpertTS.A, order=od) + \
                        reg_loss_r(mbpertTS.r)
        return loss
    
    optimizer = torch.optim.Adam(mbpertTS.parameters())

    # Model training
    THRESH = 1e-5 # threshold for setting a_ij to 0 in the case of 1-norm 
                  # regularization
    N_EPOCHS = 400 # this will be max number of epochs if `stop_training_early` is set
                   # in mbp.train
    mbp = MBP(mbpertTS, loss_fn_ts, optimizer, ts_mode=True)
    mbp.set_loaders(dataloader)
    mbp.train(n_epochs=N_EPOCHS, verbose=False, seed=did*50+1 if set_seed else None,\
              stop_training_early={'epochs':10, 'eps':0.05}) 

    if mbp.total_epochs < N_EPOCHS:
        print(f"Model for MTIST dataset {did} stopped early at epoch {mbp.total_epochs}\n"
              f"dut to stopping criterion set in mbp.train")

    A_est = mbp.model.state_dict()['A'].cpu().numpy()
    r_est = mbp.model.state_dict()['r'].cpu().numpy()

    if n_species >= 10:
        A_est[np.abs(A_est) < THRESH] = 0

    if save_model:
        mbp.save_checkpoint(OUT_DIR + f"mbp_mtist_dataset_{did}.pth")

    return A_est, r_est


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped
    

if __name__ == '__main__':
    # Test MBPert on MTIST datasets (a suite of realisticly simulated human gut
    # microbiome time series datasets for testing new inference algorithms)

    # Get ground truth ids corresponding to the datasets
    df_gt = pd.read_csv(DATA_DIR + f"mtist_datasets/mtist_metadata.csv")
    gts = dict(zip(df_gt['did'], df_gt['ground_truth'])) # dict mapping dataset id to ground truth id

    es_score_all = {} # dict mapping dataset id to the calculated ES score
    inferred_aij_all = {} # dict mapping dataset id to the inferred A matrix

    @background
    def score_inferred_dataset(did, gts):
        gt = gts[did].replace('gt', 'aij')
        true_aij = np.loadtxt(DATA_DIR + f"ground_truths/interaction_coefficients/{gt}.csv", delimiter=',')
        inferred_aij, _ = infer_from_did_mbpert(did)
        es_score_all[did] = mtist_es_score(true_aij, inferred_aij)
        inferred_aij_all[did] = inferred_aij

    loop = asyncio.get_event_loop()                                              
    looper = asyncio.gather(*[score_inferred_dataset(i, gts) for i in range(0, 648)])
    results = loop.run_until_complete(looper) 

    #for did in range(0, 648): 
    #    gt = gts[did].replace('gt', 'aij')
    #    true_aij = np.loadtxt(DATA_DIR + f"ground_truths/interaction_coefficients/{gt}.csv", delimiter=',')
    #    inferred_aij, _ = infer_from_did_mbpert(did)
    #    es_score_all[did] = mtist_es_score(true_aij, inferred_aij)
    #    inferred_aij_all[did] = inferred_aij

    # Save ES scores
    es_score_df = pd.DataFrame(es_score_all, index=['ES_score']).T #.reset_index(names='did')
    es_score_df.reset_index().rename(columns={'index':'did'}).to_csv(OUT_DIR + "es_score.csv", index=False)

    # Save inferred A matrix
    with open(OUT_DIR + 'inferred_aij_dict.pkl', 'wb') as f:
        pickle.dump(inferred_aij_all, f)
            
    # with open(OUT_DIR + 'inferred_aij_dict.pkl', 'rb') as f:
    #     inferred_aij_all = pickle.load(f)

   
    # Load checkpoint
    # n_species = 10
    # mbpertTS = MBPertTS(n_species, P)
    # mbp = MBP(mbpertTS, loss_fn_ts, optimizer, ts_mode=True)
    # mbp.load_checkpoint(OUT_DIR + "sp_simu_multi_starts.pth")
    # mbp.plot_losses()
