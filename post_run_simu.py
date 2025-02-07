import numpy as np
import pandas as pd
import itertools as it
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

# RUN_ID = "pp_simu_random_split"
RUN_ID = "sp_simu_multi_starts"

if RUN_ID == "pp_simu_random_split":
    DATA_DIR = "data/pp_simu_random_split/"
    OUT_DIR = "output/pp_simu_random_split/"

if RUN_ID == "sp_simu_multi_starts":
    DATA_DIR = "data/sp_simu_multi_starts/"
    OUT_DIR = "output/sp_simu_multi_starts/"

A_est = np.load(OUT_DIR + "A_est.npy")
A_exact = np.loadtxt(DATA_DIR + "A.txt")
r_est = np.load(OUT_DIR + "r_est.npy")
r_exact = np.loadtxt(DATA_DIR + "r.txt")
eps_est = np.load(OUT_DIR + "eps_est.npy")
eps_exact = np.loadtxt(DATA_DIR + "eps.txt")

# For sequential perturbation, instead of suscep vector we have a suscep matrix
if RUN_ID == "sp_simu_multi_starts" and eps_exact.ndim == 1:
    eps_exact = eps_exact.reshape(-1, 1)

val_pred = np.load(OUT_DIR + "val_pred.npy")

def to_df(arr, prefix, long=True):
    """Convert a numpy array of shape either (N, n) or (N, n1, n2) to 
       a data frame of N rows and n (n1 * n2) columns, with each column
       named "prefix" + i for i in range(n) or "prefix" + ij
       where i, j run to n1 and n2, respectively.
       If long = True, then will subsequently convert to a long data frame
    """
    if arr.ndim not in [2, 3]:
        raise ValueError("This function works for array dimension 2 or 3.")

    if arr.ndim == 2:
        N, n = arr.shape    
        colnames = [prefix + str(i) for i in range(n)]
    
    if arr.ndim == 3:
        N, n1, n2 = arr.shape
        colnames = [prefix + str(i) + str(j) for i, j in it.product(range(n1), range(n2))]

    arr = arr.reshape(N, -1)
    df = pd.DataFrame(arr, columns=colnames)

    if long:
        df = df.melt(var_name='name')
    return df

def plot_errbar(df_est, df_exact, startwith='', errtype='sd'):
    """Point plot with error bars, overlayed by exact data points.
       df_est: long data frame with a column 'name' containing the 
               names of the estimated quantities, and a column 'value'
               the corresponding values.
       df_exact: long data frame with column 'name' and 'value' containing
                 the names of the quantities and the exact values
       startwith: a string used to subset the data frames, so can show a 
                  selection of quantities and avoid clutter the plot.
       errtype: the same 'errorbar' argument in seaborn.pointplot(), 
                default is one standard deviation.
    """
    plt.figure()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.pointplot(data=df_est[df_est.name.str.startswith(startwith)],\
                x='value', y='name',\
                errorbar=errtype, join=False, capsize=.4, color='.6', ax=ax)
    sns.scatterplot(data=df_exact[df_exact.name.str.startswith(startwith)],\
                    x='value', y='name',\
                    color='.0', marker='X', s=150, ax=ax)
    return ax

def print_stats(est, exact, prefix):
    """Print number of parameters contained within one or two standard deviations
       of the mean estimates
    est: array of estimated parameters values where the first dimension is N 
         (number of runs/repeats)
    exact: exact parameters, same shape as est[0]
    """
    if est[0].size != exact.size:
        raise ValueError("Parameter size does not match between est and exact.")
    exact = to_df(exact[np.newaxis,...], prefix=prefix, long=False).iloc[0]
    est_df = to_df(est, prefix=prefix, long=False)
    sd1 = est_df.apply(np.std, axis=0)
    sd2 = 2 * sd1
    est_mean = est_df.apply(np.mean, axis=0)

    in_1sd = (est_mean - sd1 < exact) & (exact < est_mean + sd1)
    in_2sd = (est_mean - sd2 < exact) & (exact < est_mean + sd2)

    print(f'Parameters contained within 1 sd of the mean: {sum(in_1sd)}/{len(exact)}')
    if sum(in_1sd) < len(exact):
        print(f'Parameters outside 1 sd of the mean: {exact[~in_1sd].index}')

    print(f'Parameters contained within 2 sd of the mean: {sum(in_2sd)}/{len(exact)}')
    if sum(in_2sd) < len(exact):
        print(f'Parameters outside 2 sd of the mean: {exact[~in_2sd].index}')

# Compare estimated and exact parameters
print_stats(A_est, A_exact, prefix='a')
print_stats(r_est, r_exact, prefix='r')
print_stats(eps_est, eps_exact, prefix='eps')

# Plot errorbars for the estimated interaction matrix A, growth rate vector r,
# and susceptibility vector eps
n_species = A_exact.shape[0]
A_est = to_df(A_est, prefix='a')
A_exact = to_df(A_exact[np.newaxis, ...], prefix='a')

for i in range(n_species):
    ax = plot_errbar(A_est, A_exact, startwith='a' + str(i))
    ax.set_title("Species interactions on species " + str(i))
    ax.figure.savefig(OUT_DIR + f"A_a{i}.png")
    plt.clf()

r_est = to_df(r_est, prefix='r')
r_exact = to_df(r_exact[np.newaxis, ...], prefix='r')
ax = plot_errbar(r_est, r_exact)
ax.set_title("Growth rate")
ax.figure.savefig(OUT_DIR + "r.png")

eps_est = to_df(eps_est, prefix='eps')
eps_exact = to_df(eps_exact[np.newaxis,...], prefix='eps')
ax = plot_errbar(eps_est, eps_exact)
ax.set_title("Susceptibility to perturbations")
ax.figure.savefig(OUT_DIR + "eps.png")

if RUN_ID == "pp_simu_random_split":
    # Plot distribution of correlations between predicted and true steady states 
    # in the validation sets of all training rounds
    ax = sns.histplot(data=np.array([stats.pearsonr(x[:,0], x[:, 1])[0] for x in val_pred]),\
                    binwidth=0.01)
    ax.set_title("Correlation between predicted and true steady states\n under validation set perturbations")
    ax.set_xlabel("Pearson correlation")
    ax.figure.savefig(OUT_DIR + "pcorr_hist.png")

if RUN_ID == "sp_simu_multi_starts":
    # Plot distribution of correlations between predicted and true states 
    # at test time points, over all initial species concentrations
    ax = sns.histplot(data=np.array([stats.pearsonr(x[:,2], x[:, 3])[0] for x in val_pred]),\
                    binwidth=0.002)
    ax.set_title("Histogram of Pearson correlations between predicted and\n true states across test time points")
    ax.set_xlabel("")
    ax.figure.savefig(OUT_DIR + "pcorr_hist.png")