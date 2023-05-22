# Utility plot functions

import numpy as np
import pandas as pd
import torch
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


# Plot error heatmap for the interaction matrix A
def plot_error_A(A_est, A_exact, file_out=None, relative=False):
    A = np.loadtxt(A_exact, dtype=np.float32) if isinstance(A_exact, str) else A_exact.astype(np.float32)
    err_abs = np.abs(A - A_est)
    err = err_abs / A if relative else err_abs

    plt.figure()
    A_error_heatmap = sns.heatmap(err, center=0, annot=True, fmt='.2f')
    A_error_heatmap = A_error_heatmap.set_title(("Relative" if relative else "Absolute") +\
        " error for A")

    if file_out:
        A_error_heatmap.get_figure().savefig(file_out)

# Estimated and exact growth rate and susceptibility vectors r and eps
def plot_r_eps(r_est, eps_est, r_exact, eps_exact, file_out=None):
    r = np.loadtxt(r_exact, dtype=np.float32) if isinstance(r_exact, str) else r_exact.astype(np.float32)
    eps = np.loadtxt(eps_exact, dtype=np.float32) if isinstance(eps_exact, str) else eps_exact.astype(np.float32)

    r_df = pd.DataFrame(data={'est': r_est,
                            'exact': r,
                            'param': 'r'})
    eps_df = pd.DataFrame(data={'est': eps_est.ravel(),
                                'exact': eps.ravel(),
                                'param': 'eps'})

    plt.figure()
    fig, ax = plt.subplots(figsize=(6, 4))
    r_eps = sns.scatterplot(data=pd.concat([r_df, eps_df]), x='est', y='exact', hue='param', ax=ax)
    r_eps = sns.lineplot(x=np.linspace(-0.5, 1.5), y=np.linspace(-0.5, 1.5), color='g', ax=ax)

    if file_out:
        r_eps.get_figure().savefig(file_out)

# Plot predicted vs true steady states in test set, taking a data frame with predicted 
# and true values (e.g. model.predict_val())
def plot_ss_test(df, file_out=None):
    df['value'] = 'x'
    
    plt.figure()
    ss_test = sns.relplot(data=df, x='pred', y='true', hue='value', palette={'x':'.4'}, legend=False)
    plt.plot(np.linspace(0.0, 2.5), np.linspace(0.0, 2.5), color='g')

    def annotate(data, **kwargs):
        r, _ = stats.pearsonr(data['pred'], data['true'])
        ax = plt.gca()
        ax.text(0.5, 0.1, 'Pearson correlation r={:.3f}'.format(r),
                transform=ax.transAxes)
        
    ss_test.map_dataframe(annotate)
    plt.title("Predicted and true steady states in test set \nacross all conditions")

    if file_out:
        ss_test.savefig(file_out)

# Plot train and validation loss across folds
def plot_loss_folds(df_loss, facet_by, file_out=None):
    plt.figure()
    g_loss = sns.relplot(x='epoch', y='value', hue='Loss', col=facet_by,  
                        data=df_loss.melt(id_vars=[facet_by, 'epoch'], 
                                        value_vars=['loss_train', 'loss_val'], 
                                        var_name='Loss'),
                        kind='line', col_wrap=5, height=2, linewidth=2)
    g_loss.set_ylabels('')

    if file_out:
        g_loss.savefig(file_out)

# Plot predicted and true steady states for each left-out test set (for leave-one-species-out CV)
def plot_ss_folds(val_pred, file_out=None):
    def annotate(x, y, **kwargs):
        plt.axline((0, 0), (1, 1), color='darkgrey', linestyle='dashed')
        r, _ = stats.pearsonr(x, y)
        plt.annotate(f"r = {r:.3f}", xy=(0.7, 0.1), 
                     xycoords=plt.gca().get_yaxis_transform())

    plt.figure()    
    g_ss = sns.FacetGrid(val_pred, col='leftout_species', col_wrap=4, height=2)
    g_ss.map(sns.scatterplot, 'pred', 'true')
    g_ss.map(annotate, 'pred', 'true')

    if file_out:
        g_ss.savefig(file_out)

# Barplot of correlations between predicted and true steady states for each left-out test set 
# (for leave-one-species-out CV)
def barplot_pcorr_folds(val_pred, file_out=None):
    vg = val_pred.groupby('leftout_species', group_keys=True)
    pcorr = vg.apply(lambda x: stats.pearsonr(x.pred, x.true)[0])

    plt.figure()
    ax = pcorr.plot(kind='bar',\
                    title="Pearson correlation between\npredicted and true steady states")

    if file_out:
        ax.figure.savefig(file_out)

# Plot predicted and true states, by test time points or by group (for time series data with multiple groups)
# If by group, then predictions at all time points are merged for each hold-out group.
def plot_ts_error(val_pred, facet_by, color_by=None, file_out=None):
    def annotate(x, y, **kwargs):
        plt.axline((0, 0), (1, 1), color='darkgrey', linestyle='dashed')
        r, _ = stats.pearsonr(x, y)
        plt.annotate(f"r = {r:.3f}", xy=(0.7, 0.1), 
                     xycoords=plt.gca().get_yaxis_transform())
    
    plt.figure()
    g_ts = sns.FacetGrid(val_pred, col=facet_by, col_wrap=4, height=2.5, aspect=1)
    g_ts.map_dataframe(sns.scatterplot, x='pred', y='true', hue=color_by,\
                       palette=sns.color_palette('flare', as_cmap=True) if color_by else None)
    g_ts.map(annotate, 'pred', 'true')

    if file_out:
        g_ts.savefig(file_out)

# Plot predicted and true trajectory for each hold-out group (for time series data with multiple groups)
def plot_traj_groups(val_pred, file_out=None):
    plt.figure()
    trajgps = sns.relplot(x='t', y='value', hue='key', row='species_id', col='leftout_group',  
                        data=val_pred.melt(id_vars=['t', 'leftout_group', 'species_id'], 
                                        value_vars=['pred', 'true'], 
                                        var_name='key'),
                        kind='line', height=2, linewidth=2)
    trajgps.set_ylabels('')

    if file_out:
        trajgps.savefig(file_out)
