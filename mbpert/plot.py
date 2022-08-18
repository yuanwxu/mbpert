# Utility plot functions

import numpy as np
import pandas as pd
import torch
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


# Absolute error in the interaction matrix A
def plot_error_A(mbp, file_in, file_out=None):
    A = np.loadtxt(file_in, dtype=np.float32)

    plt.figure()
    A_error_heatmap = sns.heatmap(np.abs(A - mbp.model.state_dict()['A'].numpy()), center=0, annot=True, fmt='.2f')
    A_error_heatmap = A_error_heatmap.set_title("Absolute error for A")

    if file_out:
        A_error_heatmap.get_figure().savefig(file_out)

# Estimated and exact growth rate and susceptibility vectors r and eps
def plot_r_eps(mbp, file_in_1, file_in_2, file_out=None):
    r = np.loadtxt(file_in_1, dtype=np.float32)
    eps = np.loadtxt(file_in_2, dtype=np.float32)

    r_df = pd.DataFrame(data={'est': mbp.model.state_dict()['r'].numpy(),
                            'exact': r,
                            'param': 'r'})
    eps_df = pd.DataFrame(data={'est': mbp.model.state_dict()['eps'].numpy().ravel(),
                                'exact': eps.ravel(),
                                'param': 'eps'})

    plt.figure()
    fig, ax = plt.subplots(figsize=(6, 4))
    r_eps = sns.scatterplot(data=pd.concat([r_df, eps_df]), x='est', y='exact', hue='param', ax=ax)
    r_eps = sns.lineplot(x=np.linspace(-0.5, 1.5), y=np.linspace(-0.5, 1.5), color='g', ax=ax)

    if file_out:
        r_eps.get_figure().savefig(file_out)

# Plot predicted vs true steady states in test set
def plot_ss_test(mbp, file_out=None):
    x_df = mbp.predict_val()
    x_df['value'] = 'x'
    
    plt.figure()
    ss_test = sns.relplot(data=x_df, x='pred', y='true', hue='value', palette={'x':'.4'}, legend=False)
    plt.plot(np.linspace(0.0, 2.5), np.linspace(0.0, 2.5), color='g')

    def annotate(data, **kwargs):
        r, _ = stats.pearsonr(data['pred'], data['true'])
        ax = plt.gca()
        ax.text(0.5, 0.1, 'Pearson correlation r={:.3f}'.format(r),
                transform=ax.transAxes)
        
    ss_test.map_dataframe(annotate)
    plt.title("Predicted and true steady states in test set \nacross all conditions")
    # plt.show()

    if file_out:
        ss_test.savefig(file_out)

# Plot train and validation loss across folds (for leave-one-species-out CV)
def plot_loss_folds(df_loss, file_out=None):
    plt.figure()
    g_loss = sns.relplot(x='epoch', y='value', hue='Loss', col='fold',  
                        data=df_loss.melt(id_vars=['fold', 'epoch'], 
                                        value_vars=['loss_train', 'loss_val'], 
                                        var_name='Loss'),
                        kind='line', col_wrap=5, height=2, linewidth=2)
    g_loss.set_ylabels('')

    if file_out:
        g_loss.savefig(file_out)

# Plot predicted and true steady states for each left-out test set (for leave-one-species-out CV)
def plot_ss_folds(df_ss, file_out=None):
    def annotate(x, y, **kwargs):
        plt.axline((0, 0), (1, 1), color='k', linestyle='dashed')
        r, _ = stats.pearsonr(x, y)
        plt.annotate(f"r = {r:.3f}", xy=(0.7, 0.1), 
                     xycoords=plt.gca().get_yaxis_transform())

    plt.figure()    
    g_ss = sns.FacetGrid(df_ss, col='fold', col_wrap=4, height=2)
    g_ss.map(sns.scatterplot, 'pred', 'true')
    g_ss.map(annotate, 'pred', 'true')

    if file_out:
        g_ss.savefig(file_out)

# Plot predicted and true states at test time points (for time series data)
def plot_pred_ts(mbp, file_out=None):
    df_ts = mbp.predict_val_ts()

    def annotate(x, y, **kwargs):
        plt.axline((0, 0), (1, 1), color='k', linestyle='dashed')
        r, _ = stats.pearsonr(x, y)
        plt.annotate(f"r = {r:.3f}", xy=(0.7, 0.1), 
                     xycoords=plt.gca().get_yaxis_transform())
    
    plt.figure()
    g_ts = sns.FacetGrid(df_ts, col='t', col_wrap=4, height=2)
    g_ts.map(sns.scatterplot, 'pred', 'true')
    g_ts.map(annotate, 'pred', 'true')

    if file_out:
        g_ts.savefig(file_out) 