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
    eps_df = pd.DataFrame(data={'est': mbp.model.state_dict()['eps'].numpy(),
                                'exact': eps,
                                'param': 'eps'})

    plt.figure()
    fig, ax = plt.subplots(figsize=(6, 4))
    r_eps = sns.scatterplot(data=pd.concat([r_df, eps_df]), x='est', y='exact', hue='param', ax=ax)
    r_eps = sns.lineplot(x=np.linspace(-0.5, 1.5), y=np.linspace(-0.5, 1.5), color='g', ax=ax)

    if file_out:
        r_eps.get_figure().savefig(file_out)

# Plot predicted vs true steady states in test set
def plot_ss_test(mbp, testloader, file_out=None):
    x_ss_test = []
    x_pred_test = []
    with torch.no_grad():
        for testdata in testloader:
            (x0, p), responses = testdata
            x0 = torch.ravel(x0)
            p = torch.t(p)
            x_ss = torch.ravel(responses).numpy()
            x_pred = mbp.predict(x0, p)

            x_ss_test.append(x_ss)
            x_pred_test.append(x_pred)
    
    x_ss_test = np.concatenate(x_ss_test)
    x_pred_test = np.concatenate(x_pred_test)
    x_df = pd.DataFrame(data={'pred': x_pred_test, 'true': x_ss_test, 'value': 'x'})
    
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

