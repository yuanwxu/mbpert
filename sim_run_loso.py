# Leave-one-species-out: 10 speices, all conditions involving one species left out
# the rest will be used for training

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from mbpert.loss import reg_loss_interaction, reg_loss_r, reg_loss_eps
from mbpert.mbpert import MBPertDataset, MBPert
from mbpert.simhelper import pert_mat, get_ode_params

if __name__ == '__main__':
    n_species = 10
    P = pert_mat(n_species, list(range(1, n_species+1)))
    r, A, eps, X_ss = get_ode_params(n_species, P, seed=0)
    n_conds = P.shape[1]
    X_0 = 0.1 * np.ones((n_species, n_conds))  # initial state chosen arbitrary

    loss_history = [] # to store loss over epochs across all folds
    ss_prediction = [] # to store steady state prediction on the withheld set across all folds
    for i in range(n_species):
        P_train_foldi = P[:, ~P[i]]
        P_test_foldi = P[:, P[i]]
        xss_train_foldi = np.ravel(X_ss[:, ~P[i]], order='F')
        xss_test_foldi = np.ravel(X_ss[:, P[i]], order='F')
        x0_train_foldi = np.ravel(X_0[:, ~P[i]], order='F')
        x0_test_foldi = np.ravel(X_0[:, P[i]], order='F')

        trainset = MBPertDataset(x0_train_foldi, xss_train_foldi, P_train_foldi)  
        testset = MBPertDataset(x0_test_foldi, xss_test_foldi, P_test_foldi)
        trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
        testloader = DataLoader(testset, batch_size=32, shuffle=True) 

        mbpert = MBPert(n_species)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(mbpert.parameters())

        loss_history_foldi = {'Epoch': [], 'Loss_train': [], 'Loss_test': []}
        ss_prediction_foldi = {}
        for epoch in range(500): 

            running_loss = 0.0 # printing loss statistics per batch
            epoch_loss = 0.0 # plotting training loss curve
            for j, data in enumerate(trainloader, 0):
                # Get the input batch
                (x0, p), responses = data

                # Get the inputs that model.forward() accepts
                x0 = torch.ravel(x0)
                p = torch.t(p)

                # and responses
                x_ss = torch.ravel(responses)

                # Forward pass
                x_pred = mbpert(x0, p)

                # Compute loss (MSE + reg)
                loss = criterion(x_pred, x_ss)
                loss = loss + reg_loss_interaction(mbpert.A) + reg_loss_r(mbpert.r) + reg_loss_eps(mbpert.eps)

                # Zero gradients, perform a backward pass, and update parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if epoch % 10 == 9 and j % 5 == 4:    # print every 5 mini-batches every 10 epochs
                    print('[Fold: %5d, Epoch: %5d, Batch: %5d] loss: %.3f' %
                        (i + 1, epoch + 1, j + 1, running_loss / 5))
                    running_loss = 0.0

                epoch_loss += loss.item()

            # Log epoch loss (per batch)
            loss_history_foldi['Epoch'].append(epoch)
            loss_history_foldi['Loss_train'].append(epoch_loss/(j+1))

            # Log test set loss
            epoch_loss_test = 0.0
            with torch.no_grad():
                for k, testdata in enumerate(testloader, 0):
                    (x0, p), responses = testdata
                    x0 = torch.ravel(x0)
                    p = torch.t(p)
                    x_ss = torch.ravel(responses)
                    x_pred = mbpert(x0, p)

                    # Compute loss (MSE + reg)
                    loss = criterion(x_pred, x_ss)
                    loss = loss + reg_loss_interaction(mbpert.A) + reg_loss_r(mbpert.r) + reg_loss_eps(mbpert.eps)

                    epoch_loss_test += loss.item()

            loss_history_foldi['Loss_test'].append(epoch_loss_test/(k+1))    
            
        loss_history_foldi['Fold'] = i
        loss_history.append(pd.DataFrame(loss_history_foldi))

        # Record predicted vs true steady states for the withheld fold i
        ss_prediction_foldi['Fold'] = i
        x_ss_foldi = []
        x_pred_foldi = []
        with torch.no_grad():
            for _, testdata in enumerate(testloader, 0):
                (x0, p), responses = testdata
                x0 = torch.ravel(x0)
                p = torch.t(p)
                x_ss = torch.ravel(responses)
                x_pred = mbpert(x0, p)

                x_ss_foldi.append(x_ss)
                x_pred_foldi.append(x_pred)

        x_ss_foldi = torch.cat(x_ss_foldi, dim=0)
        x_pred_foldi = torch.cat(x_pred_foldi, dim=0)
        ss_prediction_foldi['Pred'] = x_pred_foldi
        ss_prediction_foldi['Exact'] = x_ss_foldi
        ss_prediction.append(pd.DataFrame(ss_prediction_foldi))

    df_loss = pd.concat(loss_history)
    df_ss = pd.concat(ss_prediction)
    df_loss.to_csv('data/loss_loso.csv', index=False)   
    df_ss.to_csv('data/ss_loso.csv', index=False) 

    # Make plots 
    # Train and test loss across folds
    plt.figure()
    g_loss = sns.relplot(x='Epoch', y='value', hue='Loss', col='Fold',  
                        data=df_loss.melt(id_vars=['Fold', 'Epoch'], 
                                        value_vars=['Loss_train', 'Loss_test'], 
                                        var_name='Loss'),
                        kind='line', col_wrap=5, height=2, linewidth=2)
    g_loss.set_ylabels('')
    g_loss.savefig('data/figs/loss_loso.png')

    # Predicted and true steady states for each left-out test set
    def annotate(x, y, **kwargs):
        plt.axline((0, 0), (1, 1), color='k', linestyle='dashed')
        r, _ = stats.pearsonr(x, y)
        plt.annotate(f"r = {r:.3f}", xy=(0.7, 0.1), 
                     xycoords=plt.gca().get_yaxis_transform())
        
    g_ss = sns.FacetGrid(df_ss, col='Fold', col_wrap=4, height=2)
    g_ss.map(sns.scatterplot, 'Pred', 'Exact')
    g_ss.map(annotate, 'Pred', 'Exact')
    g_ss.savefig('data/figs/ss_loso.png')

    







