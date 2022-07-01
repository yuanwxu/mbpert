import os
import torch
from torch.utils.data import DataLoader
from mbpert.loss import reg_loss_interaction, reg_loss_r, reg_loss_eps_mat
from mbpert.mbpertTS import MBPertTSDataset, MBPertTS
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

if __name__ == '__main__':
    trainset = MBPertTSDataset("data/ts/X_train.txt", 
                               "data/ts/P.txt",
                               "data/ts/tobs_train.txt")
    testset = MBPertTSDataset("data/ts/X_test.txt", 
                               "data/ts/P.txt",
                               "data/ts/tobs_test.txt")
    trainloader = DataLoader(trainset, batch_size=4, shuffle=False)
    testloader = DataLoader(testset, batch_size=4, shuffle=False)

    n_species, P = trainset.n_species, trainset.P
    mbpertTS = MBPertTS(n_species, P)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(mbpertTS.parameters())

    # Log history
    loss_history = {'Epoch': [], 'Loss_train': [], 'Loss_test': []}

    for epoch in range(100): 

        # running_loss = 0.0 # printing loss statistics per batch
        epoch_loss = 0.0 # plotting training loss curve
        for i, data in enumerate(trainloader, 0):
            # Get the input batch
            (x0, t), xt = data

            # Compute loss (MSE + reg) for the minibatch
            xpred_t = []
            for x0j, tj in zip(x0, t):
                xpred_tj = mbpertTS(x0j, tj)
                xpred_t.append(xpred_tj)
            xpred_t = torch.cat(xpred_t, dim=0)
            xt = torch.ravel(xt)

            loss = criterion(xpred_t, xt)
            loss = loss + (reg_loss_interaction(mbpertTS.A) + 
                          reg_loss_r(mbpertTS.r) + 
                          reg_loss_eps_mat(mbpertTS.eps))

            # Zero gradients, perform a backward pass, and update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics for last batch in every epoch
            if i == len(trainloader) - 1:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
            # running_loss += loss.item()
            # if i % 5 == 4:    # print every 5 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #         (epoch + 1, i + 1, running_loss / 5))
            #     running_loss = 0.0

            epoch_loss += loss.item()

        # Log epoch loss (per batch)
        loss_history['Epoch'].append(epoch)  
        loss_history['Loss_train'].append(epoch_loss/(i+1))

        # Log test set loss
        epoch_loss_test = 0.0
        with torch.no_grad():
            for i, testdata in enumerate(testloader, 0):
                (x0, t), xt = testdata

                xpred_t = []
                for x0j, tj in zip(x0, t):
                    xpred_tj = mbpertTS(x0j, tj)
                    xpred_t.append(xpred_tj)
                xpred_t = torch.cat(xpred_t, dim=0)
                xt = torch.ravel(xt)

                loss = criterion(xpred_t, xt)
                loss = loss + (reg_loss_interaction(mbpertTS.A) + 
                              reg_loss_r(mbpertTS.r) + 
                              reg_loss_eps_mat(mbpertTS.eps))

                epoch_loss_test += loss.item()

        loss_history['Loss_test'].append(epoch_loss_test/(i+1))    

    df_loss = pd.DataFrame(loss_history)
    df_loss.to_csv("data/ts/loss.csv", index=False)

    # Record prediction at test time points
    xpred_test = {}
    df_xpred_test = []
    with torch.no_grad():
        for _, testdata in enumerate(testloader, 0):
            (x0, t), xt = testdata
            for x0j, tj, xtj in zip(x0, t, xt):
                xpred_tj = mbpertTS(x0j, tj)
                xpred_test['t'] = tj.item()
                xpred_test['xpred'] = xpred_tj
                xpred_test['xtrue'] = xtj
                df_xpred_test.append(pd.DataFrame(xpred_test))

    df_xpred_test = pd.concat(df_xpred_test)
    df_xpred_test.to_csv("data/ts/xpred_test.csv", index=False)

    # Make plots
    # Train and test loss
    if not os.path.exists('data/ts/figs'):
        os.makedirs('data/ts/figs')

    plt.figure()
    g_loss = sns.relplot(x='Epoch', y='value', hue='Loss', 
                        data=df_loss.melt(id_vars='Epoch', 
                                          value_vars=['Loss_train', 'Loss_test'], 
                                          var_name='Loss'),
                        kind='line', linewidth=2)
    g_loss.set_ylabels('')
    g_loss.savefig('data/ts/figs/loss.png')

    # Predicted and true states at test time points
    def annotate(x, y, **kwargs):
        plt.axline((0, 0), (1, 1), color='k', linestyle='dashed')
        r, _ = stats.pearsonr(x, y)
        plt.annotate(f"r = {r:.3f}", xy=(0.7, 0.1), 
                     xycoords=plt.gca().get_yaxis_transform())
        
    g_xpred_test = sns.FacetGrid(df_xpred_test, col='t', col_wrap=4, height=2)
    g_xpred_test.map(sns.scatterplot, 'xpred', 'xtrue')
    g_xpred_test.map(annotate, 'xpred', 'xtrue')
    g_xpred_test.savefig('data/ts/figs/xpred_test.png') 

    # Comparing true and predicted A, r, eps
    A = np.loadtxt("data/ts/A.txt", dtype=np.float32)
    r = np.loadtxt("data/ts/r.txt", dtype=np.float32)
    eps = np.loadtxt("data/ts/eps.txt", dtype=np.float32)

    plt.figure()
    A_error_heatmap = sns.heatmap(np.abs(A - mbpertTS.A.detach().numpy()), center=0, annot=True, fmt='.2f')
    A_error_heatmap = A_error_heatmap.set_title("Absolute error for A")
    A_error_heatmap.get_figure().savefig("data/ts/figs/A_error_heatmap.png")

    r_df = pd.DataFrame(data={'pred': mbpertTS.r.detach().numpy(),
                            'true': r,
                            'param': 'r'})
    eps_df = pd.DataFrame(data={'pred': mbpertTS.eps.detach().numpy().ravel(),
                                'true': eps.ravel(),
                                'param': 'eps'})

    plt.figure()
    fig, ax = plt.subplots(figsize=(6, 4))
    r_eps_pred_vs_true = sns.scatterplot(data=pd.concat([r_df, eps_df]), x='pred', y='true', hue='param', ax=ax)
    r_eps_pred_vs_true = sns.lineplot(x=np.linspace(-0.5, 1.5), y=np.linspace(-0.5, 1.5), color='g', ax=ax)
    r_eps_pred_vs_true.get_figure().savefig("data/ts/figs/r_eps_pred_vs_true.png")

    
    # Save model
    if not os.path.exists('data/ts/model'):
        os.makedirs('data/ts/model')
    torch.save(mbpertTS.state_dict(), "data/ts/model/mbpertts_sim.pth")




