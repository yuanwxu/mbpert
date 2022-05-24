from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from mbpert.loss import reg_loss_interaction, reg_loss_r, reg_loss_eps
from mbpert.mbpert import MBPertDataset, MBPert
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

if __name__ == '__main__':
    trainset = MBPertDataset("data/x0_train.txt", 
                            "data/x_ss_train.txt", 
                            "data/P_train.txt")
    testset = MBPertDataset("data/x0_test.txt", 
                            "data/x_ss_test.txt", 
                            "data/P_test.txt")
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=True)  

    n_species = trainset.n_species
    mbpert = MBPert(n_species)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(mbpert.parameters())

    # Log history
    writer = SummaryWriter()

    for epoch in range(500): 

        running_loss = 0.0 # printing loss statistics per batch
        epoch_loss = 0.0 # plotting training loss curve
        for i, data in enumerate(trainloader, 0):
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
            if i % 5 == 4:    # print every 5 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0

            epoch_loss += loss.item()

        # Log epoch loss (per batch)
        writer.add_scalar("Loss/train", epoch_loss/(i+1), epoch)

        # Log test set loss
        epoch_loss_test = 0.0
        with torch.no_grad():
            for i, testdata in enumerate(testloader, 0):
                (x0, p), responses = testdata
                x0 = torch.ravel(x0)
                p = torch.t(p)
                x_ss = torch.ravel(responses)
                x_pred = mbpert(x0, p)

                # Compute loss (MSE + reg)
                loss = criterion(x_pred, x_ss)
                loss = loss + reg_loss_interaction(mbpert.A) + reg_loss_r(mbpert.r) + reg_loss_eps(mbpert.eps)

                epoch_loss_test += loss.item()
            
        writer.add_scalar("Loss/test", epoch_loss_test/(i+1), epoch)

    writer.flush()
    writer.close()

    # Comparing true and predicted A, r, eps
    A = np.loadtxt("data/A.txt", dtype=np.float32)
    r = np.loadtxt("data/r.txt", dtype=np.float32)
    eps = np.loadtxt("data/eps.txt", dtype=np.float32)

    plt.figure()
    A_error_heatmap = sns.heatmap(np.abs(A - mbpert.A.detach().numpy()), center=0, annot=True, fmt='.2f')
    A_error_heatmap = A_error_heatmap.set_title("Absolute error for A")
    A_error_heatmap.get_figure().savefig("data/figs/A_error_heatmap.png")

    r_df = pd.DataFrame(data={'pred': mbpert.r.detach().numpy(),
                            'true': r,
                            'param': 'r'})
    eps_df = pd.DataFrame(data={'pred': mbpert.eps.detach().numpy(),
                                'true': eps,
                                'param': 'eps'})

    plt.figure()
    fig, ax = plt.subplots(figsize=(6, 4))
    r_eps_pred_vs_true = sns.scatterplot(data=pd.concat([r_df, eps_df]), x='pred', y='true', hue='param', ax=ax)
    r_eps_pred_vs_true = sns.lineplot(x=np.linspace(-0.5, 1.5), y=np.linspace(-0.5, 1.5), color='g', ax=ax)
    r_eps_pred_vs_true.get_figure().savefig("data/figs/r_eps_pred_vs_true.png")

    # Plot predicted vs true steady states in test set
    x_ss_test = []
    x_pred_test = []
    with torch.no_grad():
        for i, testdata in enumerate(testloader, 0):
            (x0, p), responses = testdata
            x0 = torch.ravel(x0)
            p = torch.t(p)
            x_ss = torch.ravel(responses)
            x_pred = mbpert(x0, p)

            x_ss_test.append(x_ss)
            x_pred_test.append(x_pred)

    x_ss_test = torch.cat(x_ss_test, dim=0)
    x_pred_test = torch.cat(x_pred_test, dim=0)

    x_df = pd.DataFrame(data={'pred': x_pred_test, 'true': x_ss_test, 'value': 'x'})
    plt.figure()
    # fig, ax = plt.subplots(figsize=(6, 4))
    # x_test_pred_vs_true = sns.scatterplot(data=x_df, x='pred', y='true', hue='value', palette={'x':'.4'}, ax=ax)
    x_test_pred_vs_true = sns.relplot(data=x_df, x='pred', y='true', hue='value', palette={'x':'.4'}, legend=False)
    plt.plot(np.linspace(0.0, 2.5), np.linspace(0.0, 2.5), color='g')
    # x_test_pred_vs_true = sns.lineplot(x=np.linspace(0.0, 2.5), y=np.linspace(0.0, 2.5), color='g', ax=ax)

    def annotate(data, **kwargs):
        r, _ = stats.pearsonr(data['pred'], data['true'])
        ax = plt.gca()
        ax.text(0.5, 0.1, 'Pearson correlation r={:.3f}'.format(r),
                transform=ax.transAxes)
        
    x_test_pred_vs_true.map_dataframe(annotate)
    plt.title("Predicted and true steady states in test set \nacross all conditions")
    plt.show()

    x_test_pred_vs_true.savefig("data/figs/x_test_pred_vs_true.png")

    # Save model
    torch.save(mbpert.state_dict(), "data/model/mbpert_sim.pth")

    # Load model
    # mbpert = MBPert(n_species)
    # mbpert.load_state_dict(torch.load("data/model/mbpert_sim.pth"))




