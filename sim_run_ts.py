import torch
from torch.utils.data import DataLoader
from mbpert.main import MBP
from mbpert.loss import reg_loss_interaction, reg_loss_r, reg_loss_eps_mat
from mbpert.mbpertTS import MBPertTSDataset, MBPertTS
from mbpert.plot import plot_pred_ts, plot_error_A, plot_r_eps


if __name__ == '__main__':
    # Build Dataset and Dataloader
    trainset = MBPertTSDataset("data/ts/X_train.txt", 
                               "data/ts/P.txt",
                               "data/ts/tobs_train.txt")
    testset = MBPertTSDataset("data/ts/X_test.txt", 
                               "data/ts/P.txt",
                               "data/ts/tobs_test.txt")
    trainloader = DataLoader(trainset, batch_size=4, shuffle=False)
    testloader = DataLoader(testset, batch_size=4, shuffle=False)

    # Model configuration
    n_species, P = trainset.n_species, trainset.P

    torch.manual_seed(42)
    mbpertTS = MBPertTS(n_species, P)

    def loss_fn_ts(y_hat, y, mbpertTS):
        # Compute loss (MSE + reg)
        criterion = torch.nn.MSELoss()
        loss = criterion(y_hat, y)
        loss = loss + reg_loss_interaction(mbpertTS.A) + \
                        reg_loss_r(mbpertTS.r) + \
                        reg_loss_eps_mat(mbpertTS.eps)
        return loss
    
    optimizer = torch.optim.Adam(mbpertTS.parameters())

    # Model training
    mbp = MBP(mbpertTS, loss_fn_ts, optimizer, ts_mode=True)
    mbp.set_loaders(trainloader, testloader)
    mbp.set_tensorboard('sim_sequential_pert')

    mbp.train(n_epochs=100, verbose=True)

    # Visualize results
    # Predicted and true states at test time points
    plot_pred_ts(mbp)

    # Plot exact and estimated parameters
    plot_error_A(mbp, "data/ts/A.txt")
    plot_r_eps(mbp, "data/ts/r.txt", "data/ts/eps.txt")
        
    # Save checkpoint
    mbp.save_checkpoint("checkpoint/sim_sequential_pert.pth")

    # Load checkpoint
    # torch.manual_seed(42)
    # mbpertTS = MBPertTS(n_species, P)
    # mbp = MBP(mbpertTS, loss_fn_ts, optimizer, ts_mode=True)
    # mbp.load_checkpoint("checkpoint/sim_sequential_pert.pth")
