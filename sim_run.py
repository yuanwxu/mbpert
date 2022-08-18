import torch
from torch.utils.data import DataLoader
from mbpert.main import MBP
from mbpert.loss import reg_loss_interaction, reg_loss_r, reg_loss_eps
from mbpert.mbpert import MBPertDataset, MBPert
from mbpert.plot import plot_error_A, plot_r_eps, plot_ss_test


if __name__ == '__main__':
    # Build Dataset and Dataloader
    trainset = MBPertDataset("data/x0_train.txt", 
                            "data/x_ss_train.txt", 
                            "data/P_train.txt")
    testset = MBPertDataset("data/x0_test.txt", 
                            "data/x_ss_test.txt", 
                            "data/P_test.txt")
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)  

    # Model configuration
    n_species = trainset.n_species

    torch.manual_seed(42)
    mbpert = MBPert(n_species)

    def loss_fn(y_hat, y, mbpert):
        # Compute loss (MSE + reg)
        criterion = torch.nn.MSELoss()
        loss = criterion(y_hat, y)
        loss = loss + reg_loss_interaction(mbpert.A) + \
                      reg_loss_r(mbpert.r) + \
                      reg_loss_eps(mbpert.eps)
        return loss

    optimizer = torch.optim.Adam(mbpert.parameters())

    # Model training
    mbp = MBP(mbpert, loss_fn, optimizer)
    mbp.set_loaders(trainloader, testloader)
    mbp.set_tensorboard('sim_parallel_pert')

    mbp.train(n_epochs=200, verbose=True) # 500

    # Visualize results
    # Plot exact and estimated parameters
    plot_error_A(mbp, "data/A.txt")
    plot_r_eps(mbp, "data/r.txt", "data/eps.txt")

    # Plot predicted vs true steady states in test set
    plot_ss_test(mbp)

    # Save checkpoint
    mbp.save_checkpoint("checkpoint/sim_parallel_pert.pth")

    # Load checkpoint
    # torch.manual_seed(42)
    # mbpert = MBPert(n_species)
    # mbp = MBP(mbpert, loss_fn, optimizer)
    # mbp.load_checkpoint("checkpoint/sim_parallel_pert.pth")




