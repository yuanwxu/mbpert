import numpy as np
import pandas as pd
import datetime
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

plt.style.use('fivethirtyeight')

class MBP(object):
    def __init__(self, model, loss_fn, optimizer, ts_mode=False):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Let's send the model to the specified device right away
        self.model.to(self.device)

        # Set mode: default is "parallel perturbation", alternative is
        # "sequential perturbation" for time series data with time-dependent perturbations
        self.ts_mode = ts_mode

        # These attributes are not informed at the moment of creation, 
        # we keep them None
        self.train_loader = None
        self.val_loader = None
        self.writer = None
        
        # These attributes are going to be computed internally
        self.losses = []
        self.val_losses = []
        self.total_epochs = 0

        # Creates the train_step function for our model, 
        # loss function and optimizer
        self.train_step_fn = self._make_train_step_fn()
        # Creates the val_step function for our model and loss
        self.val_step_fn = self._make_val_step_fn()

    def to(self, device):
        # This method allows the user to specify a different device
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Couldn't send it to {device}, sending it to {self.device} instead.")
            self.model.to(self.device)

    def set_loaders(self, train_loader, val_loader=None):
        # This method allows the user to define which train_loader (and val_loader, optionally) to use
        # Both loaders are then assigned to attributes of the class
        # So they can be referred to later
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_tensorboard(self, name, folder='runs'):
        # This method allows the user to define a SummaryWriter to interface with TensorBoard
        suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')

    def _make_train_step_fn(self):
        # This method does not need ARGS... it can refer to
        # the attributes: self.model, self.loss_fn and self.optimizer
        
        # Builds function that performs a step in the train loop
        def perform_train_step_fn(x, y):
            # Sets model to TRAIN mode
            self.model.train()

            # Step 1 - forward pass
            if self.ts_mode:
                yhat = []
                for x0, t in zip(*x):
                    y_t = self.model(x0, t)
                    yhat.append(y_t)

                yhat = torch.cat(yhat, dim=0)
            else: 
                yhat = self.model(*x)
            # Step 2 - Computes the loss
            loss = self.loss_fn(yhat, y, self.model)
            # Step 3 - Computes gradients 
            loss.backward()
            # Step 4 - Updates parameters using gradients and the learning rate
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Returns the loss
            return loss.item()

        # Returns the function that will be called inside the train loop
        return perform_train_step_fn
    
    def _make_val_step_fn(self):
        # Builds function that performs a step in the validation loop
        def perform_val_step_fn(x, y):
            # Sets model to EVAL mode
            self.model.eval()

            if self.ts_mode:
                yhat = []
                for x0, t in zip(*x):
                    y_t = self.model(x0, t)
                    yhat.append(y_t)

                yhat = torch.cat(yhat, dim=0)
            else: 
                yhat = self.model(*x)
            loss = self.loss_fn(yhat, y, self.model)
           
            return loss.item()

        return perform_val_step_fn
            
    def _mini_batch(self, validation=False):
        # The mini-batch can be used with both loaders
        # The argument `validation`defines which loader and 
        # corresponding step function is going to be used
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn

        if data_loader is None:
            return None
 
        mini_batch_losses = []
        for data in data_loader:
            if self.ts_mode: # sequential perturbation
                # Unpack input batch
                (x0_batch, t_batch), y_batch = data
                x0_batch = x0_batch.to(self.device)
                t_batch = t_batch.to(self.device)
                x_batch = (x0_batch, t_batch)
                y_batch = torch.ravel(y_batch).to(self.device)

            else: # parallel perturbation
                # Unpack input batch
                (x0_batch, p_batch), y_batch = data 

                # Get the input that model() accepts
                x0_batch = torch.ravel(x0_batch).to(self.device)
                p_batch = torch.t(p_batch).to(self.device)
                x_batch = (x0_batch, p_batch)

                # and responses
                y_batch = torch.ravel(y_batch).to(self.device)

            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

        loss = np.mean(mini_batch_losses)
        return loss

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False    
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def train(self, n_epochs, seed=42, verbose=False):
        # To ensure reproducibility of the training process
        if seed:
            self.set_seed(seed)

        for epoch in range(n_epochs):
            # Keeps track of the numbers of epochs
            # by updating the corresponding attribute
            self.total_epochs += 1

            # inner loop
            # Performs training using mini-batches
            loss = self._mini_batch(validation=False)
            self.losses.append(loss)

            # VALIDATION
            # no gradients in validation!
            with torch.no_grad():
                # Performs evaluation using mini-batches
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

            # If a SummaryWriter has been set...
            if self.writer:
                scalars = {'training': loss}
                if val_loss is not None:
                    scalars.update({'validation': val_loss})
                # Records both losses for each epoch under the main tag "loss"
                self.writer.add_scalars(main_tag='loss',
                                        tag_scalar_dict=scalars,
                                        global_step=epoch)

            # If verbose=True, will print train and validation losses per mini-batch
            if verbose:
                if self.val_loader is None:
                    print(f'Epoch: {epoch + 1} | Loss train: {loss:.4f} ')
                else:
                    print(f'Epoch: {epoch + 1} | Loss train/test: {loss:.4f}/{val_loss:.4f} ')

        if self.writer:
            # Closes the writer
            self.writer.close()

    def save_checkpoint(self, filename):
        # Builds dictionary with all elements for resuming training
        checkpoint = {'epoch': self.total_epochs,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'loss': self.losses,
                      'val_loss': self.val_losses}

        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        # Loads dictionary
        checkpoint = torch.load(filename)

        # Restore state for model and optimizer
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']

        self.model.train() # always use TRAIN for resuming training   

    def predict(self, x0, p):
        if self.ts_mode:
            return None
        # Set is to evaluation mode for predictions
        self.model.eval() 
        # Takes numpy input (long initial state vector and species-by-condition
        # perturbation matrix) and make it a float tensor
        x0_tensor = torch.as_tensor(x0).float()
        p_tensor = torch.as_tensor(p, dtype=torch.bool)
        y_hat_tensor = self.model(x0_tensor.to(self.device), p_tensor.to(self.device))
        # Set it back to train mode
        self.model.train()
        # Detaches it, brings it to CPU and back to Numpy
        return y_hat_tensor.detach().cpu().numpy()

    def predict_ts(self, x0, t):
        if not self.ts_mode:
            return None

        self.model.eval()
        # Takes numpy input (initial species absolute abundances and final time point t)
        x0_tensor = torch.as_tensor(x0).float()
        t_tensor = torch.as_tensor(t).float()
        y_hat_tensor = self.model(x0_tensor.to(self.device), t_tensor.to(self.device))
        self.model.train()

        return y_hat_tensor.detach().cpu().numpy()


    # Make predictions on validation set, return a two-column data frame 
    # with predicted and true steady states
    def predict_val(self):
        if self.ts_mode or self.val_loader is None:
            return None

        self.model.eval()

        val_true = []
        val_pred = []
        with torch.no_grad():
            for testdata in self.val_loader:
                (x0, p), responses = testdata
                x0 = torch.ravel(x0)
                p = torch.t(p)
                x_ss = torch.ravel(responses).numpy()
                x_pred = self.predict(x0, p)

                val_true.append(x_ss)
                val_pred.append(x_pred)
        
        val_true = np.concatenate(val_true)
        val_pred = np.concatenate(val_pred)
        x_df = pd.DataFrame(data={'pred': val_pred, 'true': val_true})

        self.model.train()
        return x_df

    # Make predictions on validation set, for time series data, return a
    # data frame of the (scaled) time points, predicted state at the corresponding
    # time points and the observed, true state.
    def predict_val_ts(self):
        if not self.ts_mode or self.val_loader is None:
            return None
        
        self.model.eval()

        pred_val = {}
        df_val = []
        with torch.no_grad():
            for testdata in self.val_loader:
                (x0_batch, t_batch), y_batch = testdata
                for x0, t, y in zip(x0_batch, t_batch, y_batch):
                    y_t = self.predict_ts(x0, t)
                    pred_val['species_id'] = range(len(y_t))
                    pred_val['t'] = t.item()
                    pred_val['pred'] = y_t
                    pred_val['true'] = y
                    df_val.append(pd.DataFrame(pred_val))
        df_val = pd.concat(df_val)

        self.model.train()
        return df_val

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Loss', c='b')
        plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        return fig

    def add_graph(self):
        # Fetches a single mini-batch so we can use add_graph
        if self.train_loader and self.writer:
            if self.ts_mode:
                (x0_sample, t_sample), _ = next(iter(self.train_loader))
                for x0, t in zip(x0_sample, t_sample):
                    self.writer.add_graph(self.model, (x0.to(self.device, t.to(self.device))))
            else: 
                (x0_sample, p_sample), _ = next(iter(self.train_loader))
                x0_sample = torch.ravel(x0_sample).to(self.device)
                p_sample = torch.t(p_sample).to(self.device)
                self.writer.add_graph(self.model, (x0_sample, p_sample))

