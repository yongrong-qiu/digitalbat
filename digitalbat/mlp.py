import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import copy


def data_prep_for_mlp(all_input, all_label, train_num, valid_num):
    """
    Data preprocessing for MLP.
    Args:
        all_input: 2D numpy array with a shape of (num_of_data_points, num_of_input_features)
        all_label: 2D numpy array with a shape of (num_of_data_points, num_of_output_dimensions)
        train_num: number of data points for training
        valid_num: number of data points for validation

    Return:
        Pytorch tensor for training models
        
    """

    # shuffle the data 
    np.random.seed(100)
    data_indice = np.random.choice(all_input.shape[0], all_input.shape[0], replace=False)
    all_input = all_input[data_indice, :]
    all_label = all_label[data_indice, :]

    # split the data to train, validation and test
    train_input = all_input[:train_num, :]
    train_label = all_label[:train_num, :]
    valid_input = all_input[train_num : train_num + valid_num, :]
    valid_label = all_label[train_num : train_num + valid_num, :]
    test_input = all_input[train_num + valid_num :, :]
    test_label = all_label[train_num + valid_num :, :]

    # numpy to torch
    train_input = torch.from_numpy(train_input).float()
    train_label = torch.from_numpy(train_label).float()
    valid_input = torch.from_numpy(valid_input).float()
    valid_label = torch.from_numpy(valid_label).float()
    test_input = torch.from_numpy(test_input).float()
    test_label = torch.from_numpy(test_label).float()

    # normalize input
    train_input_std = torch.std(train_input, dim=0, keepdim=True)
    train_input = train_input/train_input_std
    valid_input = valid_input/train_input_std
    test_input = test_input/train_input_std

    return train_input, valid_input, test_input, train_label, valid_label, test_label
    
    
class ELU1(nn.Module):
    """
    Adapted from <https://github.com/sinzlab/neuralpredictors/blob/main/neuralpredictors/layers/activations.py>
    Elu activation function shifted by 1 to ensure that the
    output stays positive. That is:
    Elu1(x) = Elu(x) + 1
    """

    @staticmethod
    def elu1(x, inplace=False, eps=0.0):
        return F.elu(x, inplace=inplace) + 1.0 + eps

    def forward(self, x, inplace=False, eps=0):
        return ELU1.elu1(x, inplace, eps)
    

class MLP(nn.Module):
    """
    MLP model with dynamically created linear layers based on input_dim and latent_dims_list.
    """
    def __init__(self, input_dim=10, output_dim=10, latent_dims_list=[100, 100], linear_flag=False):
        super().__init__()

        # Create a list to hold all the layers
        self.layers = nn.ModuleList()
        
        if len(latent_dims_list)>0:
            # Add the first layer (input_dim -> first latent dimension)
            self.layers.append(nn.Linear(input_dim, latent_dims_list[0]))
            self.layers.append(nn.ReLU()) if not linear_flag else self.layers.append(nn.Identity())
            
            # Add intermediate layers (latent_dim[i] -> latent_dim[i+1])
            if len(latent_dims_list)>1:
                for ii in range(len(latent_dims_list) - 1):
                    self.layers.append(nn.Linear(latent_dims_list[ii], latent_dims_list[ii+1]))
                    self.layers.append(nn.ReLU()) if not linear_flag else self.layers.append(nn.Identity())
            
            # Add the final output layer (last latent dimension -> output_dim)
            self.layers.append(nn.Linear(latent_dims_list[-1], output_dim))

        else: # no hidden layer
            self.layers.append(nn.Linear(input_dim, output_dim))

        self.layers.append(ELU1())
    
    def forward(self, x):
        # Pass the input through all layers
        for layer in self.layers:
            x = layer(x)
        return x
    

def model_train(
    model, 
    X_train, # torch tensor
    y_train, # torch tensor
    X_val, # torch tensor
    y_val, # torch tensor
    device,
    n_epochs=200, # # number of epochs to run
    batch_size=100,  # size of each batch
    lr=0.0001, # learning rate
    verbose=False, # not display the progress
):
    """
    Model training
    """

    model = model.to(device)

    # loss function and optimizer
    loss_fn = nn.PoissonNLLLoss(log_input=False)  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=lr)
    batch_start = np.arange(0, len(X_train), batch_size)
 
    # Hold the best model
    best_logl = - np.inf   # the best log likelihood, init to negative infinity
    best_weights = None
    logl_vals = []
 
    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=not verbose) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size].float().to(device)
                y_batch = y_train[start:start+batch_size].float().to(device)
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                # acc = (y_pred.round() == y_batch).float().mean().detach().cpu().data.numpy()
                # bar.set_postfix(
                #     loss=float(loss).detach().cpu().data.numpy(),
                #     acc=acc
                # )
        # evaluate accuracy at end of each epoch
        model.eval()
        X_val = X_val.float().to(device)
        y_val = y_val.float().to(device)
        y_pred = model(X_val)
        logl =  -loss_fn(y_pred, y_val).detach().cpu().data.numpy() # (y_pred.round() == y_val).float().mean().detach().cpu().data.numpy()
        logl_vals.append(logl)
        if logl > best_logl:
            best_logl = logl
            best_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_logl, best_weights, logl_vals