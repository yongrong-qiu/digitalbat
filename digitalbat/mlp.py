import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import copy
from sklearn.metrics import average_precision_score

from .utils import corr


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

        # self.layers.append(ELU1())
        self.layers.append(nn.Identity())
    
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
    pos_weight=1.0, # torch tensor, for nn.BCEWithLogitsLoss()
    verbose=False, # not display the progress
):
    """
    Model training
    """

    model = model.to(device)

    # loss function and optimizer
    # loss_fn = nn.PoissonNLLLoss(log_input=False)  # binary cross entropy
    pos_weight = pos_weight.to(device)  # Assuming spike fraction ~5%, weight the positive class more, 95% / 5% = 19
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # loss_fn = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    batch_start = np.arange(0, len(X_train), batch_size)
 
    # Hold the best model
    best_auprc = - np.inf   # the best area under the precision-recall curve, init to negative infinity
    best_weights = None
    auprc_vals = []
 
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
        # if epoch % 5 == 0:
        model.eval()
        X_val = X_val.float().to(device)
        y_val = y_val.float().to(device)
        y_pred = model(X_val)
        #
        # logl =  -loss_fn(y_pred, y_val).detach().cpu().data.numpy() # (y_pred.round() == y_val).float().mean().detach().cpu().data.numpy()
        #
        # auprc_score = -loss_fn(y_pred, y_val).detach().cpu().numpy()
        #
        # y_true = y_val.detach().cpu().numpy()
        # y_pred = torch.sigmoid(y_pred).detach().cpu().numpy()
        # auprc_score = np.mean(auprc_per_unit(y_true, y_pred))
        #
        # auprc_score = average_precision_score(y_true.flatten(), y_pred.flatten())
        #
        auprc_score = corr(y_val.detach().cpu().numpy(), torch.sigmoid(y_pred).detach().cpu().numpy().round(), axis=0)
        auprc_score = np.nanmean(auprc_score)
        auprc_vals.append(auprc_score)
        if auprc_score > best_auprc:
            best_auprc = auprc_score
            best_weights = copy.deepcopy(model.state_dict())

    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_auprc, best_weights, auprc_vals


def auprc_per_unit(
    y_true_np: np.ndarray,  # shape: (batch_size, num_units)
    y_pred_np: np.ndarray   # shape: (batch_size, num_units)
) -> np.ndarray:
    """
    Compute the Area Under the Precision-Recall Curve (AUPRC) for each unit (column) independently.

    Parameters:
    ----------
    y_true_np : np.ndarray
        A 2D NumPy array of binary ground truth labels (0 or 1),
        with shape (batch_size, number_of_units).

    y_pred_np : np.ndarray
        A 2D NumPy array of predicted probabilities (after sigmoid),
        with shape (batch_size, number_of_units).

    Returns:
    -------
    np.ndarray
        A 1D NumPy array of AUPRC scores for each unit.
        Units with no positive examples in `y_true_np` will have `np.nan`.
    """

    num_units = y_true_np.shape[1]
    auprc_scores = []

    for i in range(num_units):
        y_true_unit = y_true_np[:, i]
        y_pred_unit = y_pred_np[:, i]

        if np.sum(y_true_unit) == 0:
            # AUPRC is undefined when there are no positive samples
            auprc_scores.append(np.nan)
        else:
            score = average_precision_score(y_true_unit, y_pred_unit)
            auprc_scores.append(score)

    return np.array(auprc_scores)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()
