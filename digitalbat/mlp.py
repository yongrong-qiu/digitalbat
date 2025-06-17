import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import copy
from sklearn.metrics import average_precision_score
import math

from typing import List, Optional

from .utils import corr


def data_prep_for_mlp(all_input, all_label, train_num, valid_num, use_fourier=False):
    """
    Data preprocessing for MLP.

    Args:
        all_input: np.ndarray, shape (n_samples, n_features)
            Full input data. Positions assumed to be in columns 1 to 3.
        all_label: np.ndarray, shape (n_samples, n_outputs)
            Corresponding labels.
        train_num: int
            Number of training samples.
        valid_num: int
            Number of validation samples.
        use_fourier: bool, default=False
            If True, normalize positions (columns 1–3) to [0, 1] range instead of std.

    Returns:
        Tuple of torch.Tensors:
            train_input, valid_input, test_input,
            train_label, valid_label, test_label
    """

    # Shuffle the data
    np.random.seed(100)
    data_indices = np.random.permutation(all_input.shape[0])
    all_input = all_input[data_indices]
    all_label = all_label[data_indices]

    # Split data
    train_input = all_input[:train_num]
    train_label = all_label[:train_num]
    valid_input = all_input[train_num:train_num + valid_num]
    valid_label = all_label[train_num:train_num + valid_num]
    test_input = all_input[train_num + valid_num:]
    test_label = all_label[train_num + valid_num:]

    # Normalize input
    if use_fourier:
        # Normalize position dims (1:4) to [0, 1]
        pos_min = train_input[:, 1:4].min(axis=0)
        pos_max = train_input[:, 1:4].max(axis=0)
        pos_range = pos_max - pos_min + 1e-8  # avoid div by zero

        def normalize_pos(x):
            x[:, 1:4] = (x[:, 1:4] - pos_min) / pos_range
            return x

        train_input = normalize_pos(train_input)
        valid_input = normalize_pos(valid_input)
        test_input = normalize_pos(test_input)

        # Standardize other dims (0, 4:10)
        keep_dims = [0] + list(range(4, all_input.shape[1]))
        train_std = np.std(train_input[:, keep_dims], axis=0, keepdims=True)
        train_input[:, keep_dims] /= train_std
        valid_input[:, keep_dims] /= train_std
        test_input[:, keep_dims] /= train_std
    else:
        # Standardize all dimensions
        train_std = np.std(train_input, axis=0, keepdims=True)
        train_input /= train_std
        valid_input /= train_std
        test_input /= train_std

    # Convert to torch tensors
    train_input = torch.from_numpy(train_input).float()
    valid_input = torch.from_numpy(valid_input).float()
    test_input = torch.from_numpy(test_input).float()
    train_label = torch.from_numpy(train_label).float()
    valid_label = torch.from_numpy(valid_label).float()
    test_label = torch.from_numpy(test_label).float()

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


class Sin(nn.Module):
    """
    The sine of the input
    """

    def forward(self, x):
        return torch.sin(x)


class MLP(nn.Module):
    """
    Flexible MLP model with optional Fourier features for spatial inputs.

    Parameters
    ----------
    input_dim : int
        Original dimensionality of the input (e.g., 10).

    output_dim : int
        Dimensionality of the output.

    latent_dims_list : List[int]
        List of hidden layer sizes. If empty, the model becomes a single linear layer.

    latent_activation : nn.Module
        Activation function applied after each hidden layer (e.g., nn.ReLU()).

    final_activation : nn.Module
        Activation function applied after the output layer (e.g., nn.Identity()).

    use_fourier : bool
        Whether to apply Fourier features to position inputs (dimensions 1–3).

    fourier_features_dim : int
        Number of Fourier frequencies for each positional input dimension.

    fourier_scale : float
        Scale factor for the Fourier feature frequencies.
    """
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 10,
        latent_dims_list: List[int] = [100, 100],
        latent_activation: nn.Module = nn.ReLU(),
        final_activation: nn.Module = nn.Identity(),
        use_fourier: bool = False,
        fourier_features_dim: int = 16,
        fourier_scale: float = 10.0,
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.use_fourier = use_fourier
        self.fourier_features_dim = fourier_features_dim
        self.fourier_scale = fourier_scale

        # Number of Fourier features added per input dimension
        fourier_expand = 2 * fourier_features_dim if use_fourier else 0

        # Dimensions 1-3 (positions) → add Fourier features
        num_fourier_input_dims = 3 if use_fourier else 0
        self.total_fourier_dim = num_fourier_input_dims * fourier_expand

        self.input_dim_no_fourier = input_dim
        self.effective_input_dim = input_dim - num_fourier_input_dims + self.total_fourier_dim

        # Create learnable layers
        self.layers = nn.ModuleList()

        if latent_dims_list:
            self.layers.append(nn.Linear(self.effective_input_dim, latent_dims_list[0]))
            self.layers.append(nn.BatchNorm1d(latent_dims_list[0]))
            self.layers.append(latent_activation)
            self.layers.append(nn.Dropout(dropout_prob))
            for i in range(len(latent_dims_list) - 1):
                self.layers.append(nn.Linear(latent_dims_list[i], latent_dims_list[i + 1]))
                self.layers.append(nn.BatchNorm1d(latent_dims_list[i + 1]))
                self.layers.append(latent_activation)
                self.layers.append(nn.Dropout(dropout_prob))
            self.layers.append(nn.Linear(latent_dims_list[-1], output_dim))
        else:
            self.layers.append(nn.Linear(self.effective_input_dim, output_dim))

        self.layers.append(final_activation)

        # Register Fourier frequencies as buffers
        if self.use_fourier:
            freqs = torch.exp2(torch.arange(self.fourier_features_dim).float()) * self.fourier_scale
            self.register_buffer("fourier_freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, input_dim)

        Returns
        -------
        torch.Tensor
            Shape (batch_size, output_dim)
        """
        if self.use_fourier:
            # Extract spatial dimensions (1–3)
            pos = x[:, 1:4]  # shape (B, 3)
            freqs = self.fourier_freqs[None, None, :]  # shape (1, 1, F)
            pos = pos[:, :, None] * freqs  # shape (B, 3, F)

            pos_enc = torch.cat([torch.sin(pos), torch.cos(pos)], dim=-1)  # (B, 3, 2F)
            pos_enc = pos_enc.flatten(1)  # shape (B, 3*2F)

            # Concatenate: [flying | pos_enc | accel/gyro]
            x = torch.cat([x[:, :1], pos_enc, x[:, 4:]], dim=-1)

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
