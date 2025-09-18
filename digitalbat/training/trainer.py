import numpy as np
import torch
from torch import nn
import torch.optim as optim
import tqdm
import copy
from typing import Callable

from ..utility.utils import corr
from ..models.layers import FocalLoss, BCEFocalCombinedLoss


def model_train_mlp(
    model, 
    X_train, # torch tensor
    y_train, # torch tensor
    X_val, # torch tensor
    y_val, # torch tensor
    device,
    n_epochs=200, # # number of epochs to run
    batch_size=100,  # size of each batch
    lr=0.0001, # learning rate
    pos_weight=torch.tensor(1.0), # torch tensor, for nn.BCEWithLogitsLoss()
    loss_fn: nn.Module | None = None, 
    valid_fn: Callable[[np.ndarray, np.ndarray], float] | None = None, # validation during training, larger number, better performance
    verbose=False, # not display the progress
):
    """
    Model training
    Arguments:
        valid_fn: validation during training, larger number, better performance
                  e.g., valid_fn=lambda y_true, y_pred: np.nanmean(corr(y_true, y_pred, axis=0))
                  y_true.shape: (N, spiking_unit_number)
    """

    model = model.to(device)

    # loss function and optimizer
    # loss_fn = nn.PoissonNLLLoss(log_input=False)  # binary cross entropy
    # loss_fn = FocalLoss(alpha = 0.9, gamma = 2.0) 
    # loss_fn = BCEFocalCombinedLoss(
    #     alpha=0.25,     # aggressive positive emphasis
    #     gamma=2.0,
    #     focal_weight=0.99,   # focal 60%, BCE 40%
    #     pos_weight=pos_weight,  # optional, helps with imbalance
    # )
    if loss_fn is None:
        pos_weight = pos_weight.to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Default if not provided
    loss_fn = loss_fn.to(device)  # Make sure the loss function is on the correct device

    # Default validation function: corr
    def default_valid_fn(y_true, y_pred):
        """
        y_true: 0 or 1.0
        y_pred: logit
        """
        sigmoid = lambda xx: 1/(1 + np.exp(-xx))
        return np.nanmean(corr(y_true, sigmoid(y_pred).round(), axis=0))
    valid_fn = valid_fn or default_valid_fn

    optimizer = optim.Adam(model.parameters(), lr=lr)
    batch_start = np.arange(0, len(X_train), batch_size)
 
    # Hold the best model
    best_auprc = - np.inf   # the best area under the precision-recall curve, init to negative infinity
    best_weights = None
    val_scores = []
 
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
                loss = loss_fn(y_pred, y_batch) # + torch.sigmoid(y_pred).mean() * 10.0
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
        # score = corr(y_val.detach().cpu().numpy(), torch.sigmoid(y_pred).detach().cpu().numpy().round(), axis=0)
        # score = np.nanmean(score)
        y_val_np = y_val.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
        score = valid_fn(y_val_np, y_pred_np)
        val_scores.append(score)
        if score > best_auprc:
            best_auprc = score
            best_weights = copy.deepcopy(model.state_dict())

    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_auprc, best_weights, val_scores


def bat_mlp_trainer(
    model, 
    dataloaders, 
    device,
    n_epochs=200, # # number of epochs to run
    lr=0.0001, # learning rate
    loss_fn: nn.Module | None = None, 
    valid_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None, 
    # verbose=False, # not display the progress
):
    """
    Model training
    Arguments:
        valid_fn: validation during training, larger number, better performance
                  Compared to the valid_fn used for model_train_mlp(), here we return an array of correlation values for all units, without average.
                  e.g., valid_fn=lambda y_true, y_pred: corr(y_true, y_pred, axis=0)
                  y_true.shape: (N, spiking_unit_number)
    """

    model = model.to(device)

    if loss_fn is None:
        pos_weight = pos_weight.to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Default if not provided
    loss_fn = loss_fn.to(device)  # Make sure the loss function is on the correct device

    # Default validation function: corr
    def default_valid_fn(y_true, y_pred):
        """
        y_true: 0 or 1.0
        y_pred: logit
        """
        sigmoid = lambda xx: 1/(1 + np.exp(-xx))
        return np.nanmean(corr(y_true, sigmoid(y_pred).round(), axis=0))
    valid_fn = valid_fn or default_valid_fn

    optimizer = optim.Adam(model.parameters(), lr=lr)
 
    # Hold the best model
    best_score = - np.inf   # the best area under the precision-recall curve, init to negative infinity
    best_weights = None
    val_scores = []
 
    for epoch in tqdm.tqdm(range(n_epochs), desc="Training"):
        model.train()
        for session_name, session_data in dataloaders['train'].items():
            # print(session_name, session_data)
            for batch_id, batch in enumerate(session_data):
                # print(batch[0].shape, batch[1].shape) # (N, behavior_feature), (N, spiking_unit_number)
                # take a batch
                X_batch = batch[0].to(device)
                y_batch = batch[1].to(device)
                # forward pass
                y_pred = model(session_name, X_batch)
                loss = loss_fn(y_pred, y_batch) # + torch.sigmoid(y_pred).mean() * 10.0
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()

        # evaluate accuracy at end of each epoch
        model.eval()
        temp_scores_list = []
        for session_name, session_data in dataloaders['valid'].items():
            # print(session_name, session_data)
            for batch_id, batch in enumerate(session_data):
                X_val = batch[0].to(device)
                y_val = batch[1].to(device)
                y_pred = model(session_name, X_val)
                y_val_np = y_val.detach().cpu().numpy()
                y_pred_np = y_pred.detach().cpu().numpy()
                temp_scores = valid_fn(y_val_np, y_pred_np) # an array of scores for spiking units in one batch data within one session of data
                temp_scores_list.append(temp_scores)
        val_score = np.nanmean(np.hstack(temp_scores_list))
        val_scores.append(val_score)
        if val_score > best_score:
            best_score = val_score
            best_weights = copy.deepcopy(model.state_dict())

    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_score, best_weights, val_scores