import numpy as np
import torch
from torch import nn
import torch.optim as optim
import tqdm
import copy
from typing import Callable
from transformers import get_cosine_schedule_with_warmup

from ..utility.utils import corr, poisson_fde_population, round_robin_session_batches
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

    optimizer = optim.AdamW(model.parameters(), lr=lr)
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
    valid_stat='mean', # after computing validation scores for all neurons, we calculate a single number (statistics) as 'val_score' for these neurons
    # verbose=False, # not display the progress
):
    """
    Model training
    Arguments:
        valid_fn: validation during training, larger number, better performance
                  Compared to the valid_fn used for model_train_mlp(), here we return an array of correlation values for all units, without average.
                  e.g., valid_fn=lambda y_true, y_pred: corr(y_true, y_pred, axis=0)
                  e.g., valid_fn=lambda y_true, y_pred: poisson_fde_population(y_true, y_pred)
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

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Hold the best model
    best_score = - np.inf   # the best area under the precision-recall curve, init to negative infinity
    best_weights = None
    val_scores = []
 
    for epoch in tqdm.tqdm(range(n_epochs), desc="Training"):
        model.train()
        # Round-robin per-session batches: see `round_robin_session_batches`.
        for session_name, batch in round_robin_session_batches(dataloaders['train']):
            # batch[0]: (N, behavior_feature)  batch[1]: (N, spiking_unit_number)
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
        if valid_stat == 'mean':
            val_score = np.nanmean(np.hstack(temp_scores_list))
        elif valid_stat == 'median':
            val_score = np.nanmedian(np.hstack(temp_scores_list))
        val_scores.append(val_score)
        if val_score > best_score:
            best_score = val_score
            best_weights = copy.deepcopy(model.state_dict())

    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_score, best_weights, val_scores

def bat_mlp_behavior_trainer(
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
    Model training for the model Bat_MLP_Behavior()
    Arguments:
        valid_fn: validation during training, larger number, better performance
                  Compared to the valid_fn used for bat_mlp_trainer(), here we return an array of minus squared difference values for each time point and behavior_feature, without average.
                  e.g., valid_fn=lambda y_true, y_pred: -(y_true - y_pred)**2
                  y_true.shape: (N, behavior_feature)
                  Here we add "minus" as we want to use model with higher score
    """

    model = model.to(device)
    loss_fn = loss_fn.to(device)  # Make sure the loss function is on the correct device

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Hold the best model
    best_score = - np.inf
    best_weights = None
    val_scores = []
 
    for epoch in tqdm.tqdm(range(n_epochs), desc="Training"):
        model.train()
        # Round-robin per-session batches: see `round_robin_session_batches`.
        for session_name, batch in round_robin_session_batches(dataloaders['train']):
            # batch[0]: (N, behavior_feature)  batch[1]: (N, spiking_unit_number)
            # use spikes as model input, behavior as target
            X_batch = batch[1].to(device)
            y_batch = batch[0].to(device)
            # forward pass
            y_pred = model(session_name, X_batch)
            loss = loss_fn(y_pred, y_batch)
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
                X_val = batch[1].to(device) # # use spikes as model input
                y_val = batch[0].to(device)
                y_pred = model(session_name, X_val)
                y_val_np = y_val.detach().cpu().numpy()
                y_pred_np = y_pred.detach().cpu().numpy()
                temp_scores = valid_fn(y_val_np, y_pred_np) # an array of scores for spiking units in one batch data within one session of data
                temp_scores_list.append(temp_scores)
        val_score = np.nanmean(np.vstack(temp_scores_list))
        val_scores.append(val_score)
        if val_score > best_score:
            best_score = val_score
            best_weights = copy.deepcopy(model.state_dict())

    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_score, best_weights, val_scores

def bat_former_trainer(
    model, 
    dataloaders, 
    device,
    is_causal: bool = False, 
    is_bigbird: bool = False,
    attention_window_size=3, # bigbird style sparse attention
    num_global=2, # bigbird style sparse attention
    num_random=1, # bigbird style sparse attention
    n_epochs=200, # number of epochs to run
    use_cosine_scheduler: bool = False, # use cosine scheduler or not
    n_warmup=None, # number of warmup steps for learning rate
    lr=0.0001, # learning rate
    weight_decay=0.0, # L2 regularization
    loss_fn: nn.Module | None = None, 
    valid_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None, 
    valid_stat='mean', # after computing validation scores for all neurons, we calculate a single number (statistics) as 'val_score' for these neurons
    early_stopping_patience: int | None = None, # stop if no improvement for this many epochs
    save_model: bool = False, # whether to save model or not
    model_name: str = 'best_model_weights.pth',
    # verbose=False, # not display the progress
):
    """
    Training transfomer models for predicting neural activity
    Arguments:
        valid_fn: validation during training, larger number, better performance
                  Compared to the valid_fn used for model_train_mlp(), here we return an array of correlation values for all units, without average.
                  e.g., valid_fn=lambda y_true, y_pred: corr(y_true, y_pred, axis=0)
                  e.g., valid_fn=lambda y_true, y_pred: poisson_fde_population(y_true, y_pred)
                  y_true.shape: (N, unit_number)
    """

    model = model.to(device)
    loss_fn = loss_fn.to(device)  # Make sure the loss module is on the correct device

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if use_cosine_scheduler:
        if n_warmup is None:
            n_warmup = int(0.1 * n_epochs)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=n_warmup,
            num_training_steps=n_epochs,
            num_cycles=1.5,
        )
 
    # Hold the best model
    best_score = - np.inf   # correlation coefficient, larger scores indicate better model
    best_weights = None
    val_scores = []
    epochs_without_improvement = 0

    for epoch in tqdm.tqdm(range(n_epochs), desc="Training"):
        model.train()
        # Round-robin per-session batches: see `round_robin_session_batches`.
        for session_name, batch in round_robin_session_batches(dataloaders['train']):
            # behavior (N, window_size, behavior_feature)
            # spike    (N, window_size, unit_number)
            # mask     (N, window_size)
            X_batch = batch.behavior.to(device)
            y_batch = batch.spike.to(device)
            mask_batch = batch.mask.to(device)
            # forward pass
            y_pred = model(
                session_name,
                X_batch,
                mask_batch,
                is_causal=is_causal,
                is_bigbird=is_bigbird,
                attention_window_size=attention_window_size,
                num_global=num_global,
                num_random=num_random,
            )
            loss = loss_fn(y_pred[mask_batch == 1.0], y_batch[mask_batch == 1.0])
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            if use_cosine_scheduler:
                scheduler.step()

        # evaluate accuracy at end of each epoch
        model.eval()
        temp_scores_list = []
        for session_name, session_data in dataloaders['valid'].items():
            # print(session_name, session_data)
            y_val_np_list, y_pred_np_list = [], []
            for batch_id, batch in enumerate(session_data):
                X_val = batch.behavior.to(device)
                y_val = batch.spike.to(device)
                mask_val = batch.mask.to(device)
                y_pred = model(
                    session_name, 
                    X_val, 
                    mask_val, 
                    is_causal=is_causal, 
                    is_bigbird=is_bigbird,
                    attention_window_size=attention_window_size,
                    num_global=num_global,
                    num_random=num_random,
                )
                y_val_np = y_val.detach()[mask_val == 1.0].cpu().numpy() # compute scores only on valid timesteps
                y_pred_np = y_pred.detach()[mask_val == 1.0].cpu().numpy()
                # y_pred.shape: torch.Size([48, 70, 95]), y_pred_np.shape: (1732, 95) ... 95: number of neurons, 48: batch_size
                y_val_np_list.append(y_val_np)
                y_pred_np_list.append(y_pred_np)
            y_val_np = np.vstack(y_val_np_list)
            y_pred_np = np.vstack(y_pred_np_list)
            temp_scores = valid_fn(y_val_np, y_pred_np) # an array of scores for spiking units in one batch data within one session of data
            temp_scores_list.append(temp_scores)
        if valid_stat == 'mean':
            val_score = np.nanmean(np.hstack(temp_scores_list))
        elif valid_stat == 'median':
            val_score = np.nanmedian(np.hstack(temp_scores_list))
        val_scores.append(val_score)
        if val_score > best_score:
            best_score = val_score
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {early_stopping_patience} epochs)")
                break

    # save model, restore model and return best accuracy
    if save_model:
        torch.save(model.state_dict(), model_name)
    model.load_state_dict(best_weights)
    return best_score, best_weights, val_scores


def finetune_readout_trainer(
    model,
    dataloaders,
    device,
    target_session: str,
    is_causal: bool = False,
    is_bigbird: bool = False,
    attention_window_size=3,
    num_global=2,
    num_random=1,
    n_epochs=200,
    use_cosine_scheduler: bool = False,
    n_warmup=None,
    lr=0.0001,
    weight_decay=0.0,
    loss_fn: nn.Module | None = None,
    valid_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    valid_stat='mean',
    early_stopping_patience: int | None = None,
    save_model: bool = False,
    model_name: str = 'best_finetune_weights.pth',
):
    """
    Fine-tune only `model.readouts[target_session]` while the rest of the model stays frozen.

    Expects callers to freeze non-target params (requires_grad=False) beforehand. This trainer
    additionally puts the whole model in .eval() every epoch, then flips ONLY the target readout
    to .train(), which disables dropout in the transformer core and other readouts during fine-tune.

    Iterates only over dataloaders['train'|'valid'][target_session].
    Returns (best_score, best_weights, val_scores) as in bat_former_trainer.
    """
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    target_params = list(model.readouts[target_session].parameters())
    optimizer = optim.AdamW(target_params, lr=lr, weight_decay=weight_decay)
    if use_cosine_scheduler:
        if n_warmup is None:
            n_warmup = int(0.1 * n_epochs)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=n_warmup,
            num_training_steps=n_epochs,
            num_cycles=1.5,
        )

    best_score = -np.inf
    best_weights = None
    val_scores = []
    epochs_without_improvement = 0

    train_loader = dataloaders['train'][target_session]
    valid_loader = dataloaders['valid'][target_session]

    for epoch in tqdm.tqdm(range(n_epochs), desc=f"Finetune[{target_session}]"):
        model.eval()
        model.readouts[target_session].train()
        for batch in train_loader:
            X_batch = batch.behavior.to(device)
            y_batch = batch.spike.to(device)
            mask_batch = batch.mask.to(device)
            y_pred = model(
                target_session,
                X_batch,
                mask_batch,
                is_causal=is_causal,
                is_bigbird=is_bigbird,
                attention_window_size=attention_window_size,
                num_global=num_global,
                num_random=num_random,
            )
            loss = loss_fn(y_pred[mask_batch == 1.0], y_batch[mask_batch == 1.0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if use_cosine_scheduler:
                scheduler.step()

        model.eval()
        y_val_np_list, y_pred_np_list = [], []
        for batch in valid_loader:
            X_val = batch.behavior.to(device)
            y_val = batch.spike.to(device)
            mask_val = batch.mask.to(device)
            y_pred = model(
                target_session,
                X_val,
                mask_val,
                is_causal=is_causal,
                is_bigbird=is_bigbird,
                attention_window_size=attention_window_size,
                num_global=num_global,
                num_random=num_random,
            )
            y_val_np_list.append(y_val.detach()[mask_val == 1.0].cpu().numpy())
            y_pred_np_list.append(y_pred.detach()[mask_val == 1.0].cpu().numpy())
        y_val_np = np.vstack(y_val_np_list)
        y_pred_np = np.vstack(y_pred_np_list)
        temp_scores = valid_fn(y_val_np, y_pred_np)
        if valid_stat == 'mean':
            val_score = np.nanmean(temp_scores)
        elif valid_stat == 'median':
            val_score = np.nanmedian(temp_scores)
        val_scores.append(val_score)
        if val_score > best_score:
            best_score = val_score
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {early_stopping_patience} epochs)")
                break

    if save_model:
        torch.save(model.state_dict(), model_name)
    if best_weights is not None:
        model.load_state_dict(best_weights)
    return best_score, best_weights, val_scores