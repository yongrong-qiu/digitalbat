"""Default ``Bat_Transformer`` config + training scaffolding used across the
analysis notebooks.

Centralizes the architectural defaults (latent_dim=32, ff_dim=32, num_heads=4,
num_layers=1, dropout_prob=0.3, ELU1 final activation, trajectory branch only),
the optimizer hyperparameters, the on-disk checkpoint naming convention, and
the trainer-config dict so notebooks can stay in sync without copy-pasting.
"""
from torch import nn

from ..utility.utils import poisson_fde_population
from .former import Bat_Transformer
from .layers import ELU1


DEFAULT_LR = 0.01
DEFAULT_WEIGHT_DECAY = 0.01


def make_model_config(behavior_dim, unit_num_dict, meta_info_dict, session_to_room=None):
    """Return the standard ``model_config`` dict for ``Bat_Transformer``.

    Pass ``session_to_room`` (``{session_name: room_label}``) to give each
    environment its own trajectory projection (``embed_traj``); ``None`` (default)
    keeps the single shared projection.
    """
    return dict(
        input_dim=behavior_dim,
        output_dim_dict=unit_num_dict,
        use_place_as_input=False,
        spatial_centers_sigma_dict=meta_info_dict['spatial_centers_sigma'],
        latent_dim_place=32,
        use_trajectory_as_input=True,
        latent_dim=32,
        ff_dim=32,
        num_heads=4,
        num_layers=1,
        dropout_prob=0.3,
        final_activation=ELU1(),
        session_to_room=session_to_room,
    )


def make_bat_former(behavior_dim, unit_num_dict, meta_info_dict):
    """Construct a fresh ``Bat_Transformer`` from :func:`make_model_config`."""
    return Bat_Transformer(**make_model_config(behavior_dim, unit_num_dict, meta_info_dict))


def make_model_name(seed, lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY,
                    out_dir='./out', suffix=''):
    """Standard checkpoint path: ``<out_dir>/bat_former_seed_{seed}_lr_<lr>_wd_<wd>{suffix}.pth``."""
    return f'{out_dir}/bat_former_seed_{seed}_lr_{lr:.2e}_wd_{weight_decay:.2e}{suffix}.pth'


def make_trainer_config(model, dataloaders, device, model_name,
                        lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY,
                        n_epochs=1000, early_stopping_patience=100,
                        save_model=True):
    """Return the standard ``trainer_config`` dict for ``bat_former_trainer``."""
    return dict(
        model=model,
        dataloaders=dataloaders,
        device=device,
        is_causal=False,
        is_bigbird=False,
        attention_window_size=9,
        num_global=2,
        n_epochs=n_epochs,
        lr=lr,
        weight_decay=weight_decay,
        loss_fn=nn.PoissonNLLLoss(log_input=False),
        valid_fn=poisson_fde_population,
        valid_stat='median',
        early_stopping_patience=early_stopping_patience,
        save_model=save_model,
        model_name=model_name,
    )
