"""Joint masked encoder-decoder over neural + behaviour modalities.

A single transformer with one shared latent that unifies neural *encoding*
(behaviour -> spikes) and behavioural *decoding* (spikes -> behaviour) — the
NEDS / POYO+ / OmniMouse lineage adapted to bat hippocampal recordings. Encoding,
decoding, within-/cross-modal infilling, and forecasting are all the *same* weights
under different **masking** schemes; there are no task-specific heads.

Tokenization (per 0.1 s bin, NEDS-style per-timestep):
  - neural: ``spike_readin`` per-session ``Linear(K_session -> D)`` of (optionally
    sqrt-transformed) spike counts. The read-in weight columns are interpretable
    per-neuron embeddings.
  - behaviour: ``beh_readin`` ``Linear(behavior_dim -> D)``, **per-room** (env-keyed)
    when ``session_to_room`` is given so physically-different arenas don't share a
    coordinate frame, else shared. Per-session z-scoring (upstream) handles scale;
    the per-room map handles geometry / remapping.

Each token gets a learned modality embedding (neural vs behaviour) and, if it is a
prediction *target* for the current mask, its content is swapped for a single learned
mask token shared across modalities (kept in the sequence, MAE/NEDS-style); the
modality embedding added afterward is what distinguishes the two masked streams. The two streams are
concatenated into a length-``2T`` sequence and processed by ``num_layers`` pre-norm
transformer blocks with full attention and **timestamp-aware RoPE**
(``build_rotary_emb_from_timestamps``) so a neural and a behaviour token at the same
time bin share rotary phase. After a final ``LayerNorm`` (pre-norm convention),
per-session ``Linear`` + ``ELU1`` reads the neural latent back to a non-negative Poisson
rate; a (per-room) ``Linear`` reads the behaviour latent back to position.

Tasks are selected by masking (see :data:`SCHEMES` / :func:`sample_mask`) and trained
with :func:`joint_loss` (Poisson NLL on masked neural targets + ``lam`` * MSE on masked
behaviour targets). Both losses are computed only on masked *and* valid timesteps.
"""

import numpy as np
import torch
from torch import nn

from .layers import MultiheadAttentionRoPE, ELU1


__all__ = [
    'build_rotary_emb_from_timestamps',
    'JointBlock',
    'JointMaskedTransformer',
    'SCHEMES',
    'sample_mask',
    'within_double_batch',
    'forecast_input_mask',
    'joint_loss',
    'JointEncoderAdapter',
    'encode_eval_records',
]


def build_rotary_emb_from_timestamps(timestamps, dim_head, device):
    """RoPE sin/cos from an explicit per-token timestamp vector (L,) -> (L, dim_head//2).

    Unlike :func:`digitalbat.models.layers.build_rotary_emb` (which keys off the
    sequence index), this keys off explicit timestamps, so a neural token and a
    behaviour token at the same time bin in the concatenated ``2T`` sequence share
    rotary phase.
    """
    half = dim_head // 2
    freq = 1.0 / (10000 ** (torch.arange(half, device=device, dtype=torch.float32) / half))
    angles = torch.einsum('t,f->tf', timestamps.to(device=device, dtype=torch.float32), freq)
    return angles.sin(), angles.cos()


class JointBlock(nn.Module):
    """Timestamp-RoPE multi-head attention + FFN with **pre-norm** residuals
    (LayerNorm inside each residual branch; takes externally-built sin/cos)."""
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout, nonlin=nn.GELU()):
        super().__init__()
        self.attn = MultiheadAttentionRoPE(hidden_dim, num_heads)
        self.ff = nn.Sequential(nn.Linear(hidden_dim, ff_dim), nonlin,
                                nn.Linear(ff_dim, hidden_dim))
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sin, cos, attn_mask, return_attn_weights=False):
        h = self.norm1(x)
        if return_attn_weights:
            a, aw = self.attn(h, h, h, sin=sin, cos=cos, attn_mask=attn_mask,
                              return_attn_weights=True)
        else:
            a, aw = self.attn(h, h, h, sin=sin, cos=cos, attn_mask=attn_mask), None
        x = x + self.dropout(a)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return (x, aw) if return_attn_weights else x


class JointMaskedTransformer(nn.Module):
    """Joint masked encoder-decoder over neural + 3-D-position modalities.

    Per-session spike read-in/read-out (read-in weight columns are per-neuron
    embeddings); per-room (env-keyed) or shared behaviour read-in/read-out. Masked
    tokens -> a single shared learned mask token, kept in the sequence (MAE/NEDS);
    the per-modality modality embedding added afterward distinguishes the streams.

    Parameters
    ----------
    unit_num_dict : dict[str, int]
        ``{session_name: n_neurons}``; one spike read-in/read-out per session.
    behavior_dim : int
        Behaviour channel count (default 3 = x, y, z position).
    latent_dim, num_heads, ff_dim, num_layers, dropout :
        Transformer hyperparameters (``latent_dim`` must be divisible by
        ``num_heads``, and ``latent_dim // num_heads`` must be even for RoPE).
    input_transform : {'identity', 'sqrt'}
        Optional variance-stabilising transform on raw spike counts.
    session_to_room : dict[str, str] | None
        ``{session_name: room_label}``. When given, behaviour read-in/read-out are
        per-room ``nn.ModuleDict``s (different arenas get independent position<->latent
        maps); ``None`` uses a single shared behaviour map.

    Forward
    -------
    ``forward(session_name, spike, behavior, valid_mask, neural_target, beh_target,
    return_attn_weights=False)`` where ``spike`` is ``(B, T, K)``, ``behavior`` is
    ``(B, T, behavior_dim)``, and ``valid_mask`` / ``neural_target`` / ``beh_target``
    are ``(B, T)`` (1.0 = valid / masked-as-target). Returns
    ``(spike_rate (B,T,K), beh_pred (B,T,behavior_dim), latent (B,2T,D))`` — plus a
    list of per-block attention weights when ``return_attn_weights=True``.
    """
    def __init__(self, unit_num_dict, behavior_dim=3, latent_dim=64, num_heads=4,
                 ff_dim=64, num_layers=2, dropout=0.2, input_transform='sqrt',
                 session_to_room=None):
        super().__init__()
        if input_transform not in ('identity', 'sqrt'):
            raise ValueError(f"input_transform must be 'identity' or 'sqrt', got {input_transform!r}")
        self.latent_dim, self.num_heads = latent_dim, num_heads
        self.input_transform = input_transform
        self.session_to_room = session_to_room
        self.spike_readin = nn.ModuleDict({s: nn.Linear(k, latent_dim)
                                           for s, k in unit_num_dict.items()})
        self.spike_readout = nn.ModuleDict({s: nn.Linear(latent_dim, k)
                                            for s, k in unit_num_dict.items()})
        # Behaviour read-in/read-out: per-room (env-keyed) when session_to_room is
        # given, else shared. Rooms differ in geometry, so the position<->latent maps
        # are room-specific; neuron maps stay per-session (they already carry room id).
        if session_to_room is None:
            self.beh_readin = nn.Linear(behavior_dim, latent_dim)
            self.beh_readout = nn.Linear(latent_dim, behavior_dim)
        else:
            rooms = sorted(set(session_to_room.values()))
            self.beh_readin = nn.ModuleDict({r: nn.Linear(behavior_dim, latent_dim) for r in rooms})
            self.beh_readout = nn.ModuleDict({r: nn.Linear(latent_dim, behavior_dim) for r in rooms})
        self.modality_embed = nn.Embedding(2, latent_dim)         # 0 neural, 1 behaviour
        # Single shared mask token: the per-modality modality_embed added afterward is
        # what distinguishes masked neural from masked behaviour tokens, so a per-modality
        # mask token would be redundant. Initialised at zero -> a masked token starts as
        # exactly its modality embedding.
        self.mask_token = nn.Parameter(torch.zeros(latent_dim))
        self.readin_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([JointBlock(latent_dim, num_heads, ff_dim, dropout)
                                     for _ in range(num_layers)])
        # Pre-LN final norm: the residual stream is otherwise un-normalised going
        # into the read-outs. Applied to the full (B, 2T, D) latent before splitting.
        self.final_norm = nn.LayerNorm(latent_dim)
        self.spike_act = ELU1()

    def forward(self, session_name, spike, behavior, valid_mask,
                neural_target, beh_target, return_attn_weights=False):
        B, T, _ = behavior.shape
        dev = behavior.device
        sp = spike.float()
        if self.input_transform == 'sqrt':
            sp = torch.sqrt(torch.clamp(sp, min=0.0))
        beh_readin = (self.beh_readin if self.session_to_room is None
                      else self.beh_readin[self.session_to_room[session_name]])
        beh_readout = (self.beh_readout if self.session_to_room is None
                       else self.beh_readout[self.session_to_room[session_name]])
        z_n = self.readin_dropout(self.spike_readin[session_name](sp))
        z_b = self.readin_dropout(beh_readin(behavior))
        z_n = torch.where(neural_target.bool().unsqueeze(-1), self.mask_token, z_n)
        z_b = torch.where(beh_target.bool().unsqueeze(-1), self.mask_token, z_b)
        z_n = z_n + self.modality_embed(torch.zeros(1, dtype=torch.long, device=dev))
        z_b = z_b + self.modality_embed(torch.ones(1, dtype=torch.long, device=dev))
        x = torch.cat([z_n, z_b], dim=1)                          # (B, 2T, D)

        ts = torch.cat([torch.arange(T, device=dev), torch.arange(T, device=dev)])
        sin, cos = build_rotary_emb_from_timestamps(ts, self.latent_dim // self.num_heads, dev)
        vm = torch.cat([valid_mask, valid_mask], dim=1).float()
        attn_mask = torch.einsum('bi,bj->bij', vm, vm).bool()[:, None]

        attn_all = []
        for blk in self.blocks:
            if return_attn_weights:
                x, aw = blk(x, sin, cos, attn_mask, return_attn_weights=True)
                attn_all.append(aw)
            else:
                x = blk(x, sin, cos, attn_mask)
        x = self.final_norm(x)
        h_n, h_b = x[:, :T], x[:, T:]
        rate = self.spike_act(self.spike_readout[session_name](h_n))
        beh = beh_readout(h_b)
        if return_attn_weights:
            return rate, beh, x, attn_all
        return rate, beh, x


# ---------------------------------------------------------------------------
# Masking schemes: the task is a choice of which tokens are prediction targets.
# ---------------------------------------------------------------------------
SCHEMES = ['encode', 'decode', 'within', 'cross', 'forecast', 'forecast_stim']


def sample_mask(valid_mask, scheme, frac=0.5):
    """Build ``(neural_target, beh_target)`` ``(B, T)`` float **loss-target** masks.

    ``*_target == 1`` marks tokens that are *scored* by the loss (and, for most
    schemes, also masked on input). Padded timesteps (``valid_mask == 0``) are never
    targets. Schemes:
    ``encode`` (spikes <- behaviour), ``decode`` (behaviour <- spikes),
    ``within`` (random infill of a *single* modality chosen per row, the other
    left fully visible), ``cross`` (same random tokens in both modalities),
    ``forecast`` / ``forecast_stim`` (predict the **2nd half of neural** activity;
    behaviour is *not* scored in either).

    The two forecasting variants (after Willeke et al. 2026, Fig. 5) return the
    **same** loss target here — 2nd-half neural, no behaviour — because they differ
    only in what the model *sees on input*, not in what is scored:

    - ``forecast``       : behaviour fully masked on input (no stimulus).
    - ``forecast_stim``  : behaviour fully visible on input (+ stimulus).

    That input-visibility difference is applied by the trainer (which passes the
    input mask to ``forward`` separately from the loss target it passes to
    :func:`joint_loss`); see :func:`forecast_input_mask`.
    """
    B, T = valid_mask.shape
    v = valid_mask.bool()
    z = torch.zeros_like(valid_mask)
    if scheme == 'encode':
        return valid_mask.clone(), z.clone()
    if scheme == 'decode':
        return z.clone(), valid_mask.clone()
    if scheme == 'within':
        # True within-modal infill: each *sequence* targets ONE modality, the
        # other left fully visible, chosen independently per row. If both were
        # masked at once (even at different times), a masked token could lean on
        # the same-time token in the other modality — the cross-modal crutch this
        # scheme avoids. Per-row choice lets a doubled 2B batch (each flight
        # stacked twice) carry every flight once as a neural target and once as a
        # behaviour target; see :func:`within_double_batch`.
        m = (torch.rand_like(valid_mask) < frac) & v
        pick_neural = torch.rand(B, 1, device=valid_mask.device) < 0.5
        return (m & pick_neural).float(), (m & ~pick_neural).float()
    if scheme == 'cross':
        m = (torch.rand_like(valid_mask) < frac) & v
        return m.float(), m.float()
    if scheme in ('forecast', 'forecast_stim'):
        # Predict the 2nd half of NEURAL activity from the 1st half; behaviour is
        # never a loss target (it is only context, present or absent per variant).
        tgt = torch.zeros_like(valid_mask)
        for b in range(B):
            idx = torch.where(v[b])[0]
            if len(idx) >= 2:
                tgt[b, idx[len(idx) // 2]:] = valid_mask[b, idx[len(idx) // 2]:]
        return tgt.clone(), z.clone()
    raise ValueError(scheme)


def forecast_input_mask(valid_mask, scheme):
    """Behaviour **input** mask for a forecasting scheme (``(B, T)`` float).

    Distinct from the loss target of :func:`sample_mask`: this is what the model
    *cannot see*, not what it is scored on. ``forecast`` (no stimulus) hides all
    valid behaviour timesteps; ``forecast_stim`` (+ stimulus) hides none. Neural
    input masking is the same as the neural loss target (2nd-half neural) for both,
    so only the behaviour input mask differs and is returned here.
    """
    if scheme == 'forecast':
        return valid_mask.clone()                 # hide all behaviour (no stimulus)
    if scheme == 'forecast_stim':
        return torch.zeros_like(valid_mask)       # behaviour fully visible (+ stimulus)
    raise ValueError(scheme)


def within_double_batch(valid_mask, frac=0.5):
    """Deterministic within-modal doubling of a ``(B, T)`` batch to ``2B``.

    Stacks the batch on top of itself so every flight is seen *twice* under
    within-modal infill: the first ``B`` rows mask random valid **neural**
    timepoints (behaviour fully visible), the next ``B`` rows mask random valid
    **behaviour** timepoints (neural fully visible). Unlike ``sample_mask('within')``
    (which picks one modality per row at random, so a flight may land as the same
    modality every time), this guarantees each flight contributes once as a neural
    target and once as a behaviour target.

    Returns ``(neural_target, beh_target)`` each ``(2B, T)``. Pair with model inputs
    doubled the same way (``torch.cat([x, x], dim=0)`` for spike / behaviour / mask).
    The two halves draw independent random timepoints, so the neural and behaviour
    views of a flight are not forced to hide the same bins.
    """
    v = valid_mask.bool()
    z = torch.zeros_like(valid_mask)
    mn = ((torch.rand_like(valid_mask) < frac) & v).float()
    mb = ((torch.rand_like(valid_mask) < frac) & v).float()
    neural_target = torch.cat([mn, z], dim=0)
    beh_target = torch.cat([z, mb], dim=0)
    return neural_target, beh_target


def joint_loss(rate, beh, spike, behavior, valid_mask, neural_target, beh_target,
               lam, pois_fn, pos_weight=1.0, pos_dim=3):
    """Poisson NLL on masked+valid neural targets + ``lam`` * weighted behaviour MSE.

    The behaviour MSE is split into a **position** term (first ``pos_dim`` channels) and
    an **auxiliary** term (any remaining kinematic channels — speed / heading sin-cos /
    gyro / accel)::

        behaviour_loss = pos_weight * MSE_pos + (1 - pos_weight) * MSE_aux

    so ``pos_weight`` (in ``[0, 1]``) trades position reconstruction against the auxiliary
    variables without the auxiliary channels diluting position by sheer count. With no
    auxiliary channels (``behavior_dim == pos_dim``) it reduces to ``MSE_pos`` — i.e.
    ``pos_weight`` is a no-op and the position-only case is unchanged (default
    ``pos_weight=1.0`` is fully backward compatible). The whole behaviour term is scaled
    by ``lam`` against the neural NLL.

    ``pois_fn`` is a ``nn.PoissonNLLLoss(log_input=False, reduction='none')`` instance.
    Only masked *and* valid timesteps contribute to each term.
    """
    nsel = valid_mask.bool() & neural_target.bool()
    bsel = valid_mask.bool() & beh_target.bool()
    loss = rate.new_zeros(())
    if nsel.any():
        loss = loss + pois_fn(rate[nsel], spike.float()[nsel]).mean()
    if bsel.any():
        diff2 = (beh[bsel] - behavior[bsel]) ** 2           # (n_masked, behavior_dim)
        beh_loss = diff2[:, :pos_dim].mean()                # position
        if diff2.shape[1] > pos_dim:                        # auxiliary kinematic channels present
            beh_loss = pos_weight * beh_loss + (1.0 - pos_weight) * diff2[:, pos_dim:].mean()
        loss = loss + lam * beh_loss
    return loss


class JointEncoderAdapter(nn.Module):
    """Expose a trained :class:`JointMaskedTransformer` with the standalone
    ``Bat_Transformer`` encoder interface (behaviour in -> spike rate out) plus
    per-session ``readouts``.

    The encode masking scheme is applied internally: neural tokens are the
    prediction targets (masked) and behaviour tokens are seen, so a single call
    ``adapter(session_name, behaviour, mask)`` returns the predicted Poisson rate,
    exactly like the encoder-only model. ``readouts[session]`` wraps the joint
    model's per-session spike readout, so the encoding / embedding / readout-weight
    analyses can treat the joint model as a drop-in encoder.
    """

    def __init__(self, joint, unit_num_dict):
        super().__init__()
        self.joint = joint
        self.unit_num_dict = dict(unit_num_dict)
        self.readouts = {s: nn.Sequential(joint.spike_readout[s])
                         for s in joint.spike_readout}
        self.final_activation = joint.spike_act

    def forward(self, session_name, x, mask, return_attn_weights=False, **kwargs):
        B, T, _ = x.shape
        K = self.unit_num_dict[session_name]
        spike = x.new_zeros(B, T, K)
        neural_target = mask.to(x.dtype)        # encode: neural tokens are targets
        beh_target = torch.zeros_like(neural_target)   # behaviour is seen
        if return_attn_weights:
            _, _, xfull, attn = self.joint(session_name, spike, x, mask,
                                           neural_target, beh_target,
                                           return_attn_weights=True)
            h_n = xfull[:, :T]
            rate = self.final_activation(self.readouts[session_name](h_n))
            return rate, [a[:, :, :T, T:] for a in attn]
        _, _, xfull = self.joint(session_name, spike, x, mask, neural_target, beh_target)
        h_n = xfull[:, :T]
        return self.final_activation(self.readouts[session_name](h_n))


@torch.no_grad()
def encode_eval_records(model, loaders, device, *, split='test'):
    """Per-(session, neuron) encoding metrics for an encoder-like ``model`` on
    ``loaders[split]``.

    Returns a tidy ``pandas.DataFrame`` with columns ``session, bat, unit, corr,
    fde`` (raw FDE, not clipped). ``model`` is anything with the
    ``model(session, behaviour, mask) -> rate`` interface (e.g.
    :class:`JointEncoderAdapter` or ``Bat_Transformer``).
    """
    import pandas as pd
    from ..utility.utils import corr, poisson_fde_population, parse_bat

    rows = []
    for sn, sd in loaders[split].items():
        yv, yp = [], []
        for batch in sd:
            m = batch.mask.to(device)
            rate = model(sn, batch.behavior.to(device), m)
            yv.append(batch.spike[batch.mask == 1.0].numpy())
            yp.append(rate[m == 1.0].detach().cpu().numpy())
        yv, yp = np.vstack(yv), np.vstack(yp)
        c = corr(yv, yp, axis=0)
        f = poisson_fde_population(yv, yp)
        for u in range(yv.shape[1]):
            rows.append(dict(session=sn, bat=parse_bat(sn), unit=u,
                             corr=float(c[u]), fde=float(f[u])))
    return pd.DataFrame(rows)
