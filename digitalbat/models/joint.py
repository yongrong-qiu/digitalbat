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
prediction *target* for the current mask, its content is swapped for a learned
per-modality mask token (kept in the sequence, MAE/NEDS-style). The two streams are
concatenated into a length-``2T`` sequence and processed by ``num_layers`` pre-norm
transformer blocks with full attention and **timestamp-aware RoPE**
(``build_rotary_emb_from_timestamps``) so a neural and a behaviour token at the same
time bin share rotary phase. Per-session ``Linear`` + ``ELU1`` reads the neural latent
back to a non-negative Poisson rate; a (per-room) ``Linear`` reads the behaviour latent
back to position.

Tasks are selected by masking (see :data:`SCHEMES` / :func:`sample_mask`) and trained
with :func:`joint_loss` (Poisson NLL on masked neural targets + ``lam`` * MSE on masked
behaviour targets). Both losses are computed only on masked *and* valid timesteps.
"""

import torch
from torch import nn

from .layers import MultiheadAttentionRoPE, ELU1


__all__ = [
    'build_rotary_emb_from_timestamps',
    'JointBlock',
    'JointMaskedTransformer',
    'SCHEMES',
    'sample_mask',
    'joint_loss',
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
    """Timestamp-RoPE multi-head attention + FFN, pre-norm residual (mirrors
    ``TransformerEncoderRoPE`` but takes externally-built sin/cos)."""
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout, nonlin=nn.GELU()):
        super().__init__()
        self.attn = MultiheadAttentionRoPE(hidden_dim, num_heads)
        self.ff = nn.Sequential(nn.Linear(hidden_dim, ff_dim), nonlin,
                                nn.Linear(ff_dim, hidden_dim))
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sin, cos, attn_mask, return_attn_weights=False):
        if return_attn_weights:
            a, aw = self.attn(x, x, x, sin=sin, cos=cos, attn_mask=attn_mask,
                              return_attn_weights=True)
        else:
            a, aw = self.attn(x, x, x, sin=sin, cos=cos, attn_mask=attn_mask), None
        x = self.norm1(x + self.dropout(a))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return (x, aw) if return_attn_weights else x


class JointMaskedTransformer(nn.Module):
    """Joint masked encoder-decoder over neural + 3-D-position modalities.

    Per-session spike read-in/read-out (read-in weight columns are per-neuron
    embeddings); per-room (env-keyed) or shared behaviour read-in/read-out. Masked
    tokens -> learned per-modality mask token, kept in the sequence (MAE/NEDS).

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
        self.mask_token = nn.Parameter(torch.zeros(2, latent_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.readin_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([JointBlock(latent_dim, num_heads, ff_dim, dropout)
                                     for _ in range(num_layers)])
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
        z_n = torch.where(neural_target.bool().unsqueeze(-1), self.mask_token[0], z_n)
        z_b = torch.where(beh_target.bool().unsqueeze(-1), self.mask_token[1], z_b)
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
        h_n, h_b = x[:, :T], x[:, T:]
        rate = self.spike_act(self.spike_readout[session_name](h_n))
        beh = beh_readout(h_b)
        if return_attn_weights:
            return rate, beh, x, attn_all
        return rate, beh, x


# ---------------------------------------------------------------------------
# Masking schemes: the task is a choice of which tokens are prediction targets.
# ---------------------------------------------------------------------------
SCHEMES = ['encode', 'decode', 'within', 'cross', 'forecast']


def sample_mask(valid_mask, scheme, frac=0.5):
    """Build ``(neural_target, beh_target)`` ``(B, T)`` float masks for a scheme.

    ``*_target == 1`` marks tokens masked on input (i.e. prediction targets). Padded
    timesteps (``valid_mask == 0``) are never targets. Schemes:
    ``encode`` (spikes <- behaviour), ``decode`` (behaviour <- spikes),
    ``within`` (independent random infill per modality), ``cross`` (same random tokens
    in both modalities), ``forecast`` (2nd half of valid timesteps in both modalities).
    """
    B, T = valid_mask.shape
    v = valid_mask.bool()
    z = torch.zeros_like(valid_mask)
    if scheme == 'encode':
        return valid_mask.clone(), z.clone()
    if scheme == 'decode':
        return z.clone(), valid_mask.clone()
    if scheme == 'within':
        nm = (torch.rand_like(valid_mask) < frac) & v
        bm = (torch.rand_like(valid_mask) < frac) & v
        return nm.float(), bm.float()
    if scheme == 'cross':
        m = (torch.rand_like(valid_mask) < frac) & v
        return m.float(), m.float()
    if scheme == 'forecast':
        tgt = torch.zeros_like(valid_mask)
        for b in range(B):
            idx = torch.where(v[b])[0]
            if len(idx) >= 2:
                tgt[b, idx[len(idx) // 2]:] = valid_mask[b, idx[len(idx) // 2]:]
        return tgt.clone(), tgt.clone()
    raise ValueError(scheme)


def joint_loss(rate, beh, spike, behavior, valid_mask, neural_target, beh_target,
               lam, pois_fn):
    """Poisson NLL on masked+valid neural targets + ``lam`` * MSE on behaviour targets.

    ``pois_fn`` is a ``nn.PoissonNLLLoss(log_input=False, reduction='none')`` instance.
    Only masked *and* valid timesteps contribute to each term.
    """
    nsel = valid_mask.bool() & neural_target.bool()
    bsel = valid_mask.bool() & beh_target.bool()
    loss = rate.new_zeros(())
    if nsel.any():
        loss = loss + pois_fn(rate[nsel], spike.float()[nsel]).mean()
    if bsel.any():
        loss = loss + lam * ((beh[bsel] - behavior[bsel]) ** 2).mean()
    return loss
