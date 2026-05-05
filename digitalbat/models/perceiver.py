"""Perceiver IO trajectory decoder: maps a single population vector to a 3D trajectory.

Two variants share the same architecture but differ in temporal query construction:
  - "relative": query grid centered at t=0 (moment of PV sampling), spanning [-T_max, +T_max]
  - "absolute": query grid from t=0 (flight onset) to T_max

The model follows encode-process-decode:
  1. Encode: cross-attention from learned latent array to neuron tokens
  2. Process: self-attention stack refining latents
  3. Decode: cross-attention from temporal Fourier queries to latents → (x, y, z, logit_valid)
"""

from dataclasses import dataclass
from math import pi

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class PerceiverIOConfig:
    n_neurons: int = 2000
    C: int = 16
    L: int = 64
    D: int = 128
    n_process_layers: int = 4
    n_heads: int = 4
    D_ff_multiplier: int = 2
    time_mode: str = "relative"
    T_max: float = 8.0
    dt: float = 0.1
    n_fourier_bands: int = 16
    lambda_pos: float = 1.0
    lambda_valid: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    dropout_rate: float = 0.1
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 20
    warmup_steps: int = 500
    gradient_clip: float = 1.0

    def __post_init__(self):
        if self.D % self.n_heads != 0:
            raise ValueError(f"D={self.D} must be divisible by n_heads={self.n_heads}")
        if self.D // self.n_heads < 4:
            raise ValueError(
                f"d_head={self.D // self.n_heads} must be >= 4 "
                f"(D={self.D}, n_heads={self.n_heads})"
            )


def fourier_features(t: Tensor, min_freq: float, max_freq: float, n_bands: int) -> Tensor:
    """Fourier positional encoding of scalar time values.

    Returns shape (..., 2 * n_bands + 1): sin/cos pairs + raw time.
    """
    freqs = torch.linspace(min_freq, max_freq, n_bands, device=t.device, dtype=t.dtype)
    angles = 2 * pi * t.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(angles), torch.cos(angles), t.unsqueeze(-1)], dim=-1)


class CrossAttentionBlock(nn.Module):
    """Multi-head cross-attention with optional residual connection."""

    def __init__(self, q_dim: int, kv_dim: int, d_model: int, n_heads: int,
                 D_ff_multiplier: int, dropout: float, use_residual: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_residual = use_residual

        self.norm_q = nn.LayerNorm(q_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)

        self.W_Q = nn.Linear(q_dim, d_model)
        self.W_K = nn.Linear(kv_dim, d_model)
        self.W_V = nn.Linear(kv_dim, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)

        self.norm_ff = nn.LayerNorm(d_model)
        D_ff = d_model * D_ff_multiplier
        self.mlp = nn.Sequential(
            nn.Linear(d_model, D_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query: Tensor, kv: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (output, attn_weights). attn_weights shape: (B, n_heads, Q, KV)."""
        B, Q, _ = query.shape
        KV = kv.shape[1]

        q = self.W_Q(self.norm_q(query))
        kv_normed = self.norm_kv(kv)
        k = self.W_K(kv_normed)
        v = self.W_V(kv_normed)

        q = q.view(B, Q, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, KV, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, KV, self.n_heads, self.d_head).transpose(1, 2)

        scale = self.d_head ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights_out = attn_weights.detach()
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, Q, -1)
        out = self.W_O(out)

        if self.use_residual:
            out = query + out
        x = out
        x = x + self.mlp(self.norm_ff(x))
        return x, attn_weights_out


class SelfAttentionBlock(nn.Module):
    """Multi-head self-attention block."""

    def __init__(self, d_model: int, n_heads: int, D_ff_multiplier: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        D_ff = d_model * D_ff_multiplier
        self.mlp = nn.Sequential(
            nn.Linear(d_model, D_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, L, _ = x.shape
        h = self.norm1(x)

        q = self.W_Q(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_K(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_V(h).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        scale = self.d_head ** -0.5
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, -1)
        out = self.W_O(out)

        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x


class PerceiverIOTrajectoryDecoder(nn.Module):
    """Perceiver IO that decodes a population vector into a 3D trajectory segment."""

    def __init__(self, config: PerceiverIOConfig):
        super().__init__()
        self.config = config
        N = config.n_neurons
        C = config.C
        D = config.D
        L = config.L

        # Module 1: Input tokenization
        if C == 1:
            self.rate_mlp = None
        else:
            self.rate_mlp = nn.Sequential(
                nn.Linear(1, C),
                nn.GELU(),
                nn.Linear(C, C),
            )

        # Per-neuron identity embedding: lets the encoder distinguish *which*
        # neuron fired, not just the distribution of rates. Without this the
        # rate_mlp is shared across neurons and the encoder cross-attention is
        # permutation-invariant in N — destroying place-coding information.
        self.neuron_embed = nn.Parameter(torch.randn(N, C) * 0.02)

        self.input_dropout = nn.Dropout(config.dropout_rate)

        # Module 2: Encoder (cross-attention from latent to neurons)
        self.latent_array = nn.Parameter(torch.randn(L, D) * 0.02)
        self.encoder = CrossAttentionBlock(
            q_dim=D, kv_dim=C, d_model=D, n_heads=config.n_heads,
            D_ff_multiplier=config.D_ff_multiplier, dropout=config.dropout_rate,
            use_residual=True,
        )

        # Module 3: Processor (self-attention stack)
        self.processor_layers = nn.ModuleList([
            SelfAttentionBlock(
                d_model=D, n_heads=config.n_heads,
                D_ff_multiplier=config.D_ff_multiplier, dropout=config.dropout_rate,
            )
            for _ in range(config.n_process_layers)
        ])

        # Module 4: Temporal query construction
        query_times = self._build_query_grid(config)
        self.register_buffer("query_times", query_times)
        query_features = self._build_query_features(config, query_times)
        self.register_buffer("query_features", query_features)
        E = query_features.shape[-1]  # 2 * n_fourier_bands + 1

        # Module 5: Decoder (cross-attention from queries to latent)
        self.decoder = CrossAttentionBlock(
            q_dim=E, kv_dim=D, d_model=D, n_heads=config.n_heads,
            D_ff_multiplier=config.D_ff_multiplier, dropout=config.dropout_rate,
            use_residual=False,
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 4),
        )

    @staticmethod
    def _build_query_grid(config: PerceiverIOConfig) -> Tensor:
        if config.time_mode == "relative":
            n_bins = round(2 * config.T_max / config.dt) + 1
            times = torch.linspace(-config.T_max, config.T_max, n_bins)
        elif config.time_mode == "absolute":
            n_bins = round(config.T_max / config.dt) + 1
            times = torch.linspace(0, config.T_max, n_bins)
        else:
            raise ValueError(f"time_mode must be 'relative' or 'absolute', got {config.time_mode!r}")
        return times

    @staticmethod
    def _build_query_features(config: PerceiverIOConfig, query_times: Tensor) -> Tensor:
        if config.time_mode == "relative":
            window = 2 * config.T_max
        else:
            window = config.T_max
        # T_max = 0 → single-query (Δt=0) ablation. window is otherwise unused
        # (Fourier features at t=0 collapse to a constant regardless of freq),
        # but we still need a finite min_freq to avoid 1/0.
        window = max(window, config.dt)
        min_freq = 1.0 / window
        max_freq = 1.0 / (2 * config.dt)
        return fourier_features(query_times, min_freq, max_freq, config.n_fourier_bands)

    def forward(self, population_vector: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            population_vector: (batch, N) firing rates

        Returns:
            position_pred: (batch, O, 3)
            valid_logit:   (batch, O)
            encoder_attn:  (batch, n_heads, L, N)
            decoder_attn:  (batch, n_heads, O, L)
        """
        B = population_vector.shape[0]
        N = population_vector.shape[1]

        # Neuron dropout (training only)
        pv = self.input_dropout(population_vector)

        # Tokenize
        if self.rate_mlp is not None:
            neuron_tokens = self.rate_mlp(pv.unsqueeze(-1))  # (B, N, C)
        else:
            neuron_tokens = pv.unsqueeze(-1)  # (B, N, 1)
        neuron_tokens = neuron_tokens + self.neuron_embed  # broadcast over batch

        # Encode
        latent = self.latent_array.unsqueeze(0).expand(B, -1, -1)  # (B, L, D)
        latent, enc_attn = self.encoder(latent, neuron_tokens)

        # Process
        for layer in self.processor_layers:
            latent = layer(latent)

        # Decode
        queries = self.query_features.unsqueeze(0).expand(B, -1, -1)  # (B, O, E)
        decoded, dec_attn = self.decoder(queries, latent)

        # Output head
        output = self.output_head(decoded)  # (B, O, 4)
        position_pred = output[:, :, :3]
        valid_logit = output[:, :, 3]

        return position_pred, valid_logit, enc_attn, dec_attn

    def compute_loss(
        self,
        position_pred: Tensor,
        valid_logit: Tensor,
        position_targets: Tensor,
        validity_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute combined position + validity loss.

        Args:
            position_pred: (B, O, 3)
            valid_logit: (B, O)
            position_targets: (B, O, 3)
            validity_mask: (B, O) float, 1.0 = valid

        Returns:
            total_loss, pos_loss, valid_loss
        """
        mask_bool = validity_mask.bool()
        if mask_bool.any():
            pos_loss = F.mse_loss(position_pred[mask_bool], position_targets[mask_bool])
        else:
            # Unreachable via PVTrajectoryDataset (qt=0 always valid), but guards
            # against NaN gradients if called with an all-zero mask externally.
            pos_loss = torch.tensor(0.0, device=position_pred.device)

        valid_loss = F.binary_cross_entropy_with_logits(valid_logit, validity_mask)
        total = self.config.lambda_pos * pos_loss + self.config.lambda_valid * valid_loss
        return total, pos_loss, valid_loss
