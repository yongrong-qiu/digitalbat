import torch
from torch import nn
import torch.nn.functional as F
import math


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
    

class BCEFocalCombinedLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 2.0,
        focal_weight: float = 0.5,
        pos_weight: torch.Tensor = None,
        reduction: str = 'mean',
    ):
        """
        Combines BCEWithLogitsLoss and Focal Loss for imbalanced binary classification.

        Parameters
        ----------
        alpha : float
            Scaling factor for focal loss.
        gamma : float
            Focusing parameter for focal loss.
        focal_weight : float
            Weight for focal loss in the final combination. (1 - focal_weight) is used for BCE.
        pos_weight : torch.Tensor, optional
            A weight of positive examples. Passed to BCEWithLogitsLoss.
        reduction : str
            'mean' | 'sum' | 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight
        self.reduction = reduction

        self.bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE with logits part
        bce_loss = self.bce_loss_fn(logits, targets)

        # Focal loss part
        bce_no_reduce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_no_reduce)  # probability for the correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_no_reduce

        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        # else: keep as is

        # Combine the two
        combined_loss = self.focal_weight * focal_loss + (1 - self.focal_weight) * bce_loss
        return combined_loss


# ============================================================
# 1. Rotary Positional Embedding (RoPE)
# ============================================================

def apply_rotary_pos_emb(x, sin, cos):
    """
    Apply rotary embedding to q or k tensor.
    x: (B, T, H, D_head)
    sin, cos: (T, D_head)
    """
    # split last dim into two halves
    x1, x2 = x[..., ::2], x[..., 1::2]
    sin, cos = sin.unsqueeze(0).unsqueeze(0), cos.unsqueeze(0).unsqueeze(0)
    x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot


def build_rotary_emb(seq_len, dim_head, device):
    """Precompute RoPE sine/cosine embeddings."""
    # dim_head must be even
    half_dim = dim_head // 2
    freq_seq = torch.arange(half_dim, device=device, dtype=torch.float32)
    freq = 1.0 / (10000 ** (freq_seq / half_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.einsum('t,f->tf', t, freq)
    sin, cos = angles.sin(), angles.cos()
    return sin, cos


class TransformerRoPE(nn.Module):
    """
    Transformer Encoder Layer with RoPE
    """
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim)
        )

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, attn_mask=None, sin=None, cos=None):
        """
        x: (B, T, D)
        attn_mask: (B, T) bool (True for valid)
        sin, cos: RoPE embeddings
        """
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)

        # Apply RoPE
        if sin is not None and cos is not None:
            q = apply_rotary_pos_emb(q, sin, cos)
            k = apply_rotary_pos_emb(k, sin, cos)

        # Scaled dot-product attention
        attn = torch.einsum('bthd,bThd->bhtT', q, k) / math.sqrt(self.head_dim)

        # mask: 1=valid, 0=pad
        if attn_mask is not None:
            mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,T)
            attn = attn.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(attn, dim=-1)
        attn_out = torch.einsum('bhtT,bThd->bthd', attn_weights, v)
        attn_out = attn_out.reshape(B, T, D)
        x = x + self.dropout(self.o_proj(attn_out))
        x = self.norm1(x)

        x = x + self.dropout(self.ff(x))
        x = self.norm2(x)
        return x