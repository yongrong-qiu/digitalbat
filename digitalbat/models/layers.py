import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


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


def apply_rotary_pos_emb(x, sin, cos):
    """
    Apply rotary positional embedding (RoPE) to q or k tensor.
    x: (B, num_head, T, D_head)
    sin, cos: (T, D_head)
    """
    # split last dim into two halves
    x1, x2 = x[..., ::2], x[..., 1::2]
    sin, cos = sin.unsqueeze(0).unsqueeze(0), cos.unsqueeze(0).unsqueeze(0)
    x_rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    x_rot = x_rot.reshape(x.shape[:-1] + (-1,))
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


def apply_rotary_emb_cis(
    xq: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Adapted from LLAMA
    Apply RoPE to xq, perform computations in complex domain

    Example:
        batch_size=10
        seq_len=3
        d=8
        xq = torch.ones(batch_size, seq_len, d)
        freqs_cis = precompute_freqs_cis(d, seq_len)
        xq_out = apply_rotary_emb_cis(xq, freqs_cis)
    """

    # xq.shape = [batch_size, num_head, seq_len, dim] or [batch_size, seq_len, dim]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    
    # complex domain
    xq_ = torch.view_as_complex(xq_)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq)

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    """
    Adapted from LLAMA
    Precompute RoPE, in complex domain
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2] 
    freqs = torch.outer(t, freqs).float()  # m * \theta

    # complex tensor
    # assume freqs = [x, y]
    # then: freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
    return freqs_cis


class MultiheadAttentionRoPE(nn.Module):
    """
    Multihead attention using rotary position embedding

    Example:
        B, T, C, H = 2, 16, 64, 8
        x = torch.randn(B, T, C)
        attn = MultiHeadAttention(embed_dim=C, num_heads=H)

        # Build rotary embeddings
        sin, cos = build_rotary_emb(seq_len=T, dim_head=C // H, device=x.device)

        # Forward with RoPE
        out, weights = attn(x, x, x, sin=sin, cos=cos, return_attn_weights=True)
        print(out.shape, weights.shape)
        # (2, 16, 64), (2, 8, 16, 16)

    """
    def __init__(self, embed_dim, num_heads, bias=True):
        """
        embed_dim: total embedding dimension (e.g., 512)
        num_heads: number of attention heads (e.g., 8)
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim % 2 == 0, "RoPE requires an even head_dim."

        # Linear projections for query, key, value
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self, 
        query, 
        key, 
        value, 
        sin=None, 
        cos=None, 
        attn_mask: Optional[torch.BoolTensor] = None, 
        dropout_p=0.0, 
        is_causal=False, 
        return_attn_weights=False,
        is_bigbird=False, # whether to use bigbird style sparse attention
        attention_window_size=3, # bigbird style sparse attention
        num_global=2, # bigbird style sparse attention
        num_random=1, # bigbird style sparse attention
    ):
        """
        query, key, value: (batch_size, seq_len, embed_dim)
        attn_mask: optional (batch_size, 1, seq_len_query, seq_len_key) or broadcastable form, bool
        return_attn_weights: if True, also return attention probabilities (B, num_heads, L, L)
        """
        B, L, _ = query.shape

        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Reshape to multi-head form: (B, num_heads, L, head_dim)
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings if provided
        if sin is not None and cos is not None:
            Q = apply_rotary_pos_emb(Q, sin, cos)
            K = apply_rotary_pos_emb(K, sin, cos)

        d_k = Q.size(-1)
        L_q, L_k = Q.size(-2), K.size(-2)

        # Apply big bird style sparse attention mask if needed
        if is_bigbird:
            # sliding window attention
            assert attention_window_size%2==1, "attention_window_size should be an odd number"
            shift = (attention_window_size-1)//2
            mask1 = torch.tril(torch.ones(L_q, L_k, device=Q.device, dtype=torch.bool), diagonal=shift)
            mask2 = ~torch.tril(torch.ones(L_q, L_k, device=Q.device, dtype=torch.bool), diagonal=-shift-1)
            bigbird_mask = mask1 * mask2
            # fig, ax = plt.subplots(figsize=(3,3))
            # ax.imshow(mask.cpu().numpy())
            #
            # global attention
            if num_global > 0:
                global_idx = torch.arange(num_global, device=Q.device)
                bigbird_mask[:, global_idx] = True  # all tokens attend to globals
                bigbird_mask[global_idx, :] = True  # globals attend to all tokens
            # fig, ax = plt.subplots(figsize=(3,3))
            # ax.imshow(mask.cpu().numpy())
            #
            # random attention
            if num_random > 0:
                torch.manual_seed(1) 
                rand = torch.randint(0, L_k, (L_q, num_random), device=Q.device)
                bigbird_mask.scatter_(1, rand, True)
            # fig, ax = plt.subplots(figsize=(3,3))
            # ax.imshow(mask.cpu().numpy())
            #
            attn_mask = attn_mask & bigbird_mask[None, None, :, :]

        if not return_attn_weights:
            # Efficient attention (no weights returned)
            attn_output = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
            attn_weights = None
        else: # TODO: Somehow, the following code yielded NaNs for val_score, so could not be used for training.
            # Compute attention scores manually to extract weights
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k**0.5

            attn_bias = torch.zeros(Q.size(0), Q.size(1), L_q, L_k, dtype=Q.dtype, device=Q.device)

            # Apply causal mask if needed
            if is_causal:
                # assert attn_mask is None  # TODO: do we need this? can we have attn_mask together with causal
                causal_mask = torch.tril(torch.ones(L_q, L_k, device=Q.device, dtype=torch.bool))
                attn_bias.masked_fill_(~causal_mask, float('-inf'))

            if attn_mask is not None:
                attn_bias.masked_fill_(~attn_mask, float('-inf'))

            attn_scores += attn_bias
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.embed_dim)

        # Final output projection
        out = self.W_o(attn_output)

        if return_attn_weights:
            return out, attn_weights  # (B, L, D), (B, num_heads, L, L)
        else:
            return out


class TransformerEncoderRoPE(nn.Module):
    """
    Transformer encoder layer using RoPE
    Example:
        B, T, C, H, ff = 2, 16, 64, 8, 128
        x = torch.randn(B, T, C)
        enc = TransformerEncoderRoPE(hidden_dim=C, num_head=H, ff_dim=128)

        # Forward with RoPE
        out = enc(x, x, x)
        print(out.shape)
        # (2, 16, 64)
    """
    def __init__(
        self,
        hidden_dim,
        num_head,
        ff_dim,
        nonlinear_module = nn.GELU(),
        dropout = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.head_dim = hidden_dim // num_head

        self.attention = MultiheadAttentionRoPE(hidden_dim, num_head)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nonlinear_module,
            nn.Linear(ff_dim, hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query, 
        key, 
        value, 
        attn_mask=None, 
        is_causal=False, 
        return_attn_weights=False,
        is_bigbird=False, # whether to use bigbird style sparse attention
        attention_window_size=3, # bigbird style sparse attention
        num_global=2, # bigbird style sparse attention
        num_random=1, # bigbird style sparse attention
    ):
        # Build rotary embeddings
        batch_size, seq_len, D = query.shape
        sin, cos = build_rotary_emb(seq_len=seq_len, dim_head=self.head_dim, device=query.device)

        # Forward with RoPE
        if return_attn_weights:
            out, attn_weights= self.attention(
                query, 
                key, 
                value, 
                sin=sin, 
                cos=cos, 
                attn_mask=attn_mask, 
                is_causal=is_causal, 
                return_attn_weights=return_attn_weights,
                is_bigbird=is_bigbird,
                attention_window_size=attention_window_size,
                num_global=num_global,
                num_random=num_random,
            )
        else:
            out = self.attention(
                query, 
                key, 
                value, 
                sin=sin, 
                cos=cos, 
                attn_mask=attn_mask, 
                is_causal=is_causal, 
                return_attn_weights=return_attn_weights,
                is_bigbird=is_bigbird,
                attention_window_size=attention_window_size,
                num_global=num_global,
                num_random=num_random,
            )
        
        out = out + self.dropout(out)
        out = self.norm1(out)

        out = out + self.dropout(self.ff(out))
        out = self.norm2(out)

        if return_attn_weights:
            return out, attn_weights 
        else:
            return out
        

class RBFBasis(torch.nn.Module):
    """
    Spatial basis
    RBF (Gaussian place basis)
    """
    def __init__(self, centers, sigma):
        """
        centers: (K, 3) tensor
        sigma: float
        """
        super().__init__()
        self.register_buffer("centers", centers)
        self.sigma = sigma

    def forward(self, x):
        """
        x: (Batch_size, T, 3)
        returns: (Batch_size, T, K)
        """
        diff = x[:, :, None, :] - self.centers[None, None, :, :] # (B, T, 1, 3) - (1, 1, K, 3)
        dist2 = (diff ** 2).sum(dim=-1)
        return torch.exp(-dist2 / (2 * self.sigma ** 2))
