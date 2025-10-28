import torch
from torch import nn

from .layers import TransformerEncoderRoPE


class Bat_Transformer(nn.Module):
    """
    Flexible MLP model for spatial inputs.
    Compared to MLP(), this model uses session-wise readout layers without Fourier features.

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

    """
    def __init__(
        self,
        input_dim: int,
        output_dim_dict: dict, # a dictionary of spiking unit numbers for each session, e.g., {'Dataset_3_14543_240419': 289, 'Dataset_1_32622_231009': 95}
        latent_dim: int = 64,
        num_heads: int = 4,
        ff_dim: int = 64,
        num_layers: int = 1,
        dropout_prob: float = 0.0,
        final_activation: nn.Module = nn.Identity(),
    ):
        super().__init__()
        self.embed = nn.Linear(input_dim, latent_dim) # behavior embedding
        self.layers = nn.ModuleList([
            TransformerEncoderRoPE(hidden_dim=latent_dim, num_head=num_heads, ff_dim=ff_dim, dropout=dropout_prob) for _ in range(num_layers)
        ])
        self.num_heads = num_heads

        # session-specific readouts
        self.readouts = nn.ModuleDict({
            session_name: nn.Linear(latent_dim, unit_num)
            for session_name, unit_num in output_dim_dict.items()
        })

        self.final_activation = final_activation

    def forward(self, session_name: str, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, T, input_dim)
        mask: (B, T) bool, True for valid, False for pad

        Returns
        -------
        torch.Tensor
            Shape (batch_size, T, output_dim)
        """

        # attn_mask
        mask = mask.to(x.device)
        attn_mask = torch.einsum("bi,bj->bij", mask.float(), mask.float()).bool() # (B, T, T), BOOL
        attn_mask = attn_mask[:, None, :, :] # (B, 1, T, T)
        # attn_mask = attn_mask.expand(-1, self.num_heads, -1, -1) # (B, num_heads, T, T)

        x = self.embed(x)

        for layer in self.layers:
            x = layer(x, x, x, attn_mask=attn_mask)
        x = self.readouts[session_name](x)
        x = self.final_activation(x)

        return x