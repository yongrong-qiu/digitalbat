import torch
from torch import nn

from .layers import TransformerEncoderRoPE, RBFBasis


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
        add_spatial_basis: bool = False, # whether to add spatial bases to the model input or not
        basis_type: str = "RBF", # type of spatial basis
        # basis_num: int = 32, # number of basis
        spatial_centers_sigma_dict = None, # a dictionary of spatial centers and sigma for each session
        latent_dim: int = 64,
        num_heads: int = 4,
        ff_dim: int = 64,
        num_layers: int = 1,
        dropout_prob: float = 0.0,
        use_low_rank: bool = False, # whether to use low rank matrix multiplication or not
        use_low_rank_readout: bool = False, # whether to use low rank matrix multiplication for readout modules or not
        low_rank_r: int = 4,
        final_activation: nn.Module = nn.Identity(),
    ):
        super().__init__()
        
        self.add_spatial_basis = add_spatial_basis
        if self.add_spatial_basis:
            self.embed = nn.Linear(input_dim, latent_dim//2) # behavior embedding
            self.embed_dropout = nn.Dropout(0.1)
            # transformer
            self.layers = nn.ModuleList([
                TransformerEncoderRoPE(hidden_dim=latent_dim//2, num_head=num_heads, ff_dim=ff_dim//2,
                                       dropout=dropout_prob, use_low_rank=use_low_rank, low_rank_r=low_rank_r) for _ in range(num_layers)
            ])

            # discrete spatial embdding
            if basis_type == 'RBF':
                self.spatial_basis = nn.ModuleDict({
                    session_name: RBFBasis(torch.from_numpy(centers).float(), sigma)
                    for session_name, (centers, sigma) in spatial_centers_sigma_dict.items()
                })
            self.basis_num = len(list(spatial_centers_sigma_dict.values())[0][0]) # number of basis
            if use_low_rank:
                # self.embed_place = nn.ModuleDict({
                #     session_name: nn.Sequential(nn.Linear(self.basis_num, low_rank_r), nn.Linear(low_rank_r, latent_dim//2), nn.Dropout(0.1))
                #     for session_name in spatial_centers_sigma_dict.keys()
                # })
                self.embed_place = nn.Sequential(
                    nn.Linear(self.basis_num, low_rank_r), 
                    nn.Linear(low_rank_r, latent_dim//2), 
                    nn.ReLU(),
                    nn.Linear(latent_dim//2, low_rank_r), 
                    nn.Linear(low_rank_r, latent_dim//2), 
                    nn.ReLU(),
                    nn.Dropout(0.1),
                )

            else:
                # self.embed_place = nn.ModuleDict({
                #     session_name: nn.Sequential(nn.Linear(self.basis_num, latent_dim//2), nn.Dropout(0.3))
                #     for session_name in spatial_centers_sigma_dict.keys()
                # })
                self.embed_place = nn.Sequential(
                    nn.Linear(self.basis_num, latent_dim//2), 
                    nn.ReLU(),
                    nn.Linear(latent_dim//2, latent_dim//2), 
                    nn.ReLU(),
                    nn.Dropout(0.3),
                )

        else:
            self.embed = nn.Linear(input_dim, latent_dim) # behavior embedding
            self.embed_dropout = nn.Dropout(0.1)
            # transformer
            self.layers = nn.ModuleList([
                TransformerEncoderRoPE(hidden_dim=latent_dim, num_head=num_heads, ff_dim=ff_dim, 
                                       dropout=dropout_prob, use_low_rank=use_low_rank, low_rank_r=low_rank_r) for _ in range(num_layers)
            ])
        self.num_heads = num_heads

        # session-specific readouts
        if use_low_rank_readout:
            self.readouts = nn.ModuleDict({
                session_name: nn.Sequential(nn.Linear(latent_dim, low_rank_r), nn.Linear(low_rank_r, unit_num))
                for session_name, unit_num in output_dim_dict.items()
            })
        else:
            self.readouts = nn.ModuleDict({
                session_name: nn.Sequential(nn.Linear(latent_dim, unit_num), nn.Dropout(0.1))
                for session_name, unit_num in output_dim_dict.items()
            })
        self.final_activation = final_activation

    def forward(
        self, 
        session_name: str, 
        x: torch.Tensor, 
        mask: torch.Tensor,
        is_causal: bool = False, 
        return_attn_weights: bool = False,
        is_bigbird=False, # whether to use bigbird style sparse attention
        attention_window_size=3, # bigbird style sparse attention
        num_global=2, # bigbird style sparse attention
        num_random=1, # bigbird style sparse attention
    ) -> torch.Tensor:
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
        
        # spatial basis
        if self.add_spatial_basis:
            phi_x = self.spatial_basis[session_name](x[:, :, :3]) #
            x_place = self.embed_place(phi_x) # self.embed_place[session_name](phi_x)
        # behavior embedding
        x = self.embed_dropout(self.embed(x)) 

        # transformer
        if return_attn_weights:
            attn_weights_list = []
            for layer in self.layers:
                x, attn_weights = layer(
                    x, 
                    x, 
                    x, 
                    attn_mask=attn_mask, 
                    is_causal=is_causal, 
                    return_attn_weights=return_attn_weights,
                    is_bigbird=is_bigbird,
                    attention_window_size=attention_window_size,
                    num_global=num_global,
                    num_random=num_random,
                )
                attn_weights_list.append(attn_weights)
        else:
            for layer in self.layers:
                x = layer(
                    x, 
                    x, 
                    x, 
                    attn_mask=attn_mask, 
                    is_causal=is_causal, 
                    return_attn_weights=return_attn_weights,
                    is_bigbird=is_bigbird,
                    attention_window_size=attention_window_size,
                    num_global=num_global,
                    num_random=num_random,
                )

        if self.add_spatial_basis:
            x = torch.cat((x, x_place), dim=-1)

        # readout
        x = self.readouts[session_name](x)
        x = self.final_activation(x)

        if return_attn_weights:
            return x, attn_weights_list
        else:
            return x