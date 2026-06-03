import torch
from torch import nn

from .layers import TransformerEncoderRoPE, RBFBasis


class _PerTimestepFFN(nn.Module):
    """Per-timestep position-wise MLP block, structurally matched to
    TransformerEncoderRoPE's FFN sub-block (LayerNorm + Linear-GELU-Linear +
    dropout + residual). No attention, no temporal mixing. Used by
    Bat_Transformer when use_attention=False to ablate temporal attention
    while preserving per-timestep capacity."""
    def __init__(self, hidden_dim, ff_dim, dropout):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, *args, **kwargs):
        return self.norm(x + self.dropout(self.ff(x)))


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
        # place
        use_place_as_input: bool = True, # use place as input
        # add_spatial_basis: bool = False, # whether to add spatial bases to the model input or not
        basis_type: str = "RBF", # type of spatial basis
        # basis_num: int = 32, # number of basis
        spatial_centers_sigma_dict = None, # a dictionary of spatial centers and sigma for each session
        latent_dim_place: int = 32, # latent_dim for embed_place
        # trajectory
        use_trajectory_as_input: bool = True, # use trajectory as input
        latent_dim: int = 32,
        num_heads: int = 4,
        ff_dim: int = 32,
        num_layers: int = 1,
        dropout_prob: float = 0.3,
        #
        use_low_rank: bool = False, # whether to use low rank matrix multiplication or not for embed_place
        use_low_rank_readout: bool = False, # whether to use low rank matrix multiplication for readout modules or not
        low_rank_r: int = 4,
        final_activation: nn.Module = nn.Identity(),
        use_attention: bool = True, # if False, replace transformer blocks with per-timestep FFN blocks (ablate temporal attention)
        # environment-keyed trajectory projection: maps each session_name -> a
        # "room"/environment label.  Sessions sharing a room share one embed_traj
        # Linear, so the same (per-session normalised) position is projected
        # differently per environment (remapping handled in the trajectory
        # encoder).  None (default) keeps a single shared embed_traj — fully
        # backward compatible with existing checkpoints.
        session_to_room: dict = None,
    ):
        super().__init__()

        self.use_place_as_input = use_place_as_input
        self.use_trajectory_as_input = use_trajectory_as_input
        self.use_attention = use_attention
        self.session_to_room = session_to_room
        if session_to_room is not None:
            missing = set(output_dim_dict) - set(session_to_room)
            assert not missing, f'session_to_room missing entries for sessions: {sorted(missing)}'
        assert (self.use_place_as_input or self.use_trajectory_as_input), 'Use at least one input (place or trajectory)!'
        
        if self.use_place_as_input:
            # discrete spatial embdding
            if basis_type == 'RBF':
                self.spatial_basis = nn.ModuleDict({
                    session_name: RBFBasis(torch.from_numpy(centers).float(), sigma)
                    for session_name, (centers, sigma) in spatial_centers_sigma_dict.items()
                })
            self.basis_num = len(list(spatial_centers_sigma_dict.values())[0][0]) # number of basis
            if use_low_rank:
                # self.embed_place = nn.ModuleDict({
                #     session_name: nn.Sequential(nn.Linear(self.basis_num, low_rank_r), nn.Linear(low_rank_r, latent_dim_place), nn.Dropout(0.1))
                #     for session_name in spatial_centers_sigma_dict.keys()
                # })
                self.embed_place = nn.Sequential(
                    nn.Linear(self.basis_num, low_rank_r), 
                    nn.Linear(low_rank_r, latent_dim_place), 
                    nn.ReLU(),
                    nn.Linear(latent_dim_place, low_rank_r), 
                    nn.Linear(low_rank_r, latent_dim_place), 
                    nn.ReLU(),
                    nn.Dropout(0.1),
                )
            else:
                # self.embed_place = nn.ModuleDict({
                #     session_name: nn.Sequential(nn.Linear(self.basis_num, latent_dim_place), nn.Dropout(0.3))
                #     for session_name in spatial_centers_sigma_dict.keys()
                # })
                self.embed_place = nn.Sequential(
                    nn.Linear(self.basis_num, latent_dim_place), 
                    nn.ReLU(),
                    nn.Linear(latent_dim_place, latent_dim_place), 
                    nn.ReLU(),
                    nn.Dropout(0.3),
                )

        if self.use_trajectory_as_input:
            if session_to_room is not None:
                # one trajectory projection per unique room/environment
                rooms = sorted(set(session_to_room.values()))
                self.embed_traj = nn.ModuleDict({
                    room: nn.Linear(input_dim, latent_dim) for room in rooms
                })
            else:
                self.embed_traj = nn.Linear(input_dim, latent_dim) # trajectory embedding
            self.embed_traj_dropout = nn.Dropout(0.1)
            # transformer or per-timestep FFN ablation
            if self.use_attention:
                self.layers = nn.ModuleList([
                    TransformerEncoderRoPE(hidden_dim=latent_dim, num_head=num_heads, ff_dim=ff_dim,
                                           dropout=dropout_prob, use_low_rank=use_low_rank, low_rank_r=low_rank_r) for _ in range(num_layers)
                ])
            else:
                self.layers = nn.ModuleList([
                    _PerTimestepFFN(hidden_dim=latent_dim, ff_dim=ff_dim, dropout=dropout_prob)
                    for _ in range(num_layers)
                ])

        # session-specific readouts
        readout_latent_dim = latent_dim_place * int(self.use_place_as_input) + latent_dim * int(self.use_trajectory_as_input)
        if use_low_rank_readout:
            self.readouts = nn.ModuleDict({
                session_name: nn.Sequential(nn.Linear(readout_latent_dim, low_rank_r), nn.Linear(low_rank_r, unit_num))
                for session_name, unit_num in output_dim_dict.items()
            })
        else:
            self.readouts = nn.ModuleDict({
                session_name: nn.Sequential(nn.Linear(readout_latent_dim, unit_num), nn.Dropout(0.1))
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
        
        # place
        if self.use_place_as_input:
            x_place_input = self.spatial_basis[session_name](x[:, :, :3]) #
            x_place_embed = self.embed_place(x_place_input) # self.embed_place[session_name](phi_x)

        # trajectory
        if self.use_trajectory_as_input:
            # embedding (environment-keyed projection when session_to_room is set)
            x_traj_input = x.clone()
            embed_traj = (self.embed_traj[self.session_to_room[session_name]]
                          if self.session_to_room is not None else self.embed_traj)
            x_traj_embed = self.embed_traj_dropout(embed_traj(x_traj_input))
            if not self.use_attention:
                assert not return_attn_weights, 'return_attn_weights requires use_attention=True'
                for layer in self.layers:
                    x_traj_embed = layer(x_traj_embed)
            elif return_attn_weights:
                attn_weights_list = []
                for layer in self.layers:
                    x_traj_embed, attn_weights = layer(
                        x_traj_embed,
                        x_traj_embed,
                        x_traj_embed,
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
                    x_traj_embed = layer(
                        x_traj_embed,
                        x_traj_embed,
                        x_traj_embed,
                        attn_mask=attn_mask,
                        is_causal=is_causal,
                        return_attn_weights=return_attn_weights,
                        is_bigbird=is_bigbird,
                        attention_window_size=attention_window_size,
                        num_global=num_global,
                        num_random=num_random,
                    )

        if self.use_place_as_input and (not self.use_trajectory_as_input):
            x_embed = x_place_embed.clone()
        elif (not self.use_place_as_input) and self.use_trajectory_as_input:
            x_embed = x_traj_embed.clone()
        elif self.use_place_as_input and self.use_trajectory_as_input:
            x_embed = torch.cat((x_place_embed, x_traj_embed), dim=-1)

        # readout
        x_embed = self.readouts[session_name](x_embed)
        x_out = self.final_activation(x_embed)

        if return_attn_weights:
            return x_out, attn_weights_list
        else:
            return x_out