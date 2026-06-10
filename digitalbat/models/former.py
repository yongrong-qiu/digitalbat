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
        # alternative environment conditioning for the trajectory branch, keyed
        # off the SAME session_to_room mapping but with a SHARED embed_traj
        # Linear.  A learnable per-room embedding (nn.Embedding over the unique
        # rooms) is injected into the trajectory sequence:
        #   'segment' : added to every timestep after embed_traj (BERT segment-
        #               embedding style), broadcast over T.
        #   'token'   : prepended as an extra token at position 0 (BERT [CLS]
        #               style); attention conditions all timesteps on it and the
        #               token output is dropped before the readout.
        #   'film'    : FiLM conditioning -- the per-room embedding generates a
        #               feature-wise (gamma, beta) for EACH transformer block, and
        #               the block output is modulated `h <- gamma * h + beta`.
        #               Unlike 'segment' (additive bias only) this is multiplicative
        #               and interleaved with the encoder nonlinearities, so a config
        #               can learn a nonlinear/region-dependent deformation of the
        #               position->latent map.  Initialised to identity (gamma=1,
        #               beta=0) so an untrained film model matches the shared model.
        #   None      : no room embedding (default; backward compatible).
        # Requires session_to_room.  When set, embed_traj is shared (the per-room
        # projection above is disabled) so the room embedding is the sole carrier
        # of environment identity.
        room_embed_mode: str = None,
        # optional readout sharing: maps each session_name -> a readout key.
        # Sessions mapped to the same key share ONE readout head (they must have
        # the same unit count, and row i must be the same neuron across them).
        # Use this to tie matched neurons recorded across configurations to a
        # single readout, so the only per-config degree of freedom is the
        # trajectory-branch warp -- forcing a config-invariant latent.  None
        # (default) gives every session its own readout (backward compatible).
        session_to_readout: dict = None,
    ):
        super().__init__()

        self.use_place_as_input = use_place_as_input
        self.use_trajectory_as_input = use_trajectory_as_input
        self.use_attention = use_attention
        self.session_to_room = session_to_room
        self.room_embed_mode = room_embed_mode
        if session_to_room is not None:
            missing = set(output_dim_dict) - set(session_to_room)
            assert not missing, f'session_to_room missing entries for sessions: {sorted(missing)}'
        if room_embed_mode is not None:
            assert room_embed_mode in ('segment', 'token', 'film'), \
                f"room_embed_mode must be 'segment', 'token', 'film', or None (got {room_embed_mode!r})"
            assert session_to_room is not None, 'room_embed_mode requires session_to_room'
            assert use_trajectory_as_input, 'room_embed_mode requires use_trajectory_as_input'
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
            if session_to_room is not None and room_embed_mode is None:
                # one trajectory projection per unique room/environment
                rooms = sorted(set(session_to_room.values()))
                self.embed_traj = nn.ModuleDict({
                    room: nn.Linear(input_dim, latent_dim) for room in rooms
                })
            else:
                # shared trajectory embedding (also used by room_embed_mode, where
                # environment identity is carried by self.room_embedding instead)
                self.embed_traj = nn.Linear(input_dim, latent_dim)
            self.embed_traj_dropout = nn.Dropout(0.1)
            if room_embed_mode is not None:
                # learnable per-room embedding injected into the trajectory branch
                rooms = sorted(set(session_to_room.values()))
                self.room_to_idx = {room: i for i, room in enumerate(rooms)}
                self.room_embedding = nn.Embedding(len(rooms), latent_dim)
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
            if room_embed_mode == 'film':
                # one FiLM head per transformer block: room embedding -> (gamma, beta).
                # Initialised to identity (weight=0, bias gamma=1/beta=0) so the model
                # starts equal to the shared-embed_traj baseline and learns warps from there.
                self.film_heads = nn.ModuleList([
                    nn.Linear(latent_dim, 2 * latent_dim) for _ in range(num_layers)
                ])
                for head in self.film_heads:
                    nn.init.zeros_(head.weight)
                    with torch.no_grad():
                        head.bias[:latent_dim].fill_(1.0)  # gamma = 1
                        head.bias[latent_dim:].zero_()     # beta = 0

        # session-specific readouts.  `session_to_readout` lets multiple sessions
        # share a head (matched neurons across configurations); default is identity.
        self.session_to_readout = session_to_readout or {s: s for s in output_dim_dict}
        missing_ro = set(output_dim_dict) - set(self.session_to_readout)
        assert not missing_ro, f'session_to_readout missing entries for sessions: {sorted(missing_ro)}'
        # resolve per-readout-key unit counts, asserting sessions that share a key agree
        readout_unit_num = {}
        for session_name, unit_num in output_dim_dict.items():
            key = self.session_to_readout[session_name]
            if key in readout_unit_num:
                assert readout_unit_num[key] == unit_num, (
                    f'sessions sharing readout {key!r} have mismatched unit counts '
                    f'({readout_unit_num[key]} vs {unit_num})')
            else:
                readout_unit_num[key] = unit_num
        readout_latent_dim = latent_dim_place * int(self.use_place_as_input) + latent_dim * int(self.use_trajectory_as_input)
        if use_low_rank_readout:
            self.readouts = nn.ModuleDict({
                key: nn.Sequential(nn.Linear(readout_latent_dim, low_rank_r), nn.Linear(low_rank_r, unit_num))
                for key, unit_num in readout_unit_num.items()
            })
        else:
            self.readouts = nn.ModuleDict({
                key: nn.Sequential(nn.Linear(readout_latent_dim, unit_num), nn.Dropout(0.1))
                for key, unit_num in readout_unit_num.items()
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
            # embedding: environment-keyed per-room projection only when
            # session_to_room is set AND room_embed_mode is None; otherwise shared.
            x_traj_input = x.clone()
            use_per_room_proj = (self.session_to_room is not None
                                 and self.room_embed_mode is None)
            embed_traj = (self.embed_traj[self.session_to_room[session_name]]
                          if use_per_room_proj else self.embed_traj)
            x_traj_embed = self.embed_traj_dropout(embed_traj(x_traj_input))

            # room embedding injection (segment / token / film), keyed off session_to_room
            attn_mask_traj = attn_mask
            film_params = None  # set below for room_embed_mode == 'film'
            if self.room_embed_mode is not None:
                room_idx = torch.tensor(self.room_to_idx[self.session_to_room[session_name]],
                                        device=x.device)
                room_vec = self.room_embedding(room_idx)  # (latent_dim,)
                if self.room_embed_mode == 'segment':
                    # add to every timestep, broadcast over (B, T, latent_dim)
                    x_traj_embed = x_traj_embed + room_vec
                elif self.room_embed_mode == 'film':
                    # per-block (gamma, beta) from the room embedding; applied after
                    # each transformer block in the loops below.
                    film_params = [head(room_vec).chunk(2, dim=-1) for head in self.film_heads]
                elif self.room_embed_mode == 'token':
                    # prepend as an extra token at position 0 (BERT [CLS] style)
                    B, T, D = x_traj_embed.shape
                    room_tok = room_vec.expand(B, 1, D)
                    x_traj_embed = torch.cat([room_tok, x_traj_embed], dim=1)  # (B, T+1, D)
                    # extend the attention mask: the room token is always valid
                    mask_tok = torch.cat([torch.ones(B, 1, device=mask.device), mask.float()], dim=1)
                    attn_mask_traj = torch.einsum("bi,bj->bij", mask_tok, mask_tok).bool()[:, None]

            if not self.use_attention:
                assert not return_attn_weights, 'return_attn_weights requires use_attention=True'
                for li, layer in enumerate(self.layers):
                    x_traj_embed = layer(x_traj_embed)
                    if film_params is not None:
                        gamma, beta = film_params[li]
                        x_traj_embed = gamma * x_traj_embed + beta
            elif return_attn_weights:
                attn_weights_list = []
                for li, layer in enumerate(self.layers):
                    x_traj_embed, attn_weights = layer(
                        x_traj_embed,
                        x_traj_embed,
                        x_traj_embed,
                        attn_mask=attn_mask_traj,
                        is_causal=is_causal,
                        return_attn_weights=return_attn_weights,
                        is_bigbird=is_bigbird,
                        attention_window_size=attention_window_size,
                        num_global=num_global,
                        num_random=num_random,
                    )
                    if film_params is not None:
                        gamma, beta = film_params[li]
                        x_traj_embed = gamma * x_traj_embed + beta
                    attn_weights_list.append(attn_weights)
            else:
                for li, layer in enumerate(self.layers):
                    x_traj_embed = layer(
                        x_traj_embed,
                        x_traj_embed,
                        x_traj_embed,
                        attn_mask=attn_mask_traj,
                        is_causal=is_causal,
                        return_attn_weights=return_attn_weights,
                        is_bigbird=is_bigbird,
                        attention_window_size=attention_window_size,
                        num_global=num_global,
                        num_random=num_random,
                    )
                    if film_params is not None:
                        gamma, beta = film_params[li]
                        x_traj_embed = gamma * x_traj_embed + beta

            # drop the prepended room token so the trajectory output is back to
            # (B, T, D) and aligns with the per-timestep place branch / readout
            if self.room_embed_mode == 'token':
                x_traj_embed = x_traj_embed[:, 1:, :]

        if self.use_place_as_input and (not self.use_trajectory_as_input):
            x_embed = x_place_embed.clone()
        elif (not self.use_place_as_input) and self.use_trajectory_as_input:
            x_embed = x_traj_embed.clone()
        elif self.use_place_as_input and self.use_trajectory_as_input:
            x_embed = torch.cat((x_place_embed, x_traj_embed), dim=-1)

        # readout (shared across sessions that map to the same readout key)
        x_embed = self.readouts[self.session_to_readout[session_name]](x_embed)
        x_out = self.final_activation(x_embed)

        if return_attn_weights:
            return x_out, attn_weights_list
        else:
            return x_out