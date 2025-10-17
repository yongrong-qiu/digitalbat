import torch
from torch import nn

from typing import List, Optional


class MLP(nn.Module):
    """
    Flexible MLP model with optional Fourier features for spatial inputs.

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

    use_fourier : bool
        Whether to apply Fourier features to position inputs (dimensions 1–3).

    fourier_features_dim : int
        Number of Fourier frequencies for each positional input dimension.

    fourier_scale : float
        Scale factor for the Fourier feature frequencies.
    """
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 10,
        latent_dims_list: List[int] = [100, 100],
        latent_activation: nn.Module = nn.ReLU(),
        final_activation: nn.Module = nn.Identity(),
        use_fourier: bool = False,
        fourier_features_dim: int = 16,
        fourier_scale: float = 10.0,
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.use_fourier = use_fourier
        self.fourier_features_dim = fourier_features_dim
        self.fourier_scale = fourier_scale

        # Number of Fourier features added per input dimension
        fourier_expand = 2 * fourier_features_dim if use_fourier else 0

        # Dimensions 1-3 (positions) → add Fourier features
        num_fourier_input_dims = 3 if use_fourier else 0
        self.total_fourier_dim = num_fourier_input_dims * fourier_expand

        self.input_dim_no_fourier = input_dim
        self.effective_input_dim = input_dim - num_fourier_input_dims + self.total_fourier_dim

        # Create learnable layers
        self.layers = nn.ModuleList()

        if latent_dims_list:
            self.layers.append(nn.Linear(self.effective_input_dim, latent_dims_list[0]))
            self.layers.append(nn.BatchNorm1d(latent_dims_list[0]))
            self.layers.append(latent_activation)
            self.layers.append(nn.Dropout(dropout_prob))
            for i in range(len(latent_dims_list) - 1):
                self.layers.append(nn.Linear(latent_dims_list[i], latent_dims_list[i + 1]))
                self.layers.append(nn.BatchNorm1d(latent_dims_list[i + 1]))
                self.layers.append(latent_activation)
                self.layers.append(nn.Dropout(dropout_prob))
            self.layers.append(nn.Linear(latent_dims_list[-1], output_dim))
        else:
            self.layers.append(nn.Linear(self.effective_input_dim, output_dim))

        self.layers.append(final_activation)

        # Register Fourier frequencies as buffers
        if self.use_fourier:
            freqs = torch.exp2(torch.arange(self.fourier_features_dim).float()) * self.fourier_scale
            self.register_buffer("fourier_freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, input_dim)

        Returns
        -------
        torch.Tensor
            Shape (batch_size, output_dim)
        """
        if self.use_fourier:
            # Extract spatial dimensions (0–2)
            pos = x[:, 0:3]  # shape (B, 3)
            freqs = self.fourier_freqs[None, None, :]  # shape (1, 1, F)
            pos = pos[:, :, None] * freqs  # shape (B, 3, F)

            pos_enc = torch.cat([torch.sin(pos), torch.cos(pos)], dim=-1)  # (B, 3, 2F)
            pos_enc = pos_enc.flatten(1)  # shape (B, 3*2F)

            # Concatenate: [pos_enc | flying (if used) | accel/gyro]
            x = torch.cat([pos_enc, x[:, 3:]], dim=-1)

        for layer in self.layers:
            x = layer(x)
        return x
    

class Bat_MLP(nn.Module):
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
        latent_dims_list: List[int] = [100, 100],
        latent_activation: nn.Module = nn.ReLU(),
        final_activation: nn.Module = nn.Identity(),
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        # Create learnable layers
        self.layers = nn.ModuleList()

        if latent_dims_list:
            self.layers.append(nn.Linear(input_dim, latent_dims_list[0]))
            self.layers.append(nn.BatchNorm1d(latent_dims_list[0]))
            self.layers.append(latent_activation)
            self.layers.append(nn.Dropout(dropout_prob))
            for i in range(len(latent_dims_list) - 1):
                self.layers.append(nn.Linear(latent_dims_list[i], latent_dims_list[i + 1]))
                self.layers.append(nn.BatchNorm1d(latent_dims_list[i + 1]))
                self.layers.append(latent_activation)
                self.layers.append(nn.Dropout(dropout_prob))
            
            readout_in_dim = latent_dims_list[-1]

        else:
            readout_in_dim = input_dim

        # session-specific readouts
        self.readouts = nn.ModuleDict({
            session_name: nn.Linear(readout_in_dim, unit_num)
            for session_name, unit_num in output_dim_dict.items()
        })

        self.final_activation = final_activation

    def forward(self, session_name: str, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, input_dim)

        Returns
        -------
        torch.Tensor
            Shape (batch_size, output_dim)
        """

        for layer in self.layers:
            x = layer(x)
        x = self.readouts[session_name](x)
        x = self.final_activation(x)

        return x
    

class Bat_MLP_Behavior(nn.Module):
    """
    Flexible MLP model to predict bats' behavior using neuronal activity.

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
        input_dim_dict: dict, # a dictionary of spiking unit numbers for each session, e.g., {'Dataset_3_14543_240419': 289, 'Dataset_1_32622_231009': 95}
        output_dim: int,
        latent_dims_list: List[int] = [100, 100],
        latent_activation: nn.Module = nn.ReLU(),
        final_activation: nn.Module = nn.Identity(),
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        # Create learnable layers

        # session-specific readins
        readin_out_dim = latent_dims_list[0] if latent_dims_list else output_dim
        self.readins = nn.ModuleDict({
            session_name: nn.Linear(unit_num, readin_out_dim)
            for session_name, unit_num in input_dim_dict.items()
        })

        self.layers = nn.ModuleList()
        if latent_dims_list:
            self.layers.append(nn.BatchNorm1d(readin_out_dim))
            self.layers.append(latent_activation)
            self.layers.append(nn.Dropout(dropout_prob))
            for i in range(len(latent_dims_list) - 1):
                self.layers.append(nn.Linear(latent_dims_list[i], latent_dims_list[i + 1]))
                self.layers.append(nn.BatchNorm1d(latent_dims_list[i + 1]))
                self.layers.append(latent_activation)
                self.layers.append(nn.Dropout(dropout_prob))
            self.layers.append(nn.Linear(latent_dims_list[-1], output_dim))

        else:
            self.layers.append(nn.Dropout(dropout_prob))

        self.final_activation = final_activation

    def forward(self, session_name: str, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, input_dim)

        Returns
        -------
        torch.Tensor
            Shape (batch_size, output_dim)
        """

        x = self.readins[session_name](x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_activation(x)

        return x