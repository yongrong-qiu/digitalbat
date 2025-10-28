import torch
from torch import nn


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