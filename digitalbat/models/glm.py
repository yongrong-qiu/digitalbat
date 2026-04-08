import torch
from torch import nn


class Bat_GLM(nn.Module):
    """
    Session-wise GLM for neural encoding.
    Linear model: y = activation(W @ x + b)

    Equivalent to GLM_PyTorch but with session-wise heads.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.

    output_dim_dict : dict
        Dictionary mapping session names to number of neurons.
        e.g., {'Dataset_3_14543_240419': 289, 'Dataset_1_32622_231009': 95}

    activation : str
        Activation function: 'exp', 'softplus', 'sigmoid', or 'linear'.
        Use 'exp' for Poisson regression with PoissonNLLLoss(log_input=False).
    """
    def __init__(
        self,
        input_dim: int,
        output_dim_dict: dict,
        activation: str = 'exp',
    ):
        super().__init__()

        self.readouts = nn.ModuleDict({
            session_name: nn.Linear(input_dim, unit_num)
            for session_name, unit_num in output_dim_dict.items()
        })

        if activation == 'exp':
            self.final_activation = torch.exp
        elif activation == 'softplus':
            self.final_activation = nn.functional.softplus
        elif activation == 'sigmoid':
            self.final_activation = torch.sigmoid
        elif activation == 'linear':
            self.final_activation = lambda x: x
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, session_name: str, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, input_dim) or (batch_size, T, input_dim)

        Returns
        -------
        torch.Tensor
            Shape (batch_size, num_neurons) or (batch_size, T, num_neurons)
        """
        return self.final_activation(self.readouts[session_name](x))
