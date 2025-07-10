import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt


class InterpretModel(ABC):
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device).eval()
        self.device =device

    @abstractmethod
    def compute_feature_importance(self, **kwargs):
        """
        Compute feature importance. Accepts arbitrary keyword arguments.
        Each subclass must document what it expects.
        """
        pass


class Saliency(InterpretModel):
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        super().__init__(model, device)
        self.feature_importance: torch.Tensor | None = None
        self.k: int | None = None
        self.target_index: int | None = None

    def compute_feature_importance(self, input_samples: torch.Tensor, k: int, target_index: int, **kwargs) -> torch.Tensor:
        """
        Compute saliency-based feature importance by averaging over top-k most activating inputs for the given unit.
        """
        with torch.no_grad():
            outputs = self.model(input_samples.to(self.device))
            topk_idx = torch.topk(outputs[:, target_index], k=k).indices

        feature_attributions_list = []
        for idx in topk_idx:
            # Make sure input requires gradient
            sample = input_samples[idx:idx+1].to(self.device).detach().requires_grad_(True)
            # Forward pass
            output = self.model(sample)
            # Focus on the 3rd output unit (index 2)
            unit_output = output[0, target_index]
            # Backward pass to compute gradients w.r.t. input
            unit_output.backward()
            feature_attributions_list.append(sample.grad.detach().squeeze().abs())

        # Stack and average
        avg_attr = torch.stack(feature_attributions_list).mean(dim=0)
        feature_importance = avg_attr / avg_attr.sum()

        self.k = k
        self.target_index = target_index
        self.feature_importance = feature_importance

        return self.feature_importance

    def plot_feature_importance(self) -> None:
        """
        Plot the computed feature importance as a bar chart.
        """
        if self.feature_importance is None:
            raise ValueError("No feature importance computed yet. Call compute_feature_importance() first.")

        fig, ax = plt.subplots(figsize=(4, 1.5))
        ax.bar(range(len(self.feature_importance)), self.feature_importance.cpu().numpy())
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Importance")
        ax.set_title(f"Saliency: Unit {self.target_index} (avg over top-{self.k})")

    def __repr__(self) -> str:
        status = "computed" if self.feature_importance is not None else "not computed"
        return (
            f"<Saliency(model={self.model.__class__.__name__}, "
            f"device='{self.device}', target_index={self.target_index}, "
            f"k={self.k}, feature_importance={status})>"
        )

