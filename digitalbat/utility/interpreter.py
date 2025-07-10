import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, auc


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


class IntegGradi(InterpretModel):
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        super().__init__(model, device)
        self.feature_importance: torch.Tensor | None = None
        self.k: int | None = None
        self.target_index: int | None = None

    def compute_feature_importance(
        self, 
        input_samples: torch.Tensor, 
        k: int, 
        target_index: int, 
        baseline_input: torch.Tensor,
        n_steps: int = 50, 
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute integrated-gradient-based feature importance by averaging over top-k most activating inputs for the given unit.
        """
        # Initialize Integrated Gradients
        ig = IntegratedGradients(self.model)

        # Average over Top-K Inputs
        with torch.no_grad():
            outputs = self.model(input_samples.to(self.device))
            topk_idx = torch.topk(outputs[:, target_index], k=k).indices

        # Optional: baseline input 
        # baseline_input = torch.zeros((1, train_input.shape[1]), device=device)
        # baseline_input = input_samples.mean(dim=0, keepdim=True).to(self.device)
        baseline_input = baseline_input.to(self.device)
        # Collect attributions
        feature_attributions_list = []
        for idx in topk_idx:
            input_sample = input_samples[idx:idx+1].to(self.device)
            # Compute IG attribution for target output unit
            attr = ig.attribute(
                inputs=input_sample,
                baselines=baseline_input,
                target=int(target_index),
                n_steps=n_steps,
                internal_batch_size=None  # Optional: tune if you have memory issues
            )
            feature_attributions_list.append(attr.squeeze().abs())

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
        ax.set_title(f"Int. gra.: Unit {self.target_index} (avg over top-{self.k})")


class PermutationImportance(InterpretModel):
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        super().__init__(model, device)
        self.feature_importance: torch.Tensor | None = None
        self.target_index: int | None = None

    def compute_feature_importance(self, target_index: int, eval_input: torch.Tensor, eval_target: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute permutation-based feature importance, here we use AUPRC to measure the predictive performance
        """
        # === Prepare data ===
        # Use a subset of data for evaluation
        eval_input = eval_input.to(self.device)  # shape: [N, 10]
        eval_target = eval_target[:, target_index].to(self.device)  # shape: [N]
        # === Baseline performance ===
        with torch.no_grad():
            base_output = self.model(eval_input)[:, target_index]
            # base_output = base_output.sigmoid()  # since we're using BCEWithLogitsLoss
            # base_score = torch.corrcoef(torch.stack([base_output, eval_target]))[0, 1].item()
            # Alternative: base_score = roc_auc_score(eval_target.cpu(), base_output.cpu())
            y_val = eval_target.detach().cpu().numpy()
            y_pred_score = torch.sigmoid(base_output).detach().cpu().numpy()
            precision, recall, thresholds = precision_recall_curve(y_val, y_pred_score)
            base_score = auc(recall, precision)
        # print(f"Baseline correlation (unit {unit_idx + 1}): {base_score:.4f}")

        # === Permutation Importance ===
        num_features = eval_input.shape[1]
        importances = []
        for i in range(num_features):
            # Permute only feature i
            permuted_input = eval_input.clone()
            permuted_input[:, i] = permuted_input[torch.randperm(len(permuted_input)), i]
            with torch.no_grad():
                perm_output = self.model(permuted_input)[:, target_index]
                y_val = eval_target.detach().cpu().numpy()
                y_pred_score = torch.sigmoid(perm_output).detach().cpu().numpy()
                precision, recall, thresholds = precision_recall_curve(y_val, y_pred_score)
                perm_score = auc(recall, precision)
                # perm_score = torch.corrcoef(torch.stack([perm_output, eval_target]))[0, 1].item()
                # Alternative: perm_score = roc_auc_score(eval_target.cpu(), perm_output.cpu())
            importance = base_score - perm_score  # drop in performance
            importances.append(importance)
            # print(f"Feature {i}: Importance = {importance:.4f}")

        feature_importance = torch.from_numpy(np.array(importances)) 

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
        ax.set_title(f"Permutation: Unit {self.target_index}")