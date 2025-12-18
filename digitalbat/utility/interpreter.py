import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import shap
from typing import Callable


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


class Gradient_Bat_Transformer(InterpretModel):
    """
    Compute the gradient of a target_neuron's response at a time point  w.r.t a flight path, for a model Bat_Transformer(). 
    Note that the model input is a flight trajectory.
    """
    def __init__(
        self, 
        model: nn.Module, 
        session_name: str, 
        X_val: torch.Tensor, # flight path; (torch.Size([1, num_of_time_points, 3]),
        y_val: torch.Tensor, # recorded response; (torch.Size([1, num_of_time_points, num_of_neurons]),
        mask_val: torch.Tensor, # mask for the path; torch.Size([1, num_of_time_points]))
        target_neuron_index: int,
        trainer_config: dict,
        device: str = 'cpu',
    ):
        super().__init__(model, device)
        self.session_name = session_name
        self.X_val = X_val.to(self.device)
        self.mask_val = mask_val.to(self.device)
        self.trainer_config = trainer_config
        self.target_neuron_index = target_neuron_index
        self.unmask_T = int(np.sum(mask_val.detach().cpu().numpy())) # the valid time points for this flight trajectory
        self.response_pred = None # predicted responses to X_val for the target_neuron_index; numpy array; shape (T, )
        self.gradient_list = [] # a list of gradient for each time point, each gradient array has a shape of (unmask_T, 3)
        self.response_val = y_val.detach().cpu().numpy()[0, :self.unmask_T, target_neuron_index] # recorded responses to X_val for the target_neuron_index; numpy array; shape (T, )
        self.flight_path = X_val.detach().cpu().numpy()[0, :self.unmask_T, :] # transform X_val from tensor to numpy, and then only use the valid time points

    def compute_feature_importance(self, **kwargs) -> None:
        """
        Compute gradient-based feature importance for the given input and the target unit.
        """

        for ii in range(self.unmask_T):
            # Make sure input requires gradient
            sample = self.X_val.detach().requires_grad_(True)
            # Forward pass
            output = self.model(
                self.session_name, 
                sample, 
                self.mask_val, 
                is_causal=self.trainer_config['is_causal'], 
                is_bigbird=self.trainer_config['is_bigbird'], 
                attention_window_size=self.trainer_config['attention_window_size'],
                num_global=self.trainer_config['num_global'],
            ) # torch.Size([1, T, num_of_neurons])

            # Focus on the 3rd output unit (index 2)
            unit_output = output[0, ii, self.target_neuron_index]
            # Backward pass to compute gradients w.r.t. input
            unit_output.backward()
            self.gradient_list.append(sample.grad.detach().squeeze().cpu().numpy()[:self.unmask_T,:])
            if ii == 0:
                self.response_pred = output.detach().cpu().numpy()[0, :self.unmask_T, self.target_neuron_index]

        return None


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

    def compute_feature_importance(
        self, 
        target_index: int, 
        eval_input: torch.Tensor, 
        eval_target: torch.Tensor, 
        score_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
        permute_number: int = 1, # the number of permutation experiments
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute permutation-based feature importance, here we use AUPRC to measure the predictive performance
        """
        # === Prepare data ===
        # Use a subset of data for evaluation
        eval_input = eval_input.to(self.device)  # shape: [N, 10]
        eval_target = eval_target[:, target_index].to(self.device)  # shape: [N]

        # default score function: AUPRC
        def default_score_fn(y_true_np, y_pred_np):
            """
            y_true: 0 or 1.0
            y_pred: logit
            """
            sigmoid = lambda xx: 1/(1 + np.exp(-xx))
            precision, recall, thresholds = precision_recall_curve(y_true_np, sigmoid(y_pred_np))
            return auc(recall, precision)
        score_fn = score_fn or default_score_fn

        # === Baseline performance ===
        with torch.no_grad():
            base_output = self.model(eval_input)[:, target_index]
            # base_output = base_output.sigmoid()  # since we're using BCEWithLogitsLoss
            # base_score = torch.corrcoef(torch.stack([base_output, eval_target]))[0, 1].item()
            # Alternative: base_score = roc_auc_score(eval_target.cpu(), base_output.cpu())
            # y_val = eval_target.detach().cpu().numpy()
            # y_pred_score = torch.sigmoid(base_output).detach().cpu().numpy()
            # precision, recall, thresholds = precision_recall_curve(y_val, y_pred_score)
            # base_score = auc(recall, precision)
            y_val_np = eval_target.detach().cpu().numpy()
            y_pred_np = base_output.detach().cpu().numpy()
            base_score = score_fn(y_val_np, y_pred_np)

        # print(f"Baseline correlation (unit {unit_idx + 1}): {base_score:.4f}")

        # === Permutation Importance ===
        num_features = eval_input.shape[1]
        importances = []
        for i in range(num_features):
            perm_scores = np.empty((permute_number))
            for jj in range(permute_number):
                # Permute only feature i
                permuted_input = eval_input.clone()
                permuted_input[:, i] = permuted_input[torch.randperm(len(permuted_input)), i]
                with torch.no_grad():
                    perm_output = self.model(permuted_input)[:, target_index]
                    # y_val = eval_target.detach().cpu().numpy()
                    # y_pred_score = torch.sigmoid(perm_output).detach().cpu().numpy()
                    # precision, recall, thresholds = precision_recall_curve(y_val, y_pred_score)
                    # perm_score = auc(recall, precision)
                    # perm_score = torch.corrcoef(torch.stack([perm_output, eval_target]))[0, 1].item()
                    # Alternative: perm_score = roc_auc_score(eval_target.cpu(), perm_output.cpu())
                    y_val_np = eval_target.detach().cpu().numpy()
                    y_pred_np = perm_output.detach().cpu().numpy()
                    perm_score = score_fn(y_val_np, y_pred_np)
                perm_scores[jj] = perm_score
            importance = base_score - np.mean(perm_scores)  # drop in performance
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


class ActivationMaximization(InterpretModel):
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        super().__init__(model, device)
        self.feature_importance: torch.Tensor | None = None
        self.target_index: int | None = None
        self.activations: torch.Tensor | None = None

    def compute_feature_importance(self, target_index: int, input_dim: int, clamp_min: torch.Tensor, clamp_max: torch.Tensor, num_steps: int = 300) -> torch.Tensor:
        """
        Compute the input driving the activation optimally, so it is not really feature importance
        """
        # Start with a random input vector
        # optimized_input = torch.randn((1, input_dim), device=self.device, requires_grad=True)
        optimized_input = torch.empty(1, input_dim).uniform_(0.0, 1.0).to(self.device)
        clamp_min_exp, clamp_max_exp = clamp_min.unsqueeze(0).to(self.device), clamp_max.unsqueeze(0).to(self.device)
        optimized_input = clamp_min_exp + (clamp_max_exp - clamp_min_exp) * optimized_input
        optimized_input.requires_grad_(True)
        # Optimizer
        optimizer = torch.optim.Adam([optimized_input], lr=0.1)
        # Run optimization
        activations = []
        clamp_min, clamp_max = clamp_min.to(self.device), clamp_max.to(self.device)
        # clamp_min, clamp_max = train_input.abs().min(dim=0).values.to(device), train_input.abs().max(dim=0).values.to(device)
        for step in range(num_steps):
            optimizer.zero_grad()
            output = self.model(optimized_input)  # shape: [1, 205]
            loss = -output[0, target_index] + 0.01 * optimized_input.norm()  # maximize activation of unit_idx, add regularization
            loss.backward()
            optimizer.step()
            # Optional: constrain input to a valid range, e.g. [-3, 3]
            with torch.no_grad():
                optimized_input.clamp_(clamp_min, clamp_max)
            activations.append(output.detach()[0, target_index].item())
            # if step % 10 == 0:
            #     print(f"Step {step}, Activation: {output[0, unit_idx].item():.4f}")

        # Detach for inspection
        feature_importance = optimized_input.detach().cpu().squeeze()

        self.feature_importance = feature_importance
        self.target_index = target_index
        self.activations = torch.from_numpy(np.array(activations)) 

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
        ax.set_ylabel("Value")
        ax.set_title(f"Activation maximization: Unit {self.target_index}")

    def plot_activations(self) -> None:
        """
        Plot the activation against optimized steps.
        """
        if self.activations is None:
            raise ValueError("No activation computed yet. Call compute_feature_importance() first.")

        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot(range(len(self.activations)), self.activations)
        ax.set_xlabel("Step")
        ax.set_ylabel("Activation")


class ShapAnalysis(InterpretModel):
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        super().__init__(model, device)
        self.feature_importance: torch.Tensor | None = None
        self.target_index: int | None = None

    def compute_feature_importance(
        self, 
        target_index: int, 
        background: torch.Tensor, 
        test_samples: torch.Tensor,
        shap_plot_flag: bool = False,
    ) -> torch.Tensor:
        """
        Use shap to perform attribution analysis
        """
        # === Prepare data ===
        # SHAP background (reference distribution)
        background = background.to(self.device)
        # Use different samples for attribution
        test_samples = test_samples.to(self.device)
        # === Initialize SHAP Explainer ===
        def model_target_index_numpy_mode(x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()
            x = x.to(self.device)
            with torch.no_grad():
                output = self.model(x)[:, target_index].unsqueeze(1)
            return output.cpu().numpy()
        explainer = shap.Explainer(model_target_index_numpy_mode, background.detach().cpu().numpy())
        # === Compute SHAP values ===
        # This returns a tuple of 3 elements: .values (shap value with a shape of num_of_samples by num_of_features), .base_values, .data
        shap_explanation_tuple = explainer(test_samples.detach().cpu().numpy()) 
        # print(f'shap_explanation_tuple: {shap_explanation_tuple}')
        if shap_plot_flag:
            shap.plots.beeswarm(shap_explanation_tuple)
        feature_importance = torch.from_numpy(np.abs(np.array(shap_explanation_tuple.values)[:, :]).mean(axis=0)) 

        self.feature_importance = feature_importance
        self.target_index = target_index

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
        ax.set_title(f"Shap: Unit {self.target_index}")
