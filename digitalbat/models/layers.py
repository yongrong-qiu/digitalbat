import torch
from torch import nn
import torch.nn.functional as F



class ELU1(nn.Module):
    """
    Adapted from <https://github.com/sinzlab/neuralpredictors/blob/main/neuralpredictors/layers/activations.py>
    Elu activation function shifted by 1 to ensure that the
    output stays positive. That is:
    Elu1(x) = Elu(x) + 1
    """

    @staticmethod
    def elu1(x, inplace=False, eps=0.0):
        return F.elu(x, inplace=inplace) + 1.0 + eps

    def forward(self, x, inplace=False, eps=0):
        return ELU1.elu1(x, inplace, eps)


class Sin(nn.Module):
    """
    The sine of the input
    """

    def forward(self, x):
        return torch.sin(x)
    

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()
    

class BCEFocalCombinedLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 2.0,
        focal_weight: float = 0.5,
        pos_weight: torch.Tensor = None,
        reduction: str = 'mean',
    ):
        """
        Combines BCEWithLogitsLoss and Focal Loss for imbalanced binary classification.

        Parameters
        ----------
        alpha : float
            Scaling factor for focal loss.
        gamma : float
            Focusing parameter for focal loss.
        focal_weight : float
            Weight for focal loss in the final combination. (1 - focal_weight) is used for BCE.
        pos_weight : torch.Tensor, optional
            A weight of positive examples. Passed to BCEWithLogitsLoss.
        reduction : str
            'mean' | 'sum' | 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight
        self.reduction = reduction

        self.bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE with logits part
        bce_loss = self.bce_loss_fn(logits, targets)

        # Focal loss part
        bce_no_reduce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_no_reduce)  # probability for the correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_no_reduce

        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        # else: keep as is

        # Combine the two
        combined_loss = self.focal_weight * focal_loss + (1 - self.focal_weight) * bce_loss
        return combined_loss
