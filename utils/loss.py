import math
import torch
import torch.nn as nn

def empty_onehot(target: torch.Tensor, num_classes: int) -> torch.Tensor: 
    """Create an all-zero one-hot tensor for the given target shape and num_classes."""
    #주어진 target 크기에 맞춰, 0으로 채워진 원-핫 텐서 만든다.
    onehot_size = target.size() + (num_classes,)
    return target.new_zeros(onehot_size, dtype=torch.float)


def to_onehot(target: torch.Tensor, num_classes: int, src_onehot: torch.Tensor = None) -> torch.Tensor:
    """Convert integer target tensor to one-hot (optionally reusing src_onehot)."""
    #정수 인덱스 target을 (batch, seq_len, num_classes) 형태의 원-핫으로 바꿔 줍니다.
    if src_onehot is None:
        one_hot = empty_onehot(target, num_classes)
    else:
        one_hot = src_onehot

    last_dim = one_hot.dim() - 1
    return one_hot.scatter_(dim=last_dim, index=target.unsqueeze(last_dim), value=1.0)


class CrossEntropyLossWithGaussianSmoothedLabels(nn.Module):
    """
    CrossEntropy Loss with Gaussian label smoothing: assigns Gaussian weights to neighboring classes.
    Useful for tasks like pitch or RMS classification where nearby bins should be softly penalized.
    """
    def __init__(self, num_classes: int, blur_range: int = 2, sigma: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.blur_range = blur_range #GT 클래스에서 ±d 이웃까지 스무딩할지 범위
        self.sigma = sigma #가우시안 폭 (스무딩 강도)
        # precompute Gaussian decay values
        self.gaussian_decays = [math.exp(- (d ** 2) / (2 * sigma ** 2)) for d in range(blur_range + 1)]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: (batch, num_classes, seq_len) or (batch, num_classes)
        # target: (batch, seq_len) or (batch,)
        # Move classes to last dim
        if pred.dim() == 3:
            pred = pred.transpose(1, 2)  # (batch, seq_len, num_classes)
        else:
            pred = pred.unsqueeze(1).transpose(1, 2)  # (batch, 1, num_classes)
        logit = torch.log_softmax(pred, dim=-1)

        # create smoothed label
        target_smoothed = self._gaussian_smoothed_labels(target)
        # compute loss
        loss = -(logit * target_smoothed).sum(dim=-1).mean()
        return loss

    def _gaussian_smoothed_labels(self, target: torch.Tensor) -> torch.Tensor:
        # target: (batch, seq_len)
        onehot = empty_onehot(target, self.num_classes).to(target.device)
        # apply Gaussian blur around target index
        for d in range(1, self.blur_range + 1):
            decay = self.gaussian_decays[d]
            # positive direction
            pos_idx = (target + d).clamp(max=self.num_classes - 1)
            onehot.scatter_(dim=-1, index=pos_idx.unsqueeze(-1), value=decay)
            # negative direction
            neg_idx = (target - d).clamp(min=0)
            onehot.scatter_(dim=-1, index=neg_idx.unsqueeze(-1), value=decay)
        # ensure center is value=1
        onehot.scatter_(dim=-1, index=target.unsqueeze(-1), value=1.0)
        # normalize rows to sum to 1
        return onehot / onehot.sum(dim=-1, keepdim=True)


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FocalLossWithGaussianSmoothedLabels(nn.Module):
    def __init__(self, num_classes: int, gamma: float = 2.0, blur_range: int = 2, sigma: float = 1.0,
                 class_weights: torch.Tensor = None, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.blur_range = blur_range
        self.sigma = sigma
        self.reduction = reduction
        self.class_weights = class_weights
        self.gaussian_decays = [math.exp(-(d ** 2) / (2 * sigma ** 2)) for d in range(blur_range + 1)]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: (B, C, T) or (B, C)
        # target: (B, T) or (B,)
        if pred.dim() == 3:
            pred = pred.transpose(1, 2)  # (B, T, C)
        else:
            pred = pred.unsqueeze(1).transpose(1, 2)  # (B, 1, C)

        log_probs = F.log_softmax(pred, dim=-1)      # (B, T, C)
        probs = torch.exp(log_probs)

        smoothed_target = self._gaussian_smoothed_labels(target).to(pred.device)

        if self.class_weights is not None:
            weights = self.class_weights.to(pred.device)  # (C,)
            smoothed_target = smoothed_target * weights

        focal_factor = (1.0 - probs) ** self.gamma  # (B, T, C)
        loss = -(focal_factor * smoothed_target * log_probs).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def _gaussian_smoothed_labels(self, target: torch.Tensor) -> torch.Tensor:
        onehot = empty_onehot(target, self.num_classes).to(target.device)
        for d in range(1, self.blur_range + 1):
            decay = self.gaussian_decays[d]
            pos_idx = (target + d).clamp(max=self.num_classes - 1)
            neg_idx = (target - d).clamp(min=0)
            onehot.scatter_(dim=-1, index=pos_idx.unsqueeze(-1), value=decay)
            onehot.scatter_(dim=-1, index=neg_idx.unsqueeze(-1), value=decay)
        onehot.scatter_(dim=-1, index=target.unsqueeze(-1), value=1.0)
        return onehot / onehot.sum(dim=-1, keepdim=True)


# def empty_onehot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
#     shape = target.size() + (num_classes,)
#     return target.new_zeros(shape, dtype=torch.float)



def tolerant_accuracy(predictions: torch.Tensor, targets: torch.Tensor, tolerance: int = 1) -> torch.Tensor:
    """
    Calculate accuracy with tolerance for adjacent classes.

    Args:
        predictions (torch.Tensor): Tensor of shape (batch, num_classes, num_length).
                                   Predicted logits or probabilities.
        targets (torch.Tensor): Tensor of shape (batch, num_length).
                                Ground truth class indices.
        tolerance (int): Tolerance window size for adjacent classes.

    Returns:
        torch.Tensor: Accuracy with tolerance.

    Raises:
        ValueError: If inputs have incompatible shapes or invalid values.
    """
    # Validate inputs
    if predictions.ndim != 3:
        raise ValueError("predictions must be a 3D tensor of shape (batch, num_classes, num_length).")
    if targets.ndim != 2:
        raise ValueError("targets must be a 2D tensor of shape (batch, num_length).")
    if predictions.size(0) != targets.size(0):
        raise ValueError("Batch size of predictions and targets must match.")
    if predictions.size(2) != targets.size(1):
        raise ValueError("The sequence length of predictions and targets must match.")
    if tolerance < 0:
        raise ValueError("Tolerance must be a non-negative integer.")

    # Get the predicted classes (argmax along num_classes axis)
    pred_classes = torch.argmax(predictions, dim=1)  # Shape: (batch, num_length)

    batch_size, num_length = targets.shape
    num_classes = predictions.size(1)

    # Create a range tensor to handle tolerance in a vectorized way
    range_tensor = torch.arange(-tolerance, tolerance + 1, device=targets.device).view(1, -1, 1)  # Shape: (1, 2 * tolerance + 1, 1)

    # Expand targets to match the range_tensor for broadcasting
    expanded_targets = targets.unsqueeze(1) + range_tensor  # Shape: (batch, 2 * tolerance + 1, num_length)
    expanded_targets = torch.clamp(expanded_targets, min=0, max=num_classes - 1)  # Clamp to valid class range

    # Check if pred_classes match any of the valid expanded targets
    valid_matches = (pred_classes.unsqueeze(1) == expanded_targets).any(dim=1)  # Shape: (batch, num_length)

    # Calculate accuracy
    accurate_count = valid_matches.sum().float()
    total_count = targets.numel()

    return accurate_count / total_count