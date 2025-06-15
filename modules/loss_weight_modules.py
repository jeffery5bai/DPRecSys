import torch
import torch.nn as nn
import torch.nn.functional as F


class LogScaleLoss(nn.Module):
    """
    Dual-Balancing for Multi-Task Learning (Lin et al., 2023) (20 citations)
    This is only the half of DB-MTL method.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, loss):
        return torch.log(loss + self.eps)  # Apply log scaling to the input loss


class EMALossNormalizer(nn.Module):
    def __init__(self, static_weight=1, ema_decay=0.1):
        super().__init__()
        self.static_weight = static_weight
        self.ema_decay = ema_decay
        self.eps = 1e-8
        self.ema_losses = {}

    def forward(self, loss, task_name):
        # Update EMA
        if task_name not in self.ema_losses:
            self.ema_losses[task_name] = loss.detach()
        else:
            self.ema_losses[task_name] = (
                self.ema_decay * loss.detach() + (1 - self.ema_decay) * self.ema_losses[task_name]
            )

        ema = self.ema_losses[task_name].clamp(min=self.eps)  # Avoid division by zero
        # Scale to target
        return loss / (self.static_weight * ema)


