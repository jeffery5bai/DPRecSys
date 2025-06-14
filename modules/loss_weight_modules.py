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


class EMALossNormalizer:
    def __init__(self, target_scale=1e-2):
        self.target_scale = target_scale
        self.ema_losses = {}
        self.ema_decay = 0.99
    
    def normalize_loss(self, loss, task_name):
        # Update EMA
        if task_name not in self.ema_losses:
            self.ema_losses[task_name] = loss.detach()
        else:
            self.ema_losses[task_name] = (
                self.ema_decay * self.ema_losses[task_name] + 
                (1 - self.ema_decay) * loss.detach()
            )
        
        # Scale to target
        return loss * (self.target_scale / self.ema_losses[task_name])
