import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from common.init import xavier_uniform_initialization


class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super().__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class EmbLoss(nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super().__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss


class L2Loss(nn.Module):
    """L2Loss, regularization on nn.Module"""

    def __init__(self):
        super().__init__()

    def forward(self, *modules: nn.Module):

        device = None
        l2_loss = 0.0

        for module in modules:
            if not isinstance(module, nn.Module):
                raise TypeError(f"L2Loss only supports nn.Module inputs, got {type(module)}")

            for param in module.parameters():
                if param.requires_grad:
                    if device is None:
                        device = param.device
                    l2_loss += 0.5 * torch.sum(param**2)  # simplified for smoother gradients

        if device is None:
            device = torch.device("cpu")

        return l2_loss.to(device)


class DPSLoss(nn.Module):
    """Diversity Preference Scale Loss (DPSLoss), auxiliary loss for diversity preference scale task

    Args:
        - dps_weights(dict): Weights for different diversity preference scale components.

    Forward:
        - scores: dict containing scores for different components, each of shape (N,)
        - label: dict containing labels for different components, each of shape (N,)
    Returns:
        - dps_loss: scalar, the weighted sum of losses for each component
    """

    def __init__(self, dps_weights=None):
        super().__init__()
        self.dps_weights = (
            dps_weights
            if dps_weights is not None
            else {
                "actor_dps": 0.25,
                "country_dps": 0.25,
                "director_dps": 0.25,
                "genre_dps": 0.25,
            }
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, scores, label):
        dps_loss = 0.0
        for key, weight in self.dps_weights.items():
            if key in scores and key in label:
                loss = self.loss_fn(scores[key], label[key])
                dps_loss += weight * loss

        return dps_loss


class DPRLoss(nn.Module):
    """Diversity Preference Regularization Loss (DPRLoss), auxiliary loss for diversity preference regularization task

    Args:
        - dpr_weights(dict): Weights for different diversity preference scale components.

    Forward:
        - scores: dict containing scores for different components, each of shape (N,)
        - label: dict containing labels for different components, each of shape (N,)
    Returns:
        - dpr_loss: scalar, the weighted sum of losses for each component
    """

    def __init__(self, dpr_weights=None):
        super().__init__()
        self.dpr_weights = (
            dpr_weights
            if dpr_weights is not None
            else {
                "actor_dpr": 0.25,
                "country_dpr": 0.25,
                "director_dpr": 0.25,
                "genre_dpr": 0.25,
            }
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, scores, label):
        dpr_loss = 0.0
        for key, weight in self.dpr_weights.items():
            label_key = key.replace("dpr", "dps")
            if key in scores and label_key in label:
                loss = self.loss_fn(scores[key], label[label_key])
                dpr_loss += weight * loss

        return dpr_loss


class KLDivergenceLoss(nn.Module):
    """KL Divergence Loss for 2D probability distributions.

    Each input tensor is expected to have shape (N, D), where each row is a distribution.
    The inputs are first normalized so that each row sums to 1.

    Forward:
        - pred: (N, D) tensor of raw predicted probabilities
        - target: (N, D) tensor of raw target probabilities
    Returns:
        - scalar loss (mean over the batch)

    Example::

        >>> loss_fn = KLDivergenceLoss()
        >>> P = torch.rand(3, 5, requires_grad=True)
        >>> Q = torch.rand(3, 5)
        >>> loss = loss_fn(P, Q)
        >>> loss.backward()
    """

    def __init__(self, dpm_weights=None):
        super().__init__()
        self.dpm_weights = (
            dpm_weights
            if dpm_weights is not None
            else {
                "actor_pd": 0.25,
                "country_pd": 0.25,
                "director_pd": 0.25,
                "genre_pd": 0.25,
            }
        )
        self.loss_fn = nn.KLDivLoss(reduction="batchmean")

    def forward(self, pred, target):
        device = next(iter(pred.values())).device
        dpm_losses = []
        for key, weight in self.dpm_weights.items():
            if key in pred and key in target:
                pred_log_prob = F.log_softmax(pred[key], dim=1)
                target_prob = F.softmax(target[key], dim=1)

                loss = self.loss_fn(pred_log_prob, target_prob)
                dpm_losses.append(weight * loss)

        return torch.stack(dpm_losses).sum() if dpm_losses else torch.tensor(0.0, device=device)


class PersonalizedKLDivergenceLoss(nn.Module):
    """
    [Deprecated] this feature is not useful based on experiment results.
    Trainable per-user per-attribute KL divergence loss with softmax-normalized gating weights.
    Each user learns how much to weigh each attribute-specific KL loss.
    """

    def __init__(self, num_users, use_temperature=False, init_temperature=1.0):
        super().__init__()
        self.attr_keys = ["actor_pd", "country_pd", "director_pd", "genre_pd"]
        self.num_attrs = len(self.attr_keys)
        self.num_users = num_users

        # Trainable unnormalized weights (raw logits), shape: (num_users, num_attrs)
        self.raw_weights = nn.Parameter(torch.zeros(num_users, self.num_attrs))

        # Optional trainable temperature (shared across users/attributes)
        if use_temperature:
            self.temperature = nn.Parameter(torch.tensor(init_temperature))
        else:
            self.register_buffer("temperature", torch.tensor(1.0))  # constant

        # KL Divergence loss (expects log-softmax input)
        self.loss_fn = nn.KLDivLoss(reduction="none")

    def forward(self, pred, target, user_idx):
        """
        pred, target: dict[str, Tensor] of shape (batch_user_size, dim)
        user_indices: LongTensor of shape (batch_user_size,)
        """
        batch_user_size = user_idx.size(0)

        # Extract raw weights for the current batch
        raw_w = self.raw_weights[user_idx]  # (batch_user_size, num_attrs)

        # Normalize via softmax with temperature
        weights = F.softmax(raw_w / self.temperature.clamp(min=1e-5), dim=1)  # (batch_user_size, num_attrs)

        # For each attribute, compute weighted KL loss
        total_loss = 0.0
        for i, key in enumerate(self.attr_keys):
            if key not in pred or key not in target:
                continue
            pred_log_prob = F.log_softmax(pred[key], dim=1)
            target_prob = F.softmax(target[key], dim=1)

            # Individual KL loss
            kl_loss = self.loss_fn(pred_log_prob, target_prob)
            kl_loss = kl_loss.sum(dim=1)

            # Weight each sample's loss by the corresponding attribute weight
            weighted_loss = weights[:, i] * kl_loss  # (batch_user_size,)

            # Sum over batch
            total_loss += weighted_loss.sum()

        return total_loss / batch_user_size
