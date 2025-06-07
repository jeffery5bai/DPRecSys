import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self):
        super().__init__()

    def forward(self, *embeddings):
        l2_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            l2_loss += torch.sum(embedding**2) * 0.5
        return l2_loss


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
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')

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
