import torch
import torch.nn as nn


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
