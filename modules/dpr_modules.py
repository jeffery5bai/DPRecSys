# coding: utf-8
"""Auxiliary module for DPR (Diversity Preference Regularization) in a recommendation system."""

from typing import Dict, List

import torch
import torch.nn as nn
from common.utils import segment_reduce


class DPRegularizer(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_layers: int = 3,
        concat_emb: bool = True,
        features: List[str] = ["actor", "country", "director", "genre"],
    ):
        """
        Args:
            emb_dim (int): Dimension of user embeddings
            num_layers (int): Number of layers in the recommendation model
            concat_emb (bool): Whether to concatenate embeddings from all layers
            features (List[str]): List of features for which DPR scores will be predicted
        """
        super().__init__()

        # NOTE: NGCF concatenates embeddings from all layers as final embeddings
        self.features = features
        emb_dim = emb_dim * (num_layers + 1) if concat_emb else emb_dim

        # NOTE: Project item embeddings into another space as feature representation
        # Define separate pipelines (linear layer) for each feature
        self.feature_fcs = nn.ModuleDict({feature: nn.Linear(emb_dim, emb_dim) for feature in features})

    def forward(
        self, user_emb: torch.Tensor, user_idx: torch.Tensor, item_emb: torch.Tensor, item_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            user_emb (Tensor): [n_users, emb_dim] user embeddings
            user_idx (Tensor): [batch_size, 1] user idx in the batch
            item_emb (Tensor): [n_items, emb_dim] item embeddings
            item_idx (Tensor): [batch_size, 1] item idx in the batch
        Returns:
            A dict with predicted DPR scores (in range [0, 1]) for each feature
        """
        device = user_emb.device
        input_user_emb = user_emb[user_idx]  # (batch_size, emb_dim)
        input_item_emb = item_emb[item_idx]  # (batch_size, emb_dim)

        dpr_scores = {}

        for feature in self.features:
            projected_item_emb = self.feature_fcs[feature](input_item_emb)  # (batch_size, emb_dim)
            distance = torch.norm(input_user_emb - projected_item_emb, dim=-1)  # (batch_size,)
            mean_feature_distance_by_user = segment_reduce(distance, user_idx, "mean", device)  # (n_users,)
            dpr_scores[f"{feature}_dpr"] = torch.sigmoid(mean_feature_distance_by_user).squeeze(-1)

        return dpr_scores
