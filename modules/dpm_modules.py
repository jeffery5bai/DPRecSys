# coding: utf-8
"""Auxiliary module for DPM (Diversity Preference Matching) in a recommendation system."""

from typing import Dict, List

import torch
import torch.nn as nn
from common.utils import segment_reduce


class DPMatcher(nn.Module):
    def __init__(
        self,
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
        self.features = features

    def forward(
        self, 
        yp_scores: torch.Tensor,
        user_idx: torch.Tensor,  
        item_vec: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            yp_scores (Tensor): [batch_size, 1] predicted scores for the user-item pair
            user_idx (Tensor): [batch_size, 1] user indices in the batch
            item_vec (Dict[str, Tensor]): Dictionary containing item feature vectors,
                                          each of shape [batch_size, emb_dim]
        Returns:
            A dict with expected DPM probability distributions for each feature (num_users, D)
        """
        device = user_idx.device
        dpm_prob_dist = {}

        for feature in self.features:
            feat_vec = item_vec[f"{feature}_vec"]
            expected_feature_vec = torch.sigmoid(yp_scores).unsqueeze(1) * feat_vec
            agg_expected_feature_vec_by_user = segment_reduce(expected_feature_vec, user_idx, "sum", device)  # (n_users,)
            dpm_prob_dist[f"{feature}_pd"] = agg_expected_feature_vec_by_user

        return dpm_prob_dist
