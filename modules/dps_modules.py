# coding: utf-8
"""Auxiliary module for DPS (Diversity Preference Scale) prediction in a recommendation system."""

from typing import List

import torch
import torch.nn as nn


class DPSPredictor(nn.Module):
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
        """
        super().__init__()

        # NOTE: NGCF concatenates embeddings from all layers as final embeddings
        self.features = features
        emb_dim = emb_dim * (num_layers + 1) if concat_emb else emb_dim

        # Define a separate pipeline (linear layer) for each feature
        self.projection_fcs = nn.ModuleDict({feature: nn.Linear(emb_dim, 1) for feature in features})

    def forward(self, user_emb: torch.Tensor, user_idx: torch.Tensor) -> dict:
        """
        Args:
            user_emb (Tensor): [n_users, emb_dim] user embeddings
            user_idx (Tensor): [batch_size, 1] user idx in the batch

        Returns:
            A dict with predicted DPS scores (in range [0, 1]) for each feature
        """
        input_user_emb = user_emb[user_idx]
        return {
            f"{feature}_dps": torch.sigmoid(self.projection_fcs[feature](input_user_emb)).squeeze(-1)
            for feature in self.features
        }
