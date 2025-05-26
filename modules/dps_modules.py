# coding: utf-8
"""Auxiliary module for DPS (Diversity Preference Scale) prediction in a recommendation system."""

import torch
import torch.nn as nn


class DPSPredictor(nn.Module):
    def __init__(self, emb_dim: int):
        """
        Args:
            emb_dim (int): Dimension of user embeddings
        """
        super().__init__()

        # Define a separate pipeline (linear layer) for each feature
        self.actor_fc = nn.Linear(emb_dim, 1)
        self.country_fc = nn.Linear(emb_dim, 1)
        self.director_fc = nn.Linear(emb_dim, 1)
        self.genre_fc = nn.Linear(emb_dim, 1)

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
            "actor_dps": torch.sigmoid(self.actor_fc(input_user_emb)).squeeze(-1),
            "country_dps": torch.sigmoid(self.country_fc(input_user_emb)).squeeze(-1),
            "director_dps": torch.sigmoid(self.director_fc(input_user_emb)).squeeze(-1),
            "genre_dps": torch.sigmoid(self.genre_fc(input_user_emb)).squeeze(-1),
        }
