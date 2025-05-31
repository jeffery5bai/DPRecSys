import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import torch
from lightning_models.ngcf import NGCFRec
from modules.dps_modules import DPSPredictor
from modules.loss import DPSLoss


class MTDPRecV1(NGCFRec):
    """
    `NGCF` model with Auxiliary Tasks.
    - Diversity Preference Scale Prediction (DPS)
    """

    def __init__(self, dps_weights=None, mt_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.mt_weights = (
            mt_weights
            if mt_weights is not None
            else {
                "rec_loss": 1.0,
                "dps_loss": 1.0,
            }
        )

        # NOTE: Auxiliary task 1 - Diversity Preference Scale (DPS)
        self.dps_module = DPSPredictor(emb_dim=self.embedding_dim, num_layers=self.num_layers)
        self.dps_loss_fn = DPSLoss(dps_weights=dps_weights)
        self.dps_weights = self.dps_loss_fn.dps_weights

    def _get_first_occurrence_indices(self, tensor: torch.Tensor):
        """Get the first occurrence indices of unique values in a tensor."""
        unique_vals, inverse_indices = torch.unique(tensor, return_inverse=True)
        unique_vals = unique_vals.to(self.device)
        inverse_indices = inverse_indices.to(self.device)

        # NOTE: Find the first occurrence of each unique value
        first_occurrence_indices = torch.full(
            (unique_vals.size(0),), tensor.size(0), dtype=torch.long, device=self.device
        )
        first_occurrence_indices = first_occurrence_indices.scatter_reduce_(
            0,
            inverse_indices,
            torch.arange(tensor.size(0), device=self.device),
            reduce="amin",
        )
        return unique_vals, first_occurrence_indices

    def training_step(self, batch, batch_idx):
        user, pos_item, neg_item = batch["user"], batch["pos_item"], batch["neg_item"]
        dps_label = {
            "actor_dps": batch["actor_dps"],
            "country_dps": batch["country_dps"],
            "director_dps": batch["director_dps"],
            "genre_dps": batch["genre_dps"],
        }

        # NOTE: Main task: BPR (Pair-wise + L2 regularization loss)
        user_emb, item_emb = self.ngcf_model(training=True)  # Compute embedding for training
        yp_scores = self.forward(user_emb, item_emb, user, pos_item)
        yn_scores = self.forward(user_emb, item_emb, user, neg_item)

        pair_loss = self.bpr_loss(yp_scores, yn_scores)
        reg_loss = self.reg_loss(user_emb, item_emb[pos_item], item_emb[neg_item])
        bpr_loss = pair_loss + self.reg_weight * reg_loss
        self.log_dict(
            {
                "train_bpr_loss": bpr_loss,
                "train_pair_loss": pair_loss,
                "train_reg_loss": self.reg_weight * reg_loss,
            },
            on_epoch=True,
            on_step=True,
        )

        # NOTE: Auxiliary task 1 - Diversity Preference Scale (DPS)
        unique_users, first_indices = self._get_first_occurrence_indices(user)
        unique_dps_label = {k: v[first_indices] for k, v in dps_label.items()}
        dps_scores = self.dps_module(user_emb, unique_users)
        dps_loss = self.dps_loss_fn(dps_scores, unique_dps_label)
        self.log_dict({"train_dps_loss": dps_loss}, on_epoch=True, on_step=True)

        # TODO: weighing the main task and auxiliary task losses
        total_loss = self.mt_weights["rec_loss"] * bpr_loss + self.mt_weights["dps_loss"] * dps_loss
        self.log_dict({"train_loss": total_loss}, on_epoch=True, on_step=True)

        return total_loss
