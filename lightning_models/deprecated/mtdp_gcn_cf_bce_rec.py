"""This file has been deprecated."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

import torch
from lightning_models.bce.gcn_cf_bce_rec import GCNRecCF
from modules.dps_modules import DPSPredictor
from modules.loss import DPSLoss


class MTDPRecGCNCF(GCNRecCF):
    """
    Wrap up `GraphConvModule` model into Lightning Module as base recommendation model.
    Integrated with multi-task learning for recommendation and collaborative filtering.
    - Auxiliary Task 1: Diversity Preference Scale (DPS)
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

        # NOTE: AT1 - Diversity Preference Scale (DPS)
        self.dps_module = DPSPredictor(emb_dim=self.dim_id, concat_emb=False)
        self.dps_loss_fn = DPSLoss(dps_weights=dps_weights)
        self.dps_weights = self.dps_loss_fn.dps_weights

    def _get_first_occurrence_indices(self, tensor: torch.Tensor):
        """Get the first occurrence indices of unique values in a tensor."""
        unique_vals, inverse_indices = torch.unique(tensor, return_inverse=True)

        # NOTE: Find the first occurrence of each unique value
        first_occurrence_indices = torch.full((unique_vals.size(0),), tensor.size(0), dtype=torch.long)
        first_occurrence_indices = first_occurrence_indices.scatter_reduce_(
            0, inverse_indices, torch.arange(tensor.size(0)), reduce="amin"
        )
        return unique_vals, first_occurrence_indices

    def training_step(self, batch, batch_idx):
        user, item, label = batch["user"], batch["item"], batch["label"]
        dps_label = {
            "actor_dps": batch["actor_dps"],
            "country_dps": batch["country_dps"],
            "director_dps": batch["director_dps"],
            "genre_dps": batch["genre_dps"],
        }

        # NOTE: main task
        user_emb, item_emb = self.forward()
        scores = self._compute_scores(user_emb, item_emb, user, item)
        loss, acc, prec, rec, f1 = self._evaluate(scores, label)
        self.log_dict(
            {"train_rec_loss": loss, "train_acc": acc, "train_prec": prec, "train_rec": rec, "train_f1": f1}
        )

        # NOTE: auxiliary task 1 - Diversity Preference Scale (DPS)
        unique_users, first_indices = self._get_first_occurrence_indices(user)
        unique_dps_label = {k: v[first_indices] for k, v in dps_label.items()}
        dps_scores = self.dps_module(user_emb, unique_users)
        dps_loss = self.dps_loss_fn(dps_scores, unique_dps_label)
        self.log_dict({"train_dps_loss": dps_loss})

        # TODO: weighing the main task and auxiliary task losses
        total_loss = self.mt_weights["rec_loss"] * loss + self.mt_weights["dps_loss"] * dps_loss
        self.log_dict({"train_loss": total_loss})

        return total_loss
