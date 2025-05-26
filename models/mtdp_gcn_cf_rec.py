import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import torch
import torch.nn.functional as F
from models.gcn_cf_rec import GCNRecCF
from modules.dps_modules import DPSPredictor
from pytorch_lightning import LightningModule


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
        self.dps_module = DPSPredictor(emb_dim=self.dim_id)
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

    def _evaluate_dps(self, scores, label):
        loss_fn = F.mse_loss

        actor_loss = loss_fn(scores["actor_dps"], label["actor_dps"])
        country_loss = loss_fn(scores["country_dps"], label["country_dps"])
        director_loss = loss_fn(scores["director_dps"], label["director_dps"])
        genre_loss = loss_fn(scores["genre_dps"], label["genre_dps"])
        dps_loss = (
            self.dps_weights["actor_dps"] * actor_loss + 
            self.dps_weights["country_dps"] * country_loss + 
            self.dps_weights["director_dps"] * director_loss + 
            self.dps_weights["genre_dps"] * genre_loss
        )

        return dps_loss

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
            {"train_loss": loss, "train_acc": acc, "train_prec": prec, "train_rec": rec, "train_f1": f1}
        )

        # NOTE: auxiliary task 1 - Diversity Preference Scale (DPS)
        dps_scores = self.dps_module(user_emb, user)
        dps_loss = self._evaluate_dps(dps_scores, dps_label)
        self.log_dict({"train_dps_loss": dps_loss})

        # TODO: weighing the main task and auxiliary task losses
        total_loss = self.mt_weights["rec_loss"] * loss + self.mt_weights["dps_loss"] * dps_loss

        return total_loss
