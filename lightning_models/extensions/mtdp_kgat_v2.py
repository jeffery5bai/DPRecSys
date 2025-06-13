import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import torch
import torch.nn.functional as F
from lightning_models.extensions.kgat_v2 import KGATRecV2
from modules.dpm_modules import DPMatcher
from modules.dpr_modules import DPRegularizer
from modules.dps_modules import DPSPredictor
from modules.loss import DPRLoss, DPSLoss, KLDivergenceLoss


class MTDPRec(KGATRecV2):
    """
    `KGAT` model with Auxiliary Tasks.
    - Diversity Preference Scale Prediction (DPS) (S)
    - Diversity Preference Regularization (DPR) (R)
    - Diversity Preference Matching (DPM) (M)
    """

    def __init__(self, dps_weights=None, dpr_weights=None, dpm_weights=None, mt_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.mt_weights = (
            mt_weights
            if mt_weights is not None
            else {
                "rec_loss": 1.0,
                "dps_loss": 1.0,
                "dpr_loss": 1.0,
                "dpm_loss": 1.0,
            }
        )

        # NOTE: Auxiliary task 1 - Diversity Preference Scale (DPS)
        self.dps_module = DPSPredictor(emb_dim=self.embedding_dim, num_layers=self.num_layers)
        self.dps_loss_fn = DPSLoss(dps_weights=dps_weights)
        self.dps_weights = self.dps_loss_fn.dps_weights

        # NOTE: Auxiliary task 2 - Diversity Preference Regularization (DPR)
        self.dpr_module = DPRegularizer(emb_dim=self.embedding_dim, num_layers=self.num_layers)
        self.dpr_loss_fn = DPRLoss(dpr_weights=dpr_weights)
        self.dpr_weights = self.dpr_loss_fn.dpr_weights

        # NOTE: Auxiliary task 3 - Diversity Preference Matching (DPM)
        self.dpm_module = DPMatcher()
        self.dpm_loss_fn = KLDivergenceLoss(dpm_weights=dpm_weights)
        self.dpm_weights = self.dpm_loss_fn.dpm_weights

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
        dpm_label = {
            "actor_pd": batch["actor_wvec"],
            "country_pd": batch["country_wvec"],
            "director_pd": batch["director_wvec"],
            "genre_pd": batch["genre_wvec"],
        }
        dpm_vec = {
            "actor_vec": batch["actor_vec"],
            "country_vec": batch["country_vec"],
            "director_vec": batch["director_vec"],
            "genre_vec": batch["genre_vec"],
        }

        # NOTE: Build mini-batch subgraph (centered to "user")
        if self.use_mini_batch:
            input_nodes = ("user", user)
            subgraph = self.get_subgraph(input_nodes)
            emb_dict = self.kgat_model(subgraph, training=True)
        else:
            emb_dict = self.kgat_model(self.hetero_data, training=True)  # Compute embedding for training

        user_emb = emb_dict["user"]
        item_emb = emb_dict["movie"]

        # NOTE: Main task: BPR (Pair-wise + L2 regularization loss)
        yp_scores = self.forward(user_emb, item_emb, user, pos_item)
        yn_scores = self.forward(user_emb, item_emb, user, neg_item)

        pair_loss = self.bpr_loss(yp_scores, yn_scores)
        reg_loss = self.reg_loss(user_emb, item_emb[pos_item], item_emb[neg_item])
        bpr_loss = pair_loss + self.reg_weight * reg_loss
        self.log_dict(
            {
                "train_bpr_loss": self.mt_weights["rec_loss"] * bpr_loss,
                "train_pair_loss": pair_loss,
                "train_reg_loss": self.reg_weight * reg_loss,
            },
            on_epoch=True,
            on_step=True,
        )

        # NOTE: Calculate KG loss for embedding regularization (TransR)
        kg_loss = self.calc_kg_loss(user_emb, item_emb, user, pos_item, neg_item)
        self.log("train_kg_loss", kg_loss, on_step=True)

        # NOTE: Auxiliary task 1 - Diversity Preference Scale (DPS)
        unique_users, first_indices = self._get_first_occurrence_indices(user)
        unique_dps_label = {k: v[first_indices] for k, v in dps_label.items()}
        dps_scores = self.dps_module(user_emb, unique_users)
        dps_loss = self.dps_loss_fn(dps_scores, unique_dps_label)
        self.log_dict({"train_dps_loss": self.mt_weights["dps_loss"] * dps_loss}, on_epoch=True, on_step=True)

        # NOTE: Auxiliary task 2 - Diversity Preference Regularization (DPR)
        unique_users, first_indices = self._get_first_occurrence_indices(user)
        unique_dps_label = {k: v[first_indices] for k, v in dps_label.items()}
        dpr_scores = self.dpr_module(user_emb, user, item_emb, pos_item)
        dpr_loss = self.dpr_loss_fn(dpr_scores, unique_dps_label)
        self.log_dict({"train_dpr_loss": self.mt_weights["dpr_loss"] * dpr_loss}, on_epoch=True, on_step=True)

        # NOTE: Auxiliary task 3 - Diversity Preference Matching (DPM)
        unique_users, first_indices = self._get_first_occurrence_indices(user)
        unique_dpm_label = {k: v[first_indices] for k, v in dpm_label.items()}
        dpm_prob_dist = self.dpm_module(yp_scores, user, dpm_vec)
        dpm_loss = self.dpm_loss_fn(dpm_prob_dist, unique_dpm_label)
        self.log_dict({"train_dpm_loss": self.mt_weights["dpm_loss"] * dpm_loss}, on_epoch=True, on_step=True)

        # TODO: weighing the main task and auxiliary task losses
        total_loss = (
            self.mt_weights["rec_loss"] * (bpr_loss + kg_loss)
            + self.mt_weights["dps_loss"] * dps_loss
            + self.mt_weights["dpr_loss"] * dpr_loss
            + self.mt_weights["dpm_loss"] * dpm_loss
        )
        self.log_dict({"train_loss": total_loss}, on_epoch=True, on_step=True)

        # NOTE: Gradient Conflict Test
        if self.global_step % 50 == 0:
            # --- Gradient extraction ---
            g_main = self.get_flat_grads(bpr_loss)
            g_dps = self.get_flat_grads(dps_loss)
            g_dpr = self.get_flat_grads(dpr_loss)
            g_dpm = self.get_flat_grads(dpm_loss)

            # --- Cosine similarities ---
            cos = F.cosine_similarity

            cos_main_dps = cos(g_main.unsqueeze(0), g_dps.unsqueeze(0)).item()
            cos_main_dpr = cos(g_main.unsqueeze(0), g_dpr.unsqueeze(0)).item()
            cos_main_dpm = cos(g_main.unsqueeze(0), g_dpm.unsqueeze(0)).item()
            cos_dps_dpr = cos(g_dps.unsqueeze(0), g_dpr.unsqueeze(0)).item()
            cos_dps_dpm = cos(g_dps.unsqueeze(0), g_dpm.unsqueeze(0)).item()
            cos_dpr_dpm = cos(g_dpr.unsqueeze(0), g_dpm.unsqueeze(0)).item()

            self.log_dict(
                {
                    "cos_main_dps": cos_main_dps,
                    "cos_main_dpr": cos_main_dpr,
                    "cos_main_dpm": cos_main_dpm,
                    "cos_dps_dpr": cos_dps_dpr,
                    "cos_dps_dpm": cos_dps_dpm,
                    "cos_dpr_dpm": cos_dpr_dpm,
                },
                prog_bar=True,
            )

        return total_loss

    def get_flat_grads(self, loss):
        self.zero_grad(set_to_none=True)  # Clear existing gradients.
        loss.backward(retain_graph=True)  # Compute grads for analysis only
        grads = []
        for p in self.parameters():
            # If a parameter did not contribute to the loss, substitute with zeros.
            if p.grad is not None:
                grads.append(p.grad.detach().flatten())
            else:
                grads.append(torch.zeros_like(p).flatten())
        flat_grad = torch.cat(grads)
        self.zero_grad(set_to_none=True)
        return flat_grad

    def validation_step(self, batch, batch_idx):
        """We use user-item triplets for validation, so we need to compute scores for each triplet."""
        user, pos_item, neg_item = batch["user"], batch["pos_item"], batch["neg_item"]
        dps_label = {
            "actor_dps": batch["actor_dps"],
            "country_dps": batch["country_dps"],
            "director_dps": batch["director_dps"],
            "genre_dps": batch["genre_dps"],
        }
        dpm_label = {
            "actor_pd": batch["actor_wvec"],
            "country_pd": batch["country_wvec"],
            "director_pd": batch["director_wvec"],
            "genre_pd": batch["genre_wvec"],
        }
        dpm_vec = {
            "actor_vec": batch["actor_vec"],
            "country_vec": batch["country_vec"],
            "director_vec": batch["director_vec"],
            "genre_vec": batch["genre_vec"],
        }

        # NOTE: Main task: BPR (Pair-wise + L2 regularization + KG loss)
        yp_scores = self.forward(self.user_emb, self.item_emb, user, pos_item)
        yn_scores = self.forward(self.user_emb, self.item_emb, user, neg_item)

        pair_loss = self.bpr_loss(yp_scores, yn_scores)
        reg_loss = self.reg_loss(self.user_emb, self.item_emb[pos_item], self.item_emb[neg_item])
        bpr_loss = pair_loss + self.reg_weight * reg_loss
        self.log("val_bpr_loss", bpr_loss, on_epoch=True, on_step=True)

        # Calculate KG loss for embedding regularization (TransR)
        kg_loss = self.calc_kg_loss(self.user_emb, self.item_emb, user, pos_item, neg_item)
        self.log("val_kg_loss", kg_loss, on_epoch=True, on_step=True)

        # NOTE: Auxiliary task 1 - Diversity Preference Scale (DPS)
        unique_users, first_indices = self._get_first_occurrence_indices(user)
        unique_dps_label = {k: v[first_indices] for k, v in dps_label.items()}
        dps_scores = self.dps_module(self.user_emb, unique_users)
        dps_loss = self.dps_loss_fn(dps_scores, unique_dps_label)
        self.log_dict({"val_dps_loss": self.mt_weights["dps_loss"] * dps_loss}, on_epoch=True, on_step=True)

        # NOTE: Auxiliary task 2 - Diversity Preference Regularization (DPR)
        unique_users, first_indices = self._get_first_occurrence_indices(user)
        unique_dps_label = {k: v[first_indices] for k, v in dps_label.items()}
        dpr_scores = self.dpr_module(self.user_emb, user, self.item_emb, pos_item)
        dpr_loss = self.dpr_loss_fn(dpr_scores, unique_dps_label)
        self.log_dict({"val_dpr_loss": self.mt_weights["dpr_loss"] * dpr_loss}, on_epoch=True, on_step=True)

        # NOTE: Auxiliary task 3 - Diversity Preference Matching (DPM)
        unique_users, first_indices = self._get_first_occurrence_indices(user)
        unique_dpm_label = {k: v[first_indices] for k, v in dpm_label.items()}
        dpm_prob_dist = self.dpm_module(yp_scores, user, dpm_vec)
        dpm_loss = self.dpm_loss_fn(dpm_prob_dist, unique_dpm_label)
        self.log_dict({"val_dpm_loss": self.mt_weights["dpm_loss"] * dpm_loss}, on_epoch=True, on_step=True)

        # TODO: weighing the main task and auxiliary task losses
        total_loss = (
            self.mt_weights["rec_loss"] * (bpr_loss + kg_loss)
            + self.mt_weights["dps_loss"] * dps_loss
            + self.mt_weights["dpr_loss"] * dpr_loss
            + self.mt_weights["dpm_loss"] * dpm_loss
        )
        self.log_dict({"val_loss": total_loss}, on_epoch=True, on_step=True)

        self.val_step_outputs.append({})

    def on_validation_epoch_end(self):
        # TODO: Save anything you want to log after model training for analysis
        outputs = self.val_step_outputs
        self.val_results = {}
