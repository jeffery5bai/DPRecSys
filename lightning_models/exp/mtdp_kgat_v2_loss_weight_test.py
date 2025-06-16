"""
This experiment file is used to test the loss weights strategies for the MTDPRec model.
strategies:
- static weights (baseline)
- log-scaling weights
- EMA-scaling weights
- Hybrid (loss scaling + gradient normalization) weights?
"""

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
from modules.loss_weight_modules import EMALossNormalizer, LogScaleLoss


class MTDPRec(KGATRecV2):
    """
    `KGAT` model with Auxiliary Tasks.
    - Diversity Preference Scale Prediction (DPS) (S)
    - Diversity Preference Regularization (DPR) (R)
    - Diversity Preference Matching (DPM) (M)
    """

    def __init__(self, dps_weights=None, dpr_weights=None, dpm_weights=None, mt_weights=None, rescale_method=None, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # # NOTE: [IMPORTANT] set this to manually optimize the model
        # self.automatic_optimization = False
        self.rescale_method = rescale_method

        self.mt_weights = (
            mt_weights
            if mt_weights is not None
            else {
                "bpr_loss": 1.0,
                "kg_loss": 1.0,
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

        # NOTE: Loss scaling module
        self.loss_scaling_module = LogScaleLoss()
        self.ema_normalizer = EMALossNormalizer(static_weight=1.0, ema_decay=0.1)

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
                "train_bpr_loss": self.mt_weights["bpr_loss"] * bpr_loss,
                # "train_pair_loss": pair_loss,
                # "train_reg_loss": self.reg_weight * reg_loss,
            },
            # on_epoch=True,
            on_step=True,
        )

        # NOTE: Calculate KG loss for embedding regularization (TransR)
        kg_loss = self.calc_kg_loss(user_emb, item_emb, user, pos_item, neg_item)
        self.log_dict(
            {
                "train_kg_loss": kg_loss,
            },
            on_step=True,
        )

        # NOTE: Auxiliary task 1 - Diversity Preference Scale (DPS)
        unique_users, first_indices = self._get_first_occurrence_indices(user)
        unique_dps_label = {k: v[first_indices] for k, v in dps_label.items()}
        dps_scores = self.dps_module(user_emb, unique_users)
        dps_loss = self.dps_loss_fn(dps_scores, unique_dps_label)
        self.log_dict({"train_dps_loss": self.mt_weights["dps_loss"] * dps_loss}, on_step=True)

        # NOTE: Auxiliary task 2 - Diversity Preference Regularization (DPR)
        unique_users, first_indices = self._get_first_occurrence_indices(user)
        unique_dps_label = {k: v[first_indices] for k, v in dps_label.items()}
        dpr_scores = self.dpr_module(user_emb, user, item_emb, pos_item)
        dpr_loss = self.dpr_loss_fn(dpr_scores, unique_dps_label)
        self.log_dict({"train_dpr_loss": self.mt_weights["dpr_loss"] * dpr_loss}, on_step=True)

        # NOTE: Auxiliary task 3 - Diversity Preference Matching (DPM)
        unique_users, first_indices = self._get_first_occurrence_indices(user)
        unique_dpm_label = {k: v[first_indices] for k, v in dpm_label.items()}
        dpm_prob_dist = self.dpm_module(yp_scores, user, dpm_vec)
        dpm_loss = self.dpm_loss_fn(dpm_prob_dist, unique_dpm_label)
        self.log_dict({"train_dpm_loss": self.mt_weights["dpm_loss"] * dpm_loss}, on_step=True)

        # TODO: weighing the main task and auxiliary task losses
        total_loss = (
            self.mt_weights["bpr_loss"] * bpr_loss
            + self.mt_weights["kg_loss"] * kg_loss
            + self.mt_weights["dps_loss"] * dps_loss
            + self.mt_weights["dpr_loss"] * dpr_loss
            + self.mt_weights["dpm_loss"] * dpm_loss
        )
        self.log_dict({"train_loss": total_loss}, on_epoch=True, on_step=True)

        # NOTE: Apply loss scaling
        loss_dict = {
            "bpr_loss": bpr_loss,
            "kg_loss": kg_loss,
            "dps_loss": dps_loss,
            "dpr_loss": dpr_loss,
            "dpm_loss": dpm_loss,
        }
        rescaled_total_loss, rescaled_loss_dict = self.rescale_loss_weighting(loss_dict, method="ema")

        # balanced_bpr_loss = self.ema_normalizer(bpr_loss, "bpr_loss")  # self.loss_scaling_module(bpr_loss)
        # balanced_kg_loss = self.ema_normalizer(kg_loss, "kg_loss")  # self.loss_scaling_module(kg_loss)
        # balanced_dps_loss = self.ema_normalizer(dps_loss, "dps_loss")  # self.loss_scaling_module(dps_loss)
        # balanced_dpr_loss = self.ema_normalizer(dpr_loss, "dpr_loss")  # self.loss_scaling_module(dpr_loss)
        # balanced_dpm_loss = self.ema_normalizer(dpm_loss, "dpm_loss")  # self.loss_scaling_module(dpm_loss)

        # self.log_dict(
        #     {
        #         "train_normalized_bpr_loss": balanced_bpr_loss,
        #         "train_normalized_kg_loss": balanced_kg_loss,
        #         "train_normalized_dps_loss": balanced_dps_loss,
        #         "train_normalized_dpr_loss": balanced_dpr_loss,
        #         "train_normalized_dpm_loss": balanced_dpm_loss,
        #     },
        #     # on_epoch=True,
        #     on_step=True,
        # )

        # balanced_total_loss = (
        #     self.mt_weights["bpr_loss"] * balanced_bpr_loss
        #     + self.mt_weights["kg_loss"] * balanced_kg_loss
        #     + self.mt_weights["dps_loss"] * balanced_dps_loss
        #     + self.mt_weights["dpr_loss"] * balanced_dpr_loss
        #     + self.mt_weights["dpm_loss"] * balanced_dpm_loss
        # )
        # self.log_dict({"normalized_train_loss": balanced_total_loss}, on_epoch=True, on_step=True)

        # return balanced_total_loss
        return rescaled_total_loss

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

    def rescale_loss_weighting(self, loss_dict, method: str = "ema"):
        """Dynamically adjust the loss weight based on the loss value."""
        # NOTE: Apply loss scaling
        if method == "ema":
            rescaled_loss_dict = {
                loss_name: self.ema_normalizer(loss, loss_name) for loss_name, loss in loss_dict.items()
            }
        elif method == "log_scale":
            rescaled_loss_dict = {
                loss_name: self.loss_scaling_module(loss) for loss_name, loss in loss_dict.items()
            }
        else:
            raise NotImplementedError(f"Loss normalization method '{method}' is not implemented.")

        rescaled_total_loss = 0
        for loss_name, normalized_loss in rescaled_loss_dict.items():
            rescaled_total_loss += self.mt_weights[loss_name] * normalized_loss

        # NOTE: Log the rescaled losses
        self.log_dict(
            {
                f"train_{method}_{loss_name}": normalized_loss
                for loss_name, normalized_loss in rescaled_loss_dict.items()
            },
            # on_epoch=True,
            on_step=True,
        )
        self.log_dict({"normalized_train_loss": rescaled_total_loss}, on_epoch=True, on_step=True)

        return rescaled_total_loss, rescaled_loss_dict

