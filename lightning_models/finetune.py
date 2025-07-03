import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from common.eval import Evaluator
from modules.dpm_modules import DPMatcher
from modules.dpr_modules import DPRegularizer
from modules.dps_modules import DPSPredictor
from modules.emb_modules import EmbeddingWrapper
from modules.loss import (
    BPRLoss,
    DPRLoss,
    DPSLoss,
    EmbLoss,
    KLDivergenceLoss,
    L2Loss,
    PersonalizedKLDivergenceLoss,
)
from modules.loss_weight_modules import EMALossNormalizer, LogScaleLoss


class FTDPRec(LightningModule):
    """
    `ANY` model with Downsteam Tasks.
    - Diversity Preference Scale Prediction (DPS) (S)
    - Diversity Preference Regularization (DPR) (R)
    - Diversity Preference Matching (DPM) (M)
    """

    def __init__(
        self,
        user_emb_tensor,
        item_emb_tensor,
        strategy,  # whether to train the original embeddings
        lora_r=None,  # rank for LoRA, if applicable
        lr=1e-3,
        l2_reg_weight=1e-3,
        rel_dim=16,
        dps_weights=None,
        dpr_weights=None,
        dpm_weights=None,
        mt_weights=None,
        rescale_method=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.lr = lr
        self.l2_reg_weight = l2_reg_weight
        self.rel_dim = rel_dim

        self.user_emb = None
        self.item_emb = None

        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_results = dict()
        self.test_results = dict()

        self.evaluator = Evaluator()

        self.embedding_model = EmbeddingWrapper(user_emb_tensor, item_emb_tensor, strategy=strategy, r=lora_r)
        self.strategy = strategy
        self.lora_r = lora_r
        self.embedding_dim = self.embedding_model.pretrained_user_emb.size(1)

        self.mt_weights = (
            mt_weights
            if mt_weights is not None
            else {
                "dps_loss": 1.0,
                "dpr_loss": 1.0,
                "dpm_loss": 1.0,
            }
        )

        # # NOTE: [IMPORTANT] set this to manually optimize the model
        # self.automatic_optimization = False
        self.rescale_method = rescale_method
        # NOTE: Loss scaling module
        self.l2_regularizer = L2Loss()
        self.loss_scaling_module = LogScaleLoss()
        self.ema_normalizer = EMALossNormalizer(static_weight=1.0, ema_decay=0.1)
        if rescale_method == "gradnorm":
            self.gradnorm_log_w = nn.Parameter(torch.ones(len(self.mt_weights)), requires_grad=True)
            self.gradnorm_initialized_losses = None

        # NOTE: Auxiliary task 1 - Diversity Preference Scale (DPS)
        self.dps_module = DPSPredictor(emb_dim=self.embedding_dim, concat_emb=False)
        self.dps_loss_fn = DPSLoss(dps_weights=dps_weights)
        self.dps_weights = self.dps_loss_fn.dps_weights

        # NOTE: Auxiliary task 2 - Diversity Preference Regularization (DPR)
        self.dpr_module = DPRegularizer(emb_dim=self.embedding_dim, rel_dim=rel_dim, concat_emb=False)
        self.dpr_loss_fn = DPRLoss(dpr_weights=dpr_weights)
        self.dpr_weights = self.dpr_loss_fn.dpr_weights

        # NOTE: Auxiliary task 3 - Diversity Preference Matching (DPM)
        self.dpm_module = DPMatcher()
        self.dpm_loss_fn = KLDivergenceLoss(dpm_weights=dpm_weights)
        # self.dpm_loss_fn = PersonalizedKLDivergenceLoss(num_users=user_emb_tensor.size(0), use_temperature=True, init_temperature=0.5) # TODO: remove this in final version
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

    def forward(self, user_emb, item_emb, user_idx, item_idx):
        # NOTE: the dimension of user_emb = [num_user + 1, emb_dim], item_emb = [num_item + 1, emb_dim]
        # two towers dot product
        return (user_emb[user_idx] * item_emb[item_idx]).sum(dim=1)

    def training_step(self, batch, batch_idx):
        user, pos_item, neg_item = batch["user"], batch["pos_item"], batch["neg_item"]
        # user, item = batch["user"], batch["item"] # TODO: dataset exp (failed to improve performance)

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

        # NOTE: Main task: BPR (Pair-wise + L2 regularization loss)
        user_emb, item_emb = self.embedding_model()  # Compute embedding for training
        yp_scores = self.forward(user_emb, item_emb, user, pos_item)
        yn_scores = self.forward(user_emb, item_emb, user, neg_item)

        loss_dict = self.task_forward(
            user_emb,
            item_emb,
            user,
            pos_item,
            neg_item,
            yp_scores,
            yn_scores,
            dps_label,
            dpm_label,
            dpm_vec,
            mode="train",
        )

        # TODO: weighing the main task and auxiliary task losses
        total_loss = 0
        for loss_name, loss in loss_dict.items():
            total_loss += self.mt_weights[loss_name] * loss if loss_name in self.mt_weights else loss
        self.log_dict({"train_loss": total_loss}, on_epoch=True, on_step=True)

        # NOTE: Apply loss scaling
        if self.rescale_method is not None:
            total_loss, rescaled_loss_dict = self.rescale_loss_weighting(
                loss_dict, method=self.rescale_method
            )

        # NOTE: Gradient Conflict Test
        if self.global_step % 50 == 0:
            self.log_dict(
                self.gradient_conflict_test(
                    loss_dict=rescaled_loss_dict if self.rescale_method else loss_dict
                ),
                prog_bar=True,
            )

        return total_loss

    def task_forward(
        self,
        user_emb,
        item_emb,
        user,
        pos_item,
        neg_item,
        yp_scores,
        yn_scores,
        dps_label,
        dpm_label,
        dpm_vec,
        mode="train",
    ):
        # NOTE: Auxiliary task 1 - Diversity Preference Scale (DPS)
        unique_users, first_indices = self._get_first_occurrence_indices(user)
        unique_dps_label = {k: v[first_indices] for k, v in dps_label.items()}
        dps_scores = self.dps_module(user_emb, unique_users)
        dps_loss = self.dps_loss_fn(dps_scores, unique_dps_label)
        self.log_dict(
            {f"{mode}_dps_loss": self.mt_weights["dps_loss"] * dps_loss}, on_epoch=True, on_step=True
        )

        # NOTE: Auxiliary task 2 - Diversity Preference Regularization (DPR)
        unique_users, first_indices = self._get_first_occurrence_indices(user)
        unique_dps_label = {k: v[first_indices] for k, v in dps_label.items()}
        dpr_scores = self.dpr_module(user_emb, user, item_emb, pos_item)
        dpr_loss = self.dpr_loss_fn(dpr_scores, unique_dps_label)
        self.log_dict(
            {f"{mode}_dpr_loss": self.mt_weights["dpr_loss"] * dpr_loss}, on_epoch=True, on_step=True
        )

        # NOTE: Auxiliary task 3 - Diversity Preference Matching (DPM)
        unique_users, first_indices = self._get_first_occurrence_indices(user)
        unique_dpm_label = {k: v[first_indices] for k, v in dpm_label.items()}
        dpm_prob_dist = self.dpm_module(yp_scores, user, dpm_vec)
        dpm_loss = self.dpm_loss_fn(dpm_prob_dist, unique_dpm_label)
        # dpm_loss = self.dpm_loss_fn(dpm_prob_dist, unique_dpm_label, unique_users) # TODO: remove this in final version
        self.log_dict(
            {f"{mode}_dpm_loss": self.mt_weights["dpm_loss"] * dpm_loss}, on_epoch=True, on_step=True
        )

        # NOTE: L2 regularization loss
        l2_loss = self.l2_regularizer(
            *self.dps_module.projection_fcs.values(), *self.dpr_module.relation_fcs.values()
        )
        self.log_dict({f"{mode}_l2_loss": self.l2_reg_weight * l2_loss}, on_epoch=True, on_step=True)

        return {
            "dps_loss": dps_loss,
            "dpr_loss": dpr_loss,
            "dpm_loss": dpm_loss,
            "l2_loss": self.l2_reg_weight * l2_loss,
        }

    # NOTE: Validation
    def on_validation_epoch_start(self):
        user_emb, item_emb = self.embedding_model()  # Get embedding once for val phase
        self.user_emb, self.item_emb = user_emb.detach(), item_emb.detach()

    def validation_step(self, batch, batch_idx):
        """We use user-item pairs for inference, so we need to compute scores for each pair."""
        user, item, label = batch["user"], batch["item"], batch["label"]
        scores = self.forward(self.user_emb, self.item_emb, user, item)

        self.val_step_outputs.append(
            {
                "users": user,
                "items": item,
                "scores": scores,
                "labels": label,
            }
        )

    def on_validation_epoch_end(self):
        # pass
        outputs = self.val_step_outputs
        all_users = torch.cat([x["users"] for x in outputs])
        all_items = torch.cat([x["items"] for x in outputs])
        all_scores = torch.cat([x["scores"] for x in outputs])
        all_labels = torch.cat([x["labels"] for x in outputs])

        self.val_results = {
            "user": all_users.cpu(),
            "item": all_items.cpu(),
            "score": all_scores.cpu(),
            "label": all_labels.cpu(),
            "user_emb": self.user_emb.detach().cpu(),
            "item_emb": self.item_emb.detach().cpu(),
        }

        eval_df = self.evaluator.prepare_evaluation_data(self.val_results)
        eval_score_df = self.evaluator.evaluate(eval_df, K=5)
        eval_score_df = self.evaluator.evaluate(eval_score_df, K=10)
        eval_score_df = self.evaluator.evaluate(eval_score_df, K=20)
        metrics = {
            f"val_{metric}{k}": eval_score_df[f"{metric}@{k}"].mean()
            for metric in ["ndcg", "precision", "recall"]
            for k in [5, 10, 20]
        }
        self.log_dict(metrics, prog_bar=True)
        self.val_results["eval_score_df"] = eval_score_df

    def rescale_loss_weighting(self, loss_dict, method: str = "ema"):
        """Dynamically adjust the loss weight based on the loss value."""
        # NOTE: Apply loss scaling
        if method == "ema":
            rescaled_loss_dict = {
                loss_name: self.ema_normalizer(loss, loss_name) for loss_name, loss in loss_dict.items()
            }
        elif method == "log":
            rescaled_loss_dict = {
                loss_name: self.loss_scaling_module(loss) for loss_name, loss in loss_dict.items()
            }
        elif method == "gradnorm":
            return self.gradnorm_step(loss_dict)
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

    def gradnorm_step(self, loss_dict):
        """Implements GradNorm loss rescaling."""
        task_names = list(self.mt_weights.keys())
        task_losses = torch.stack([loss_dict[k] for k in task_names])
        N = len(task_names)

        if self.gradnorm_initialized_losses is None:
            self.gradnorm_initialized_losses = task_losses.detach()

        # Compute weighted loss
        w = torch.exp(self.gradnorm_log_w)
        weighted_losses = w * task_losses
        total_loss = weighted_losses.sum()

        # Compute gradient norms for shared parameters (embedding model only)
        shared_params = list(self.embedding_model.parameters())
        G = []
        for i in range(N):
            grad = torch.autograd.grad(
                w[i] * task_losses[i], shared_params, retain_graph=True, create_graph=True
            )
            norm = torch.sqrt(sum(torch.sum(g**2) for g in grad))
            G.append(norm)
        G = torch.stack(G)
        G_avg = G.mean()

        # Compute inverse training rate
        with torch.no_grad():
            L_ratio = (task_losses / self.gradnorm_initialized_losses).detach()
            r = L_ratio / L_ratio.mean()

        alpha = 1.5  # GradNorm paper default
        target_G = G_avg * r**alpha
        gradnorm_loss = F.l1_loss(G, target_G)

        # Log weights and gradnorm loss
        for i, name in enumerate(task_names):
            self.log(f"gradnorm_weight/{name}", w[i], prog_bar=True, on_step=True)
        self.log("gradnorm_loss", gradnorm_loss, prog_bar=True, on_step=True)

        return (total_loss + gradnorm_loss), {
            name: w[i] * task_losses[i] for i, name in enumerate(task_names)
        }

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

    def gradient_conflict_test(self, loss_dict):
        """
        Test for gradient conflict by computing cosine similarities between gradients of different losses.
        This is useful to understand if the auxiliary tasks are conflicting with the main task.
        """
        # --- Gradient extraction ---
        g_dps = self.get_flat_grads(loss_dict["dps_loss"])
        g_dpr = self.get_flat_grads(loss_dict["dpr_loss"])
        g_dpm = self.get_flat_grads(loss_dict["dpm_loss"])

        # --- Cosine similarities ---
        cos = F.cosine_similarity

        cos_dps_dpr = cos(g_dps.unsqueeze(0), g_dpr.unsqueeze(0)).item()
        cos_dps_dpm = cos(g_dps.unsqueeze(0), g_dpm.unsqueeze(0)).item()
        cos_dpr_dpm = cos(g_dpr.unsqueeze(0), g_dpm.unsqueeze(0)).item()

        return {
            "cos_dps_dpr": cos_dps_dpr,
            "cos_dps_dpm": cos_dps_dpm,
            "cos_dpr_dpm": cos_dpr_dpm,
        }

    # NOTE: Testing/Inference
    def on_test_epoch_start(self):
        user_emb, item_emb = self.embedding_model()
        self.user_emb, self.item_emb = user_emb.detach(), item_emb.detach()

    def test_step(self, batch, batch_idx):
        """We use user-item pairs for inference, so we need to compute scores for each pair."""
        user, item, label = batch["user"], batch["item"], batch["label"]
        scores = self.forward(self.user_emb, self.item_emb, user, item)

        self.test_step_outputs.append(
            {
                "users": user,
                "items": item,
                "scores": scores,
                "labels": label,
            }
        )

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        all_users = torch.cat([x["users"] for x in outputs])
        all_items = torch.cat([x["items"] for x in outputs])
        all_scores = torch.cat([x["scores"] for x in outputs])
        all_labels = torch.cat([x["labels"] for x in outputs])

        self.test_results = {
            "user": all_users.cpu(),
            "item": all_items.cpu(),
            "score": all_scores.cpu(),
            "label": all_labels.cpu(),
            "user_emb": self.user_emb.detach().cpu(),
            "item_emb": self.item_emb.detach().cpu(),
        }

        eval_df = self.evaluator.prepare_evaluation_data(self.test_results)
        eval_score_df = self.evaluator.evaluate(eval_df, K=5)
        eval_score_df = self.evaluator.evaluate(eval_score_df, K=10)
        eval_score_df = self.evaluator.evaluate(eval_score_df, K=20)
        metrics = {
            f"test_{metric}{k}": eval_score_df[f"{metric}@{k}"].mean()
            for metric in ["ndcg", "precision", "recall"]
            for k in [5, 10, 20]
        }
        self.log_dict(metrics, prog_bar=True)
        self.test_results["eval_score_df"] = eval_score_df

    def configure_optimizers(self):
        params = list(self.parameters())
        if self.rescale_method == "gradnorm":
            other_params = [p for p in params if p is not self.gradnorm_log_w]
            return torch.optim.Adam(
                [{"params": other_params}, {"params": [self.gradnorm_log_w], "lr": self.lr}], lr=self.lr
            )
        else:
            return torch.optim.Adam(params, lr=self.lr)
