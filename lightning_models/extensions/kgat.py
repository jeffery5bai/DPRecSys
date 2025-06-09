import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

import torch
from common.eval import Evaluator
from modules.kgat_modules import KGAT
from modules.loss import BPRLoss, EmbLoss
from pytorch_lightning import LightningModule


class KGATRec(LightningModule):
    """
    Wrap up `KGAT` model into Lightning Module as base recommendation model.
    """

    def __init__(
        self,
        hetero_data,
        num_users,
        num_items,
        num_nodes_dict,
        embedding_dim=64,
        num_layers=3,
        node_dropout=0.0,
        mess_dropout=0.1,
        lr=1e-3,
        reg_weight=1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.hetero_data = hetero_data

        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes_dict = {k: v + 1 for k, v in num_nodes_dict.items()}  # +1 for OOV
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.node_dropout = node_dropout
        self.mess_dropout = mess_dropout

        self.lr = lr
        self.reg_weight = reg_weight

        self.user_emb = None
        self.item_emb = None

        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_results = dict()
        self.test_results = dict()

        self.evaluator = Evaluator()

        self.kgat_model = KGAT(
            hetero_data=self.hetero_data,
            num_nodes_dict=self.num_nodes_dict,
            embed_dim=self.embedding_dim,
            num_layers=self.num_layers,
            aggr="bi-interaction",  # Use bi-interaction aggregation
        )

        # NOTE: Main loss functions for recommendation
        self.bpr_loss = BPRLoss(gamma=1e-10)
        self.reg_loss = EmbLoss(norm=2)
        self.kg_loss = None  # calculate in `training_step`

    def forward(self, user_emb, item_emb, user_idx, item_idx):
        # NOTE: the dimension of user_emb = [num_user + 1, emb_dim], item_emb = [num_item + 1, emb_dim]
        # two towers dot product
        return (user_emb[user_idx] * item_emb[item_idx]).sum(dim=1)

    def training_step(self, batch, batch_idx):
        emb_dict = self.kgat_model()  # Compute embedding for training
        user_emb = emb_dict["user"]
        item_emb = emb_dict["movie"]

        user, pos_item, neg_item = batch["user"], batch["pos_item"], batch["neg_item"]
        yp_scores = self.forward(user_emb, item_emb, user, pos_item)
        yn_scores = self.forward(user_emb, item_emb, user, neg_item)

        # Main task: BPR (Pair-wise + L2 regularization loss)
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

        # NOTE: Calculate KG loss for embedding regularization (TransR)
        kg_loss = self.calc_kg_loss(self, user_emb, item_emb, user, pos_item, neg_item)
        self.log("train_kg_loss", kg_loss, on_step=True)

        return bpr_loss + kg_loss

    def calc_kg_loss(self, user_emb, item_emb, user_idx, pos_t, neg_t):
        """Only calculate Pairwise TransR Loss for User-Item Triplets"""
        r_emb_ui = self.kgat_model.r_embs["interacts_with"]
        trans_r_ui = self.kgat_model.trans_m["interacts_with"]

        emb_u = trans_r_ui(user_emb[user_idx])
        emb_i = trans_r_ui(item_emb[pos_t])
        emb_neg_i = trans_r_ui(item_emb[neg_t])

        pos_score = torch.norm(emb_u + r_emb_ui - emb_i, p=2)
        neg_score = torch.norm(emb_u + r_emb_ui - emb_neg_i, p=2)

        loss = -torch.log(torch.sigmoid(neg_score - pos_score)).mean()
        return loss

    # NOTE: Validation
    def on_validation_epoch_start(self):
        emb_dict = self.kgat_model()  # Compute embedding once for val phase
        self.user_emb = emb_dict["user"]
        self.item_emb = emb_dict["movie"]

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

    # NOTE: Testing/Inference
    def on_test_epoch_start(self):
        emb_dict = self.kgat_model()
        self.user_emb = emb_dict["user"]
        self.item_emb = emb_dict["movie"]

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
            "user_emb": self.user_emb.cpu(),
            "item_emb": self.item_emb.cpu(),
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
        return torch.optim.Adam(self.parameters(), lr=self.lr)
