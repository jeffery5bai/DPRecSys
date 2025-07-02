import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import torch
from common.eval import Evaluator
from modules.lightgcn_modules import LightGCN
from modules.loss import BPRLoss, EmbLoss
from pytorch_lightning import LightningModule


class LightGCNRec(LightningModule):
    """
    Wrap up `LightGCN` model into Lightning Module as base recommendation model.
    """

    def __init__(
        self,
        graph_data,
        num_users,
        num_items,
        embedding_dim=64,
        num_layers=3,
        lr=1e-3,
        reg_weight=1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.graph_data = graph_data

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.lr = lr
        self.reg_weight = reg_weight

        self.user_emb = None
        self.item_emb = None

        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_results = dict()
        self.test_results = dict()

        self.evaluator = Evaluator()

        self.lightgcn_model = LightGCN(
            num_users=self.num_users,
            num_items=self.num_items,
            edge_index=self.graph_data.edge_index,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
        )

        # NOTE: Main loss functions for recommendation
        self.bpr_loss = BPRLoss(gamma=1e-10)
        self.reg_loss = EmbLoss(norm=2)

    def forward(self, user_emb, item_emb, user_idx, item_idx):
        # NOTE: the dimension of user_emb = [num_user + 1, emb_dim], item_emb = [num_item + 1, emb_dim]
        # two towers dot product
        return (user_emb[user_idx] * item_emb[item_idx]).sum(dim=1)

    def training_step(self, batch, batch_idx):
        user_emb, item_emb = self.lightgcn_model()  # Compute embedding for training

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
        return bpr_loss

    # NOTE: Validation
    def on_validation_epoch_start(self):
        self.user_emb, self.item_emb = self.lightgcn_model()  # Compute embedding once for val phase

    def validation_step(self, batch, batch_idx):
        """We use user-item pairs for inference, so we need to compute scores for each pair."""
        user, pos_item, neg_item = batch["user"], batch["pos_item"], batch["neg_item"]
        yp_scores = self.forward(self.user_emb, self.item_emb, user, pos_item)
        yn_scores = self.forward(self.user_emb, self.item_emb, user, neg_item)

        # Main task: BPR (Pair-wise + L2 regularization loss)
        pair_loss = self.bpr_loss(yp_scores, yn_scores)
        reg_loss = self.reg_loss(self.user_emb, self.item_emb[pos_item], self.item_emb[neg_item])
        bpr_loss = pair_loss + self.reg_weight * reg_loss

        self.log_dict({"val_bpr_loss": bpr_loss}, on_epoch=True, on_step=True)

        self.val_step_outputs.append({"bpr_loss": bpr_loss})

    def on_validation_epoch_end(self):
        outputs = self.val_step_outputs
        all_bpr_loss = torch.cat([x["bpr_loss"] for x in outputs])

        self.val_results = {
            "bpr_loss": all_bpr_loss.cpu(),
        }

    # NOTE: Testing/Inference
    def on_test_epoch_start(self):
        self.user_emb, self.item_emb = self.lightgcn_model()

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
        return torch.optim.Adam(self.parameters(), lr=self.lr)
