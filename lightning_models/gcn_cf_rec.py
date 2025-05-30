import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import torch
from modules.gcn_modules import GraphConvModule
from modules.loss import BPRLoss, EmbLoss
from pytorch_lightning import LightningModule


class GCNRec(LightningModule):
    """
    Wrap up `GraphConvModule` model into Lightning Module as base recommendation model.
    """

    def __init__(
        self, num_users, num_items, graph_data, dim_id=64, num_layers=3, concat=True, lr=1e-3, reg_weight=1e-5
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_users = num_users
        self.num_items = num_items
        self.graph_data = graph_data
        self.dim_id = dim_id
        self.num_layers = num_layers
        self.concat = concat
        self.lr = lr
        self.reg_weight = reg_weight
        self.user_emb = None
        self.item_emb = None

        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_results = dict()
        self.test_results = dict()

        self.gcn_model = GraphConvModule(
            num_users=self.num_users,
            num_items=self.num_items,
            edge_index=self.graph_data.edge_index,
            dim_id=self.dim_id,
            num_layers=self.num_layers,
            concat=self.concat,
        )

        # NOTE: Main loss functions for recommendation
        self.bpr_loss = BPRLoss(gamma=1e-10)
        self.reg_loss = EmbLoss(norm=2)

    def forward(self, user_emb, item_emb, user_idx, item_idx):
        # NOTE: the dimension of user_emb = [num_user + 1, emb_dim], item_emb = [num_item + 1, emb_dim]
        # two towers dot product
        return (user_emb[user_idx] * item_emb[item_idx]).sum(dim=1)

    def training_step(self, batch, batch_idx):
        user_emb, item_emb = self.gcn_model()

        user, pos_item, neg_item = batch["user"], batch["pos_item"], batch["neg_item"]
        yp_scores = self.forward(user_emb, item_emb, user, pos_item)
        yn_scores = self.forward(user_emb, item_emb, user, neg_item)

        bpr_loss = self.bpr_loss(yp_scores, yn_scores)
        reg_loss = self.reg_loss(user_emb, item_emb[pos_item], item_emb[neg_item])
        loss = bpr_loss + self.reg_weight * reg_loss
        self.log_dict(
            {"train_loss": loss, "train_bpr_loss": bpr_loss, "train_reg_loss": self.reg_weight * reg_loss},
            on_epoch=True,
            on_step=True,
        )
        return loss

    # NOTE: Validation
    def on_validation_epoch_start(self):
        self.user_emb, self.item_emb = self.gcn_model()  # Compute embedding once for val phase

    def validation_step(self, batch, batch_idx):
        user, pos_item, neg_item = batch["user"], batch["pos_item"], batch["neg_item"]
        yp_scores = self.forward(self.user_emb, self.item_emb, user, pos_item)
        yn_scores = self.forward(self.user_emb, self.item_emb, user, neg_item)

        bpr_loss = self.bpr_loss(yp_scores, yn_scores)
        reg_loss = self.reg_loss(self.user_emb, self.item_emb[pos_item], self.item_emb[neg_item])
        loss = bpr_loss + self.reg_weight * reg_loss
        self.log_dict({"train_loss": loss, "train_bpr_loss": bpr_loss, "train_reg_loss": self.reg_weight * reg_loss})

        self.val_step_outputs.append(
            {
                "loss": loss,
                "bpr_loss": bpr_loss,
                "reg_loss": reg_loss,
            }
        )

    def on_validation_epoch_end(self):
        outputs = self.val_step_outputs
        avg_loss = torch.tensor([x["loss"] for x in outputs]).mean()
        avg_bpr_loss = torch.tensor([x["bpr_loss"] for x in outputs]).mean()
        avg_reg_loss = torch.tensor([x["reg_loss"] for x in outputs]).mean()

        metrics = {"val_loss": avg_loss, "val_bpr_loss": avg_bpr_loss, "val_reg_loss": avg_reg_loss}
        self.log_dict(metrics, prog_bar=True)

    # NOTE: Testing/Inference
    def on_test_epoch_start(self):
        self.user_emb, self.item_emb = self.gcn_model()

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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
