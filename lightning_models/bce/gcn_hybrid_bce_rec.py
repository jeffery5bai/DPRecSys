import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "....")))

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.gcn_modules import GraphConvModule
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch_geometric.data import Data


class GCNRecHybrid(LightningModule):
    """
    Wrap up `GraphConvModule` model and `Side Information` into Lightning Module as base recommendation model.
    """

    def __init__(
        self,
        feat_cardinalities: Dict[str, int],
        graph_data: Data,
        dim_id: int = 64,
        dim_feature: int = 4,
        num_layers: int = 3,
        concat: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()

        num_items = feat_cardinalities["item"]
        num_users = feat_cardinalities["user"]
        num_actors = feat_cardinalities["actor"]
        num_countries = feat_cardinalities["country"]
        num_directors = feat_cardinalities["director"]
        num_genres = feat_cardinalities["genre"]

        # GNN model
        self.gcn_model = GraphConvModule(
            num_users=num_users,
            num_items=num_items,
            edge_index=graph_data.edge_index,
            dim_id=dim_id,
            dim_feature=dim_feature,
            num_layers=num_layers,
            concat=concat,
        )

        # Side information: (+2 for padding and unknown)
        self.feature_embeds = nn.ModuleDict(
            {
                name: nn.Embedding(cardinality + 2, dim_feature, padding_idx=0)
                for name, cardinality in feat_cardinalities.items()
            }
        )
        

        # NOTE: Two-tower prediction used in _compute_scores
        self.user_mlp = nn.Sequential(
            nn.Linear(dim_id, dim_id), nn.LeakyReLU(), nn.Dropout(p=dropout), nn.Linear(dim_id, dim_id)
        )

        self.item_mlp = nn.Sequential(
            nn.Linear(dim_id + dim_feature * 4, dim_id),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim_id, dim_id),
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.user_emb = None
        self.item_emb = None

        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_results = dict()
        self.test_results = dict()

    def gnn_forward(self):
        return self.gcn_model()

    def _compute_scores(self, user_emb, item_emb, user_idx, item_idx):
        # NOTE: the dimension of user_emb = [num_user + 1, emb_dim], item_emb = [num_item + 1, emb_dim]
        # return U_MLP(user_emb) * I_MLP(item_emb || item_feat_emb)
        user_vec = self.user_mlp(user_emb[user_idx])
        item_vec = self.item_mlp(torch.cat([item_emb[item_idx], self.item_feat_emb[item_idx]], dim=1))
        return (user_vec * item_vec).sum(dim=1)

    def _evaluate(self, scores, label):
        loss = F.binary_cross_entropy_with_logits(scores, label)
        preds = torch.sigmoid(scores) > 0.5
        acc = accuracy_score(label.cpu(), preds.cpu())
        prec = precision_score(label.cpu(), preds.cpu(), zero_division=0)
        rec = recall_score(label.cpu(), preds.cpu(), zero_division=0)
        f1 = f1_score(label.cpu(), preds.cpu(), zero_division=0)

        return loss, acc, prec, rec, f1

    def training_step(self, batch, batch_idx):
        user, item, label = batch["user"], batch["item"], batch["label"]
        user_emb, item_emb = self.gnn_forward()
        scores = self._compute_scores(user_emb, item_emb, user, item)
        loss, acc, prec, rec, f1 = self._evaluate(scores, label)
        self.log_dict(
            {"train_loss": loss, "train_acc": acc, "train_prec": prec, "train_rec": rec, "train_f1": f1}
        )
        return loss

    def on_validation_epoch_start(self):
        self.user_emb, self.item_emb = self.gnn_forward()  # Compute embedding once for val phase

    def validation_step(self, batch, batch_idx):
        user, item, label = batch["user"], batch["item"], batch["label"]
        scores = self._compute_scores(self.user_emb, self.item_emb, user, item)

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

        loss, acc, prec, rec, f1 = self._evaluate(all_scores, all_labels)
        metrics = {"val_loss": loss.item(), "val_acc": acc, "val_prec": prec, "val_rec": rec, "val_f1": f1}
        self.log_dict(metrics, prog_bar=True)

        self.val_results = {
            "user": all_users,
            "item": all_items,
            "score": all_scores,
            "label": all_labels,
            "metric": metrics,
            "user_emb": self.user_emb,
            "item_emb": self.item_emb,
        }

    # Testing
    def on_test_epoch_start(self):
        self.user_emb, self.item_emb = self.gnn_forward()

    def test_step(self, batch, batch_idx):
        user, item, label = batch["user"], batch["item"], batch["label"]
        scores = self._compute_scores(self.user_emb, self.item_emb, user, item)

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

        loss, acc, prec, rec, f1 = self._evaluate(all_scores, all_labels)
        metrics = {
            "test_loss": loss.item(),
            "test_acc": acc,
            "test_prec": prec,
            "test_rec": rec,
            "test_f1": f1,
        }
        self.log_dict(metrics, prog_bar=True)

        self.test_results = {
            "user": all_users.cpu(),
            "item": all_items.cpu(),
            "score": all_scores.cpu(),
            "label": all_labels.cpu(),
            "metric": metrics,
            "user_emb": self.user_emb.cpu(),
            "item_emb": self.item_emb.cpu(),
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
