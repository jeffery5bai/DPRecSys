"""This file has been deprecated."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

import torch
import torch.nn.functional as F
from modules.gcn_modules import GCN_ID_MODULE, GraphConvModule
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class GCNRecCF(LightningModule):
    """
    Wrap up `GraphConvModule` model into Lightning Module as base recommendation model.
    """

    def __init__(self, num_users, num_items, graph_data, dim_id=64, num_layers=3, concat=True, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.gcn_model = GraphConvModule(
            num_users=num_users,
            num_items=num_items,
            edge_index=graph_data.edge_index,
            dim_id=dim_id,
            num_layers=num_layers,
            concat=concat,
        )

        self.num_users = num_users
        self.num_items = num_items
        self.graph_data = graph_data
        self.dim_id = dim_id
        self.num_layers = num_layers
        self.concat = concat
        self.lr = lr
        self.user_emb = None
        self.item_emb = None

        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_results = dict()
        self.test_results = dict()

    def forward(self):
        return self.gcn_model()

    def _compute_scores(self, user_emb, item_emb, user_idx, item_idx):
        # NOTE: the dimension of user_emb = [num_user + 1, emb_dim], item_emb = [num_item + 1, emb_dim]
        # return dot-product of user and item embeddings
        return (user_emb[user_idx] * item_emb[item_idx]).sum(dim=1)

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
        user_emb, item_emb = self.forward()
        scores = self._compute_scores(user_emb, item_emb, user, item)
        loss, acc, prec, rec, f1 = self._evaluate(scores, label)
        self.log_dict(
            {"train_loss": loss, "train_acc": acc, "train_prec": prec, "train_rec": rec, "train_f1": f1}
        )
        return loss

    def on_validation_epoch_start(self):
        self.user_emb, self.item_emb = self.forward()  # Compute embedding once for val phase

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
        self.user_emb, self.item_emb = self.forward()

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
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# Deprecated: the base module does not support bi-partite graph edge_index
class GCNRecID(LightningModule):
    """
    Wrap up `GCN_ID_MODULE` model into Lightning Module as base recommendation model.
    """

    def __init__(self, edge_index, num_user, num_item, dim_id=64, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = GCN_ID_MODULE(
            edge_index=edge_index,
            num_user=num_user,
            num_item=num_item,
            dim_id=dim_id,
            aggr_mode="mean",
            concate=True,
        )

        self.num_user = num_user
        self.num_item = num_item
        self.lr = lr
        self.emb = None

        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_results = dict()
        self.test_results = dict()

    def forward(self):
        return self.model().to(self.device)

    def _compute_scores(self, emb, user_idx, item_idx):
        # NOTE: the dimension of emb = [(num_user+num_item), emb_dim]
        # return (emb[user_idx] * emb[self.num_user + item_idx]).sum(dim=1)
        return torch.sigmoid((emb[user_idx] * emb[self.num_user + item_idx]).sum(dim=1))

    def _evaluate(self, scores, label):
        # loss = F.binary_cross_entropy_with_logits(scores, label)
        # preds = torch.sigmoid(scores) > 0.5
        loss = F.binary_cross_entropy(scores, label)
        preds = scores > 0.5
        acc = accuracy_score(label.cpu(), preds.cpu())
        prec = precision_score(label.cpu(), preds.cpu())
        rec = recall_score(label.cpu(), preds.cpu())
        f1 = f1_score(label.cpu(), preds.cpu())

        return loss, acc, prec, rec, f1

    def training_step(self, batch, batch_idx):
        user, item, label = batch["user"], batch["item"], batch["label"]
        emb = self.forward()
        scores = self._compute_scores(emb, user, item)
        loss, acc, prec, rec, f1 = self._evaluate(scores, label)
        self.log_dict(
            {"train_loss": loss, "train_acc": acc, "train_prec": prec, "train_rec": rec, "train_f1": f1}
        )
        return loss

    def on_validation_epoch_start(self):
        self.emb = self.forward()  # Compute embedding once for val phase

    def validation_step(self, batch, batch_idx):
        user, item, label = batch["user"], batch["item"], batch["label"]
        scores = self._compute_scores(self.emb, user, item)

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
        metrics = {"val_loss": loss, "val_acc": acc, "val_prec": prec, "val_rec": rec, "val_f1": f1}
        self.log_dict(metrics, prog_bar=True)

        self.val_results = {
            "user": all_users,
            "item": all_items,
            "score": all_scores,
            "label": all_labels,
            "metric": metrics,
            "emb": self.emb,
        }

    # Testing
    def on_test_epoch_start(self):
        self.emb = self.forward()

    def test_step(self, batch, batch_idx):
        user, item, label = batch["user"], batch["item"], batch["label"]
        scores = self._compute_scores(self.emb, user, item)

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
            "emb": self.emb.cpu(),
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
