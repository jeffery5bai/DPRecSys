import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

import torch
from common.eval import Evaluator
from common.utils import seed_worker
from modules.kgat_modules import KGAT
from modules.loss import BPRLoss, EmbLoss
from pytorch_lightning import LightningModule
from torch_geometric.loader import NeighborLoader

# NOTE: To handle random seed for NeighborLoader
RANDOM_SEED = 42
g = torch.Generator()
g.manual_seed(RANDOM_SEED)


class KGATRecV2(LightningModule):
    """
    Wrap up `KGAT` model into Lightning Module as base recommendation model.
    In V2, we use TripletDataLoader for validation in order to monitor early stopping on BPR loss.
    """

    def __init__(
        self,
        hetero_data,
        embedding_dim=64,
        rel_emb_dim=8,
        num_layers=3,
        node_dropout=0.0,
        mess_dropout=0.0,
        num_neighbors=-1,  # -1 means all neighbors
        lr=1e-3,
        reg_weight=1e-5,
        use_mini_batch=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.hetero_data = hetero_data

        self.num_users = hetero_data["user"].num_nodes
        self.num_items = hetero_data["movie"].num_nodes
        self.embedding_dim = embedding_dim
        self.rel_emb_dim = rel_emb_dim
        self.num_layers = num_layers
        self.node_dropout = node_dropout
        self.mess_dropout = mess_dropout
        self.num_neighbors = num_neighbors
        self.use_mini_batch = use_mini_batch

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
            embed_dim=self.embedding_dim,
            rel_emb_dim=self.rel_emb_dim,
            num_layers=self.num_layers,
            node_dropout=self.node_dropout,
            mess_dropout=self.mess_dropout,
            aggr="bi-interaction",  # Use bi-interaction aggregation
            device=self.device,
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
        user, pos_item, neg_item = batch["user"], batch["pos_item"], batch["neg_item"]

        # NOTE: Build mini-batch subgraph (centered to "user")
        if self.use_mini_batch:
            input_nodes = ("user", user)
            subgraph = self.get_subgraph(input_nodes)
            emb_dict = self.kgat_model(subgraph, training=True)
        else:
            emb_dict = self.kgat_model(self.hetero_data, training=True)  # Compute embedding for training

        user_emb = emb_dict["user"]
        item_emb = emb_dict["movie"]

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
        kg_loss = self.calc_kg_loss(user_emb, item_emb, user, pos_item, neg_item)
        self.log("train_kg_loss", kg_loss, on_step=True)

        return bpr_loss + kg_loss

    def get_subgraph(self, input_nodes):
        """Get subgraph for the given input nodes."""
        loader = NeighborLoader(
            self.hetero_data,
            num_neighbors=[self.num_neighbors] * self.num_layers,
            input_nodes=input_nodes,
            batch_size=len(set(input_nodes[1])),
            shuffle=False,
            directed=True,
            worker_init_fn=seed_worker,
            generator=g,
            num_workers=4,
        )
        subgraph = next(iter(loader))

        # NOTE: Map local IDs back to global for each edge_index
        for edge_type in subgraph.edge_types:
            src_type, rel_type, dst_type = edge_type
            src_global_ids, src_num_nodes = subgraph[src_type].n_id, subgraph[src_type].num_nodes
            dst_global_ids, dst_num_nodes = subgraph[dst_type].n_id, subgraph[dst_type].num_nodes
            src_map = {i: src_global_ids[i] for i in range(src_num_nodes)}
            dst_map = {i: dst_global_ids[i] for i in range(dst_num_nodes)}

            src_edge_local = subgraph.edge_index_dict[edge_type][0]
            dst_edge_local = subgraph.edge_index_dict[edge_type][1]

            src_edge_global = torch.tensor([src_map[i.item()] for i in src_edge_local], dtype=torch.long)
            dst_edge_global = torch.tensor([dst_map[i.item()] for i in dst_edge_local], dtype=torch.long)
            subgraph[edge_type].edge_index = torch.stack([src_edge_global, dst_edge_global], dim=0)

        return subgraph

    def calc_kg_loss(self, user_emb, item_emb, user_idx, pos_t, neg_t):
        """Only calculate Pairwise TransR Loss for User-Item Triplets"""
        r_emb_ui = self.kgat_model.r_embs["interacts_with"]
        trans_r_ui = self.kgat_model.trans_m["interacts_with"]
        emb_dim = self.embedding_dim

        emb_u = trans_r_ui(user_emb[user_idx].reshape(-1, emb_dim))
        emb_i = trans_r_ui(item_emb[pos_t].reshape(-1, emb_dim))
        emb_neg_i = trans_r_ui(item_emb[neg_t].reshape(-1, emb_dim))

        pos_score = torch.norm(emb_u + r_emb_ui - emb_i, p=2)
        neg_score = torch.norm(emb_u + r_emb_ui - emb_neg_i, p=2)

        loss = -torch.log(torch.sigmoid(neg_score - pos_score)).mean()
        return loss

    # NOTE: Validation
    def on_validation_epoch_start(self):
        emb_dict = self.kgat_model(self.hetero_data)  # Compute embedding once for val phase
        self.user_emb = emb_dict["user"]
        self.item_emb = emb_dict["movie"]

    def validation_step(self, batch, batch_idx):
        """We use user-item triplets for validation, so we need to compute scores for each triplet."""
        user, pos_item, neg_item = batch["user"], batch["pos_item"], batch["neg_item"]
        yp_scores = self.forward(self.user_emb, self.item_emb, user, pos_item)
        yn_scores = self.forward(self.user_emb, self.item_emb, user, neg_item)

        # Main task: BPR (Pair-wise + L2 regularization loss)
        pair_loss = self.bpr_loss(yp_scores, yn_scores)
        reg_loss = self.reg_loss(self.user_emb, self.item_emb[pos_item], self.item_emb[neg_item])
        bpr_loss = pair_loss + self.reg_weight * reg_loss
        self.log("val_bpr_loss", bpr_loss, on_epoch=True, on_step=True)

        # NOTE: Calculate KG loss for embedding regularization (TransR)
        kg_loss = self.calc_kg_loss(self.user_emb, self.item_emb, user, pos_item, neg_item)
        self.log("val_kg_loss", kg_loss, on_epoch=True, on_step=True)

        self.val_step_outputs.append({"bpr_loss": bpr_loss, "kg_loss": kg_loss})

    def on_validation_epoch_end(self):
        pass
        # outputs = self.val_step_outputs
        # all_bpr_loss = torch.cat([x["bpr_loss"] for x in outputs])
        # all_kg_loss = torch.cat([x["kg_loss"] for x in outputs])
        # avg_bpr_loss = all_bpr_loss.mean().item()
        # avg_kg_loss = all_kg_loss.mean().item()

        # self.val_results = {
        #     "bpr_loss": avg_bpr_loss,
        #     "kg_loss": avg_kg_loss,
        # }

    # NOTE: Testing/Inference
    def on_test_epoch_start(self):
        emb_dict = self.kgat_model(self.hetero_data)
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
