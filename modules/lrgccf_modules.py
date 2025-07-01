"""
LR-GCCF (Linear Residual Graph Convolutional Collaborative Filtering) Implementation with PyTorch Geometric.
Chen, Lei, et al. "Revisiting graph based collaborative filtering: A linear residual graph convolutional network approach."
Proceedings of the AAAI conference on artificial intelligence. Vol. 34. No. 01. 2020.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import add_remaining_self_loops, scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes


class LRGCCF(nn.Module):
    def __init__(self, edge_index, num_users, num_items, embedding_dim, num_layers):
        super().__init__()
        self.register_buffer("edge_index", edge_index)
        self.edge_index = edge_index
        self.num_users = num_users + 1  # +1 for OOV
        self.num_items = num_items + 1  # +1 for OOV
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Initialize user and item embeddings
        self.user_embedding = nn.Embedding(self.num_users, embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

        # Define NGCF layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(LRGCCFConv())

    def forward(self):
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [x]

        for conv in self.convs:
            x = conv(x, self.edge_index)
            all_embeddings.append(x)

        # Concatenate embeddings from all layers
        final_embedding = torch.cat(all_embeddings, dim=1)
        user_embeddings = final_embedding[: self.num_users]
        item_embeddings = final_embedding[self.num_users :]
        return user_embeddings, item_embeddings


class LRGCCFConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, x, edge_index):
        # Compute symmetric normalization in classic GCN
        edge_index, edge_weight, target_weight = self.custom_gcn_norm(edge_index, add_self_loops=True)

        return self.propagate(edge_index, x=x, edge_weight=edge_weight, target_weight=target_weight)

    def message(self, x_i, x_j, edge_weight, target_weight):
        # Propagate normalized messages
        return edge_weight.view(-1, 1) * x_j + target_weight.view(-1, 1) * x_i

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=1)

    def custom_gcn_norm(
        self,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        num_nodes: Optional[int] = None,
        add_self_loops: bool = True,
        flow: str = "source_to_target",
        dtype: Optional[torch.dtype] = None,
    ):
        fill_value = 1.0

        assert flow in ["source_to_target", "target_to_source"]
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if add_self_loops:
            edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        row, col = edge_index[0], edge_index[1]
        idx = col if flow == "source_to_target" else row
        deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce="sum")
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        target_weight = edge_weight * deg_inv_sqrt[col]

        return edge_index, edge_weight, target_weight
