"""
LR-GCCF (Linear Residual Graph Convolutional Collaborative Filtering) Implementation with PyTorch Geometric.
Chen, Lei, et al. "Revisiting graph based collaborative filtering: A linear residual graph convolutional network approach."
Proceedings of the AAAI conference on artificial intelligence. Vol. 34. No. 01. 2020.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


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
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

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
        edge_index, edge_weight = gcn_norm(edge_index, add_self_loops=True)

        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        # Propagate normalized messages
        return edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out
