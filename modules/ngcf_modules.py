"""
NGCF (Neural Graph Collaborative Filtering) Module in PyTorch Geometric
Wang, Xiang, et al. "Neural graph collaborative filtering."
Proceedings of the 42nd international ACM SIGIR conference on Research and development in Information Retrieval. 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import dropout_adj


class NGCF(nn.Module):
    def __init__(
        self, edge_index, num_users, num_items, embedding_dim, num_layers, node_dropout=0.0, mess_dropout=0.0
    ):
        super().__init__()
        self.edge_index = edge_index
        self.num_users = num_users + 1  # +1 for OOV
        self.num_items = num_items + 1  # +1 for OOV
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.node_dropout = node_dropout
        self.mess_dropout = mess_dropout

        # Initialize user and item embeddings
        self.user_embedding = nn.Embedding(self.num_users, embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Define NGCF layers
        self.convs = nn.ModuleList()
        for _ in range(len(num_layers)):
            self.convs.append(NGCFConv(embedding_dim, embedding_dim, dropout=node_dropout))

        # Dropout layer for message dropout
        self.dropout = nn.Dropout(mess_dropout)

    def forward(self, training=False):
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [x]

        for conv in self.convs:
            x = conv(x, self.edge_index, training)
            x = self.dropout(x) if training else x
            all_embeddings.append(x)

        # Concatenate embeddings from all layers
        final_embedding = torch.cat(all_embeddings, dim=1)
        user_embeddings = final_embedding[: self.num_users]
        item_embeddings = final_embedding[self.num_users :]
        return user_embeddings, item_embeddings


class NGCFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__(aggr="add")  # "Add" aggregation.
        self.linear = nn.Linear(in_channels, out_channels)
        self.linear_bi = nn.Linear(in_channels, out_channels)
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, training=False):
        if self.dropout > 0:
            edge_index, _ = dropout_adj(
                edge_index, p=self.dropout, force_undirected=True, num_nodes=x.size(0), training=training
            )
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # Sum component
        sum_message = self.linear(x_j)
        # Bi-interaction component
        bi_message = self.linear_bi(x_i * x_j)
        return self.leaky_relu(sum_message + bi_message)

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=1)
