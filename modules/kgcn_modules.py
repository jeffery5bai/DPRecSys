"""
KGCN (Knowledge Graph Convolutional Network) Module in PyTorch Geometric
Wang, Hongwei, et al. "Knowledge graph convolutional networks for recommender systems."
The world wide web conference. 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import HeteroConv, MessagePassing


class KGCN(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        attr_dims,
        embedding_dim,
        train_graph: Data,
        hetero_data: HeteroData,
        num_layers: int = 1,
    ):
        super().__init__()
        self.num_users = num_users + 1  # +1 for OOV user
        self.num_items = num_items + 1  # +1 for OOV item
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.edge_index_ui = train_graph.edge_index
        self.hetero_data = hetero_data

        # Initialize user, item, attribute and relation embeddings
        self.user_embedding = nn.Embedding(self.num_users, embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.attr_embeddings = nn.ParameterDict(
            {attr: nn.Parameter(torch.randn(size, embedding_dim)) for attr, size in attr_dims.items()}
        )
        self.rel_embeddings = nn.ParameterDict(
            {rel_type[1]: nn.Parameter(torch.randn(1, embedding_dim)) for rel_type in hetero_data.edge_types}
        )

        # Initialize the KGCN layers
        self.kg_convs = nn.ModuleList(
            [
                HeteroConv(
                    {edge_type[1]: KGConv(embedding_dim) for edge_type in hetero_data.edge_types}, aggr="mean"
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self):
        x_user = self.user_embedding
        x_item = self.item_embedding

        for layer in self.kg_convs:
            out_dict = {}
            for edge_type in self.hetero_data.edge_types:
                src_type, rel_name, dst_type = edge_type
                edge_index = self.hetero_data[edge_type].edge_index

                # NOTE: we only propagate from attributes to movie
                # movie nodes are always the target (dst_type)
                x_src = self.attr_embeddings[src_type][edge_index[0]]
                x_dst = x_item[edge_index[1]]
                rel = self.rel_embeddings[rel_name]

                # NOTE: To get Vu: user embeddings for the interacted source (movie) nodes
                user_interacted_src = self.edge_index_ui[0][
                    self.edge_index_ui[1] == edge_index[1].unsqueeze(-1)
                ].squeeze(-1)
                user_emb = x_user[user_interacted_src]

                out_dict[rel_name] = layer.convs[rel_name](
                    x_src, x_dst, rel=rel, user_emb=user_emb, index=edge_index
                )

            updated = layer(out_dict, self.hetero_data.edge_index_dict)

            for node_type, x_updated in updated.items():
                if node_type == "movie":
                    x_item = x_updated
                elif node_type in self.attr_embeddings:
                    self.attr_embeddings[node_type] = x_updated

        self.user_embedding = x_user
        self.item_embedding = x_item

        return x_user, x_item


class KGConv(MessagePassing):
    def __init__(self, dim):
        super().__init__(aggr="add", flow="source_to_target")
        self.dim = dim
        self.linear = nn.Linear(dim, dim)

    def forward(self, x_i, x_j, rel, user_emb, index):
        user_emb = user_emb[index[0]]  # Get corresponding users who interacted with source nodes
        user_emb = user_emb.view(-1, 1, 1, self.dim)  # [E,1,1,D]
        rel = rel.view(-1, 1, self.dim)  # [E,1,D] for broadcasting
        score = (user_emb * rel).sum(dim=-1)  # [E,1]
        score = F.softmax(score, dim=0)  # normalize per edge
        x_j = x_j.view(-1, 1, self.dim)  # [E,1,D]
        agg = (score.unsqueeze(-1) * x_j).sum(dim=0)  # [1,D]
        return F.relu(self.linear(x_i + agg))
