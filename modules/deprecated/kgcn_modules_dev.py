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

    def forward(self, user_idx, item_idx):
        x_user = self.user_embedding[user_idx]
        x_item = self.item_embedding[item_idx]

        for layer in self.kg_convs:
            out_dict = {}
            for edge_type in self.hetero_data.edge_types:
                src_type, rel_name, dst_type = edge_type
                edge_index = self.hetero_data[edge_type].edge_index

                # NOTE: sample neighbors from attributes to items
                # TODO: we skipped the entity sampling strategy here since movielens dataset has few attributes
                src, dst = edge_index
                mask = torch.isin(dst, item_idx) # attributes (source) -> items (target) (B,)
                sampled_src = src[mask] # (nunique item in batch,)
                sampled_dst = dst[mask] # (nunique item in batch,), may contains duplicated item_idx

                # NOTE: we only propagate from attributes to movie
                # movie nodes are always the target (dst_type)
                x_src = self.attr_embeddings[src_type][sampled_src]
                x_dst = x_item[sampled_dst]
                rel = self.rel_embeddings[rel_name]

                # NOTE: To get Vu: user embeddings for the interacted source (movie) nodes
                # user_interacted_src = self.edge_index_ui[0][
                #     self.edge_index_ui[1] == edge_index[1].unsqueeze(-1)
                # ].squeeze(-1)
                # user_emb = x_user[user_interacted_src]

                out_dict[rel_name] = layer.convs[rel_name](
                    x_dst, x_src, rel=rel, user_emb=user_emb, index=edge_index
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

    def sample_neighbors(self, item_indices, rel_name):
        """
        Samples 1-hop neighbors using HeteroData.edge_index.
        Returns: neighbor_item_indices, relation_types
        """
        all_neighbors = []
        all_relations = []

        for i, (src_type, edge_type, dst_type) in enumerate(self.data.edge_types):
            src, dst = self.data[edge_type].edge_index

            mask = torch.isin(dst, item_indices) # attributes (source) -> items (target) (B,)
            sampled_dst = dst[mask] # (nunique item in batch,), may contains duplicated item_idx
            sampled_src = src[mask] # (nunique item in batch,)

            if sampled_src.numel() == 0:
                continue

            # NOTE: by items, sample to K neighbors with the edge_type
            # TODO: we skipped the entity sampling strategy here since movielens dataset has few attributes
            # perm = torch.randperm(sampled_src.size(0))[:self.neighbor_sample_size]
            # sampled_src = sampled_src[perm]
            # sampled_dst = sampled_dst[perm]

            all_neighbors.append(sampled_src)
            all_relations.append(torch.full_like(sampled_src, fill_value=i))

        if len(all_neighbors) == 0:
            return item_indices, torch.zeros_like(item_indices)

        return torch.cat(all_neighbors), torch.cat(all_relations)


class KGConv(MessagePassing):
    def __init__(self, dim):
        super().__init__(aggr="add", flow="source_to_target")
        self.dim = dim
        self.linear = nn.Linear(dim, dim)

    def forward(self, x_i, x_j, rel, user_emb, edge_index_ui):
        # Get user emb
        user_emb = user_emb[edge_index_ui[0]]  # Get corresponding users who interacted with source nodes
        user_emb = user_emb.view(-1, 1, 1, self.dim)  # [E,1,1,D]

        # Get item neighbors (attributes)

        # Get rels from neighbors

        # calculate user*rel importance
        rel = rel.view(-1, 1, self.dim)  # [E,1,D] for broadcasting
        score = (user_emb * rel).sum(dim=-1)  # [E,1]
        score = F.softmax(score, dim=0)  # normalize per edge

        # sum message from all attributes
        x_j = x_j.view(-1, 1, self.dim)  # [E,1,D]
        agg = (score.unsqueeze(-1) * x_j).sum(dim=0)  # [1,D]

        # add back to item emb -> (Vu)
        return F.relu(self.linear(x_i + agg))
