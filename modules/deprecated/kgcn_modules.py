import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData


class KGCN(nn.Module):
    def __init__(self, args, num_users: int, num_items: int, hetero_data: HeteroData):
        super(KGCN, self).__init__()
        self.parse_args(args)
        self.data = hetero_data
        self.num_users = num_users + 1  # +1 for OOV user
        self.num_items = num_items + 1  # +1 for OOV item

        self.user_emb_matrix = nn.Embedding(self.num_users, self.dim)
        self.item_emb_matrix = nn.Embedding(self.num_items, self.dim)
        self.entity_emb_matrix = nn.Embedding(self.data["item"].num_nodes, self.dim)

        self.relation_emb_matrix = nn.Embedding(len(self.data.edge_types), self.dim)

        nn.init.xavier_uniform_(self.user_emb_matrix.weight)
        nn.init.xavier_uniform_(self.entity_emb_matrix.weight)
        nn.init.xavier_uniform_(self.relation_emb_matrix.weight)

        # Aggregator class
        self.aggregator_class = SumAggregator

    def parse_args(self, args):
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.neighbor_sample_size = args.neighbor_sample_size

    def forward(self, user_indices, item_indices):
        """
        user_indices: [batch_size]
        item_indices: [batch_size]
        """
        user_emb = self.user_emb_matrix(user_indices)
        item_vu_emb = self.aggregate(user_emb, item_indices)

        scores = torch.sum(user_emb * item_vu_emb, dim=1)
        return scores, torch.sigmoid(scores)

    def aggregate(self, user_emb, item_indices):
        """
        Multi-hop aggregation from HeteroData graph.
        item_indices: [batch_size]
        """
        entity_vectors = [self.item_emb_matrix(item_indices)]

        for i in range(self.n_iter):
            neighbors = self.sample_neighbors(item_indices)
            neighbor_items, edge_types = neighbors

            neighbor_embs = self.entity_emb_matrix(neighbor_items)
            rel_embs = self.relation_emb_matrix(edge_types)

            aggregator = self.aggregator_class(
                batch_size=self.batch_size,
                dim=self.dim,
                activation=torch.tanh if i == self.n_iter - 1 else None
            )

            out = aggregator(
                self_vectors=entity_vectors[-1],
                neighbor_vectors=neighbor_embs,
                neighbor_relations=rel_embs,
                user_embeddings=user_emb
            )
            entity_vectors.append(out)

        return entity_vectors[-1]

    def sample_neighbors(self, item_indices):
        """
        Samples 1-hop neighbors using HeteroData.edge_index.
        Returns: neighbor_item_indices, relation_types
        """
        all_neighbors = []
        all_relations = []

        for i, edge_type in enumerate(self.data.edge_types):
            src, dst = self.data[edge_type[1]].edge_index

            mask = torch.isin(src, item_indices)
            sampled_dst = dst[mask]
            sampled_src = src[mask]

            if sampled_dst.numel() == 0:
                continue

            # NOTE: sample neighbors from attributes to items
            # TODO: we skipped the entity sampling strategy here since movielens dataset has few attributes

            all_neighbors.append(sampled_dst)
            all_relations.append(torch.full_like(sampled_dst, fill_value=i))

        if len(all_neighbors) == 0:
            return item_indices, torch.zeros_like(item_indices)

        return torch.cat(all_neighbors), torch.cat(all_relations)

    def compute_loss(self, scores, labels):
        bce = F.binary_cross_entropy_with_logits(scores, labels)

        l2_loss = self.user_emb_matrix.weight.norm(2) + self.entity_emb_matrix.weight.norm(2) + self.relation_emb_matrix.weight.norm(2)
        return bce + self.l2_weight * l2_loss

import torch
import torch.nn as nn
import torch.nn.functional as F


class SumAggregator(nn.Module):
    def __init__(self, batch_size, dim, dropout=0.0, activation=F.relu):
        super(SumAggregator, self).__init__()
        self.batch_size = batch_size
        self.dim = dim
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation

        self.linear = nn.Linear(dim, dim)  # PyTorch uses bias=True by default

    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        """
        self_vectors: Tensor [batch_size, dim]
        neighbor_vectors: Tensor [batch_size * n_sample, dim]
        neighbor_relations: Tensor [batch_size * n_sample, dim]
        user_embeddings: Tensor [batch_size, dim]
        """

        neighbors_agg = self.mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        # Sum self vector and neighbor aggregation
        output = self_vectors + neighbors_agg  # shape: [batch_size, dim]

        output = self.dropout(output)
        output = self.linear(output)  # shape: [batch_size, dim]

        return self.activation(output)

    def mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        """
        Perform relation-aware attention-based mixing for neighbors.

        neighbor_vectors: [batch_size * n_sample, dim]
        neighbor_relations: [batch_size * n_sample, dim]
        user_embeddings: [batch_size, dim]
        """

        # Reshape: [batch_size, n_sample, dim]
        n_sample = neighbor_vectors.shape[0] // self.batch_size
        neighbor_vectors = neighbor_vectors.view(self.batch_size, n_sample, self.dim)
        neighbor_relations = neighbor_relations.view(self.batch_size, n_sample, self.dim)

        # [batch_size, 1, dim] → broadcasted
        user_embeddings = user_embeddings.unsqueeze(1)

        # Relation-aware attention score (dot product between (u ◦ r) and v)
        # where ◦ is element-wise product, v is neighbor vector
        scores = torch.sum((user_embeddings * neighbor_relations) * neighbor_vectors, dim=2)  # [batch_size, n_sample]

        attention = F.softmax(scores, dim=1)  # [batch_size, n_sample]
        attention = attention.unsqueeze(2)  # [batch_size, n_sample, 1]

        # Weighted sum of neighbor vectors
        neighbors_agg = torch.sum(attention * neighbor_vectors, dim=1)  # [batch_size, dim]
        return neighbors_agg
