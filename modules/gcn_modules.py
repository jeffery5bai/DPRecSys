# coding: utf-8
"""
MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video.
In ACM MM`19,
"""

import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from common.loss import BPRLoss, EmbLoss
from torch_geometric.nn import GATConv, GCNConv, GraphConv, SAGEConv
from torch_geometric.nn.conv import MessagePassing


class GraphConvModule(nn.Module):
    def __init__(self, num_users, num_items, edge_index, dim_id, num_layers=3, concat=True):
        super().__init__()
        self.register_buffer("edge_index", edge_index)
        self.num_users = num_users
        self.num_items = num_items
        self.edge_index = edge_index
        self.dim_id = dim_id
        self.concat = concat
        self.num_layers = num_layers

        # Embeddings for user and item IDs (+1 for OOV)
        self.user_embedding = nn.Parameter(torch.empty(self.num_users + 1, dim_id))
        self.item_embedding = nn.Parameter(torch.empty(self.num_items + 1, dim_id))
        nn.init.xavier_normal_(self.user_embedding)
        nn.init.xavier_normal_(self.item_embedding)

        # GraphConv layers
        self.user_to_item_gconvs = nn.ModuleList()
        self.item_to_user_gconvs = nn.ModuleList()
        # Linear layers for user and item embeddings
        self.user_linears = nn.ModuleList()
        self.item_linears = nn.ModuleList()
        # Skip connection layers for user and item embeddings
        self.user_fuse_linears = nn.ModuleList()
        self.item_fuse_linears = nn.ModuleList()

        for _ in range(num_layers):
            self.user_to_item_gconvs.append(GraphConv(dim_id, dim_id))
            self.item_to_user_gconvs.append(GraphConv(dim_id, dim_id))
            self.user_linears.append(nn.Linear(dim_id, dim_id))
            self.item_linears.append(nn.Linear(dim_id, dim_id))
            self.user_fuse_linears.append(
                nn.Linear(2 * dim_id, dim_id) if self.concat else nn.Linear(dim_id, dim_id)
            )
            self.item_fuse_linears.append(
                nn.Linear(2 * dim_id, dim_id) if self.concat else nn.Linear(dim_id, dim_id)
            )

    def forward(self):
        # Initial embeddings
        u_emb = F.normalize(self.user_embedding)
        i_emb = F.normalize(self.item_embedding)

        for layer in range(self.num_layers):
            # NOTE: GraphConv message passing
            # Edge: User -> Item
            i_msg = F.leaky_relu(self.user_to_item_gconvs[layer]((u_emb, i_emb), self.edge_index))
            i_proj = F.leaky_relu(self.item_linears[layer](i_emb))
            i_emb = F.leaky_relu(
                self.item_fuse_linears[layer](
                    torch.cat([i_msg, i_proj], dim=1) if self.concat else i_msg + i_proj
                )
            )

            # Reverse Edges: Item -> User
            reversed_edge_index = self.edge_index.flip(0)
            u_msg = F.leaky_relu(self.item_to_user_gconvs[layer]((i_emb, u_emb), reversed_edge_index))
            u_proj = F.leaky_relu(self.user_linears[layer](u_emb))
            u_emb = F.leaky_relu(
                self.user_fuse_linears[layer](
                    torch.cat([u_msg, u_proj], dim=1) if self.concat else u_msg + u_proj
                )
            )

        return u_emb, i_emb


# NOTE: simplified version of GCN_ID_MODULE + BaseModel
# NOTE: (ERROR, the egde_index does not directly match the id embeddings)
class GCNConvModule(torch.nn.Module):
    def __init__(self, edge_index, num_user, num_item, dim_id, concate=True):
        super().__init__()
        self.register_buffer("edge_index", edge_index)
        # NOTE: each user/item/OOV has a trainable ID embedding
        self.edge_index = edge_index
        self.id_embedding = nn.Parameter(torch.randn(num_user + num_item + 2, dim_id))

        self.concate = concate

        self.conv1 = GCNConv(dim_id, dim_id)
        self.linear1 = nn.Linear(dim_id, dim_id)
        self.g_layer1 = nn.Linear(2 * dim_id if concate else dim_id, dim_id)

        self.conv2 = GCNConv(dim_id, dim_id)
        self.linear2 = nn.Linear(dim_id, dim_id)
        self.g_layer2 = nn.Linear(2 * dim_id if concate else dim_id, dim_id)

        self.conv3 = GCNConv(dim_id, dim_id)
        self.linear3 = nn.Linear(dim_id, dim_id)
        self.g_layer3 = nn.Linear(2 * dim_id if concate else dim_id, dim_id)

    def forward(self):
        x = F.normalize(self.id_embedding)

        h = F.leaky_relu(self.conv1(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(
            self.g_layer1(torch.cat([h, x_hat], dim=1)) if self.concate else self.g_layer1(h + x_hat)
        )

        h = F.leaky_relu(self.conv2(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(
            self.g_layer2(torch.cat([h, x_hat], dim=1)) if self.concate else self.g_layer2(h + x_hat)
        )

        h = F.leaky_relu(self.conv3(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear3(x))
        x = F.leaky_relu(
            self.g_layer3(torch.cat([h, x_hat], dim=1)) if self.concate else self.g_layer3(h + x_hat)
        )

        return x


# NOTE: (ERROR, the egde_index does not directly match the id embeddings)
class GCN_ID_MODULE(torch.nn.Module):
    def __init__(self, edge_index, num_user, num_item, dim_id, aggr_mode="add", concate=True):
        super(GCN_ID_MODULE, self).__init__()
        # NOTE: This makes edge_index move together with the model across devices automatically!
        self.register_buffer("edge_index", edge_index)
        self.edge_index = edge_index

        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.concate = concate

        # NOTE: ID embedding for all users and items
        # dim: batch_size * (num_user + num_item + 2([OOV])) * dim_id
        self.id_embedding = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_user + num_item + 2, dim_id)))

        self.conv1 = BaseModel(dim_id, dim_id, aggr=aggr_mode)
        self.linear1 = nn.Linear(dim_id, dim_id)
        self.g_layer1 = nn.Linear(2 * dim_id if concate else dim_id, dim_id)

        self.conv2 = BaseModel(dim_id, dim_id, aggr=aggr_mode)
        self.linear2 = nn.Linear(dim_id, dim_id)
        self.g_layer2 = nn.Linear(2 * dim_id if concate else dim_id, dim_id)

        self.conv3 = BaseModel(dim_id, dim_id, aggr=aggr_mode)
        self.linear3 = nn.Linear(dim_id, dim_id)
        self.g_layer3 = nn.Linear(2 * dim_id if concate else dim_id, dim_id)

    def forward(self):
        x = F.normalize(self.id_embedding)

        h = F.leaky_relu(self.conv1(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(
            self.g_layer1(torch.cat([h, x_hat], dim=1)) if self.concate else self.g_layer1(h + x_hat)
        )

        h = F.leaky_relu(self.conv2(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(
            self.g_layer2(torch.cat([h, x_hat], dim=1)) if self.concate else self.g_layer2(h + x_hat)
        )

        h = F.leaky_relu(self.conv3(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear3(x))
        x = F.leaky_relu(
            self.g_layer3(torch.cat([h, x_hat], dim=1)) if self.concate else self.g_layer3(h + x_hat)
        )

        return x


class BaseModel(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr="add", **kwargs):
        super(BaseModel, self).__init__(aggr=aggr, **kwargs)
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch_geometric.nn.inits.uniform(self.weight.size(0), self.weight)

    def forward(self, x, edge_index, size=None):
        x = torch.matmul(x, self.weight)
        return self.propagate(edge_index, x=x, size=(x.size(0), x.size(0)))

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out
