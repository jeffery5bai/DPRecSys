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
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import torch_geometric

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization


class MMGCN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGCN, self).__init__(config, dataset)
        self.num_user = self.n_users
        self.num_item = self.n_items
        num_user = self.n_users
        num_item = self.n_items
        dim_x = config['embedding_size']
        num_layer = config['n_layers']
        batch_size = config['train_batch_size']         # not used
        self.aggr_mode = 'mean'
        self.concate = 'False'
        has_id = True
        self.weight = torch.tensor([[1.0], [-1.0]]).to(self.device)
        self.reg_weight = config['reg_weight']

        # packing interaction in training into edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = torch.tensor(self.pack_edge_index(train_interactions), dtype=torch.long)
        self.edge_index = edge_index.t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)
        self.num_modal = 0

        if self.v_feat is not None:
            self.v_gcn = GCN(self.edge_index, batch_size, num_user, num_item, self.v_feat.size(1), dim_x, self.aggr_mode,
                             self.concate, num_layer=num_layer, has_id=has_id, dim_latent=256, device=self.device)
            self.num_modal += 1

        if self.t_feat is not None:
            self.t_gcn = GCN(self.edge_index, batch_size, num_user, num_item, self.t_feat.size(1), dim_x,
                             self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id, device=self.device)
            self.num_modal += 1

        self.id_embedding = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x), requires_grad=True)).to(self.device)
        self.result = nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_x))).to(self.device)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))

    def forward(self):
        representation = None
        if self.v_feat is not None:
            representation = self.v_gcn(self.v_feat, self.id_embedding)
        if self.t_feat is not None:
            if representation is None:
                representation = self.t_gcn(self.t_feat, self.id_embedding)
            else:
                representation += self.t_gcn(self.t_feat, self.id_embedding)

        representation /= self.num_modal

        self.result = representation
        return representation

    def calculate_loss(self, interaction):
        batch_users = interaction[0]
        pos_items = interaction[1] + self.n_users
        neg_items = interaction[2] + self.n_users

        user_tensor = batch_users.repeat_interleave(2)
        stacked_items = torch.stack((pos_items, neg_items))
        item_tensor = stacked_items.t().contiguous().view(-1)

        out = self.forward()
        user_score = out[user_tensor]
        item_score = out[item_tensor]
        score = torch.sum(user_score * item_score, dim=1).view(-1, 2)
        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight))))
        reg_embedding_loss = (self.id_embedding[user_tensor]**2 + self.id_embedding[item_tensor]**2).mean()
        if self.v_feat is not None:
            reg_embedding_loss += (self.v_gcn.preference**2).mean()
        reg_loss = self.reg_weight * reg_embedding_loss
        return loss + reg_loss

    def full_sort_predict(self, interaction):
        user_tensor = self.result[:self.n_users]
        item_tensor = self.result[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

class GCN_MODULE(torch.nn.Module):
    def __init__(self, graph_data, num_user, num_item, dim_id, aggr_mode='add', concate=True):
        super(GCN_MODULE, self).__init__()
        # NOTE: This makes edge_index move together with the model across devices automatically!
        self.register_buffer('graph_data', graph_data)
        self.edge_index = graph_data.edge_index

        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.concate = concate

        # ID embedding for all users and items
        self.id_embedding = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(num_user + num_item, dim_id))
        )

        # Actor embedding
        self.actor_embedding = nn.Embedding(graph_data.num_actor_ids, dim_id, padding_idx=0)

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
        x = F.leaky_relu(self.g_layer1(torch.cat([h, x_hat], dim=1)) if self.concate else self.g_layer1(h + x_hat))

        h = F.leaky_relu(self.conv2(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.g_layer2(torch.cat([h, x_hat], dim=1)) if self.concate else self.g_layer2(h + x_hat))
    
        h = F.leaky_relu(self.conv3(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear3(x))
        x = F.leaky_relu(self.g_layer3(torch.cat([h, x_hat], dim=1)) if self.concate else self.g_layer3(h + x_hat))

        return x


class GCN_ID_MODULE(torch.nn.Module):
    def __init__(self, edge_index, num_user, num_item, dim_id, aggr_mode='add', concate=True):
        super(GCN_ID_MODULE, self).__init__()
        # NOTE: This makes edge_index move together with the model across devices automatically!
        self.register_buffer('edge_index', edge_index)
        self.edge_index = edge_index

        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.concate = concate

        self.id_embedding = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(num_user + num_item, dim_id))
        )

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
        x = F.leaky_relu(self.g_layer1(torch.cat([h, x_hat], dim=1)) if self.concate else self.g_layer1(h + x_hat))

        h = F.leaky_relu(self.conv2(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.g_layer2(torch.cat([h, x_hat], dim=1)) if self.concate else self.g_layer2(h + x_hat))
    
        h = F.leaky_relu(self.conv3(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear3(x))
        x = F.leaky_relu(self.g_layer3(torch.cat([h, x_hat], dim=1)) if self.concate else self.g_layer3(h + x_hat))

        return x


class BaseModel(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add', **kwargs):
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