import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, MessagePassing


class KGAT(nn.Module):
    def __init__(self, hetero_data, num_nodes_dict, embed_dim, num_layers=3, aggr="bi-interaction"):
        super().__init__()
        self.hetero_data = hetero_data
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.aggr = aggr

        # 1. Initialize unified embedding table (no need to offset edges) and relations embeddings
        self.embeddings = nn.ModuleDict(
            {
                node_type: nn.Embedding(num_nodes_dict[node_type], embed_dim)
                for node_type in self.hetero_data.node_types
            }
        )
        self.r_embs = nn.ParameterDict(
            {edge_type[1]: nn.Parameter(torch.Tensor(embed_dim)) for edge_type in self.hetero_data.edge_types}
        )
        self.trans_m = nn.ModuleDict(
            {
                edge_type[1]: nn.Linear(embed_dim, embed_dim, bias=False)
                for edge_type in self.hetero_data.edge_types
            }
        )

        # 2. Stack of KGAT layers
        self.kgat_layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type[1]: KGATConv(self.r_embs[edge_type[1]], self.trans_m[edge_type[1]])
                    for edge_type in self.hetero_data.edge_types
                },
                aggr="sum",
            )
            self.kgat_layers.append(conv)

        # 3. Initialize aggregation modules
        self.linear1 = (
            nn.Linear(embed_dim * 2, embed_dim) if aggr == "graphsage" else nn.Linear(embed_dim, embed_dim)
        )
        self.linear_bi = nn.Linear(embed_dim, embed_dim) if aggr == "bi-interaction" else None
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self):
        """Forward pass through the KGAT model. return Dict[node_type: embeddings]"""
        x_dict = self.embeddings
        edge_index_dict = self.hetero_data.edge_index_dict
        all_embeddings = [ego_emb_dict]

        # 4. Multi-layer propagation
        for kgat_layer in self.kgat_layers:
            # 4-1. Aggregate neighbor embeddings
            ego_emb_dict = x_dict
            neighbor_emb_dict = kgat_layer(ego_emb_dict, edge_index_dict)
            # 4-2. Aggregate ego and side info (as in KGAT)
            x_dict = self.aggregate(ego_emb_dict, neighbor_emb_dict, aggr=self.aggr)
            all_embeddings.append(x_dict)

        # 5. Concatenate embeddings from all layers
        final_emb_dict = {}
        for node_type in x_dict:
            final_emb_dict[node_type] = torch.cat([emb_dict[node_type] for emb_dict in all_embeddings], dim=-1)
        
        return final_emb_dict  # final embeddings per node type

    def aggregate(self, v_dict, v_neighbor_dict, aggr="bi-interaction"):
        if aggr == "gcn":
            return {
                node_type: self.leaky_relu(self.linear1(v_dict[node_type] + v_neighbor_dict[node_type]))
                for node_type in v_dict
            }
        elif aggr == "graphsage":
            return {
                node_type: self.leaky_relu(
                    self.linear1(torch.cat([v_dict[node_type], v_neighbor_dict[node_type]]))
                )
                for node_type in v_dict
            }
        else:  # bi-interaction
            return {
                node_type: self.leaky_relu(self.linear1(v_dict[node_type] + v_neighbor_dict[node_type]))
                + self.leaky_relu(self.linear_bi(v_dict[node_type] * v_neighbor_dict[node_type]))
                for node_type in v_dict
            }


class KGATConv(MessagePassing):
    def __init__(self, r_emb, trans_r):
        super().__init__(aggr="add")  # or 'mean'
        self.r_emb = r_emb
        self.trans_r = trans_r

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # h = x_i (source), t = x_j (target)
        h = self.trans_r(x_i)
        t = self.trans_r(x_j)
        r = self.r_emb

        score = (t * torch.tanh(h + r)).sum(dim=-1)  # [num_edges]
        attn = F.softmax(score, x_i)  # softmax over neighbors of source
        return t * attn.unsqueeze(-1)

    def update(self, aggr_out):
        return aggr_out
