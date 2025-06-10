import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, MessagePassing
from torch_geometric.utils import softmax


class KGAT(nn.Module):
    def __init__(self, hetero_data, embed_dim, num_layers=3, aggr="bi-interaction", device="cuda"):
        super().__init__()
        self.hetero_data = hetero_data
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.aggr = aggr
        self.device = device

        # 1. Initialize unified embedding table (no need to offset edges) and relations embeddings
        self.embeddings = nn.ModuleDict(
            {
                node_type: nn.Embedding(hetero_data[node_type].num_nodes, embed_dim)
                for node_type in self.hetero_data.node_types
            }
        )
        self.r_embs = nn.ParameterDict(
            {edge_type[1]: nn.Parameter(torch.Tensor(embed_dim)) for edge_type in self.hetero_data.edge_types}
        )
        for param in self.r_embs.values():
            nn.init.xavier_uniform_(param.data.unsqueeze(0)).squeeze(0)

        self.trans_m = nn.ModuleDict(
            {
                edge_type[1]: nn.Linear(embed_dim, embed_dim, bias=False)
                for edge_type in self.hetero_data.edge_types
            }
        )

        self.rel_params = RelationParams(self.r_embs, self.trans_m)

        # 2. Stack of KGAT layers
        self.kgat_layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: KGATConv(self.rel_params, edge_type[1])
                    for edge_type in self.hetero_data.edge_types
                },
                aggr="sum",
            )
            self.kgat_layers.append(conv)

        # 3. Initialize aggregation modules
        self.linear = (
            nn.Linear(embed_dim * 2, embed_dim) if aggr == "graphsage" else nn.Linear(embed_dim, embed_dim)
        )
        self.linear_bi = nn.Linear(embed_dim, embed_dim) if aggr == "bi-interaction" else None
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, hetero_graph):
        """Forward pass through the KGAT model. return Dict[node_type: embeddings]"""
        device = next(self.parameters()).device
        x_dict = {k: v.weight.to(device) for k, v in self.embeddings.items()}
        edge_index_dict = {k: v.to(device) for k, v in hetero_graph.edge_index_dict.items()}
        all_embeddings = [x_dict]

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
            final_emb_dict[node_type] = torch.cat(
                [emb_dict[node_type] for emb_dict in all_embeddings], dim=-1
            )

        return final_emb_dict  # final embeddings per node type

    def aggregate(self, v_dict, v_neighbor_dict, aggr="bi-interaction"):
        if aggr == "gcn":
            return {
                node_type: self.leaky_relu(self.linear(v_dict[node_type] + v_neighbor_dict[node_type]))
                for node_type in v_dict
            }
        elif aggr == "graphsage":
            return {
                node_type: self.leaky_relu(
                    self.linear(torch.cat([v_dict[node_type], v_neighbor_dict[node_type]]))
                )
                for node_type in v_dict
            }
        else:  # bi-interaction
            return {
                node_type: self.leaky_relu(self.linear(v_dict[node_type] + v_neighbor_dict[node_type]))
                + self.leaky_relu(self.linear_bi(v_dict[node_type] * v_neighbor_dict[node_type]))
                for node_type in v_dict
            }


class RelationParams(nn.Module):
    def __init__(self, r_embs: nn.ParameterDict, trans_m: nn.ModuleDict):
        super().__init__()
        self.r_embs = r_embs
        self.trans_m = trans_m

    def get(self, rel_name):
        return self.r_embs[rel_name], self.trans_m[rel_name]


class KGATConv(MessagePassing):
    def __init__(self, rel_params, rel_name):
        super().__init__(aggr="add")  # or 'mean'
        self.rel_params = rel_params
        self.rel_name = rel_name

    def forward(self, x, edge_index):
        r_emb, trans_r = self.rel_params.get(self.rel_name)

        device = r_emb.device
        x = (x[0].to(device), x[1].to(device))
        edge_index = edge_index.to(device)
        return self.propagate(edge_index, x=x, r_emb=r_emb, trans_r=trans_r)

    def message(self, x_i, x_j, r_emb, trans_r, edge_index):
        # by PyG default, x_i is the center node (target), and x_j are its neighbors (source)
        h = trans_r(x_j)
        t = trans_r(x_i)
        r = r_emb

        score = (t * torch.tanh(h + r)).sum(dim=-1)  # [num_edges]
        attn = softmax(score, edge_index[1])  # softmax over neighbors of source
        return t * attn.unsqueeze(-1)

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=1)
