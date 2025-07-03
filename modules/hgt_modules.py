"""
HGT (Heterogeneous Graph Transformer) Implementation with PyTorch Geometric.
Hu, Ziniu, et al. "Heterogeneous graph transformer."
Proceedings of the web conference 2020. 2020.
"""

import torch.nn as nn
from common.init import xavier_uniform_initialization
from torch_geometric.nn.conv import HGTConv


class HGT(nn.Module):
    def __init__(
        self,
        hetero_data,
        embed_dim,
        num_layers=3,
        device="cuda",
    ):
        super().__init__()
        self.hetero_data = hetero_data
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.device = device

        # Initialize unified embedding table (no need to offset edges) and relations embeddings
        self.embeddings = nn.ModuleDict(
            {
                node_type: nn.Embedding(hetero_data[node_type].num_nodes, embed_dim, sparse=True)
                for node_type in self.hetero_data.node_types
            }
        )
        xavier_uniform_initialization(self.embeddings)

        # Stack of HGT layers
        self.hgt_layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                in_channels=embed_dim,
                out_channels=embed_dim,
                metadata=self.hetero_data.metadata(),
                heads=1,
            )
            self.hgt_layers.append(conv)

    def forward(self, hetero_graph):
        """Forward pass through the HGT model. return Dict[node_type: embeddings]"""
        device = next(self.parameters()).device
        x_dict = {k: v.weight.to(device) for k, v in self.embeddings.items()}
        edge_index_dict = {k: v.to(device) for k, v in hetero_graph.edge_index_dict.items()}

        # Multi-layer propagation
        for hgt_layer in self.hgt_layers:
            x_dict = hgt_layer(x_dict, edge_index_dict)  # already includes residual connection

        return x_dict  # final embeddings per node type
