import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

class MoleculeGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, graph_dim,
                 hidden_dim=64, num_layers=2):
        super().__init__()

        # embeddings
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)

        # message passing
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(mlp))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim + graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, batch, edge_attr, graph_feats):
        # --- embed ---
        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)

        # --- message passing ---
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index, edge_attr)
            x = x + norm(F.relu(h))  # residual

        # --- graph embedding ---
        g = global_mean_pool(x, batch)
        g = torch.cat([g, graph_feats], dim=1)

        # --- prediction ---
        out = self.readout(g).view(-1)

        return out, g

