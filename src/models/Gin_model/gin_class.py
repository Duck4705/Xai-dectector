import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
import numpy as np

import os
import json


class GINModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3, dropout=0.45):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim * 3),
            nn.BatchNorm1d(hidden_dim * 3), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, num_classes))

    def forward(self, x_or_data, edge_index=None, batch=None):
        if edge_index is None:
            # Called as model(data)
            x, edge_index, batch = x_or_data.x, x_or_data.edge_index, x_or_data.batch
        else:
            # Called as model(x, edge_index, batch=batch) — used by GNNExplainer
            x = x_or_data
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        readouts = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            readouts.append(global_mean_pool(x, batch))
        return self.classifier(torch.cat(readouts, dim=1))




# Load cfg riêng lẻ (dùng bởi gin_node)
def LoadCFG(sample_input_folder: str) -> dict:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sample_path = os.path.join(project_root, "input", sample_input_folder)
    cfg_path = os.path.join(sample_path, "cfg.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_data = json.load(f)  # dict có keys: "nodes", "edges"
    
    return cfg_data
