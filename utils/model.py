import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv

config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)
    MODEL_CONFIG = config['model']

class SingleFeatureGNNModel(nn.Module):
    def __init__(self, node_feature_dim):
        super().__init__()
        hidden_dim = MODEL_CONFIG['hidden_dim']  # Get hidden_dim from config
        self.dropout = nn.Dropout(MODEL_CONFIG['dropout_rate'])
        
        # First GCN layer
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        if MODEL_CONFIG['batch_norm']:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
        if MODEL_CONFIG['layer_norm']:
            self.ln1 = nn.LayerNorm(hidden_dim)
        
        # Second GCN layer
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        if MODEL_CONFIG['batch_norm']:
            self.bn2 = nn.BatchNorm1d(hidden_dim)
        if MODEL_CONFIG['layer_norm']:
            self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.final_layer = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First layer
        x1 = self.conv1(x, edge_index)
        if hasattr(self, 'bn1'):
            x1 = self.bn1(x1)
        if hasattr(self, 'ln1'):
            x1 = self.ln1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        
        # Second layer with residual connection
        x2 = self.conv2(x1, edge_index)
        if hasattr(self, 'bn2'):
            x2 = self.bn2(x2)
        if hasattr(self, 'ln2'):
            x2 = self.ln2(x2)
        x2 = F.relu(x2)
        if MODEL_CONFIG['residual']:
            x2 = self.dropout(x2 + x1)
        else:
            x2 = self.dropout(x2)
        
        return self.final_layer(x2)