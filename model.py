import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data

from utils.config import MODEL_CONFIG, TRAINING_CONFIG

class GNN(nn.Module):
    def __init__(self, node_feature_dim):
        super().__init__()
        hidden_dim = MODEL_CONFIG['hidden_dim']
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
        
        # Training attributes
        self.criterion = nn.MSELoss()
        self.config = TRAINING_CONFIG
        self.optimizer = None
        self.best_model = None
        
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

    #---------------
    # TRAINING CODE
    #---------------

    def train_model(self, model: nn.Module, train_data: Data, test_data: Data, target_idx: int) -> Tuple[List[Dict], nn.Module]:
        self._setup_optimizer(model)

        losses = []

        for epoch in range(self.config['num_epochs']):
            # Training Step
            train_loss = self._train_step(train_data, target_idx)

            # Test Step (test_data is data from a different graph)
            test_loss = self._test_step(test_data, target_idx)

            # Progress logging
            if epoch % 5 == 0:
                print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, "
                    f"Test Loss = {test_loss:.4f}")

            losses.append({
                'epoch': epoch,
                'train_loss': train_loss.item(),
                'test_loss': test_loss.item()
            })

        return losses, model

    def _setup_optimizer(self, model: nn.Module) -> None:
        optimizer_class = getattr(optim, self.config['optimizer']['type'])
        self.optimizer = optimizer_class(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(self.config['optimizer']['beta1'], 
                self.config['optimizer']['beta2']),
            eps=self.config['optimizer']['eps']
        )
    

    def _train_step(self, data: Data, target_idx: int) -> torch.Tensor:
        self.train()
        self.optimizer.zero_grad()
        
        output = self(data)
        loss = self.criterion(
            output, 
            data.original_features[target_idx].reshape(-1, 1)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss

    def _test_step(self, data: Data, target_idx: int) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            output = self(data)
            loss = self.criterion(
                output, 
                data.original_features[target_idx].reshape(-1, 1)
            )
            return loss

    def evaluate(self, data: Data, target_idx: int) -> Dict[str, float]:
        """Evaluate model performance and return metrics"""
        self.eval()
        metrics = {}
        
        with torch.no_grad():
            output = self(data)
            target = data.original_features[target_idx].reshape(-1, 1)
            test_loss = self.criterion(output, target)
            
            metrics['test_loss'] = test_loss.item()
            
            # Mean Absolute Error
            mae = torch.mean(torch.abs(output - target))
            metrics['mae'] = mae.item()
            
            # R-squared score
            target_mean = torch.mean(target)
            ss_tot = torch.sum((target - target_mean) ** 2)
            ss_res = torch.sum((target - output) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            metrics['r2_score'] = r2.item()

        return metrics