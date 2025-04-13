import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from typing import Dict, List, Tuple
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import time
import torch.optim.lr_scheduler as lr_scheduler

from utils.config import MODEL_CONFIG, TRAINING_CONFIG, RESIDUAL_G_FEATURES
import utils.data_utils as DataUtils

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
        
        # Third GCN layer
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        if MODEL_CONFIG['batch_norm']:
            self.bn3 = nn.BatchNorm1d(hidden_dim)
        if MODEL_CONFIG['layer_norm']:
            self.ln3 = nn.LayerNorm(hidden_dim)
            
        # Make output layer wider
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Training attributes
        self.criterion = nn.MSELoss(reduction='mean')
        self.config = TRAINING_CONFIG
        self.optimizer = None
        self.scheduler = None
        self.best_model = None
        self.best_loss = float('inf')
        self.best_accuracy = 0.0  # Track best accuracy
        
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
            x2 = x2 + x1  # Residual connection
        x2 = self.dropout(x2)
        
        # Third layer with residual connection
        x3 = self.conv3(x2, edge_index)
        if hasattr(self, 'bn3'):
            x3 = self.bn3(x3)
        if hasattr(self, 'ln3'):
            x3 = self.ln3(x3)
        x3 = F.relu(x3)
        if MODEL_CONFIG['residual']:
            x3 = x3 + x2  # Residual connection
        x3 = self.dropout(x3)
        
        return self.final_layer(x3)
    
    #---------------
    # TRAINING CODE
    #---------------

    def train_model(self, model: nn.Module, train_data: Data, test_data: Data, target_idx: int) -> Tuple[List[Dict], nn.Module]:
        self._setup_optimizer(model)
        losses = []
        
        # Get original graphs from the data objects
        G_train = nx.from_edgelist(train_data.edge_index.t().numpy())
        G_test = nx.from_edgelist(test_data.edge_index.t().numpy())
        
        # Early stopping setup
        
        # Store feature stats before training
        print("\nFeature Statistics (Initial):")
        print("Target Feature:", RESIDUAL_G_FEATURES[target_idx])
        
        print("\nTrain Graph:")
        train_residual = G_train.copy()
        selected_nodes_train = train_data.selected_nodes.tolist()
        train_residual.remove_nodes_from(selected_nodes_train)
        print(f"Min degree in G/G': {min(dict(train_residual.degree()).values()) if train_residual.number_of_edges() > 0 else 0}")
        print(f"Num nodes in G/G': {len(train_residual)}")
        
        print("\nTest Graph:")
        test_residual = G_test.copy()
        selected_nodes_test = test_data.selected_nodes.tolist()
        test_residual.remove_nodes_from(selected_nodes_test)
        print(f"Min degree in G/G': {min(dict(test_residual.degree()).values()) if test_residual.number_of_edges() > 0 else 0}")
        print(f"Num nodes in G/G': {len(test_residual)}")
        
        print("\nTraining Progress:")
        training_start = time.time()

        # Get target value range to normalize accuracy calculations
        # This helps with setting appropriate thresholds for different metrics
        target_val = test_data.y.item()

        for epoch in range(self.config['num_epochs']):
            epoch_start = time.time()
            
            # Generate new connected nodes for each graph
            selected_nodes_train = DataUtils.find_connected_subgraph(G_train)
            selected_nodes_test = DataUtils.find_connected_subgraph(G_test)
            
            # Process new data with new selected nodes (includes target value)
            epoch_train_data = DataUtils.process_graph_data(G_train, selected_nodes_train, target_idx)
            epoch_test_data = DataUtils.process_graph_data(G_test, selected_nodes_test, target_idx)

            # Training Step with new data
            train_loss = self._train_step(epoch_train_data)

            # Test Step with new data
            test_loss = self._test_step(epoch_test_data)
            
            # Update learning rate based on test loss
            if self.scheduler is not None:
                self.scheduler.step(test_loss)
            
            self.eval()

            # Print progress for EVERY epoch
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, "
                f"Test Loss = {test_loss:.4f}, "
                f"Time = {epoch_time:.2f}s")

            losses.append({
                'epoch': epoch,
                'train_loss': train_loss.item(),
                'test_loss': test_loss.item()
            })
        
        # Load best model if available
        if self.best_model is not None:
            self.load_state_dict(self.best_model)
            
        total_time = time.time() - training_start
        print(f"\nTraining completed in {total_time:.2f}s")

        # Print final feature statistics
        print("\nFeature Statistics (Final):")
        print("Target Feature:", RESIDUAL_G_FEATURES[target_idx])
        
        return losses, self

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
        
        # Add learning rate scheduler
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True
        )
    
    def _train_step(self, data: Data) -> torch.Tensor:
        self.train()
        self.optimizer.zero_grad()
        
        output = self(data)
        target = data.y.reshape(-1, 1).to(output.device)
        
        # Broadcast target to match output shape
        if output.shape[0] != target.shape[0]:
            target = target.expand(output.shape[0], -1)
            
        loss = self.criterion(output, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.config.get('clip_grad_norm', 1.0))
        self.optimizer.step()
        
        return loss

    def _test_step(self, data: Data) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            output = self(data)
            target = data.y.reshape(-1, 1).to(output.device)
            
            # Broadcast target to match output shape
            if output.shape[0] != target.shape[0]:
                target = target.expand(output.shape[0], -1)
                
            loss = self.criterion(output, target)
            return loss

    def evaluate(self, data: Data) -> Dict[str, float]:
        self.eval()
        metrics = {}
        
        with torch.no_grad():
            output = self(data)
            target = data.y.reshape(-1, 1).to(output.device)
            
            # Broadcast target to match output shape
            if output.shape[0] != target.shape[0]:
                target = target.expand(output.shape[0], -1)
            
            test_loss = self.criterion(output, target)
            
            metrics['test_loss'] = test_loss.item()
            
            # Mean Absolute Error
            mae = torch.mean(torch.abs(output - target))
            metrics['mae'] = mae.item()
            
            # R-squared score with safeguards
            target_mean = torch.mean(target)
            ss_tot = torch.sum((target - target_mean) ** 2)
            ss_res = torch.sum((target - output) ** 2)
            
            # Avoid division by zero or very small values
            if ss_tot > 1e-10:
                r2 = 1 - (ss_res / ss_tot)
                r2 = torch.clamp(r2, min=-1.0, max=1.0)
                metrics['r2_score'] = r2.item()
            else:
                metrics['r2_score'] = 0.0
                
            # Add RMSE
            rmse = torch.sqrt(test_loss)
            metrics['rmse'] = rmse.item()

        return metrics