import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from typing import Dict, List, Tuple
from torch_geometric.data import Data
import time
import random
import networkx as nx

from utils.config import TRAINING_CONFIG, RESIDUAL_G_FEATURES, FEATURE_NAMES
import utils.data_utils as DataUtils

class GNN(nn.Module):
    def __init__(self, node_feature_dim):
        super().__init__()
        hidden_dim = 64
        
        # Single GCN layer
        self.conv = GCNConv(node_feature_dim, hidden_dim)
        
        # Simple output layer
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        self.criterion = nn.MSELoss()
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)  # Since all nodes belong to one graph
        
        # Single GCN layer with activation
        x = self.conv(x, edge_index) # GCN Layer handles message passing
        x = F.relu(x)

        # Node-level predictions
        node_preds = self.final_layer(x)

        # Aggregate to graph-level prediction (mean pooling)
        return global_mean_pool(node_preds, batch)
    
    #---------------
    # TRAINING CODE
    #---------------

    def train_model(self, model: nn.Module, train_data: Data, test_data: Data, target_idx: int) -> Tuple[List[Dict], nn.Module, Data]:
        self.optimizer = torch.optim.Adam(model.parameters(), TRAINING_CONFIG['learning_rate'])
        
        losses = []
        num_epochs = TRAINING_CONFIG['num_epochs']
        num_features = train_data.x.shape[1]

        # Store original graphs so we can make new G' and G/G' graphs every epoch
        train_edge_list = train_data.edge_index.t().cpu().numpy()
        train_G = nx.Graph()
        train_G.add_nodes_from(range(train_data.x.shape[0]))
        train_G.add_edges_from(train_edge_list)
        
        test_edge_list = test_data.edge_index.t().cpu().numpy()
        test_G = nx.Graph()
        test_G.add_nodes_from(range(test_data.x.shape[0]))
        test_G.add_edges_from(test_edge_list)

        num_selected = len(train_data.selected_nodes)
        
        # Create tensors to store statistics
        feature_means = torch.zeros(num_epochs, num_features)
        actual_values = torch.zeros(num_epochs)
        predicted_values = torch.zeros(num_epochs)
        
        print("\nTraining Progress:")
        training_start = time.time()
        
        for epoch in range(num_epochs):

            # Sample new nodes to remove for this epoch
            selected_nodes_train = random.sample(list(train_G.nodes()), num_selected)
            selected_nodes_test = random.sample(list(test_G.nodes()), num_selected)

             # Recalculate everything using process_graph_data
            epoch_train_data = DataUtils.process_graph_data(train_G, selected_nodes_train, target_idx)
            epoch_test_data = DataUtils.process_graph_data(test_G, selected_nodes_test, target_idx)

            # Training step
            self.train()
            self.optimizer.zero_grad()
            output = self(epoch_train_data)
            target = epoch_train_data.y.reshape(-1, 1).to(output.device)
            train_loss = self.criterion(output, target)
            train_loss.backward()
            self.optimizer.step()
            
            # Testing step
            self.eval()
            with torch.no_grad():
                test_output = self(epoch_test_data)
                test_target = epoch_test_data.y.reshape(-1, 1).to(test_output.device)
                test_loss = self.criterion(test_output, test_target)
                
                # Store statistics
                feature_means[epoch] = torch.mean(epoch_train_data.x, dim=0)
                actual_values[epoch] = epoch_train_data.y
                predicted_values[epoch] = torch.mean(output).item()
            
            print(f"Epoch {epoch}: Train Loss = {train_loss.item():.4e}, Test Loss = {test_loss.item():.4e}")
            
            losses.append({
                'epoch': epoch,
                'train_loss': train_loss.item(),
                'test_loss': test_loss.item()
            })
        
        total_time = time.time() - training_start
        print(f"\nTraining completed in {total_time:.2f}s")
        
        # Create feature stats data object
        feature_stats = Data(
            feature_means=feature_means,           
            actual_values=actual_values,           
            predicted_values=predicted_values,     
            feature_names=FEATURE_NAMES,           
            target_feature=RESIDUAL_G_FEATURES[target_idx]  
        )
        
        return losses, self, feature_stats

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