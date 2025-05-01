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
    def __init__(self, node_feature_dim, is_continuous=False):
        super().__init__()
        hidden_dim = 64

        self.is_continuous = is_continuous
        
        # Single GCN layer
        self.conv = GCNConv(node_feature_dim, hidden_dim)
        
        # Simple output layer
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        self.criterion = nn.MSELoss()
        
    def forward(self, data):
        if self.is_continuous:
            x, edge_index = data.x, data.edge_index
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)  # Since all nodes belong to one graph
            
            # Single GCN layer with activation
            x = self.conv(x, edge_index) # GCN Layer handles message passing
            x = F.relu(x)

            # Node-level predictions
            node_preds = self.final_layer(x)

            # Aggregate to graph-level prediction (mean pooling)
            return global_mean_pool(node_preds, batch)
        else:
            x, edge_index = data.x, data.edge_index
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

            x = F.relu(self.conv(x, edge_index))
            node_preds = self.final_layer(x)
            graph_pred = global_mean_pool(node_preds, batch)

            return graph_pred.squeeze(-1)  # scalar per graph
    
    def predict_integer(self, data, clamp_range=None):
        with torch.no_grad():
            out = self.forward(data)
            pred = out.round()
            if clamp_range:
                pred = pred.clamp(min=clamp_range[0], max=clamp_range[1])
            return pred.long()
    
    #---------------
    # TRAINING CODE
    #---------------

    def train_model(self, model: nn.Module, train_data: Data, test_data: Data, target_idx: int, feature_mask=None) -> Tuple[List[Dict], nn.Module, Data]:
        self.optimizer = torch.optim.Adam(model.parameters(), TRAINING_CONFIG['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
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

            # Apply feature mask if provided
            if feature_mask is not None:
                epoch_train_data.x = epoch_train_data.x[:, feature_mask]
                epoch_test_data.x = epoch_test_data.x[:, feature_mask]

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
    
    def train_model_with_features(self, train_data, test_data, target_idx, num_epochs=300):
        # Determine if target is discrete
        target_metric = RESIDUAL_G_FEATURES[target_idx].replace("GMinus_", "")
        is_discrete = target_metric in ["ConnectedComponents", "MaxDegree", "MinDegree"]

        if "DegreeAssortativity" in target_metric:
            # Use smaller learning rate
            learning_rate = TRAINING_CONFIG['learning_rate'] * 10
            # Add weight decay for regularization
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0.00)
            # Use different loss function for training
            # Custom loss function that emphasizes sign agreement and handles spikes
            def assortativity_loss(pred, target):
                # Get absolute error
                abs_error = torch.abs(pred - target)
                
                # Basic smooth L1 loss with appropriate beta
                # Larger beta (0.1) creates a smoother transition region
                smooth_l1 = torch.where(
                    abs_error < 0.1,
                    0.5 * (abs_error ** 2) / 0.1,
                    abs_error - 0.05  # 0.5 * 0.1
                )
                
                # Sign agreement penalty
                # Strong penalty when prediction has wrong sign
                sign_pred = torch.sign(pred)
                sign_target = torch.sign(target)
                sign_penalty = torch.where(
                    sign_pred == sign_target, 
                    torch.zeros_like(pred),  
                    0.05 * torch.abs(target)  # Scale penalty by magnitude of target
                )
                
                # Combine losses
                combined_loss = smooth_l1 + sign_penalty
                
                return torch.mean(combined_loss)
            
            criterion = assortativity_loss
        # For discrete targets, use a custom loss that encourages integer predictions
        elif is_discrete:
            def discrete_loss(pred, target):
                # Base loss using Huber (robust to outliers)
                base_loss = F.huber_loss(pred, target, delta=1.0)
                
                # Add penalty for non-integer predictions
                # This encourages predictions to be close to integers
                integer_penalty = torch.mean(torch.abs(pred - pred.round()))
                
                # Combine losses, with weight on the integer penalty
                return base_loss + 0.1 * integer_penalty
            
            learning_rate = TRAINING_CONFIG['learning_rate'] 
            criterion = discrete_loss
        else:
            learning_rate = TRAINING_CONFIG['learning_rate'] 
            criterion = self.criterion

        optimizer = torch.optim.Adam(self.parameters(), learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        losses = []
        
        # Create tensors to store statistics
        num_features = train_data.x.shape[1]
        feature_means = torch.zeros(num_epochs, num_features)
        
        # Store original graphs so we can make new G' and G/G' graphs every epoch
        train_edge_list = train_data.edge_index.t().cpu().numpy()
        train_G = nx.Graph()
        train_G.add_nodes_from(range(train_data.x.shape[0]))
        train_G.add_edges_from(train_edge_list)
        
        test_edge_list = test_data.edge_index.t().cpu().numpy()
        test_G = nx.Graph()
        test_G.add_nodes_from(range(test_data.x.shape[0]))
        test_G.add_edges_from(test_edge_list)
        
        num_selected = len(train_data.selected_nodes) if hasattr(train_data, 'selected_nodes') else max(3, int(0.1 * train_G.number_of_nodes()))
        
        actual_values = torch.zeros(num_epochs)
        predicted_values = torch.zeros(num_epochs)

        # ---------- INITIAL SCALING STEP ----------
        # Do an initial forward pass to get the model's raw output scale
        self.eval()  # Set to evaluation mode
        
        # Sample nodes for initial scaling
        initial_selected_nodes = random.sample(list(train_G.nodes()), num_selected)
        initial_data = Data(
            x=train_data.x.clone(),  # Keep original features
            edge_index=train_data.edge_index.clone(),
            y=train_data.y.clone(),
            selected_nodes=initial_selected_nodes
        )
        
        with torch.no_grad():
            initial_output = self(initial_data)
            initial_target = initial_data.y.reshape(-1, 1).to(initial_output.device)
            
            # Calculate scaling factor (avoid division by zero)
            if torch.abs(torch.mean(initial_output)) > 1e-6:
                scale_factor = torch.mean(initial_target) / torch.mean(initial_output)
            else:
                # If predictions are near zero, use small positive value based on target scale
                scale_factor = torch.sign(torch.mean(initial_target)) * max(1.0, torch.abs(torch.mean(initial_target)))
            
            # Apply scaling to final layer weights and bias
            with torch.no_grad():
                self.final_layer.weight.data *= scale_factor
                if self.final_layer.bias is not None:
                    self.final_layer.bias.data *= scale_factor
        
        print(f"\nApplied initial scaling factor: {scale_factor.item():.4f}\n")
        # ---------- END INITIAL SCALING ----------

        feature_mask = getattr(train_data, 'feature_mask', None)
        
        print("\nTraining Progress:")
        for epoch in range(num_epochs):
            selected_nodes_train = random.sample(list(train_G.nodes()), num_selected)
            selected_nodes_test = random.sample(list(test_G.nodes()), num_selected)
            
            # Create the modified subgraph for this epoch
            subgraph_G_prime = train_G.copy()
            subgraph_G_prime.remove_nodes_from(selected_nodes_train)
            
            # Recalculate the target property for this specific modification
            epoch_train_data = DataUtils.process_graph_data(train_G, selected_nodes_train, target_idx, feature_mask=feature_mask)
            epoch_test_data = DataUtils.process_graph_data(test_G, selected_nodes_test, target_idx, feature_mask=feature_mask)
            
            # Training step
            self.train()
            optimizer.zero_grad()
            output = self(epoch_train_data)
            target = epoch_train_data.y.reshape(-1, 1).to(output.device)
            
            # Add small noise to discrete targets to help with generalization
            if is_discrete:
                noise_scale = 0.05 * torch.mean(target.abs())
                target_noisy = target + torch.randn_like(target) * noise_scale
                train_loss = criterion(output, target_noisy)
            else:
                train_loss = criterion(output, target)
                
            train_loss.backward()
            
            # Apply gradient clipping for stability
            # torch.nn.utils.clip_grad_norm_(self.parameters(), TRAINING_CONFIG['clip_grad_norm'])
            
            optimizer.step()
            
            # Testing step
            self.eval()
            with torch.no_grad():
                test_output = self(epoch_test_data)
                test_target = epoch_test_data.y.reshape(-1, 1).to(test_output.device)
                
                # Store statistics
                feature_means[epoch] = torch.mean(epoch_train_data.x, dim=0)
                actual_values[epoch] = epoch_train_data.y
                predicted_values[epoch] = torch.mean(output).item()
                
                # Handle discrete metrics
                if is_discrete:
                    test_rounded = test_output.round()
                    test_loss_raw = criterion(test_output, test_target)
                    test_loss_rounded = criterion(test_rounded, test_target)
                    test_loss = min(test_loss_raw.item(), test_loss_rounded.item())
                    if epoch % 10 == 0:  # Only print every 10 epochs
                        print(f"Epoch {epoch}: Train Loss = {train_loss.item():.4e}, Test Loss = {test_loss:.4e}")
                else:
                    test_loss = criterion(test_output, test_target)
                    if epoch % 10 == 0:
                        print(f"Epoch {epoch}: Train Loss = {train_loss.item():.4e}, Test Loss = {test_loss.item():.4e}")
            
            losses.append({
                'epoch': epoch,
                'train_loss': train_loss.item(),
                'test_loss': test_loss.item() if not is_discrete else test_loss
            })
        
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