from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from utils.config import TRAINING_CONFIG

class TrainingManager:
    def __init__(self, config: Dict = TRAINING_CONFIG):
        self.config = config
        self.criterion = nn.MSELoss()
        
    def train(self, 
          model: nn.Module, 
          data: Data, 
          target_idx: int) -> Tuple[List[Dict], nn.Module]:
        self._setup_optimizer(model)
        
        losses = []
        best_loss = float('inf')
        best_model = None
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            # Training, Validation and Test Steps
            train_loss = self._train_step(model, data, target_idx)
            val_loss = self._validation_step(model, data, target_idx)
            test_loss = self._test_step(model, data, target_idx)
            
            # Early stopping check
            early_stop_cfg = self.config['early_stopping']
            if early_stop_cfg['enabled']:
                if val_loss < best_loss - early_stop_cfg['min_delta']:
                    best_loss = val_loss
                    best_model = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stop_cfg['patience']:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # Progress logging
            if epoch % 5 == 0:
                print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, "
                    f"Val Loss = {val_loss:.4f}, Test Loss = {test_loss:.4f}")
            
            losses.append({
                'epoch': epoch,
                'train_loss': train_loss.item(),
                'val_loss': val_loss.item(),
                'test_loss': test_loss.item()
            })
        
        # Load best model if early stopping was enabled
        if early_stop_cfg['enabled'] and best_model is not None:
            model.load_state_dict(best_model)
        
        return losses, model

    def test(self, 
         model: nn.Module, 
         data: Data, 
         target_idx: int) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        Returns metrics dictionary containing test loss and other relevant metrics.
        """
        model.eval()
        metrics = {}
        
        with torch.no_grad():
            # Get model predictions
            output = model(data)
            test_loss = self.criterion(
                output,
                data.original_features[target_idx].reshape(-1, 1)
            )
            
            # Calculate additional metrics if needed
            predictions = output.numpy().flatten()
            targets = data.original_features[target_idx].numpy()
            
            # Mean Squared Error
            metrics['test_loss'] = test_loss.item()
            
            # Mean Absolute Error 
            mae = torch.mean(torch.abs(output - data.original_features[target_idx].reshape(-1, 1)))
            metrics['mae'] = mae.item()
            
            # R-squared score
            target_mean = torch.mean(data.original_features[target_idx].reshape(-1, 1))
            ss_tot = torch.sum((data.original_features[target_idx].reshape(-1, 1) - target_mean) ** 2)
            ss_res = torch.sum((data.original_features[target_idx].reshape(-1, 1) - output) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            metrics['r2_score'] = r2.item()

        return metrics
        
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
    
    def _train_step(self, model: nn.Module, data: Data, target_idx: int) -> torch.Tensor:
        model.train()
        self.optimizer.zero_grad()
        
        output = model(data)
        loss = self.criterion(
            output, 
            data.original_features[target_idx].reshape(-1, 1)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss
    
    def _validation_step(self, model: nn.Module, data: Data, target_idx: int) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            output = model(data)
            return self.criterion(
                output, 
                data.original_features[target_idx].reshape(-1, 1)
            )
    
    def _test_step(self, model: nn.Module, data: Data, target_idx: int) -> torch.Tensor:
        """Calculate test loss for current epoch"""
        model.eval()
        with torch.no_grad():
            output = model(data)
            return self.criterion(
                output,
                data.original_features[target_idx].reshape(-1, 1)
            )