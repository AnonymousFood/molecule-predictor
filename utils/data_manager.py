import torch
import numpy as np
from torch_geometric.data import Data
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split

class DataManager:
    def __init__(self, val_size: float = 0.15, test_size: float = 0.15):
        self.val_size = val_size
        self.test_size = test_size

    def split_data(self, data: Data) -> Dict[str, torch.Tensor]:
        """
        Split data into train, validation and test sets.
        Returns masks for each split.
        """
        num_nodes = data.x.size(0)
        indices = np.arange(num_nodes)

        # First split out test set
        train_val_idx, test_idx = train_test_split(
            indices, 
            test_size=self.test_size,
            random_state=42
        )

        # Then split remaining data into train and validation
        val_size_adjusted = self.val_size / (1 - self.test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size_adjusted,
            random_state=42
        )

        # Create boolean masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        return {
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask
        }