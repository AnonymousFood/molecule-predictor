
# Currently only plots the permutation feature importance.
# Will probably move the correlation and PCA analysis here in the future.

import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from typing import Dict, List
from datetime import datetime
from openpyxl import load_workbook


from torch_geometric.data import Data

def permutation_feature_importance(model, data: Data, feature_names: List[str], 
                                  n_repeats: int = 5, metric: str = 'test_loss',
                                  higher_is_better: bool = False, target_feature=None,
                                  save_to_excel=True):
    """
    Calculate permutation feature importance for a GNN model.
    
    Args:
        model: Trained GNN model with evaluate() method
        data: PyG Data object containing the graph and features
        feature_names: List of feature names corresponding to data.x columns
        n_repeats: Number of times to repeat permutation for each feature
        metric: Performance metric to use from model.evaluate() output
        higher_is_better: Whether higher metric values indicate better performance
    
    Returns:
        DataFrame with importance scores for each feature
    """
    print("\nCalculating permutation feature importance...")
    
    # Get baseline performance
    baseline_performance = model.evaluate(data)[metric]
    print(f"Baseline {metric}: {baseline_performance:.6f}")
    
    # Store importance scores
    importance_scores = []
    
    # For each feature
    for feat_idx, feature_name in enumerate(tqdm(feature_names, desc="Features")):
        # Skip IsSelected feature (it's not a real node feature)
        if feature_name == 'Node_Metric_IsSelected':
            continue
            
        feature_importances = []
        
        # Repeat permutation multiple times to reduce randomness effects
        for i in range(n_repeats):
            # Create a copy of the original data
            permuted_data = data.clone()
            
            # Get original feature values
            feature_values = permuted_data.x[:, feat_idx].clone()
            
            # Permute the feature values
            permuted_values = feature_values[torch.randperm(len(feature_values))]
            permuted_data.x[:, feat_idx] = permuted_values
            
            # Evaluate permuted data
            permuted_performance = model.evaluate(permuted_data)[metric]
            
            # Calculate importance (change in performance)
            if higher_is_better:
                # For metrics like accuracy where higher is better
                importance = baseline_performance - permuted_performance
            else:
                # For metrics like loss where lower is better
                importance = permuted_performance - baseline_performance
                
            feature_importances.append(importance)
        
        # Average across repeats
        mean_importance = np.mean(feature_importances)
        std_importance = np.std(feature_importances)
        
        importance_scores.append({
            'Feature': feature_name,
            'Importance': mean_importance,
            'Std': std_importance
        })
        
        print(f"  {feature_name}: {mean_importance:.6f} ± {std_importance:.6f}")
    
    # Convert to DataFrame and sort by importance
    results_df = pd.DataFrame(importance_scores)
    results_df = results_df.sort_values('Importance', ascending=False).reset_index(drop=True)

    if save_to_excel and target_feature:
        save_importance_to_excel(results_df, target_feature)
    
    return results_df

def save_importance_to_excel(importance_df, target_feature, file_path="results/feature_importances.xlsx"):
    """
    Save feature importance results to an Excel file with sheets for different target features.
    
    Args:
        importance_df: DataFrame with feature importance scores
        target_feature: Name of the target feature being analyzed (will be used as sheet name)
        file_path: Path to the Excel file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Clean sheet name (Excel has 31 character limit for sheet names and restrictions on characters)
    sheet_name = target_feature.replace("GMinus_", "")[:31]
    sheet_name = ''.join(c for c in sheet_name if c.isalnum() or c in ['_', ' '])
    
    # Add timestamp to the dataframe
    importance_df = importance_df.copy()
    importance_df['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    importance_df['Target'] = target_feature
    
    # Check if file exists
    if os.path.exists(file_path):
        try:
            # Use pandas ExcelWriter with mode='a' to append to existing file
            # if_sheet_exists='replace' will replace the sheet if it already exists
            with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
                importance_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"Updated sheet '{sheet_name}' in existing file: {file_path}")
            
        except Exception as e:
            print(f"Error appending to existing file: {e}")
            # Fallback: Create new file
            importance_df.to_excel(file_path, sheet_name=sheet_name, index=False)
            print(f"Created new file with sheet '{sheet_name}': {file_path}")
    else:
        # Create new Excel file
        importance_df.to_excel(file_path, sheet_name=sheet_name, index=False)
        print(f"Created new file with sheet '{sheet_name}': {file_path}")
    
    return file_path