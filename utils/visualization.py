import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from utils.config import FEATURE_NAMES

def visualize_results(losses, trained_model, test_data, target_feature, feature_stats=None):
    # Plot training and test loss
    epochs = [loss['epoch'] for loss in losses]
    train_losses = [loss['train_loss'] for loss in losses]
    test_losses = [loss['test_loss'] for loss in losses] 

    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, test_losses, 'r-', label='Test Loss')
    plt.title(f'Loss Over Epochs for {target_feature}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Visualize predictions vs actual
    print("\nModel Predictions Analysis:")
    trained_model.eval()
    with torch.no_grad():
        
        # Get model predictions for current test data
        predictions = feature_stats.predicted_values.cpu().numpy()
        
        # Use feature_stats for actual distribution if available
        if feature_stats is not None:
            # Extract actual values from feature_stats
            actuals = feature_stats.actual_values.cpu().numpy()
            print(f"Using {len(actuals)} sampled target values from training")
        else:
            # Fallback to single test value if feature_stats not provided
            target = test_data.y
            print("Using single test target value:", target)
            actuals = target.cpu().numpy().flatten()
        
        # Calculate average prediction and error
        avg_prediction = np.mean(predictions)
        avg_actual = np.mean(actuals)
        avg_error = np.mean(np.abs(avg_prediction - actuals))
        std_predictions = np.std(predictions)
        std_actuals = np.std(actuals)
        
        print(f"Average prediction: {avg_prediction:.6f}")
        print(f"Average actual value: {avg_actual:.6f}")
        print(f"Std dev of actual values: {std_actuals:.6f}")
        print(f"Average absolute error: {avg_error:.6f}")
        print(f"Standard deviation of predictions: {std_predictions:.6f}")
        
        # Plot distributions
        # plt.figure(figsize=(10, 6))
        # plt.hist(predictions, bins=20, alpha=0.5, color='blue', label='Predictions')
        # plt.hist(actuals, bins=20, alpha=0.5, color='red', label='Actual Values')
        # plt.axvline(avg_actual, color='red', linestyle='dashed', linewidth=2, label='True Mean Value')
        # plt.axvline(avg_prediction, color='green', linestyle='dashed', linewidth=2, label='Mean Prediction')
        # plt.xlabel('Values')
        # plt.ylabel('Frequency')
        # plt.title(f'Distribution of Predictions vs Actuals for {target_feature}')
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        # plt.show()
        plot_error_distribution(predictions, actuals, target_feature)
        
        # # If we have feature_stats with predictions, show how predictions evolved over training
        # if feature_stats is not None and hasattr(feature_stats, 'predicted_values'):
        #     plot_error_distribution(predictions, actuals, target_feature)
        #     plt.figure(figsize=(10, 6))
        #     epochs = np.arange(len(feature_stats.predicted_values))
        #     plt.plot(epochs, feature_stats.predicted_values.cpu().numpy(), 'b-', label='Predicted Values')
        #     plt.plot(epochs, feature_stats.actual_values.cpu().numpy(), 'r-', label='Actual Values')
        #     plt.xlabel('Epoch')
        #     plt.ylabel('Value')
        #     plt.title(f'Evolution of Predictions vs Actuals for {target_feature}')
        #     plt.legend()
        #     plt.grid(True, alpha=0.3)
        #     plt.tight_layout()
        #     plt.show()

def plot_error_distribution(predictions, actuals, target_feature, log_scale=True):
    """
    Plots the distribution of absolute errors between predicted and actual values.
    
    Args:
        predictions: Array of predicted values
        actuals: Array of actual values
        target_feature: Name of the feature being predicted
        log_scale: Whether to use a logarithmic scale (default: True)
    """
    # Calculate absolute errors
    abs_errors = np.abs(predictions - actuals)
    
    # Calculate error statistics
    min_error = np.min(abs_errors)
    max_error = np.max(abs_errors)
    mean_error = np.mean(abs_errors)
    median_error = np.median(abs_errors)
    
    # Create the figure
    plt.figure(figsize=(14, 8))
    
    # Increase font sizes
    plt.rcParams.update({'font.size': 16})
    title_fontsize = 26
    label_fontsize = 20
    legend_fontsize = 16
    annotation_fontsize = 16
    
    # Plot histogram with log scale if requested
    if log_scale:
        # Ensure no zeros for log scale
        nonzero_errors = abs_errors[abs_errors > 0]
        if len(nonzero_errors) == 0:
            print("All errors are zero! Perfect predictions.")
            return
            
        # Set minimum error for log scale
        min_nonzero = np.min(nonzero_errors)
        
        # Create log bins that cover the range of errors
        log_min = np.log10(min_nonzero)
        log_max = np.log10(max_error) if max_error > 0 else 0
        if log_min == log_max:  # Handle case where all errors are identical
            log_bins = np.linspace(log_min-0.1, log_min+0.1, 10)
        else:
            log_bins = np.linspace(log_min, log_max, 50)
            
        bins = 10 ** log_bins
        
        # Plot on log scale
        plt.hist(abs_errors, bins=bins, alpha=0.7)
        plt.xscale('log')
        plt.title(f'Distribution of Absolute Errors on Log Scale - {target_feature}', fontsize=title_fontsize)
    else:
        # Linear scale
        plt.hist(abs_errors, bins=30, alpha=0.7)
        plt.title(f'Distribution of Absolute Errors - {target_feature}', fontsize=title_fontsize)
    
    # Add vertical lines for statistics with increased font size
    plt.axvline(x=mean_error, color='r', linestyle='--', linewidth=2.5, 
                label=f'Mean Error: {mean_error:.8f}')
    plt.axvline(x=median_error, color='g', linestyle='--', linewidth=2.5, 
                label=f'Median Error: {median_error:.8f}')
    
    # Annotate min and max with larger font and more visible box
    plt.annotate(f'Min Error: {min_error:.8f}', 
                xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=annotation_fontsize,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.9))
    plt.annotate(f'Max Error: {max_error:.8f}', 
                xy=(0.02, 0.87), xycoords='axes fraction',
                fontsize=annotation_fontsize,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.9))
    
    # Add labels and legend with larger font sizes
    plt.xlabel('Absolute Error (|Predicted - Actual|)', fontsize=label_fontsize)
    plt.ylabel('Frequency', fontsize=label_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.grid(True, alpha=0.3, which='both')  # Grid lines for both major and minor ticks
    
    # Increase tick label size
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # Reset font size to default after plotting
    plt.rcParams.update({'font.size': plt.rcParams.get('font.size')})
    
def visualize_feature_statistics(data, show_histograms=False):
    """
    Generate a table with statistics for all node features in the provided data.
    
    Args:
        data: PyG Data object containing node features
        show_histograms: Whether to display histograms for each feature (default: False)
    """
    print("\n=== Node Feature Statistics ===")
    
    # Extract feature values
    feature_tensor = data.x.cpu().numpy()
    
    # Create a DataFrame to store statistics
    stats = {
        'Feature': [],
        'Mean': [],
        'Min': [],
        'Max': [],
        'Std Dev': [],
        'Variance': [],
        'All Identical': []
    }
    
    # Calculate statistics for each feature
    for i, feature_name in enumerate(FEATURE_NAMES):
        feature_values = feature_tensor[:, i]
        
        # Calculate statistics
        mean_val = np.mean(feature_values)
        min_val = np.min(feature_values)
        max_val = np.max(feature_values)
        std_val = np.std(feature_values)
        var_val = np.var(feature_values)
        all_identical = np.allclose(feature_values, feature_values[0], rtol=1e-5)
        
        # Store in dictionary
        stats['Feature'].append(feature_name)
        stats['Mean'].append(mean_val)
        stats['Min'].append(min_val)
        stats['Max'].append(max_val)
        stats['Std Dev'].append(std_val)
        stats['Variance'].append(var_val)
        stats['All Identical'].append(all_identical)
    
    # Create DataFrame and display
    stats_df = pd.DataFrame(stats)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 180)
    pd.set_option('display.float_format', '{:.6f}'.format)
    print(stats_df.to_string(index=False))
    
    # Reset display options
    pd.reset_option('display.max_rows')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    
    # Count features with no variance
    zero_var_features = sum(stats_df['All Identical'])
    print(f"\nFeatures with no variance: {zero_var_features} out of {len(FEATURE_NAMES)}")
    
    # Generate histograms if requested
    if show_histograms:
        # Determine grid layout
        n_features = len(FEATURE_NAMES)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
        axes = axes.flatten()
        
        for i, feature_name in enumerate(FEATURE_NAMES):
            feature_values = feature_tensor[:, i]
            
            # Skip histogram for identical values
            if stats['All Identical'][i]:
                axes[i].text(0.5, 0.5, f"All values are identical: {stats['Mean'][i]:.6f}",
                            horizontalalignment='center', verticalalignment='center')
            else:
                axes[i].hist(feature_values, bins=20)
                axes[i].axvline(stats['Mean'][i], color='red', linestyle='dashed')
            
            axes[i].set_title(feature_name)
            axes[i].grid(alpha=0.3)
        
        # Hide any unused subplots
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    return stats_df

def visualize_feature_pairs(data_or_stats, num_pairs=5, prioritize_variance=True, use_node_features=False):
    print("\nVisualizing Feature Pairs...")
    
    # Determine what data we're using
    if use_node_features and hasattr(data_or_stats, 'x'):
        # Use actual node features from PyG Data object
        print("Using raw node features (higher variance expected)")
        feature_tensor = data_or_stats.x.cpu().numpy()
        
        # Sample nodes if there are too many (max 500 for efficient plotting)
        if feature_tensor.shape[0] > 500:
            print(f"Sampling 500 nodes from {feature_tensor.shape[0]} total")
            indices = np.random.choice(feature_tensor.shape[0], 500, replace=False)
            feature_tensor = feature_tensor[indices]
            
        # Create DataFrame with all node features
        df = pd.DataFrame(feature_tensor, columns=FEATURE_NAMES)
    else:
        # Use the epoch means as before
        print("Using epoch-wise feature means (may have low variance)")
        feature_means = data_or_stats.feature_means.cpu().numpy()
        df = pd.DataFrame(feature_means, columns=FEATURE_NAMES)
    
    # Remove excluded features
    excluded_features = ['Node_Metric_CoreNumber', 'Node_Metric_IsSelected']
    included_features = [f for f in FEATURE_NAMES if f not in excluded_features]
    df_filtered = df[included_features]
    
    # Calculate and display correlation matrix
    print("\n=== Spearman Correlation Rankings ===")
    corr_matrix = df_filtered.corr(method='spearman')
    
    # Extract and rank correlations
    correlations = []
    for i, feat1 in enumerate(included_features):
        for j, feat2 in enumerate(included_features):
            if i < j:  # Only use upper triangle to avoid duplicates
                corr_value = corr_matrix.loc[feat1, feat2]
                correlations.append((feat1, feat2, corr_value, abs(corr_value)))
    
    # Sort by absolute correlation (descending)
    correlations_sorted = sorted(correlations, key=lambda x: x[3], reverse=True)
    
    # Display ranked correlations
    print(f"{'Feature Pair':<50} {'Correlation':<10} {'Abs Correlation':<15}")
    print("-" * 75)
    for feat1, feat2, corr, abs_corr in correlations_sorted:
        print(f"{feat1} — {feat2:<30} {corr:>10.4f} {abs_corr:>15.4f}")
    
    # Calculate variance for each feature
    feature_variances = {}
    for feature in included_features:
        feature_variances[feature] = df[feature].var()
    
    # Print variance for debugging
    print("\nFeature variances:")
    for feature, var in sorted(feature_variances.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {var:.8f}")
    
    # Select feature pairs based on variance
    pairs = []
    
    if prioritize_variance:
        # Get top features by variance
        sorted_features = sorted(feature_variances.items(), key=lambda x: x[1], reverse=True)
        top_features = [f for f, v in sorted_features if v > 0][:2*num_pairs]
        
        # Generate pairs from top features
        import itertools
        all_possible_pairs = list(itertools.combinations(top_features, 2))
        
        # Take the top num_pairs combinations
        pairs = all_possible_pairs[:num_pairs]
        print(f"\nSelected pairs based on highest variance features:")
    else:
        # Random selection
        import random
        
        while len(pairs) < num_pairs:
            feature1, feature2 = random.sample(included_features, 2)
            if (feature1, feature2) not in pairs and (feature2, feature1) not in pairs:
                pairs.append((feature1, feature2))
        
        print(f"\nSelected random feature pairs:")
    
    # List selected pairs
    for f1, f2 in pairs:
        print(f"  {f1} vs {f2}")
    
    # Calculate rows and columns for plotting layout (max 3 per row)
    n_cols = min(3, len(pairs))
    n_rows = (len(pairs) + n_cols - 1) // n_cols  # Ceiling division
    
    # Create the figure with multiple rows of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    # Handle axes for different scenarios
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each pair
    for i, (feature1, feature2) in enumerate(pairs):
        # Calculate row and column for this pair
        row = i // n_cols
        col = i % n_cols
        
        # Get current axis
        ax = axes[row, col]
        
        # Extract data for this feature pair
        x_data = df[feature1].values
        y_data = df[feature2].values
        
        # Handle outliers - get 5th and 95th percentiles
        x_min_p, x_max_p = np.percentile(x_data, [5, 95])
        y_min_p, y_max_p = np.percentile(y_data, [5, 95])
        
        # Get actual min/max for display purposes
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)
        
        # Calculate ranges based on percentiles to reduce outlier influence
        x_range = x_max_p - x_min_p
        y_range = y_max_p - y_min_p
        
        # Print data range
        print(f"\n{feature1} vs {feature2}:")
        print(f"  {feature1}: min={x_min:.6f}, max={x_max:.6f}, 5-95% range={x_range:.6f}")
        print(f"  {feature2}: min={y_min:.6f}, max={y_max:.6f}, 5-95% range={y_range:.6f}")
        
        # Draw the scatter plot
        scatter = ax.scatter(x_data, y_data, alpha=0.7, s=40, edgecolor='k', linewidth=0.5)
        
        # Calculate correlation coefficient
        corr = df[feature1].corr(df[feature2], method='spearman')
        
        # Set axis limits based on percentiles plus padding
        padding_x = x_range * 0.1
        padding_y = y_range * 0.1
        
        # Use percentile-based limits to reduce outlier influence
        ax.set_xlim(x_min_p - padding_x, x_max_p + padding_x)
        ax.set_ylim(y_min_p - padding_y, y_max_p + padding_y)
        
        # Add annotations
        ax.text(0.05, 0.95, f"Correlation: {corr:.3f}", transform=ax.transAxes,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        
        # Also note full range vs. displayed range
        ax.text(0.05, 0.85, 
                f"Full range {feature1}: [{x_min:.3f}, {x_max:.3f}]\n" + 
                f"Full range {feature2}: [{y_min:.3f}, {y_max:.3f}]", 
                transform=ax.transAxes,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        
        # Add labels and title
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.set_title(f"{feature1} vs {feature2}")
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add regression line
        if x_range > 0 and y_range > 0:
            # Calculate regression on trimmed data to reduce outlier influence
            mask = ((x_data >= x_min_p) & (x_data <= x_max_p) & 
                    (y_data >= y_min_p) & (y_data <= y_max_p))
            if sum(mask) > 2:  # Need at least 3 points for a meaningful line
                z = np.polyfit(x_data[mask], y_data[mask], 1)
                p = np.poly1d(z)
                # Generate points for the line that span the visible range
                x_line = np.linspace(x_min_p - padding_x, x_max_p + padding_x, 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.7)
    
    # Hide any unused subplots
    for i in range(len(pairs), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def visualize_pca(data, n_components=2, use_node_features=True):
    """
    Perform PCA on node features and visualize the results.
    
    Args:
        data: PyG Data object or feature stats
        n_components: Number of PCA components to use
        use_node_features: If True, use actual node features instead of feature means
    """
    
    print("\nPerforming PCA on node features...")
    
    # Determine what data we're using
    if use_node_features and hasattr(data, 'x'):
        # Use actual node features from PyG Data object
        print("Using raw node features")
        feature_tensor = data.x.cpu().numpy()
    else:
        # Use the epoch means
        print("Using epoch-wise feature means")
        feature_tensor = data.feature_means.cpu().numpy()
    
    # Create DataFrame with features
    df = pd.DataFrame(feature_tensor, columns=FEATURE_NAMES)
    
    # Remove any excluded features if needed
    excluded_features = ['Node_Metric_IsSelected']
    included_features = [f for f in FEATURE_NAMES if f not in excluded_features]
    df_filtered = df[included_features]
    
    # Apply PCA
    pca = PCA(n_components=min(n_components, len(included_features)))
    principal_components = pca.fit_transform(df_filtered)
    
    # Create result DataFrame
    pc_cols = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(data=principal_components, columns=pc_cols)
    
    # Print explained variance
    explained_variance = pca.explained_variance_ratio_ * 100
    cumulative_variance = np.cumsum(explained_variance)
    
    print("\nPCA Explained Variance:")
    for i, var in enumerate(explained_variance):
        print(f"  PC{i+1}: {var:.2f}% (Cumulative: {cumulative_variance[i]:.2f}%)")
    
    # Print feature contributions to top components
    print("\nFeature Contributions to Top Components:")
    components_df = pd.DataFrame(
        pca.components_[:2].T,  # Take top 2 components
        columns=['PC1', 'PC2'],
        index=included_features
    )
    print(components_df.sort_values(by='PC1', key=abs, ascending=False))
    
    # Visualization
    plt.figure(figsize=(10, 8))
    
    # If we have many points, use a scatter plot with transparency
    if principal_components.shape[0] > 50:
        plt.scatter(
            principal_components[:, 0], 
            principal_components[:, 1],
            alpha=0.5, 
            edgecolor='k', 
            s=40
        )
    else:
        # For epoch data, connect points to show progression
        plt.plot(principal_components[:, 0], principal_components[:, 1], 'o-')
        
        # Add annotations for a few epochs
        n_points = principal_components.shape[0]
        for i in [0, n_points//4, n_points//2, 3*n_points//4, n_points-1]:
            if i < n_points:
                plt.annotate(
                    f"Epoch {i}", 
                    (principal_components[i, 0], principal_components[i, 1]),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center'
                )
    
    plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2f}%)')
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2f}%)')
    plt.title('PCA of Node Features')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return pca, principal_components