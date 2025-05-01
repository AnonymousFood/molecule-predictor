import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import random

from utils.fast_features import fast_clustering_coefficient, fast_global_efficiency, fast_local_efficiency, fast_shortest_path_length
from utils.config import FEATURE_NAMES, RESIDUAL_G_FEATURES

def find_connected_subgraph(G, size=100):
    # First relabel nodes to ensure consecutive integers
    G = nx.convert_node_labels_to_integers(G)
    
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        if len(subgraph) >= size:
            start_node = np.random.choice(list(subgraph.nodes()))
            nodes = list(nx.bfs_tree(subgraph, start_node))[:size]
            return nodes
    return None

def generate_graph(num_nodes=1000, edge_prob=0.2, max_attempts=5):
    for _ in range(max_attempts):
        G = nx.erdos_renyi_graph(n=num_nodes, p=edge_prob)
        
        # Find largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        if len(largest_cc) >= 100:
            G = nx.convert_node_labels_to_integers(G)
            size = int(num_nodes / 10) # 10% of nodes
            connected_nodes = find_connected_subgraph(G, size=size)
            if connected_nodes is not None:
                return G, connected_nodes
    
    raise RuntimeError("Could not generate suitable graph after max attempts")

def process_graph_data(G, selected_nodes, target_idx, feature_mask=None):
    """
    Process graph data to extract node features and compute target values.
    
    Args:
        G: Input graph
        selected_nodes: Nodes to remove for G' calculation
        target_idx: Index of the target feature
        feature_mask: Boolean mask of which features to include (default: all features)
    
    Returns:
        Data object with node features and target value
    """
    # Relabel and map nodes
    G = nx.convert_node_labels_to_integers(G)
    node_mapping = {old: new for new, old in enumerate(G.nodes())}
    selected_nodes = [node_mapping[node] for node in selected_nodes]
    
    num_nodes = len(G)
    
    # If no mask is provided, include all features
    if feature_mask is None:
        feature_mask = [True] * len(FEATURE_NAMES)
        
    # Count active features
    active_features = sum(feature_mask)
    x = torch.zeros((num_nodes, active_features))
    
    # Get essential metrics that are always needed
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1.0
    
    # Get the specific target function to determine what we need to calculate
    target_metric = RESIDUAL_G_FEATURES[target_idx].replace("GMinus_", "")
    
    # Determine which metrics to calculate based on the features needed
    calculate_clustering = feature_mask[1] or "Clustering" in target_metric
    calculate_pagerank = feature_mask[5]
    calculate_betweenness = feature_mask[3]
    calculate_eigenvector = feature_mask[8]
    calculate_closeness = feature_mask[4]
    calculate_coreness = feature_mask[6]
    calculate_local_efficiency = feature_mask[7]
    calculate_neighbors = feature_mask[2]
    calculate_local_density = feature_mask[9]
    
    # Compute only what's needed
    clustering = fast_clustering_coefficient(G) if calculate_clustering else {node: 0.0 for node in G.nodes()}
    betweenness = nx.betweenness_centrality(G, k=min(20, num_nodes // 10)) if calculate_betweenness else {node: 0.0 for node in G.nodes()}
    closeness = nx.closeness_centrality(G) if calculate_closeness else {node: 0.0 for node in G.nodes()}
    pagerank = nx.pagerank(G, max_iter=30, tol=1e-3) if calculate_pagerank else {node: 0.0 for node in G.nodes()}
    
    # Coreness metrics
    if calculate_coreness:
        core_numbers = nx.core_number(G)
        max_core = max(core_numbers.values()) if core_numbers else 1.0
    else:
        core_numbers = {node: 0 for node in G.nodes()}
        max_core = 1.0
    
    # Eigenvector centrality
    if calculate_eigenvector:
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=50, tol=1e-3)
        except:
            eigenvector = {n: 0.1 + 0.5 * (degrees[n] / max_degree) for n in G.nodes()}
    else:
        eigenvector = {node: 0.0 for node in G.nodes()}
    
    # Pre-compute neighbor averages for efficiency
    neighbor_degrees = {}
    if calculate_neighbors:
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            neighbor_degrees[node] = (
                sum(degrees[n] for n in neighbors) / len(neighbors) 
                if neighbors else 0.1
            )
    else:
        neighbor_degrees = {node: 0.0 for node in G.nodes()}
    
    # Calculate local density for each node
    local_density = {}
    if calculate_local_density:
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if len(neighbors) < 2:
                local_density[node] = 0.0
                continue
            subgraph = G.subgraph(neighbors + [node])
            local_density[node] = nx.density(subgraph)
    else:
        local_density = {node: 0.0 for node in G.nodes()}
    
    # Map original feature indices to compressed indices
    feature_map = {}
    new_idx = 0
    for old_idx, include in enumerate(feature_mask):
        if include:
            feature_map[old_idx] = new_idx
            new_idx += 1
    
    # Assign features to nodes
    for node in G.nodes():
        # Only assign features that are in the mask
        if feature_mask[0]:  # Degree
            x[node, feature_map[0]] = degrees[node] / max_degree if max_degree > 0 else 0
        
        if feature_mask[1]:  # Clustering
            x[node, feature_map[1]] = clustering.get(node, 0.0)
        
        if feature_mask[2]:  # NeighborDeg
            x[node, feature_map[2]] = neighbor_degrees[node] / max_degree if max_degree > 0 else 0
        
        if feature_mask[3]:  # Betweenness
            x[node, feature_map[3]] = betweenness.get(node, 0.0) * 10
        
        if feature_mask[4]:  # Closeness
            x[node, feature_map[4]] = closeness.get(node, 0.0) * 10
        
        if feature_mask[5]:  # PageRank
            x[node, feature_map[5]] = pagerank.get(node, 0.0) * 10
        
        if feature_mask[6]:  # CoreNumber
            x[node, feature_map[6]] = core_numbers.get(node, 0) / max_core if max_core > 0 else 0
        
        if feature_mask[7]:  # LocalEff
            x[node, feature_map[7]] = fast_local_efficiency(G, node) if calculate_local_efficiency else 0.0
        
        if feature_mask[8]:  # Eigenvector
            x[node, feature_map[8]] = eigenvector.get(node, 0.0) * 10
        
        if feature_mask[9]:  # LocalDensity
            x[node, feature_map[9]] = local_density[node]
        
        if feature_mask[10]:  # IsSelected
            x[node, feature_map[10]] = 1.0 if node in selected_nodes else 0.0
    
    # Calculate target value
    # FIXED: Create G_prime properly by first making a copy and then removing nodes
    G_prime = G.copy()
    G_prime.remove_nodes_from(selected_nodes)
    target_value = calculate_target_property(G, G_prime, target_metric)
    
    edge_array = np.array(list(G.edges())).T
    edge_index = torch.tensor(edge_array, dtype=torch.long)
    
    # Create data object
    data = Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor(target_value, dtype=torch.float32),
        selected_nodes=torch.tensor(selected_nodes)
    )
    
    return data

def normalized_r2_score(y_true, y_pred):
    """Calculate R2 score with better handling of small variance data"""
    # Convert to numpy if they are tensors
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
        
    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Calculate means and sums of squares
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # Print diagnostic information
    print(f"Target mean: {y_mean:.8e}, Target variance: {np.var(y_true):.8e}")
    print(f"SS_tot: {ss_tot:.8e}, SS_res: {ss_res:.8e}")
    
    # Use a very small threshold for comparison
    if ss_tot < 1e-10:
        # Calculate mean absolute error
        mae = np.mean(np.abs(y_true - y_pred))
        
        # For near-constant targets, normalize by the mean value (if not near zero)
        if abs(y_mean) > 1e-10:
            normalized_error = mae / abs(y_mean)
            print(f"Target has near-zero variance, using normalized MAE: {normalized_error:.8e}")
        else:
            # If mean is also near zero, use a small constant for normalization
            normalized_error = mae / 1e-8
            print(f"Target has near-zero mean and variance, using absolute error: {mae:.8e}")
        
        # Convert to a pseudo-R2 score between 0 and 1
        r2 = max(0.0, min(1.0, 1.0 - normalized_error))
        return r2
    else:
        r2 = 1 - (ss_res / ss_tot)
        # Clamp to reasonable range
        r2 = max(min(r2, 1.0), 0.0)
        return r2
    
def calculate_target_property(G, G_prime, target_metric):
    """
    Calculate the specified metric on the modified graph G_prime.
    
    Args:
        G: Original graph
        G_prime: Graph with nodes removed
        target_metric: Name of the metric to calculate (without "GMinus_" prefix)
    
    Returns:
        float: Calculated metric value
    """
    # Handle empty graph
    if len(G_prime) == 0:
        return 0.0
        
    # Calculate metric based on the name
    if target_metric == "Density":
        return nx.density(G_prime)
    
    elif target_metric == "AvgClustering":
        try:
            clustering = nx.clustering(G_prime)
            return sum(clustering.values()) / len(G_prime) if len(G_prime) > 0 else 0
        except:
            return 0.0
    
    elif target_metric == "AvgPathLength":
        try:
            # For disconnected graphs, calculate per component and weight by component size
            if not nx.is_connected(G_prime):
                components = list(nx.connected_components(G_prime))
                total_path_length = 0.0
                total_pairs = 0
                
                for component in components:
                    if len(component) > 1:  # Need at least 2 nodes for a path
                        subgraph = G_prime.subgraph(component)
                        avg_path = nx.average_shortest_path_length(subgraph)
                        # Weight by number of node pairs in component
                        n_pairs = len(component) * (len(component) - 1) / 2
                        total_path_length += avg_path * n_pairs
                        total_pairs += n_pairs
                
                return total_path_length / total_pairs if total_pairs > 0 else 0.0
            else:
                return nx.average_shortest_path_length(G_prime)
        except:
            return 0.0
    
    elif target_metric == "DegreeAssortativity":
        try:
            # Check if graph has at least one edge and more than one degree value
            if G_prime.number_of_edges() > 0:
                degrees = dict(G_prime.degree())
                unique_degrees = set(degrees.values())
                if len(unique_degrees) > 1:
                    return nx.degree_assortativity_coefficient(G_prime)
            return 0.0
        except:
            return 0.0
    
    elif target_metric == "Transitivity":
        try:
            return nx.transitivity(G_prime)
        except:
            return 0.0
    
    elif target_metric == "ConnectedComponents":
        # Return the raw number of connected components
        return len(list(nx.connected_components(G_prime)))
    
    elif target_metric == "MaxDegree":
        if G_prime.number_of_nodes() == 0:
            return 0
        degrees = dict(G_prime.degree())
        return max(degrees.values()) if degrees else 0
    
    elif target_metric == "MinDegree":
        if G_prime.number_of_nodes() == 0:
            return 0
        degrees = dict(G_prime.degree())
        return min(degrees.values()) if degrees else 0
    
    elif target_metric == "AvgDegree":
        if G_prime.number_of_nodes() == 0:
            return 0
        total_edges = G_prime.number_of_edges()
        return 2 * total_edges / len(G_prime) if len(G_prime) > 0 else 0
    
    elif target_metric == "GlobalEfficiency":
        try:
            return nx.global_efficiency(G_prime)
        except:
            return 0.0
    
    else:
        raise ValueError(f"Unknown target metric: {target_metric}")