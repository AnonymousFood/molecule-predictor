import torch
from torch_geometric.data import Data
from tabulate import tabulate
import networkx as nx  # Added missing import
import numpy as np

from utils.config import FEATURE_NAMES

def find_connected_subgraph(G, size=4):
    """Find a connected subgraph of specified size."""
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        if len(subgraph) >= size:
            start_node = np.random.choice(list(subgraph.nodes()))
            nodes = list(nx.bfs_tree(subgraph, start_node))[:size]
            return nodes
    return None

def generate_graph(num_nodes=100, edge_prob=0.05):
    """Generate a random graph ensuring it has at least one connected component of a specified size."""
    while True:
        G = nx.erdos_renyi_graph(n=num_nodes, p=edge_prob)
        connected_nodes = find_connected_subgraph(G, size=4)
        if connected_nodes is not None:
            return G, connected_nodes
        
# Feature Computation
def compute_features(G, nodes):
    
    features = []
    n_nodes = len(G)

    def normalize_metric_dict(metric_dict, min_val=0.1):
        """Normalize dictionary values to [min_val, 1] range."""
        values = np.array(list(metric_dict.values()))
        min_metric = np.min(values)
        max_metric = np.max(values)
        
        if max_metric == min_metric:
            return {k: min_val for k in metric_dict.keys()}
            
        normalized = {
            k: min_val + (v - min_metric) * (1 - min_val) / (max_metric - min_metric)
            for k, v in metric_dict.items()
        }
        return normalized
    
    try:
        # Compute and normalize basic centrality metrics
        betweenness = normalize_metric_dict(nx.betweenness_centrality(G))
        closeness = normalize_metric_dict(nx.closeness_centrality(G))
        pagerank = normalize_metric_dict(nx.pagerank(G))
        core_numbers = normalize_metric_dict(nx.core_number(G))
        degrees = normalize_metric_dict(dict(G.degree()))
        
        # Enhanced clustering computation and normalization
        clustering = nx.clustering(G)
        clustering = normalize_metric_dict(clustering)
        
        # Handle eigenvector centrality for largest component
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        eigenvector = nx.eigenvector_centrality_numpy(subgraph)
        min_eigen = min(eigenvector.values())
        eigenvector.update({n: min_eigen for n in G.nodes() if n not in eigenvector})
        eigenvector = normalize_metric_dict(eigenvector)
        
    except Exception as e:
        print(f"Error computing centrality metrics: {str(e)}")
        raise
    
    # Process each node
    for node in nodes:
        # Calculate neighbor-based metrics
        neighbors = list(G.neighbors(node))
        avg_neighbor_degree = (
            np.mean([degrees[n] for n in neighbors]) 
            if neighbors else 0.1  # Use minimum value if no neighbors
        )
        
        # Compile node features
        node_features = [
            degrees[node],
            clustering[node],
            avg_neighbor_degree,
            betweenness[node],
            closeness[node],
            pagerank[node],
            core_numbers[node],
            compute_local_efficiency(G, node),
            eigenvector[node]
        ]
        features.extend(node_features)
    
    # Compute and normalize global features
    global_features = [
        nx.density(G),
        nx.average_clustering(G),
        nx.average_shortest_path_length(G) / n_nodes if nx.is_connected(G) else 0.1,
        nx.degree_assortativity_coefficient(G),
        nx.transitivity(G),
        len(list(nx.connected_components(G))) / n_nodes,
        max(degrees.values()),
        min(degrees.values()),
        np.mean(list(degrees.values())),
        nx.global_efficiency(G)
    ]
    
    # Normalize global features
    global_features = normalize_metric_dict(
        {str(i): v for i, v in enumerate(global_features)}
    ).values()
    
    features.extend(global_features)
    return torch.tensor(features, dtype=torch.float32)

def prepare_node_features(G):
    """Prepare node features including removal flag"""
    num_nodes = G.number_of_nodes()
    # Basic features for each node (5 base features + 1 removal flag)
    features = torch.zeros(num_nodes, 6)
    
    for i in range(num_nodes):
        features[i] = torch.tensor([
            G.degree[i],
            nx.clustering(G, i),
            np.mean([G.degree[n] for n in G.neighbors(i)]) if list(G.neighbors(i)) else 0,
            list(nx.betweenness_centrality(G).values())[i],
            list(nx.closeness_centrality(G).values())[i],
            0  # Removal flag, will be set later
        ])
    return features

def compute_local_efficiency(G, node):
    """
    Compute local efficiency for a node in G'
    """
    neighbors = list(G.neighbors(node))
    if len(neighbors) < 2:
        return 0.0
    
    # Create subgraph of node's neighbors
    subgraph = G.subgraph(neighbors)
    if len(subgraph) < 2:
        return 0.0
    
    try:
        # Calculate average shortest path length in subgraph
        avg_path_length = nx.average_shortest_path_length(subgraph)
        return 1.0 / avg_path_length if avg_path_length > 0 else 0.0
    except nx.NetworkXError:
        # Handle disconnected graphs
        return 0.0
    
def process_graph_data(G, selected_nodes, target_idx):
    # Compute features
    all_features = compute_features(G, selected_nodes)
    
    if not isinstance(all_features, torch.Tensor):
        all_features = torch.tensor(all_features, dtype=torch.float)
    
    num_nodes = len(G)
    num_features = len(FEATURE_NAMES) - 1  # Subtract 1 since we're excluding target
    x = torch.zeros((num_nodes, num_features))
    
    # Individual node-level features
    current_idx = 0
    for i in range(len(FEATURE_NAMES)):
        if i != target_idx:  # Skip target feature
            base_value = all_features[i].clone()
            # Create node-specific variations
            variations = torch.randn(num_nodes) * 0.1  # 10% variation
            node_values = base_value + variations
            # Ensure values stay in reasonable range
            node_values = torch.clamp(node_values, min=0.0, max=1.0)
            x[:, current_idx] = node_values
            current_idx += 1
    
    # Store target value separately
    target_value = all_features[target_idx]
    
    # Prepare edge index
    edge_list = list(G.edges())
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Create data object with target value
    data = Data(
        x=x,
        edge_index=edge_index,
        y=target_value,  # Add target value as y
        original_features=all_features
    )
    
    print("\nData Processing Debug:")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of input features: {num_features}")
    print(f"Feature tensor shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Target value: {target_value}")
    
    return data

def compute_features(G, nodes):
    
    features = []
    n_nodes = len(G)

    def normalize_metric_dict(metric_dict, min_val=0.1):
        """Normalize dictionary values to [min_val, 1] range."""
        values = np.array(list(metric_dict.values()))
        min_metric = np.min(values)
        max_metric = np.max(values)
        
        if max_metric == min_metric:
            return {k: min_val for k in metric_dict.keys()}
            
        normalized = {
            k: min_val + (v - min_metric) * (1 - min_val) / (max_metric - min_metric)
            for k, v in metric_dict.items()
        }
        return normalized
    
    try:
        # Compute and normalize basic centrality metrics
        betweenness = normalize_metric_dict(nx.betweenness_centrality(G))
        closeness = normalize_metric_dict(nx.closeness_centrality(G))
        pagerank = normalize_metric_dict(nx.pagerank(G))
        core_numbers = normalize_metric_dict(nx.core_number(G))
        degrees = normalize_metric_dict(dict(G.degree()))
        
        # Custering computation and normalization
        clustering = nx.clustering(G)
        clustering = normalize_metric_dict(clustering)
        
        # Handle eigenvector centrality for largest component
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        eigenvector = nx.eigenvector_centrality_numpy(subgraph)
        min_eigen = min(eigenvector.values())
        eigenvector.update({n: min_eigen for n in G.nodes() if n not in eigenvector})
        eigenvector = normalize_metric_dict(eigenvector)
        
    except Exception as e:
        print(f"Error computing centrality metrics: {str(e)}")
        raise
    
    # Process each node
    for node in nodes:
        # Calculate neighbor-based metrics
        neighbors = list(G.neighbors(node))
        avg_neighbor_degree = (
            np.mean([degrees[n] for n in neighbors]) 
            if neighbors else 0.1  # Use minimum value if no neighbors
        )
        
        # Compile node features
        node_features = [
            degrees[node],
            clustering[node],
            avg_neighbor_degree,
            betweenness[node],
            closeness[node],
            pagerank[node],
            core_numbers[node],
            compute_local_efficiency(G, node),
            eigenvector[node]
        ]
        features.extend(node_features)
    
    # Compute and normalize global features
    global_features = [
        nx.density(G),
        nx.average_clustering(G),
        nx.average_shortest_path_length(G) / n_nodes if nx.is_connected(G) else 0.1,
        nx.degree_assortativity_coefficient(G),
        nx.transitivity(G),
        len(list(nx.connected_components(G))) / n_nodes,
        max(degrees.values()),
        min(degrees.values()),
        np.mean(list(degrees.values())),
        nx.global_efficiency(G)
    ]
    
    # Normalize global features
    global_features = normalize_metric_dict(
        {str(i): v for i, v in enumerate(global_features)}
    ).values()
    
    features.extend(global_features)
    return torch.tensor(features, dtype=torch.float32)

