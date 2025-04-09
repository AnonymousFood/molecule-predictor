import torch
from torch_geometric.data import Data
from tabulate import tabulate
import networkx as nx
import numpy as np

from utils.config import FEATURE_NAMES, GRAPH_METRICS, NODE_METRICS, RESIDUAL_G_FEATURES

def find_connected_subgraph(G, size=4):
    """Find a connected subgraph of specified size."""
    # First relabel nodes to ensure consecutive integers
    G = nx.convert_node_labels_to_integers(G)
    
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
        G = nx.convert_node_labels_to_integers(G)  # Ensure consecutive integers
        connected_nodes = find_connected_subgraph(G, size=4)
        if connected_nodes is not None:
            return G, connected_nodes
        
def compute_features(G, nodes):
    features = []

    # Create subgraph G' from selected nodes
    subgraph = G.subgraph(nodes)
    
    def compute_graph_features(graph): # Used to get P(G)
        return [
            nx.density(graph),
            nx.average_clustering(graph),
            nx.average_shortest_path_length(graph) / len(graph) if nx.is_connected(graph) else 0.1,
            nx.degree_assortativity_coefficient(graph),
            nx.transitivity(graph),
            len(list(nx.connected_components(graph))) / len(graph),
            max(dict(graph.degree()).values()),
            min(dict(graph.degree()).values()),
            np.mean(list(dict(graph.degree()).values())),
            nx.global_efficiency(graph)
        ]
    
    # Compute P(G)
    g_features = compute_graph_features(G)
    features.extend(g_features)
    
    # Compute P(G')
    try:
        betweenness = nx.betweenness_centrality(subgraph)
        closeness = nx.closeness_centrality(subgraph)
        pagerank = nx.pagerank(subgraph)
        core_numbers = nx.core_number(subgraph)
        degrees = dict(subgraph.degree())
        clustering = nx.clustering(subgraph)
        eigenvector = nx.eigenvector_centrality_numpy(subgraph)
        
        for node in nodes:
            neighbors = list(subgraph.neighbors(node))
            avg_neighbor_degree = (
                np.mean([degrees[n] for n in neighbors]) 
                if neighbors else 0.1
            )
            
            node_features = [
                degrees[node],
                clustering[node],
                avg_neighbor_degree,
                betweenness[node],
                closeness[node],
                pagerank[node],
                core_numbers[node],
                compute_local_efficiency(subgraph, node),
                eigenvector[node]
            ]
            features.extend(node_features)
            
    except Exception as e:
        print(f"Error computing node metrics for subgraph: {str(e)}")
        raise

    return torch.tensor(features, dtype=torch.float32)

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
    # First relabel nodes to ensure consecutive integers from 0 to n-1
    G = nx.convert_node_labels_to_integers(G)
    
    # Map selected nodes to new indices
    node_mapping = {old: new for new, old in enumerate(G.nodes())}
    selected_nodes = [node_mapping[node] for node in selected_nodes]
    
    features = compute_features(G, selected_nodes)
    
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float)
    
    num_nodes = len(G)
    num_features = len(FEATURE_NAMES)
    x = torch.zeros((num_nodes, num_features))
    
    # Split features into their components
    num_graph_metrics = len(GRAPH_METRICS)
    g_features = features[:num_graph_metrics]
    g_prime_features = features[num_graph_metrics:]  # Remove the upper bound
    
    # Create feature matrix
    current_idx = 0
    
    # Add P(G) (repeated for each node with small variations to avoid overfitting)
    for i in range(num_graph_metrics):
        base_value = g_features[i].clone()
        variations = torch.randn(num_nodes) * 0.05  # 5% variation
        x[:, current_idx] = torch.clamp(base_value + variations, min=0.0, max=1.0)
        current_idx += 1
    
    # Add P(G') node-level features for selected nodes, zeros for others
    node_feature_size = len(NODE_METRICS)
    for i, node in enumerate(selected_nodes):
        start_idx = i * node_feature_size
        end_idx = (i + 1) * node_feature_size
        if start_idx + node_feature_size <= len(g_prime_features):  # Add bounds check
            node_values = g_prime_features[start_idx:end_idx]
            if current_idx + node_feature_size <= x.shape[1]:  # Add bounds check
                x[node, current_idx:current_idx+node_feature_size] = node_values
    
    current_idx += node_feature_size
    
    # Store target value
    features = compute_features(G, selected_nodes)
    target_value = features[target_idx]
    
    # Prepare edge index with remapped nodes
    edge_list = list(G.edges())
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Create data object
    data = Data(
        x=x,
        edge_index=edge_index,
        y=target_value,
        original_features=features,
        selected_nodes=torch.tensor(selected_nodes)
    )

    print("\nData Processing Debug:")
    print(f"Number of connected nodes: {num_nodes}")
    print(f"Number of features: {num_features}")
    print(f"Feature tensor shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Target value: {target_value}")
    print(f"Selected nodes: {selected_nodes}")
    
    return data