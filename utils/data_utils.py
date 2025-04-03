import torch
from torch_geometric.data import Data
from tabulate import tabulate
import networkx as nx  # Added missing import
import numpy as np

from utils.config import FEATURE_NAMES, GRAPH_METRICS, NODE_METRICS

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
        
def compute_features(G, nodes):
    features = []

    # Create subgraph G' from selected nodes
    subgraph = G.subgraph(nodes)
    
    # Create G/G'
    residual_graph = G.copy()
    residual_graph.remove_nodes_from(nodes)
    
    def compute_graph_features(graph): # Used to get P(G) and P(G/G')
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
    
    # Compute P(G/G')
    g_minus_features = compute_graph_features(residual_graph)
    features.extend(g_minus_features)
    
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
    # Compute features
    all_features = compute_features(G, selected_nodes)
    
    if not isinstance(all_features, torch.Tensor):
        all_features = torch.tensor(all_features, dtype=torch.float)
    
    num_nodes = len(G)
    num_features = len(FEATURE_NAMES) - 1  # Subtract 1 since we're excluding property p
    x = torch.zeros((num_nodes, num_features))
    
    # Split features into their components
    num_graph_metrics = len(GRAPH_METRICS)
    g_features = all_features[:num_graph_metrics]
    g_minus_features = all_features[num_graph_metrics:2*num_graph_metrics]
    node_features = all_features[2*num_graph_metrics:]
    
    # Create feature matrix
    current_idx = 0
    
    # Add G features (repeated for each node with small variations)
    for i in range(num_graph_metrics):
        if current_idx != target_idx:
            base_value = g_features[i].clone()
            variations = torch.randn(num_nodes) * 0.05  # 5% variation
            x[:, current_idx] = torch.clamp(base_value + variations, min=0.0, max=1.0)
            current_idx += 1
            
    # Add G/G' features
    for i in range(num_graph_metrics):
        if current_idx != target_idx:
            base_value = g_minus_features[i].clone()
            variations = torch.randn(num_nodes) * 0.05  # 5% variation
            x[:, current_idx] = torch.clamp(base_value + variations, min=0.0, max=1.0)
            current_idx += 1
    
    # Add node-level features for selected nodes, zeros for others
    node_feature_size = len(NODE_METRICS)
    for i, node in enumerate(selected_nodes):
        node_start = i * node_feature_size
        node_end = (i + 1) * node_feature_size
        node_values = node_features[node_start:node_end]
        
        # Skip the target feature when assigning node values
        if target_idx >= current_idx and target_idx < current_idx + node_feature_size:
            # Split the node values around the target feature
            target_offset = target_idx - current_idx
            x[node, current_idx:target_idx] = node_values[:target_offset]
            x[node, target_idx:current_idx+node_feature_size-1] = node_values[target_offset+1:]
        else:
            x[node, current_idx:current_idx+node_feature_size] = node_values
            
        current_idx += node_feature_size - 1  # Subtract 1 to account for removed target feature
    
    # Store target value (get it from the appropriate location in node_features)
    if target_idx < num_graph_metrics:  # Target is a G feature
        target_value = g_features[target_idx]
    elif target_idx < 2 * num_graph_metrics:  # Target is a G/G' feature
        target_value = g_minus_features[target_idx - num_graph_metrics]
    else:  # Target is a node feature
        node_idx = (target_idx - 2 * num_graph_metrics) // node_feature_size
        feature_idx = (target_idx - 2 * num_graph_metrics) % node_feature_size
        target_value = node_features[node_idx * node_feature_size + feature_idx]
    
    # Prepare edge index
    edge_list = list(G.edges())
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Create data object
    data = Data(
        x=x,
        edge_index=edge_index,
        y=target_value,
        original_features=all_features,
        selected_nodes=torch.tensor(selected_nodes)
    )

    print("\nData Processing Debug:")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of features: {num_features}")
    print(f"Feature tensor shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Target value: {target_value}")
    print(f"Selected nodes: {selected_nodes}")
    
    return data

