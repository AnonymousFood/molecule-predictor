import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import random

from utils.fast_features import fast_clustering_coefficient, fast_global_efficiency, fast_local_efficiency, fast_shortest_path_length
from utils.config import FEATURE_NAMES, RESIDUAL_G_FEATURES

def find_connected_subgraph(G, size=4):
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
        
        # Find largest connected component first - TODO make this fit problem description
        largest_cc = max(nx.connected_components(G), key=len)
        if len(largest_cc) >= 4:
            G = nx.convert_node_labels_to_integers(G)
            connected_nodes = find_connected_subgraph(G, size=4)
            if connected_nodes is not None:
                return G, connected_nodes
    
    raise RuntimeError("Could not generate suitable graph after max attempts")

def process_graph_data(G, selected_nodes, target_idx):
    # Relabel and map nodes
    G = nx.convert_node_labels_to_integers(G)
    node_mapping = {old: new for new, old in enumerate(G.nodes())}
    selected_nodes = [node_mapping[node] for node in selected_nodes]
    
    num_nodes = len(G)
    x = torch.zeros((num_nodes, len(FEATURE_NAMES)))
    
    # Can skip calculations not needed for the target feature - EXPERIMENTAL

    # Get essential metrics that are always needed
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1.0
    
    # Compute graph density - needed for most features
    # graph_density = nx.density(G)
    
    # Get the specific target function to determine what we need to calculate
    target_metric = RESIDUAL_G_FEATURES[target_idx].replace("GMinus_", "")
    
    # Determine which metrics to calculate based on the target
    calculate_clustering = True
    calculate_pagerank = True
    calculate_betweenness = True
    calculate_eigenvector = True
    calculate_closeness = True
    calculate_coreness = True
    calculate_local_efficiency = True
    
    # Only calculate if needed for node features or target metric - EXPERIMENTAL
    if "Clustering" in target_metric or "Transitivity" in target_metric or any(f in target_metric for f in ["Assortativity", "Path"]):
        calculate_clustering = True
    if "PageRank" in target_metric:
        calculate_pagerank = True
    if "Betweenness" in target_metric:
        calculate_betweenness = True
    if "Eigenvector" in target_metric:
        calculate_eigenvector = True
    if "Closeness" in target_metric or "Path" in target_metric:
        calculate_closeness = True
    if "Core" in target_metric:
        calculate_coreness = True
    if "Efficiency" in target_metric:
        calculate_local_efficiency = True
    
    # Compute only what's needed
    clustering = fast_clustering_coefficient(G) if calculate_clustering else {node: 0.0 for node in G.nodes()}
    
    # Efficient k-sampling for betweenness centrality
    betweenness = nx.betweenness_centrality(G, k=min(20, num_nodes // 10)) if calculate_betweenness else {node: 0.0 for node in G.nodes()}
    
    # Closeness optimization
    if calculate_closeness:
        # Use sampling for large graphs
        if num_nodes > 100:
            closeness_nodes = random.sample(list(G.nodes()), min(50, num_nodes))
            closeness = {}
            for node in G.nodes():
                if node in closeness_nodes:
                    closeness[node] = nx.closeness_centrality(G, node)
                else:
                    # Approximate using degree (much faster)
                    closeness[node] = 0.1 + 0.5 * (degrees[node] / max_degree)
        else:
            closeness = nx.closeness_centrality(G)
    else:
        closeness = {node: 0.0 for node in G.nodes()}
    
    # PageRank with fewer iterations
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
            # Fallback to degree-based approximation
            eigenvector = {n: 0.1 + 0.5 * (degrees[n] / max_degree) for n in G.nodes()}
    else:
        eigenvector = {node: 0.0 for node in G.nodes()}
    
    # Average clustering
    if "AvgClustering" in target_metric:
        # Use sampling for average clustering
        if num_nodes > 100:
            sample_nodes = random.sample(list(G.nodes()), min(100, num_nodes))
            avg_clustering = sum(clustering[n] for n in sample_nodes) / len(sample_nodes)
        else:
            avg_clustering = sum(clustering.values()) / len(clustering)
    else:
        avg_clustering = 0.0
    
    # Pre-compute neighbor averages for efficiency
    neighbor_degrees = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        neighbor_degrees[node] = (
            sum(degrees[n] for n in neighbors) / len(neighbors) 
            if neighbors else 0.1
        )
    
    # Calculate local density for each node
    local_density = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2:
            local_density[node] = 0.0
            continue
        # Create subgraph of node and its neighbors
        subgraph = G.subgraph(neighbors + [node])
        local_density[node] = nx.density(subgraph)
    
    # Assign features to nodes
    for node in G.nodes():
        # Set features for this node
        x[node, 0] = degrees[node] / max_degree if max_degree > 0 else 0
        x[node, 1] = clustering.get(node, 0.0)
        x[node, 2] = neighbor_degrees[node] / max_degree if max_degree > 0 else 0
        x[node, 3] = betweenness.get(node, 0.0) * 10
        x[node, 4] = closeness.get(node, 0.0) * 10
        x[node, 5] = pagerank.get(node, 0.0) * 10
        x[node, 6] = core_numbers.get(node, 0) / max_core if max_core > 0 else 0
        
        # Use approximation for local efficiency
        x[node, 7] = fast_local_efficiency(G, node) if calculate_local_efficiency else 0.0
        x[node, 8] = eigenvector.get(node, 0.0) * 10
        
        # Replace graph density with local density
        x[node, 9] = local_density[node]

        x[node, 10] = 1.0 if node in selected_nodes else 0.0  # Move IsSelected to index 10
    
    # Compute target value - G/G' metrics
    G_minus = G.copy()
    G_minus.remove_nodes_from(selected_nodes)
    
    # Calculate target value based on target idx
    target_value = 0.0
    
    # Call only the specific target function needed
    if target_idx == 0:  # GMinus_Density
        target_value = nx.density(G_minus)
    elif target_idx == 1:  # GMinus_AvgClustering
        target_value = sum(fast_clustering_coefficient(G_minus).values()) / len(G_minus) if len(G_minus) > 0 else 0
    elif target_idx == 2:  # GMinus_AvgPathLength
        target_value = fast_shortest_path_length(G_minus) if len(G_minus) > 0 else 0.1
    elif target_idx == 3:  # GMinus_DegreeAssortativity
        target_value = nx.degree_assortativity_coefficient(G_minus)
    elif target_idx == 4:  # GMinus_Transitivity
        target_value = nx.transitivity(G_minus)
    elif target_idx == 5:  # GMinus_ConnectedComponents
        target_value = len(list(nx.connected_components(G_minus))) / len(G_minus) if len(G_minus) > 0 else 0
    elif target_idx == 6:  # GMinus_MaxDegree
        target_value = max(dict(G_minus.degree()).values()) if len(G_minus) > 0 else 0
    elif target_idx == 7:  # GMinus_MinDegree
        target_value = min(dict(G_minus.degree()).values()) if len(G_minus) > 0 else 0
    elif target_idx == 8:  # GMinus_AvgDegree
        target_value = sum(dict(G_minus.degree()).values()) / len(G_minus) if len(G_minus) > 0 else 0
    elif target_idx == 9:  # GMinus_GlobalEfficiency
        target_value = fast_global_efficiency(G_minus)
    
    edge_array = np.array(list(G.edges())).T
    edge_index = torch.tensor(edge_array, dtype=torch.long)
    
    # Create data object
    data = Data(
        x=x, # Features
        edge_index=edge_index,
        y=torch.tensor(target_value, dtype=torch.float32), # Contains target value of train and test data
        selected_nodes=torch.tensor(selected_nodes) # Which nodes are removed?
    )
    
    return data