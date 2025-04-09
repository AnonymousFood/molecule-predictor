MODEL_CONFIG = {
    'hidden_dim': 32,
    'num_layers': 2,
    'dropout_rate': 0.2,
    'batch_norm': True,
    'layer_norm': False,
    'residual': True
}

TRAINING_CONFIG = {
    'num_epochs': 100,
    'learning_rate': 0.0005,
    'weight_decay': 1e-4,
    'early_stopping': {
        'enabled': False,
        'patience': 15,
        'min_delta': 1e-4
    },
    'optimizer': {
        'type': 'AdamW',
        'beta1': 0.9,
        'beta2': 0.999,
        'eps': 1e-8
    },
}

# Currently unused
ANALYSIS_CONFIG = {
    'perturbation': {
        'num_trials': 10,
        'scales': [0.05, 0.1, 0.2, 0.5],
        'feature_subset': None
    },
    'hybrid': {
        'num_trials': 5,
        'mask_ratio': 0.5
    }
}

# Feature configuration
NODE_METRICS = [
    "Degree",
    "Clustering",
    "NeighborDeg",
    "Betweenness",
    "Closeness",
    "PageRank",
    "CoreNumber",
    "LocalEff",
    "Eigenvector"
]

GRAPH_METRICS = [
    "Density",
    "AvgClustering",
    "AvgPathLength",
    "DegreeAssortativity",
    "Transitivity",
    "ConnectedComponents",
    "MaxDegree",
    "MinDegree",
    "AvgDegree",
    "GlobalEfficiency"
]

# Number of nodes in the subgraph G'
NUM_NODES = 4

# Create feature names for:
# 1. Graph-level features for G
G_FEATURES = [f"G_{metric}" for metric in GRAPH_METRICS]

# 2. Graph-level features for G/G'
RESIDUAL_G_FEATURES = [f"GMinus_{metric}" for metric in GRAPH_METRICS]

# 3. Node-level features for G'
SUBGRAPH_FEATURES = [
    f"Node{i+1}_{metric}" 
    for i in range(NUM_NODES)
    for metric in NODE_METRICS
]

# Combine all features
FEATURE_NAMES = G_FEATURES + SUBGRAPH_FEATURES