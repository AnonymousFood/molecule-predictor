MODEL_CONFIG = {
    'hidden_dim': 128,
    'num_layers': 3,
    'dropout_rate': 0.3,
    'batch_norm': True,
    'layer_norm': False,
    'residual': True
}

TRAINING_CONFIG = {
    'num_epochs': 100,
    'learning_rate': 0.0005,
    'weight_decay': 1e-4,
    'early_stopping': {
        'enabled': True,
        'patience': 25,
        'min_delta': 1e-4
    },
    'optimizer': {
        'type': 'AdamW',
        'beta1': 0.9,
        'beta2': 0.999,
        'eps': 1e-8
    },
    'clip_grad_norm': 1.0,
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
METRICS = [
    "Degree",
    "Clustering",
    "NeighborDeg", 
    "Betweenness",
    "Closeness",
    "PageRank",
    "CoreNumber",
    "LocalEff",
    "Eigenvector",
    "LocalDensity",
    "IsSelected"
]

# Isn't used right now for features, only label
NODE_METRICS = METRICS[:10]  # First 10 are node metrics now
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

# Used for indexing targets in the residual graph
RESIDUAL_G_FEATURES = [f"GMinus_{metric}" for metric in GRAPH_METRICS]

# Number of nodes in the subgraph G'
NUM_NODES = 10

# Create feature names - now all features are node features 
# (1 for node in G, plus 4 for nodes in G')
FEATURE_NAMES = [f"Node_Metric_{metric}" for metric in METRICS]