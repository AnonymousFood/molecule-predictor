MODEL_CONFIG = {
    'hidden_dim': 32,
    'num_layers': 2,
    'dropout_rate': 0.2,
    'batch_norm': True,
    'layer_norm': False,
    'residual': True
}

TRAINING_CONFIG = {
    'num_epochs': 1000,
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

GLOBAL_METRICS = [
    "GraphDensity",
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

# Feature names for each node in G'
NUM_NODES = 4
NODE_PREFIX = "Node"
SEPARATOR = "_"

FEATURE_NAMES = [
    f"{NODE_PREFIX}{i+1}{SEPARATOR}{metric}" 
    for i in range(NUM_NODES)
    for metric in NODE_METRICS
] + GLOBAL_METRICS