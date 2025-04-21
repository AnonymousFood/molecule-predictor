TRAINING_CONFIG = {
    'num_epochs': 500,
    'learning_rate': 0.0005,
    'weight_decay': 1e-4,
    'early_stopping': {
        'enabled': True,
        'patience': 25,
        'min_delta': 1e-4
    },
    'clip_grad_norm': 1.0,
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

# Used for P(G/G') Labels currently
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