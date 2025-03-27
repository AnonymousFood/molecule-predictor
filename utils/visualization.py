import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple

class VisualizationManager:
    def __init__(self):
        self.style_config = {
            'figsize': (10, 6),
            'color_palette': ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f'],
            'font_size': 12
        }

    def plot_training_progress(self, 
                             train_losses: List[float], 
                             val_losses: List[float] = None,
                             title: str = "Training Progress") -> None:
        """Plot training and test losses over epochs."""
        plt.figure(figsize=self.style_config['figsize'])
        plt.plot(train_losses, label='Training Loss', 
                 color=self.style_config['color_palette'][0])
        plt.plot(val_losses, label='Validation Loss', 
                    color=self.style_config['color_palette'][1])
        
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_method_comparison(self, 
                             results: Dict[str, List[float]], 
                             method_names: List[str],
                             title: str = "Method Comparison") -> None:
        """Plot comparison between different methods."""
        plt.figure(figsize=self.style_config['figsize'])
        
        for idx, (method, values) in enumerate(results.items()):
            plt.boxplot(values, positions=[idx+1], labels=[method],
                       patch_artist=True,
                       boxprops=dict(facecolor=self.style_config['color_palette'][idx]))
            
        plt.title(title)
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_graph_structure(self, 
                           graph: Any,
                           node_colors: List[float] = None,
                           title: str = "Graph Structure") -> None:
        """Plot molecular graph structure with optional node coloring."""
        plt.figure(figsize=self.style_config['figsize'])
        pos = nx.spring_layout(graph)
        
        nx.draw(graph, pos, 
                node_color=node_colors if node_colors else self.style_config['color_palette'][0],
                with_labels=True,
                node_size=500,
                font_size=self.style_config['font_size'])
        
        plt.title(title)
        plt.show()