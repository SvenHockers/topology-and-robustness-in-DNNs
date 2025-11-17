"""
File used to expose various methods and classes at root of the project 
"""

from .data import GiottoPointCloudDataset, make_point_clouds
from .models import SimplePointCNN, SimplePointMLP
from .topology import (
    analyze_layer_topology,
    compute_layer_topology,
    extract_persistence_stats,
    visualize_layer_topology,
)
from .training import evaluate, show_some_predictions, train_one_epoch
from .attacks import (
    find_min_adversarial_perturbation_iterative,
    find_one_correct_sample_of_class,
)
from .visualization import (
    visualize_sample_diagrams,
    plot_original_vs_adversarial,
    plot_torus_wireframe_compare,
)

__all__ = [
    # data
    "GiottoPointCloudDataset",
    "make_point_clouds",
    # models
    "SimplePointCNN",
    "SimplePointMLP",
    # topology
    "analyze_layer_topology",
    "compute_layer_topology",
    "extract_persistence_stats",
    "visualize_layer_topology",
    # training
    "evaluate",
    "show_some_predictions",
    "train_one_epoch",
    # attacks
    "find_min_adversarial_perturbation_iterative",
    "find_one_correct_sample_of_class",
    # viz
    "visualize_sample_diagrams",
    "plot_original_vs_adversarial",
    "plot_torus_wireframe_compare",
]


