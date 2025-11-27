"""
GraphToken Library

A library for training Graph Neural Networks that encode graphs into embeddings
compatible with Large Language Models. This enables LLMs to answer questions
about graph structure by conditioning on learned graph representations.

Architecture:
    1. GNN encodes graph → fixed-size embedding
    2. Embedding replaces placeholder token in LLM input
    3. LLM generates text answer conditioned on graph

Modules:
    - data_loader: Dataset loading utilities using GPDataset
    - models: GNN implementations (GCN, GIN, GAT, GraphTokenEncoder)
    - training: Training loops for GNN and GraphToken
    - sampler: GraphToken sampling for inference
    - evaluation: Metrics and visualization
"""

from .data_loader import GPDataset, Tasks, Difficulty

from .models import (
    # GNN Models
    GCN,
    GIN,
    GAT,
    GraphTokenEncoder,
    get_model,
    # Utilities
    laplacian_positional_encoding,
    networkx_to_pyg,
)

from .training import (
    # Configs
    TrainingConfig,
    GraphTokenTrainingConfig,
    TrainingInput,
    # GNN Training
    train_model,
    train_epoch,
    evaluate,
    get_predictions,
    EarlyStopping,
    # GraphToken Training
    prepare_graphtoken_dataset,
    train_graphtoken,
    graphtoken_forward_and_loss,
)

from .sampler import (
    GraphTokenSampler,
    GraphTokenModel,
    SamplerOutput,
    load_graphtoken_model,
)

from .evaluation import (
    EvaluationResults,
    evaluate_model,
    print_classification_report,
    plot_confusion_matrix,
    plot_training_history,
    compare_models,
    plot_model_comparison,
)

__all__ = [
    # Data loading
    "GPDataset",
    "Tasks",
    "Difficulty",
    # Models
    "GCN",
    "GIN",
    "GAT",
    "GraphTokenEncoder",
    "get_model",
    "laplacian_positional_encoding",
    "networkx_to_pyg",
    # Training configs
    "TrainingConfig",
    "GraphTokenTrainingConfig",
    "TrainingInput",
    # GNN Training
    "train_model",
    "train_epoch",
    "evaluate",
    "get_predictions",
    "EarlyStopping",
    # GraphToken Training
    "prepare_graphtoken_dataset",
    "train_graphtoken",
    "graphtoken_forward_and_loss",
    # Sampling
    "GraphTokenSampler",
    "GraphTokenModel",
    "SamplerOutput",
    "load_graphtoken_model",
    # Evaluation
    "EvaluationResults",
    "evaluate_model",
    "print_classification_report",
    "plot_confusion_matrix",
    "plot_training_history",
    "compare_models",
    "plot_model_comparison",
]
