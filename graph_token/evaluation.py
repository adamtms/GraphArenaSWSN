"""
Evaluation utilities for GNN models.

This module provides evaluation metrics, visualization functions,
and analysis tools for trained GNN models.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    predictions: np.ndarray
    ground_truth: np.ndarray
    probabilities: Optional[np.ndarray] = None
    
    def __repr__(self) -> str:
        return (
            f"EvaluationResults(\n"
            f"  accuracy={self.accuracy:.4f},\n"
            f"  precision={self.precision:.4f},\n"
            f"  recall={self.recall:.4f},\n"
            f"  f1={self.f1:.4f}\n"
            f")"
        )


def evaluate_model(
    model: nn.Module,
    dataset: Dataset,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32,
    average: str = "weighted",
) -> EvaluationResults:
    """
    Evaluate a trained model on a dataset.
    
    Args:
        model: The trained GNN model.
        dataset: Dataset to evaluate on.
        device: Device to use for inference.
        batch_size: Batch size for inference.
        average: Averaging method for metrics ('micro', 'macro', 'weighted').
        
    Returns:
        EvaluationResults containing all metrics.
    """
    model = model.to(device)
    model.eval()
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
            if batch.y.dim() == 1:
                all_targets.append(batch.y.cpu().numpy())
            else:
                all_targets.append(batch.y.squeeze().cpu().numpy())
    
    predictions = np.concatenate(all_preds)
    probabilities = np.concatenate(all_probs)
    ground_truth = np.concatenate(all_targets)
    
    # Compute metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, average=average, zero_division=0)
    recall = recall_score(ground_truth, predictions, average=average, zero_division=0)
    f1 = f1_score(ground_truth, predictions, average=average, zero_division=0)
    conf_matrix = confusion_matrix(ground_truth, predictions)
    
    return EvaluationResults(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=conf_matrix,
        predictions=predictions,
        ground_truth=ground_truth,
        probabilities=probabilities,
    )


def print_classification_report(
    results: EvaluationResults,
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Print a detailed classification report.
    
    Args:
        results: EvaluationResults from evaluate_model.
        class_names: Optional list of class names.
    """
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(
        classification_report(
            results.ground_truth,
            results.predictions,
            target_names=class_names,
            zero_division=0,
        )
    )
    print("=" * 60)


def plot_confusion_matrix(
    results: EvaluationResults,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "Blues",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a confusion matrix.
    
    Args:
        results: EvaluationResults from evaluate_model.
        class_names: Optional list of class names.
        figsize: Figure size.
        cmap: Colormap for the heatmap.
        save_path: Path to save the figure (optional).
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    cm = results.confusion_matrix
    n_classes = cm.shape[0]
    
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix',
    )
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Dictionary containing 'train_loss', 'train_acc', 'val_loss', 'val_acc'.
        figsize: Figure size.
        save_path: Path to save the figure (optional).
        
    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss plot
    ax1 = axes[0]
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    if 'val_loss' in history and history['val_loss']:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2 = axes[1]
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    if 'val_acc' in history and history['val_acc']:
        ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def compare_models(
    models: Dict[str, nn.Module],
    dataset: Dataset,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, EvaluationResults]:
    """
    Compare multiple models on the same dataset.
    
    Args:
        models: Dictionary of model_name -> model.
        dataset: Dataset to evaluate on.
        device: Device to use.
        
    Returns:
        Dictionary of model_name -> EvaluationResults.
    """
    results = {}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        results[name] = evaluate_model(model, dataset, device)
        print(f"  Accuracy: {results[name].accuracy:.4f}")
    
    return results


def plot_model_comparison(
    results: Dict[str, EvaluationResults],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a comparison of multiple models.
    
    Args:
        results: Dictionary of model_name -> EvaluationResults.
        figsize: Figure size.
        save_path: Path to save the figure (optional).
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    model_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    x = np.arange(len(model_names))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [getattr(results[name], metric) for name in model_names]
        ax.bar(x + i * width, values, width, label=metric.capitalize())
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
