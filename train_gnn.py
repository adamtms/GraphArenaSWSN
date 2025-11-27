#!/usr/bin/env python3
"""
Train GraphToken models (GNN + LLM) on graph reasoning tasks.

This script supports two modes:
1. GNN-only: Train standalone GNN for graph classification
2. GraphToken: Train GNN encoder with frozen LLM for graph-conditioned generation

Usage:
    # GNN-only mode
    python train_gnn.py --mode gnn --task CONNECTED --model GCN
    
    # GraphToken mode (requires HuggingFace LLM)
    python train_gnn.py --mode graphtoken --task CONNECTED --llm google/gemma-2b-it
"""

import argparse
import sys
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path

from graph_token import (
    # Data loading
    GPDataset, Tasks, Difficulty,
    # Models
    get_model, GraphTokenEncoder, laplacian_positional_encoding,
    # Training
    TrainingConfig, train_model,
    GraphTokenTrainingConfig, train_graphtoken, prepare_graphtoken_dataset,
    # Evaluation
    evaluate_model, print_classification_report,
    plot_confusion_matrix, plot_training_history,
    # Sampling
    load_graphtoken_model, GraphTokenSampler,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GNN/GraphToken models on graph tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        default="gnn",
        choices=["gnn", "graphtoken"],
        help="Training mode: 'gnn' for standalone GNN, 'graphtoken' for GNN+LLM",
    )
    
    # Task and data arguments
    parser.add_argument(
        "--task",
        type=str,
        default="CONNECTED",
        choices=[t.name for t in Tasks],
        help="Graph task to solve",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="easy",
        choices=["easy", "hard"],
        help="Dataset difficulty level",
    )
    parser.add_argument(
        "--dataset-loc",
        type=str,
        default="dataset",
        help="Path to dataset directory",
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="GIN",
        choices=["GCN", "GIN", "GAT"],
        help="GNN model architecture",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden layer dimension",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of GNN layers",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout probability",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "add", "max"],
        help="Graph pooling method",
    )
    parser.add_argument(
        "--lpe-dim",
        type=int,
        default=4,
        help="Laplacian Positional Encoding dimension",
    )
    
    # LLM arguments (for GraphToken mode)
    parser.add_argument(
        "--llm",
        type=str,
        default="google/gemma-2-2b-it",
        help="HuggingFace model name for LLM (e.g., 'google/gemma-2-2b-it', 'gpt2')",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token for gated models (e.g., Gemma). Can also set HF_TOKEN env var.",
    )
    parser.add_argument(
        "--freeze-llm",
        action="store_true",
        default=True,
        help="Freeze LLM parameters during training",
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (1 for GraphToken)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=250,
        help="Evaluate every N steps (GraphToken mode)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum sequence length (GraphToken mode)",
    )
    
    # Data split arguments
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for testing",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Fraction of training data for validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the trained model",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save training plots",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot display",
    )
    
    return parser.parse_args()


def train_gnn_mode(args):
    """Train standalone GNN for graph classification."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Dataset
    print(f"\n{'='*60}")
    print(f"Loading {args.task} dataset ({args.difficulty})...")
    print(f"{'='*60}")
    
    task = Tasks[args.task]
    difficulty = Difficulty(args.difficulty)
    
    dataset = GPDataset(
        task=task,
        difficulty=difficulty,
        dataset_loc=args.dataset_loc,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Apply Laplacian Positional Encoding
    print(f"Applying Laplacian Positional Encoding (dim={args.lpe_dim})...")
    for data in dataset:
        data.x = laplacian_positional_encoding(data, dim=args.lpe_dim)
    
    # Analyze dataset and remap labels to contiguous range starting from 0
    labels = [data.y.item() for data in dataset]
    unique_labels = sorted(set(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    
    # Remap labels
    for data in dataset:
        data.y = torch.tensor([label_map[data.y.item()]], dtype=torch.long)
    
    input_dim = dataset[0].x.shape[1]
    
    print(f"Number of classes: {num_classes}")
    print(f"Input dimension: {input_dim}")
    
    # Split dataset
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=args.seed
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=args.val_size, random_state=args.seed
    )
    
    train_dataset = [dataset[i] for i in train_idx]
    val_dataset = [dataset[i] for i in val_idx]
    test_dataset = [dataset[i] for i in test_idx]
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create Model
    print(f"\n{'='*60}")
    print(f"Creating {args.model} model...")
    print(f"{'='*60}")
    
    model = get_model(
        model_name=args.model,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling=args.pooling,
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    
    # Train
    print(f"\n{'='*60}")
    print("Training...")
    print(f"{'='*60}")
    
    config = TrainingConfig(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        device=device,
    )
    
    train_results = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        verbose=True,
    )
    
    print(f"\nTraining completed in {train_results['epochs_trained']} epochs")
    
    # Evaluate
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    print(f"{'='*60}")
    
    eval_results = evaluate_model(model, test_dataset, device=device)
    
    print(f"\nTest Results:")
    print(f"  Accuracy:  {eval_results.accuracy:.4f}")
    print(f"  F1 Score:  {eval_results.f1:.4f}")
    
    print_classification_report(eval_results)
    
    # Save outputs
    if args.save_model:
        model_path = output_dir / f"{args.model}_{args.task}_{args.difficulty}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': vars(args),
            'eval_results': {
                'accuracy': eval_results.accuracy,
                'f1': eval_results.f1,
            },
        }, model_path)
        print(f"\nModel saved to: {model_path}")
    
    if not args.no_plots or args.save_plots:
        fig_history = plot_training_history(train_results['history'])
        if args.save_plots:
            fig_history.savefig(output_dir / f"{args.model}_{args.task}_history.png")
        
        fig_cm = plot_confusion_matrix(eval_results)
        if args.save_plots:
            fig_cm.savefig(output_dir / f"{args.model}_{args.task}_confusion.png")
        
        if not args.no_plots:
            plt.show()
    
    return eval_results


def train_graphtoken_mode(args):
    """Train GraphToken model (GNN + LLM)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load LLM and create GraphToken model
    print(f"\n{'='*60}")
    print(f"Loading LLM: {args.llm}")
    print(f"{'='*60}")
    
    try:
        model, tokenizer, sampler = load_graphtoken_model(
            llm_name=args.llm,
            gnn_hidden_dim=args.hidden_dim,
            gnn_num_layers=args.num_layers,
            gnn_type=args.model,
            lpe_dim=args.lpe_dim,
            freeze_llm=args.freeze_llm,
            device=device,
            hf_token=args.hf_token,
        )
    except Exception as e:
        print(f"Error loading LLM: {e}")
        print("Make sure you have access to the model and transformers is installed.")
        print("For gated models like Gemma, use --hf-token YOUR_TOKEN or set HF_TOKEN env var.")
        return None
    
    print(f"GNN type: {args.model}")
    print(f"LLM frozen: {args.freeze_llm}")
    
    # Load Dataset
    print(f"\n{'='*60}")
    print(f"Loading {args.task} dataset ({args.difficulty})...")
    print(f"{'='*60}")
    
    task = Tasks[args.task]
    difficulty = Difficulty(args.difficulty)
    
    dataset = GPDataset(
        task=task,
        difficulty=difficulty,
        dataset_loc=args.dataset_loc,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Convert to GraphToken format
    # Note: This requires the dataset to have question/answer format
    # For now, we create simple examples from the graph data
    print("\nPreparing training examples...")
    
    examples = []
    for i, data in enumerate(dataset):
        # Create a simple question based on the task
        question = f"Q: What is the answer for this graph problem?\nA:"
        answer = str(data.y.item())
        
        # Use nx_graph stored in the data object
        if hasattr(data, 'nx_graph'):
            examples.append({
                'question': question,
                'answer': answer,
                'graph': data.nx_graph,
            })
    
    if not examples:
        print("Warning: Dataset does not contain NetworkX graphs for GraphToken training.")
        print("GraphToken requires the original graph structure.")
        return None
    
    # Split
    train_examples, test_examples = train_test_split(
        examples, test_size=args.test_size, random_state=args.seed
    )
    
    # Prepare datasets
    train_ds = prepare_graphtoken_dataset(
        train_examples,
        tokenizer,
        max_tokens=args.max_tokens,
        lpe_dim=args.lpe_dim,
    )
    
    test_ds = prepare_graphtoken_dataset(
        test_examples,
        tokenizer,
        max_tokens=args.max_tokens,
        lpe_dim=args.lpe_dim,
    )
    
    print(f"Train examples: {len(train_ds)}")
    print(f"Test examples: {len(test_ds)}")
    
    # Train
    print(f"\n{'='*60}")
    print("Training GraphToken...")
    print(f"{'='*60}")
    
    config = GraphTokenTrainingConfig(
        learning_rate=args.lr,
        num_epochs=args.epochs,
        eval_every_n=args.eval_every,
        max_tokens=args.max_tokens,
        device=device,
        freeze_llm=args.freeze_llm,
    )
    
    train_results = train_graphtoken(
        gnn=model.gnn,
        llm=model.llm,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=test_ds,
        config=config,
        verbose=True,
    )
    
    print(f"\nTraining completed: {train_results['steps_trained']} steps")
    
    # Test generation
    print(f"\n{'='*60}")
    print("Testing generation...")
    print(f"{'='*60}")
    
    for i in range(min(3, len(test_examples))):
        ex = test_examples[i]
        output = sampler.generate(
            prompts=[ex['question']],
            graphs=[ex['graph']],
            max_new_tokens=20,
        )
        print(f"\nQuestion: {ex['question']}")
        print(f"Generated: {output.text[0]}")
        print(f"Ground Truth: {ex['answer']}")
    
    # Save model
    if args.save_model:
        model_path = output_dir / f"graphtoken_{args.model}_{args.task}.pt"
        torch.save({
            'gnn_state_dict': model.gnn.state_dict(),
            'config': vars(args),
        }, model_path)
        print(f"\nGNN saved to: {model_path}")
    
    return train_results


def main():
    """Main entry point."""
    args = parse_args()
    
    print(f"\n{'='*60}")
    print(f"GraphToken Training - Mode: {args.mode.upper()}")
    print(f"{'='*60}")
    
    if args.mode == "gnn":
        return train_gnn_mode(args)
    elif args.mode == "graphtoken":
        return train_graphtoken_mode(args)
    else:
        print(f"Unknown mode: {args.mode}")
        return None


if __name__ == "__main__":
    main()
