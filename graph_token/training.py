"""
Training utilities for GraphToken models.

This module provides training loops for:
1. Standalone GNN graph classification
2. GraphToken (GNN + LLM) training for graph-conditioned text generation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
from transformers import PreTrainedTokenizer
from typing import Optional, Callable, Dict, Any, Tuple, List, Union
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np
import networkx as nx

from .models import networkx_to_pyg


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    num_epochs: int = 100
    batch_size: int = 32
    patience: int = 20
    min_delta: float = 0.001
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 10
    eval_every_n: int = 100  # For GraphToken training
    max_steps: Optional[int] = None


@dataclass
class GraphTokenTrainingConfig:
    """Configuration for GraphToken training."""
    
    learning_rate: float = 0.0001
    num_epochs: int = 3
    eval_every_n: int = 250
    batch_size: int = 1  # GraphToken typically uses batch_size=1
    max_steps: Optional[int] = None
    max_tokens: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    freeze_llm: bool = True
    optimizer: str = "adamw"  # 'adamw' or 'lion'


@dataclass
class TrainingInput:
    """
    Training input for GraphToken models.
    
    Mirrors the structure from the original notebook.
    """
    # Input token IDs [B, L]
    input_tokens: np.ndarray
    
    # Mask for target tokens (1 for tokens to predict) [B, L]
    target_mask: np.ndarray
    
    # Graph data (list of PyG Data objects)
    input_graphs: List[Data]
    
    # Parsed ground truth (optional)
    parsed_ground_truth: Optional[np.ndarray] = None


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.001,
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            return False
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
            
        if improved:
            self.best_score = score
            self.best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
    
    def restore_best_model(self, model: nn.Module) -> None:
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


# =============================================================================
# Standalone GNN Training
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch)
        
        if batch.y.dim() == 1:
            target = batch.y
        else:
            target = batch.y.squeeze()
            
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        
        if batch.y.dim() == 1:
            target = batch.y
        else:
            target = batch.y.squeeze()
            
        loss = criterion(out, target)
        total_loss += loss.item()
        
        pred = out.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
        
    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    config: Optional[TrainingConfig] = None,
    criterion: Optional[nn.Module] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Main training loop for standalone GNN.
    """
    if config is None:
        config = TrainingConfig()
        
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
        
    device = config.device
    model = model.to(device)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True
    )
    
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size, 
            shuffle=False
        )
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    early_stopping = EarlyStopping(
        patience=config.patience, 
        min_delta=config.min_delta,
        mode='min'
    )
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    epoch_iterator = tqdm(range(config.num_epochs), desc="Training") if verbose else range(config.num_epochs)
    
    for epoch in epoch_iterator:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_loss_eval, train_acc = evaluate(model, train_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        if val_dataset is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            scheduler.step(val_loss)
            
            if early_stopping(val_loss, model):
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break
                
            if verbose and (epoch + 1) % config.log_interval == 0:
                tqdm.write(
                    f"Epoch {epoch + 1}/{config.num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
        else:
            if verbose and (epoch + 1) % config.log_interval == 0:
                tqdm.write(
                    f"Epoch {epoch + 1}/{config.num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                )
    
    if val_dataset is not None:
        early_stopping.restore_best_model(model)
    
    return {
        'history': history,
        'best_val_loss': early_stopping.best_score if val_dataset else None,
        'epochs_trained': len(history['train_loss']),
    }


# =============================================================================
# GraphToken Training (GNN + LLM)
# =============================================================================

def prepare_graphtoken_dataset(
    examples: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    max_tokens: int = 100,
    placeholder_token: str = "<unused0>",
    lpe_dim: int = 4,
) -> List[TrainingInput]:
    """
    Prepare dataset for GraphToken training.
    
    Args:
        examples: List of examples with 'question', 'answer', 'graph' keys.
        tokenizer: LLM tokenizer.
        max_tokens: Maximum sequence length.
        placeholder_token: Token to replace with graph embedding.
        lpe_dim: Dimension of Laplacian Positional Encoding.
        
    Returns:
        List of TrainingInput objects.
    """
    output = []
    
    for ex in examples:
        # Extract question (after 'Q:' if present)
        question = ex['question']
        if 'Q:' in question:
            question = question[question.find('Q:'):]
        
        query = placeholder_token + question
        answer = ex['answer']
        
        # Convert graph to PyG
        graph = ex['graph']
        if isinstance(graph, nx.Graph):
            graph_data = networkx_to_pyg(graph, use_lpe=True, lpe_dim=lpe_dim)
        else:
            graph_data = graph
        
        # Tokenize
        query_tokens = tokenizer.encode(query, add_special_tokens=False)
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
        
        # Build input sequence
        input_tokens = np.array(
            #[tokenizer.bos_token_id] + query_tokens + answer_tokens,
            query_tokens + answer_tokens,
            dtype=np.int64
        )
        
        # Build target mask (1 for answer tokens)
        target_mask = np.zeros_like(input_tokens, dtype=np.int32)
        target_mask[len(query_tokens) + 1:] = 1
        
        # Pad to max_tokens
        orig_len = len(input_tokens)
        if orig_len < max_tokens:
            input_tokens = np.pad(
                input_tokens,
                (0, max_tokens - orig_len),
                constant_values=tokenizer.pad_token_id,
            )
            target_mask = np.pad(target_mask, (0, max_tokens - orig_len))
        else:
            input_tokens = input_tokens[:max_tokens]
            target_mask = target_mask[:max_tokens]
        
        output.append(
            TrainingInput(
                input_tokens=np.array([input_tokens]),
                target_mask=np.array([target_mask]),
                input_graphs=[graph_data],
                parsed_ground_truth=None,
            )
        )
    
    return output


def graphtoken_forward_and_loss(
    gnn: nn.Module,
    llm: nn.Module,
    input_tokens: torch.Tensor,
    input_mask: torch.Tensor,
    graph_data: Data,
    placeholder_token_id: int,
    device: str,
) -> torch.Tensor:
    """
    Forward pass and loss computation for GraphToken.
    
    Args:
        gnn: GNN encoder.
        llm: Language model.
        input_tokens: Input token IDs [B, L].
        input_mask: Target mask [B, L].
        graph_data: Graph data for encoding.
        placeholder_token_id: ID of placeholder token.
        device: Device.
        
    Returns:
        Cross-entropy loss.
    """
    # Get graph embedding
    graph_embedding = gnn(graph_data)  # [1, embed_dim]
    
    # Get input embeddings
    input_embeds = llm.get_input_embeddings()(input_tokens)
    
    # Replace placeholder token embedding with graph embedding
    placeholder_mask = input_tokens == placeholder_token_id
    for batch_idx in range(input_tokens.shape[0]):
        positions = placeholder_mask[batch_idx].nonzero(as_tuple=True)[0]
        for pos in positions:
            input_embeds[batch_idx, pos] = graph_embedding[batch_idx]
    
    # Forward pass through LLM
    outputs = llm(
        inputs_embeds=input_embeds,
        return_dict=True,
    )
    logits = outputs.logits
    
    # Compute loss on masked tokens
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_tokens[..., 1:].contiguous()
    shift_mask = input_mask[..., 1:].contiguous()
    
    # Flatten
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    # Apply mask and normalize
    loss = loss.view(shift_mask.shape)
    loss = (loss * shift_mask).sum() / (shift_mask.sum() + 1e-8)
    
    return loss


def train_graphtoken(
    gnn: nn.Module,
    llm: nn.Module,
    tokenizer: PreTrainedTokenizer,
    train_dataset: List[TrainingInput],
    val_dataset: Optional[List[TrainingInput]] = None,
    config: Optional[GraphTokenTrainingConfig] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Training loop for GraphToken model.
    
    Only trains the GNN parameters while keeping LLM frozen.
    
    Args:
        gnn: GNN encoder module.
        llm: Language model (frozen).
        tokenizer: LLM tokenizer.
        train_dataset: List of TrainingInput for training.
        val_dataset: Optional validation dataset.
        config: Training configuration.
        verbose: Whether to print progress.
        
    Returns:
        Dictionary with training history.
    """
    if config is None:
        config = GraphTokenTrainingConfig()
    
    device = config.device
    gnn = gnn.to(device)
    llm = llm.to(device)
    
    # Freeze LLM
    if config.freeze_llm:
        for param in llm.parameters():
            param.requires_grad = False
        llm.eval()
    
    # Get placeholder token ID
    placeholder_token = "<unused0>"
    token_ids = tokenizer.encode(placeholder_token, add_special_tokens=False)
    if len(token_ids) == 1:
        placeholder_token_id = token_ids[0]
    else:
        raise ValueError(f"Placeholder token '{placeholder_token}' not found in vocabulary")
    
    # Setup optimizer (only for GNN)
    if config.optimizer == "lion":
        try:
            from lion_pytorch import Lion
            optimizer = Lion(gnn.parameters(), lr=config.learning_rate)
        except ImportError:
            optimizer = optim.AdamW(gnn.parameters(), lr=config.learning_rate)
    else:
        optimizer = optim.AdamW(gnn.parameters(), lr=config.learning_rate)
    
    # Training loop
    history = {'train_loss': [], 'val_loss': []}
    total_steps = config.num_epochs * len(train_dataset)
    
    avg_loss = 0
    averaged_steps = 0
    
    pbar = tqdm(range(total_steps), desc="Training GraphToken") if verbose else range(total_steps)
    
    for n_steps in pbar:
        # Get example
        example = train_dataset[n_steps % len(train_dataset)]
        
        # Move data to device
        input_tokens = torch.tensor(example.input_tokens, device=device)
        target_mask = torch.tensor(example.target_mask, device=device)
        graph_data = example.input_graphs[0].to(device)
        
        # Forward and loss
        gnn.train()
        optimizer.zero_grad()
        
        loss = graphtoken_forward_and_loss(
            gnn=gnn,
            llm=llm,
            input_tokens=input_tokens,
            input_mask=target_mask,
            graph_data=graph_data,
            placeholder_token_id=placeholder_token_id,
            device=device,
        )
        
        loss.backward()
        optimizer.step()
        
        avg_loss += loss.item()
        averaged_steps += 1
        
        # Logging
        if n_steps and n_steps % config.eval_every_n == 0:
            avg_loss /= averaged_steps
            history['train_loss'].append(avg_loss)
            
            if verbose:
                tqdm.write(f'Step {n_steps} - Training loss: {avg_loss:.4f}')
            
            avg_loss = 0
            averaged_steps = 0
        
        # Max steps check
        if config.max_steps is not None and n_steps >= config.max_steps:
            break
    
    # Final logging
    if averaged_steps > 0:
        avg_loss /= averaged_steps
        history['train_loss'].append(avg_loss)
        if verbose:
            tqdm.write(f'Final - Training loss: {avg_loss:.4f}')
    
    return {
        'history': history,
        'steps_trained': n_steps + 1,
    }


def get_predictions(
    model: nn.Module,
    dataset: Dataset,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get model predictions on a dataset."""
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
    
    return (
        np.concatenate(all_preds),
        np.concatenate(all_probs),
        np.concatenate(all_targets),
    )
