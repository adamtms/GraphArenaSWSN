# GraphToken Training Guide

This guide explains how to train GraphToken models that combine Graph Neural Networks (GNNs) with Large Language Models (LLMs) for graph reasoning tasks.

## Overview

GraphToken is an approach where:
1. A **GNN** encodes the graph structure into a fixed-size embedding
2. This embedding **replaces a placeholder token** in the LLM's input
3. The **LLM generates text** answers conditioned on the graph embedding

This allows LLMs to "see" and reason about graph structures without converting them to text.

## Prerequisites

```bash
pip install torch torch_geometric scikit-learn matplotlib tqdm networkx transformers
```

For GraphToken mode with LLMs:
```bash
pip install transformers accelerate
# Optional: for Lion optimizer
pip install lion-pytorch
```

## Training Modes

### 1. GNN-Only Mode

Train a standalone GNN for graph classification:

```bash
python train_gnn.py --mode gnn --task CONNECTED --model GIN
```

### 2. GraphToken Mode

Train GNN encoder with frozen LLM for graph-conditioned generation:

```bash
python train_gnn.py --mode graphtoken --task CONNECTED --llm google/gemma-2b-it
```

## Available Tasks

| Task | Description |
|------|-------------|
| `CONNECTED` | Check if graph is connected |
| `DIAMETER` | Compute graph diameter |
| `DISTANCE` | Shortest path between nodes |
| `GED` | Graph Edit Distance |
| `MCP` | Maximum Clique Problem |
| `MCS` | Maximum Common Subgraph |
| `MIS` | Maximum Independent Set |
| `MVC` | Minimum Vertex Cover |
| `NEIGHBOR` | Neighborhood queries |
| `TSP` | Traveling Salesman Problem |

## GNN Architectures

| Model | Description | Best For |
|-------|-------------|----------|
| `GCN` | Graph Convolutional Network | General graphs |
| `GIN` | Graph Isomorphism Network | Discriminating graph structures |
| `GAT` | Graph Attention Network | Weighted node importance |

## Command Line Arguments

### Mode Selection

| Argument | Values | Description |
|----------|--------|-------------|
| `--mode` | `gnn`, `graphtoken` | Training mode |

### Task and Data

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `CONNECTED` | Graph task |
| `--difficulty` | `easy` | `easy` or `hard` |
| `--dataset-loc` | `dataset` | Dataset path |

### Model Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `GIN` | GNN architecture |
| `--hidden-dim` | `64` | Hidden dimension |
| `--num-layers` | `3` | Number of layers |
| `--dropout` | `0.5` | Dropout rate |
| `--pooling` | `mean` | Pooling method |
| `--lpe-dim` | `4` | Laplacian PE dimension |

### LLM Configuration (GraphToken mode)

| Argument | Default | Description |
|----------|---------|-------------|
| `--llm` | `google/gemma-2b-it` | HuggingFace model |
| `--freeze-llm` | `True` | Freeze LLM weights |
| `--max-tokens` | `100` | Max sequence length |

### Training

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | `100` | Training epochs |
| `--batch-size` | `32` | Batch size (1 for GraphToken) |
| `--lr` | `0.001` | Learning rate |
| `--patience` | `20` | Early stopping patience |
| `--eval-every` | `250` | Eval interval (GraphToken) |

### Output

| Argument | Description |
|----------|-------------|
| `--save-model` | Save trained model |
| `--save-plots` | Save training plots |
| `--no-plots` | Disable plot display |
| `--output-dir` | Output directory |

## Examples

### Train GIN for Graph Classification

```bash
python train_gnn.py \
    --mode gnn \
    --task MIS \
    --difficulty hard \
    --model GIN \
    --hidden-dim 128 \
    --epochs 200 \
    --save-model
```

### Train GraphToken with Gemma

```bash
python train_gnn.py \
    --mode graphtoken \
    --task CONNECTED \
    --llm google/gemma-2b-it \
    --model GIN \
    --epochs 3 \
    --lr 0.0001 \
    --eval-every 250
```

### Compare GNN Architectures

```bash
for model in GCN GIN GAT; do
    python train_gnn.py \
        --mode gnn \
        --task CONNECTED \
        --model $model \
        --save-model \
        --save-plots \
        --no-plots
done
```

## Python API

### Standalone GNN Training

```python
from graph_token import (
    GPDataset, Tasks, Difficulty,
    get_model, TrainingConfig, train_model,
    evaluate_model, print_classification_report,
)

# Load data
dataset = GPDataset(task=Tasks.CONNECTED, difficulty=Difficulty.EASY)

# Create model
model = get_model("GIN", input_dim=1, hidden_dim=64, output_dim=2)

# Train
config = TrainingConfig(num_epochs=100, batch_size=32)
results = train_model(model, dataset[:80], dataset[80:], config)

# Evaluate
eval_results = evaluate_model(model, dataset[80:])
print_classification_report(eval_results)
```

### GraphToken with LLM

```python
from graph_token import (
    load_graphtoken_model,
    GraphTokenTrainingConfig,
    prepare_graphtoken_dataset,
    train_graphtoken,
)

# Load model
model, tokenizer, sampler = load_graphtoken_model(
    llm_name="google/gemma-2b-it",
    gnn_type="GIN",
    gnn_hidden_dim=64,
)

# Prepare data
examples = [
    {"question": "Q: Is this graph connected?", "answer": "Yes", "graph": nx_graph},
    ...
]
train_ds = prepare_graphtoken_dataset(examples, tokenizer)

# Train
config = GraphTokenTrainingConfig(num_epochs=3, learning_rate=0.0001)
train_graphtoken(model.gnn, model.llm, tokenizer, train_ds, config=config)

# Generate
output = sampler.generate(
    prompts=["Q: Is this graph connected?"],
    graphs=[nx_graph],
    max_new_tokens=20,
)
print(output.text[0])
```

## Architecture Details

### GraphToken Encoder

The `GraphTokenEncoder` produces embeddings matching the LLM's token dimension:

```python
from graph_token import GraphTokenEncoder

encoder = GraphTokenEncoder(
    input_dim=4,           # LPE dimension
    hidden_dim=64,         # GNN hidden size
    output_dim=2048,       # LLM embedding dimension
    num_layers=3,
    gnn_type="GIN",
)
```

### Laplacian Positional Encoding

Node features use Laplacian Positional Encoding (LPE):

```python
from graph_token import laplacian_positional_encoding, networkx_to_pyg

# Convert NetworkX graph with LPE
pyg_data = networkx_to_pyg(nx_graph, use_lpe=True, lpe_dim=4)
```

### GraphToken Sampler

```python
from graph_token import GraphTokenSampler

sampler = GraphTokenSampler(
    gnn=gnn_encoder,
    llm=language_model,
    tokenizer=tokenizer,
)

# Generate responses
output = sampler(
    prompts=["Q: What is the diameter?"],
    graphs=[graph],
    max_new_tokens=50,
    temperature=0.7,
)
```

## Model Files

```
outputs/
├── GIN_CONNECTED_easy.pt           # GNN-only checkpoint
├── graphtoken_GIN_CONNECTED.pt     # GraphToken GNN checkpoint
├── GIN_CONNECTED_history.png       # Training curves
└── GIN_CONNECTED_confusion.png     # Confusion matrix
```

## Troubleshooting

### Out of Memory (GraphToken)

- Use smaller LLM (`google/gemma-2b` instead of `7b`)
- Reduce `--hidden-dim`
- Ensure `--batch-size 1`

### LLM Access Issues

```bash
# Login to HuggingFace for gated models
huggingface-cli login
```

### Poor GraphToken Performance

- Increase `--epochs` (try 5-10)
- Lower `--lr` (try 0.00001)
- Use `--model GIN` (most expressive)
- Increase `--lpe-dim` for complex graphs

## References

- [Let Your Graph Do the Talking](https://arxiv.org/abs/2402.05862) - Original GraphToken paper
- [Graph Isomorphism Network](https://arxiv.org/abs/1810.00826)
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
