"""
Graph Neural Network models for GraphToken.

This module provides GNN architectures that encode graphs into embeddings
compatible with LLM token embeddings. The GNN output is used to replace
a placeholder token in the LLM's embedding space.

Supports both:
1. PyTorch Geometric GNNs for standalone graph classification
2. GNN encoders that output LLM-compatible embeddings for GraphToken
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GINConv,
    GATConv,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
)
from torch_geometric.data import Data
from typing import Optional, Literal
import networkx as nx
import numpy as np


# =============================================================================
# Graph Feature Extraction
# =============================================================================

def laplacian_positional_encoding(
    graph: nx.Graph | Data,
    dim: int = 4
) -> torch.Tensor:
    """
    Compute Laplacian Positional Encoding for graph nodes.
    
    Args:
        graph: NetworkX graph or PyTorch Geometric Data object.
        dim: Dimension of positional encoding.
        
    Returns:
        Node features tensor of shape (num_nodes, dim).
    """
    # Handle PyG Data objects
    if isinstance(graph, Data):
        return _lpe_from_pyg(graph, dim)
    
    # Handle NetworkX graphs
    return _lpe_from_networkx(graph, dim)


def _lpe_from_networkx(graph: nx.Graph, dim: int) -> torch.Tensor:
    """Compute LPE from NetworkX graph."""
    if graph.number_of_nodes() == 0:
        return torch.zeros((0, dim), dtype=torch.float32)
    
    # Compute normalized Laplacian
    L = nx.normalized_laplacian_matrix(
        graph, nodelist=sorted(graph.nodes), weight=None
    ).astype(np.float32)
    
    # SVD decomposition
    try:
        U, _, _ = np.linalg.svd(L.todense(), compute_uv=True)
    except np.linalg.LinAlgError:
        # Fallback to zeros if SVD fails
        return torch.zeros((graph.number_of_nodes(), dim), dtype=torch.float32)
    
    # Pad if necessary
    if dim > U.shape[1]:
        U = np.pad(U, ((0, 0), (0, dim - U.shape[1])))
    
    return torch.from_numpy(U[:, :dim].astype(np.float32))


def _lpe_from_pyg(data: Data, dim: int) -> torch.Tensor:
    """Compute LPE from PyG Data object."""
    num_nodes = data.num_nodes
    if num_nodes == 0:
        return torch.zeros((0, dim), dtype=torch.float32)
    
    # Build adjacency matrix from edge_index
    edge_index = data.edge_index
    
    # Create sparse adjacency matrix
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    
    # Make symmetric (in case of directed edges)
    adj = (adj + adj.T) / 2
    adj = (adj > 0).float()
    
    # Compute degree matrix
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    
    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    L = torch.eye(num_nodes) - D_inv_sqrt @ adj @ D_inv_sqrt
    
    # SVD decomposition
    try:
        U, _, _ = torch.linalg.svd(L)
    except RuntimeError:
        # Fallback to zeros if SVD fails
        return torch.zeros((num_nodes, dim), dtype=torch.float32)
    
    # Pad if necessary
    if dim > U.shape[1]:
        U = F.pad(U, (0, dim - U.shape[1]))
    
    return U[:, :dim].float()


def networkx_to_pyg(
    graph: nx.Graph,
    node_features: Optional[np.ndarray] = None,
    use_lpe: bool = True,
    lpe_dim: int = 4,
) -> Data:
    """
    Convert a NetworkX graph to PyTorch Geometric Data object.
    
    Args:
        graph: NetworkX graph.
        node_features: Optional node features. If None, uses LPE or ones.
        use_lpe: Whether to use Laplacian Positional Encoding.
        lpe_dim: Dimension of LPE if used.
        
    Returns:
        PyTorch Geometric Data object.
    """
    # Create node ID mapping (original ID -> 0-indexed)
    nodes = sorted(graph.nodes())
    node_map = {node: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)
    
    # Get edges with remapped node IDs
    if graph.number_of_edges() > 0:
        edges = [(node_map[s], node_map[t]) for s, t in graph.edges()]
        # Make undirected by adding reverse edges
        if not graph.is_directed():
            edges = edges + [(t, s) for s, t in edges]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # Get node features
    if node_features is not None:
        x = torch.tensor(node_features, dtype=torch.float32)
    elif use_lpe:
        lpe = _lpe_from_networkx(graph, dim=lpe_dim)
        x = lpe
    else:
        x = torch.ones((num_nodes, 1), dtype=torch.float32)
    
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


# =============================================================================
# GNN Models for Graph Classification
# =============================================================================

class GCN(nn.Module):
    """
    Graph Convolutional Network (Kipf & Welling, 2017).
    
    A GCN with global pooling for graph-level representations.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 2,
        num_layers: int = 3,
        dropout: float = 0.5,
        pooling: Literal["mean", "add", "max"] = "mean",
    ):
        super(GCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.pooling_fn = {
            "mean": global_mean_pool,
            "add": global_add_pool,
            "max": global_max_pool,
        }[pooling]
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.pooling_fn(x, batch)
        return self.fc(x)
    
    def encode(self, data: Data) -> torch.Tensor:
        """Get graph embedding without final classification layer."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
        
        return self.pooling_fn(x, batch)


class GIN(nn.Module):
    """
    Graph Isomorphism Network (Xu et al., 2019).
    
    A powerful GNN as expressive as the Weisfeiler-Lehman test.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 2,
        num_layers: int = 3,
        dropout: float = 0.5,
        pooling: Literal["mean", "add", "max"] = "add",
        train_eps: bool = True,
    ):
        super(GIN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.pooling_fn = {
            "mean": global_mean_pool,
            "add": global_add_pool,
            "max": global_max_pool,
        }[pooling]
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs.append(GINConv(mlp1, train_eps=train_eps))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=train_eps))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.pooling_fn(x, batch)
        return self.fc(x)
    
    def encode(self, data: Data) -> torch.Tensor:
        """Get graph embedding without final classification layer."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
        
        return self.pooling_fn(x, batch)


class GAT(nn.Module):
    """
    Graph Attention Network (Veličković et al., 2018).
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 2,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.5,
        pooling: Literal["mean", "add", "max"] = "mean",
    ):
        super(GAT, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.pooling_fn = {
            "mean": global_mean_pool,
            "add": global_add_pool,
            "max": global_max_pool,
        }[pooling]
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(GATConv(input_dim, hidden_dim // heads, heads=heads, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.pooling_fn(x, batch)
        return self.fc(x)
    
    def encode(self, data: Data) -> torch.Tensor:
        """Get graph embedding without final classification layer."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
        
        return self.pooling_fn(x, batch)


# =============================================================================
# GraphToken GNN Encoder (for LLM integration)
# =============================================================================

class GraphTokenEncoder(nn.Module):
    """
    GNN encoder that produces embeddings compatible with LLM token embeddings.
    
    This encoder takes a graph and produces a fixed-size embedding that can
    replace a placeholder token in an LLM's embedding space.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 2048,  # Should match LLM embedding dimension
        num_layers: int = 3,
        gnn_type: Literal["GCN", "GIN", "GAT"] = "GIN",
        pooling: Literal["mean", "add", "max"] = "mean",
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Dimension of input node features (e.g., LPE dimension).
            hidden_dim: Hidden dimension of GNN layers.
            output_dim: Output dimension (should match LLM embedding dimension).
            num_layers: Number of GNN layers.
            gnn_type: Type of GNN to use.
            pooling: Global pooling method.
            dropout: Dropout probability.
        """
        super(GraphTokenEncoder, self).__init__()
        
        self.pooling_fn = {
            "mean": global_mean_pool,
            "add": global_add_pool,
            "max": global_max_pool,
        }[pooling]
        
        # Build GNN layers based on type
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.gnn_type = gnn_type
        
        if gnn_type == "GCN":
            self.convs.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
        elif gnn_type == "GIN":
            mlp1 = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp1))
            for _ in range(num_layers - 1):
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.convs.append(GINConv(mlp))
        elif gnn_type == "GAT":
            heads = 4
            self.convs.append(GATConv(input_dim, hidden_dim // heads, heads=heads))
            for _ in range(num_layers - 1):
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads))
        
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = dropout
        
        # Project to LLM embedding dimension
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Encode graph(s) into LLM-compatible embeddings.
        
        Args:
            data: PyTorch Geometric Data object.
            
        Returns:
            Graph embeddings of shape (batch_size, output_dim).
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # GNN message passing
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = self.pooling_fn(x, batch)
        
        # Project to LLM dimension
        x = self.projection(x)
        
        return x


# =============================================================================
# Factory Function
# =============================================================================

def get_model(
    model_name: str,
    input_dim: int = 4,
    hidden_dim: int = 64,
    output_dim: int = 2,
    num_layers: int = 3,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create a GNN model.
    
    Args:
        model_name: Name of the model ('GCN', 'GIN', 'GAT', 'GraphTokenEncoder').
        input_dim: Dimension of input node features.
        hidden_dim: Dimension of hidden layers.
        output_dim: Number of output classes/dimension.
        num_layers: Number of GNN layers.
        **kwargs: Additional model-specific arguments.
        
    Returns:
        A GNN model instance.
    """
    models = {
        "GCN": GCN,
        "GIN": GIN,
        "GAT": GAT,
        "GraphTokenEncoder": GraphTokenEncoder,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    return models[model_name](
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        **kwargs,
    )
