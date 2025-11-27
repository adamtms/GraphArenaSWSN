"""
GraphToken Sampler for LLM integration.

This module provides the GraphTokenSampler class that combines a GNN encoder
with an LLM to generate text responses conditioned on graph structure.

The approach replaces a placeholder token in the LLM's embedding space with
the GNN-encoded graph representation.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
import networkx as nx

from .models import GraphTokenEncoder, networkx_to_pyg


@dataclass
class SamplerOutput:
    """Output from GraphToken sampling."""
    
    # Generated text responses
    text: List[str]
    
    # Graph embeddings used
    graph_embeddings: List[torch.Tensor]
    
    # Input prompts
    prompts: List[str]
    
    # Token IDs generated (optional)
    tokens: Optional[List[List[int]]] = None


class GraphTokenSampler:
    """
    Sampler that combines GNN graph encoding with LLM text generation.
    
    This class implements the GraphToken approach where:
    1. A GNN encodes the graph into an embedding
    2. The embedding replaces a placeholder token in the LLM
    3. The LLM generates text conditioned on the graph embedding
    """
    
    # Default placeholder token (unused token in most vocabularies)
    PLACEHOLDER_TOKEN = "<unused0>"
    
    def __init__(
        self,
        gnn: nn.Module,
        llm: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        placeholder_token: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the GraphToken sampler.
        
        Args:
            gnn: GNN encoder that produces LLM-compatible embeddings.
            llm: HuggingFace language model for text generation.
            tokenizer: Tokenizer for the LLM.
            placeholder_token: Token to replace with graph embedding.
            device: Device to run inference on.
        """
        self.gnn = gnn.to(device)
        self.llm = llm.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        self.placeholder_token = placeholder_token or self.PLACEHOLDER_TOKEN
        
        # Get placeholder token ID
        self._setup_placeholder_token()
    
    def _setup_placeholder_token(self):
        """Set up the placeholder token in the tokenizer."""
        # Try to get existing token ID
        token_ids = self.tokenizer.encode(
            self.placeholder_token, add_special_tokens=False
        )
        
        if len(token_ids) == 1:
            self.placeholder_token_id = token_ids[0]
        else:
            # Add as special token if not found
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [self.placeholder_token]}
            )
            self.llm.resize_token_embeddings(len(self.tokenizer))
            self.placeholder_token_id = self.tokenizer.convert_tokens_to_ids(
                self.placeholder_token
            )
    
    def _get_graph_embedding(self, graph_data: Data) -> torch.Tensor:
        """Get GNN embedding for a graph."""
        graph_data = graph_data.to(self.device)
        with torch.no_grad():
            self.gnn.eval()
            embedding = self.gnn(graph_data)
        return embedding
    
    def _inject_graph_embedding(
        self,
        input_ids: torch.Tensor,
        graph_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inject graph embedding into LLM input embeddings.
        
        Replaces the placeholder token embedding with the graph embedding.
        """
        # Get input embeddings
        input_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # Find placeholder token positions
        placeholder_mask = input_ids == self.placeholder_token_id
        
        # Replace placeholder embeddings with graph embedding
        if placeholder_mask.any():
            # Expand graph embedding to match placeholder positions
            for batch_idx in range(input_ids.shape[0]):
                positions = placeholder_mask[batch_idx].nonzero(as_tuple=True)[0]
                for pos in positions:
                    input_embeds[batch_idx, pos] = graph_embedding[batch_idx]
        
        return input_embeds
    
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        graphs: List[Union[Data, nx.Graph]],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        **generate_kwargs,
    ) -> SamplerOutput:
        """
        Generate text responses conditioned on graphs.
        
        Args:
            prompts: List of text prompts (questions about the graphs).
            graphs: List of graphs (PyG Data or NetworkX graphs).
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p (nucleus) sampling parameter.
            do_sample: Whether to use sampling vs greedy decoding.
            **generate_kwargs: Additional arguments for model.generate().
            
        Returns:
            SamplerOutput with generated text and embeddings.
        """
        assert len(prompts) == len(graphs), "Must have same number of prompts and graphs"
        
        self.gnn.eval()
        self.llm.eval()
        
        generated_texts = []
        graph_embeddings = []
        all_tokens = []
        
        for prompt, graph in zip(prompts, graphs):
            # Convert NetworkX to PyG if needed
            if isinstance(graph, nx.Graph):
                graph_data = networkx_to_pyg(graph, use_lpe=True)
            else:
                graph_data = graph
            
            # Get graph embedding
            graph_data = graph_data.to(self.device)
            embedding = self.gnn(graph_data)
            graph_embeddings.append(embedding.cpu())
            
            # Prepare input with placeholder
            full_prompt = self.placeholder_token + prompt
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            
            # Get input embeddings with graph injection
            input_embeds = self._inject_graph_embedding(
                inputs["input_ids"],
                embedding,
            )
            
            # Generate
            outputs = self.llm.generate(
                inputs_embeds=input_embeds,
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **generate_kwargs,
            )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            generated_texts.append(generated_text)
            all_tokens.append(outputs[0].tolist())
        
        return SamplerOutput(
            text=generated_texts,
            graph_embeddings=graph_embeddings,
            prompts=prompts,
            tokens=all_tokens,
        )
    
    def __call__(
        self,
        prompts: List[str],
        graphs: List[Union[Data, nx.Graph]],
        **kwargs,
    ) -> SamplerOutput:
        """Alias for generate()."""
        return self.generate(prompts, graphs, **kwargs)


class GraphTokenModel(nn.Module):
    """
    End-to-end GraphToken model combining GNN encoder and LLM.
    
    This model can be trained end-to-end (or with frozen LLM) to learn
    graph representations that help the LLM answer graph-related questions.
    """
    
    PLACEHOLDER_TOKEN = "<unused0>"
    
    def __init__(
        self,
        gnn: nn.Module,
        llm: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        freeze_llm: bool = True,
        placeholder_token: Optional[str] = None,
    ):
        """
        Initialize GraphToken model.
        
        Args:
            gnn: GNN encoder.
            llm: Language model.
            tokenizer: Tokenizer.
            freeze_llm: Whether to freeze LLM parameters.
            placeholder_token: Placeholder token for graph embedding.
        """
        super().__init__()
        
        self.gnn = gnn
        self.llm = llm
        self.tokenizer = tokenizer
        self.placeholder_token = placeholder_token or self.PLACEHOLDER_TOKEN
        
        # Freeze LLM if specified
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
        
        # Setup placeholder token
        self._setup_placeholder_token()
    
    def _setup_placeholder_token(self):
        """Set up placeholder token."""
        token_ids = self.tokenizer.encode(
            self.placeholder_token, add_special_tokens=False
        )
        
        if len(token_ids) == 1:
            self.placeholder_token_id = token_ids[0]
        else:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [self.placeholder_token]}
            )
            self.llm.resize_token_embeddings(len(self.tokenizer))
            self.placeholder_token_id = self.tokenizer.convert_tokens_to_ids(
                self.placeholder_token
            )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graph_data: Data,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            input_ids: Token IDs including placeholder.
            attention_mask: Attention mask.
            graph_data: Graph data for GNN encoding.
            labels: Target token IDs for loss computation.
            
        Returns:
            Dictionary with 'loss' and 'logits'.
        """
        # Get graph embedding
        graph_embedding = self.gnn(graph_data)
        
        # Get input embeddings
        input_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # Replace placeholder with graph embedding
        batch_size = input_ids.shape[0]
        placeholder_mask = input_ids == self.placeholder_token_id
        
        for batch_idx in range(batch_size):
            positions = placeholder_mask[batch_idx].nonzero(as_tuple=True)[0]
            for pos in positions:
                if batch_idx < graph_embedding.shape[0]:
                    input_embeds[batch_idx, pos] = graph_embedding[batch_idx]
        
        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        
        return {
            "loss": outputs.loss if labels is not None else None,
            "logits": outputs.logits,
        }


def load_graphtoken_model(
    llm_name: str = "google/gemma-2-2b-it",
    gnn_hidden_dim: int = 64,
    gnn_num_layers: int = 3,
    gnn_type: str = "GIN",
    lpe_dim: int = 4,
    freeze_llm: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    hf_token: Optional[str] = None,
) -> tuple:
    """
    Load a GraphToken model with specified LLM from HuggingFace.
    
    Args:
        llm_name: HuggingFace model name (e.g., 'google/gemma-2-2b-it', 'gpt2').
        gnn_hidden_dim: Hidden dimension of GNN.
        gnn_num_layers: Number of GNN layers.
        gnn_type: Type of GNN ('GCN', 'GIN', 'GAT').
        lpe_dim: Dimension of Laplacian Positional Encoding.
        freeze_llm: Whether to freeze LLM weights.
        device: Device to load model on.
        hf_token: HuggingFace API token for accessing gated models (e.g., Gemma).
                  Can also be set via HF_TOKEN environment variable.
        
    Returns:
        Tuple of (model, tokenizer, sampler).
    """
    import os
    
    # Get token from argument, environment, or None
    token = hf_token or os.environ.get("HF_TOKEN")
    
    # Load from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(llm_name, token=token)
    llm = AutoModelForCausalLM.from_pretrained(
        llm_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        token=token,
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get LLM embedding dimension (handle different config structures)
    config = llm.config
    if hasattr(config, 'hidden_size'):
        llm_embed_dim = config.hidden_size
    elif hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
        # Gemma 3 and similar models with nested text_config
        llm_embed_dim = config.text_config.hidden_size
    elif hasattr(config, 'd_model'):
        llm_embed_dim = config.d_model
    elif hasattr(config, 'n_embd'):
        llm_embed_dim = config.n_embd
    else:
        raise ValueError(f"Cannot determine embedding dimension from config: {config}")
    
    # Create GNN encoder
    gnn = GraphTokenEncoder(
        input_dim=lpe_dim,
        hidden_dim=gnn_hidden_dim,
        output_dim=llm_embed_dim,
        num_layers=gnn_num_layers,
        gnn_type=gnn_type,
    ).to(device)
    
    # Create model and sampler
    model = GraphTokenModel(
        gnn=gnn,
        llm=llm,
        tokenizer=tokenizer,
        freeze_llm=freeze_llm,
    )
    
    sampler = GraphTokenSampler(
        gnn=gnn,
        llm=llm,
        tokenizer=tokenizer,
        device=device,
    )
    
    return model, tokenizer, sampler
