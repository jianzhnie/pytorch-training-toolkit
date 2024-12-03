"""Full definition of a GPT Language Model with improved type hints and
documentation.

Adapted from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """Configuration class for GPT model architecture.

    Attributes:
        model_type (str): Type of GPT model configuration.
        n_layer (Optional[int]): Number of transformer layers.
        n_head (Optional[int]): Number of attention heads.
        n_embd (Optional[int]): Embedding dimension.
        vocab_size (int): Size of the vocabulary.
        block_size (int): Maximum sequence length.
        embd_pdrop (float): Embedding layer dropout probability.
        resid_pdrop (float): Residual connection dropout probability.
        attn_pdrop (float): Attention dropout probability.
    """

    model_type: str = 'gpt2'
    n_layer: Optional[int] = 12
    n_head: Optional[int] = 12
    n_embd: Optional[int] = 768
    vocab_size: int = 50257
    block_size: int = 1024
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1

    def __post_init__(self):
        """Post-initialization method to set default model configurations based
        on model type if specific parameters are not provided."""
        if self.model_type and all(
                val is None
                for val in [self.n_layer, self.n_head, self.n_embd]):
            model_configs: Dict[str, Dict[str, int]] = {
                'openai-gpt': {
                    'n_layer': 12,
                    'n_head': 12,
                    'n_embd': 768,
                },  # 117M params
                'gpt2': {
                    'n_layer': 12,
                    'n_head': 12,
                    'n_embd': 768
                },  # 124M params
                'gpt2-medium': {
                    'n_layer': 24,
                    'n_head': 16,
                    'n_embd': 1024,
                },  # 350M params
                'gpt2-large': {
                    'n_layer': 36,
                    'n_head': 20,
                    'n_embd': 1280,
                },  # 774M params
                'gpt2-xl': {
                    'n_layer': 48,
                    'n_head': 25,
                    'n_embd': 1600,
                },  # 1558M params
                'gopher-44m': {
                    'n_layer': 8,
                    'n_head': 16,
                    'n_embd': 512
                },
                'gpt-mini': {
                    'n_layer': 6,
                    'n_head': 6,
                    'n_embd': 192
                },
                'gpt-micro': {
                    'n_layer': 4,
                    'n_head': 4,
                    'n_embd': 128
                },
                'gpt-nano': {
                    'n_layer': 3,
                    'n_head': 3,
                    'n_embd': 48
                },
            }
            config = model_configs.get(self.model_type, {})
            self.n_layer = config.get('n_layer', self.n_layer)
            self.n_head = config.get('n_head', self.n_head)
            self.n_embd = config.get('n_embd', self.n_embd)


@dataclass
class OptimizerConfig:
    """Configuration class for optimizer hyperparameters.

    Attributes:
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 regularization) factor.
    """

    learning_rate: float = 3e-4
    weight_decay: float = 0.1


class MultiheadAttentionLayer(nn.Module):
    """A multi-head masked self-attention layer with projection.

    Args:
        config (GPTConfig): Model configuration.
        device (str, optional): Device to place the layer. Defaults to "cpu".
        dtype (torch.dtype, optional): Data type for layer parameters. Defaults to torch.float32.
    """

    def __init__(self,
                 config: GPTConfig,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        assert (config.n_embd % config.n_head == 0
                ), 'Embedding dimension must be divisible by number of heads'

        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.c_proj = nn.Linear(config.n_embd,
                                config.n_embd,
                                device=device,
                                dtype=dtype)

        # Create a lower triangular mask for causal attention
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size),
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=config.n_embd,
            num_heads=config.n_head,
            dropout=config.attn_pdrop,
            batch_first=True,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for multi-head attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            torch.Tensor: Processed tensor after attention and projection
        """
        _, seq_size, _ = x.size()
        y = self.attn(x, x, x, attn_mask=self.mask[0,
                                                   0, :seq_size, :seq_size])[0]
        y = self.resid_drop(self.c_proj(y))
        return y


class Block(nn.Module):
    """A standard Transformer block with self-attention and feed-forward
    layers.

    Args:
        config (GPTConfig): Model configuration.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiheadAttentionLayer(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Processed tensor after self-attention and MLP
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class EmbeddingStem(nn.Module):
    """Embedding layer combining token and positional embeddings.

    Args:
        config (GPTConfig): Model configuration.
        device (str, optional): Device to place the layer. Defaults to "cpu".
        dtype (torch.dtype, optional): Data type for layer parameters. Defaults to torch.float32.
    """

    def __init__(self,
                 config: GPTConfig,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size,
                                    config.n_embd,
                                    device=device,
                                    dtype=dtype)
        self.pos_emb = nn.Parameter(
            torch.zeros(1,
                        config.block_size,
                        config.n_embd,
                        device=device,
                        dtype=dtype))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.block_size = config.block_size

    def reset_parameters(self) -> None:
        """Reset token embedding parameters."""
        self.tok_emb.reset_parameters()

    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        """Forward pass for embedding layer.

        Args:
            idx (torch.LongTensor): Input token indices

        Returns:
            torch.Tensor: Combined token and positional embeddings
        """
        b, t = idx.size()
        assert (
            t <= self.block_size
        ), f'Cannot forward sequence of length {t}, block size is only {self.block_size}'

        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        return self.drop(token_embeddings + position_embeddings)


class GPT(nn.Module):
    """GPT Language Model implementation.

    Args:
        config (GPTConfig): Model configuration.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.block_size = config.block_size

        # Ensure all model configurations are set
        config = self._set_model_config(config)

        # Model components
        self.emb_stem = EmbeddingStem(config)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight initialization
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                p.data.normal_(mean=0.0,
                               std=0.02 / math.sqrt(2 * config.n_layer))

        # Log total parameters
        n_params = sum(p.numel() for p in self.blocks.parameters())
        print(f'Number of parameters: {n_params/1e6:.2f}M')

    def _set_model_config(self, config: GPTConfig) -> GPTConfig:
        """Set or validate model configuration.

        Args:
            config (GPTConfig): Input configuration

        Returns:
            GPTConfig: Validated or updated configuration
        """
        return config

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for different types of layers.

        Args:
            module (nn.Module): Module to initialize
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        idx: torch.LongTensor,
        targets: Optional[torch.LongTensor] = None
    ) -> Union[Tuple[torch.Tensor, None], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for the GPT model.

        Args:
            idx (torch.LongTensor): Input token indices
            targets (Optional[torch.LongTensor], optional): Target token indices for loss calculation

        Returns:
            Tuple of logits and optional loss
        """
        x = self.emb_stem(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1),
                                   ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.LongTensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
    ) -> torch.LongTensor:
        """Generate new tokens based on input sequence.

        Args:
            idx (torch.LongTensor): Input token indices
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float, optional): Sampling temperature. Defaults to 1.0.
            do_sample (bool, optional): Whether to sample or take the most likely token. Defaults to False.
            top_k (Optional[int], optional): Top-k sampling parameter. Defaults to None.

        Returns:
            torch.LongTensor: Generated token sequence
        """
        for _ in range(max_new_tokens):
            # Crop context to block size
            idx_cond = (idx if idx.size(1) <= self.block_size else
                        idx[:, -self.block_size:])

            # Get logits
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample or select top token
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            # Append new token
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def create_optimizer(model: torch.nn.Module,
                     opt_config: OptimizerConfig) -> torch.optim.AdamW:
    """Create an AdamW optimizer with separate weight decay for different
    parameter types.

    Args:
        model (torch.nn.Module): PyTorch model
        opt_config (OptimizerConfig): Optimizer configuration

    Returns:
        torch.optim.AdamW: Configured optimizer
    """
    # Separate parameters for weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

    # Categorize parameters
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f'{mn}.{pn}' if mn else pn  # full param name

            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(
                    m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith('in_proj_weight'):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(
                    m, blacklist_weight_modules):
                no_decay.add(fpn)
            elif pn.endswith('pos_emb'):
                no_decay.add(fpn)

    # Validate parameter sets
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, (
        'parameters %s made it into both decay/no_decay sets!' %
        (str(inter_params), ))
    assert len(param_dict.keys() - union_params) == 0, (
        'parameters %s were not separated into either decay/no_decay set!' %
        (str(param_dict.keys() - union_params), ))

    # create the pytorch optimizer object
    optim_groups = [
        {
            'params': [param_dict[pn] for pn in sorted(list(decay))],
            'weight_decay': opt_config.weight_decay,
        },
        {
            'params': [param_dict[pn] for pn in sorted(list(no_decay))],
            'weight_decay': 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optim_groups,
                                  lr=opt_config.learning_rate,
                                  betas=(0.9, 0.95))
    return optimizer
