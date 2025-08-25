# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.checkpoint as cp


@dataclass
class ModelArgs:
    head_sp: bool = False
    tp_size: int = 1
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 32768


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # print(f'norm input: {x.shape}')
        output = self._norm(x.float()).type_as(x)
        # print(f'norm output: {output.shape}')
        return output * self.weight

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore


class sp_head(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, xq, xk, xv):
        # # (bs, n_local_heads, seqlen, head_dim)

        # print(
        #     f"in sp_head: xq.shape {xq.shape}, xk.shape {xk.shape}, xv.shape {xv.shape}, "
        # )
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        return output


class Attention(nn.Module):
    """
    Multi-head attention module.

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_local_kv_heads (int): Number of local key and value heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension size of each attention head.
        wq (Linear): Linear transformation for queries.
        wk (Linear): Linear transformation for keys.
        wv (Linear): Linear transformation for values.
        wo (Linear): Linear transformation for output.

    """

    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.tp_size = model_args.tp_size
        self.head_sp = model_args.head_sp
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.dim // model_args.n_heads

        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )
        self.sp_head = sp_head()

    def forward(
        self,
        x: torch.Tensor,
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        # print(f'in attn: hidden_states.shape {x.shape}')
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(
            bsz,
            seqlen,
            # self.n_heads // (self.tp_size if self.head_sp else 1),
            self.n_heads,
            self.head_dim,
        )
        xk = xk.view(
            bsz,
            seqlen,
            # self.n_kv_heads // (self.tp_size if self.head_sp else 1),
            self.n_kv_heads,
            self.head_dim,
        )
        xv = xv.view(
            bsz,
            seqlen,
            # self.n_kv_heads // (self.tp_size if self.head_sp else 1),
            self.n_kv_heads,
            self.head_dim,
        )
        # print(f'in attn: query_states.shape {xq.shape}')

        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        # print(
        #     f"in attn: xq.shape {xq.shape}, xk.shape {xk.shape}, xv.shape {xv.shape}, "
        # )

        # we use casual mask for training
        output = self.sp_head(xq, xk, xv)
        # output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        # print(f"in attn: after sdpa {output.shape}")
        output = output.transpose(
            1, 2
        ).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        # print(f"in attn: before output.view(bsz, seqlen, -1) {output.shape}")
        output = output.view(bsz, seqlen, -1)
        # print(f"in attn: after output.view(bsz, seqlen, -1) {output.shape}")
        output = self.wo(output)
        # print(f"in attn: after output {output.shape}")
        return output


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    def __init__(self, layer_id: int, model_args: ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.attention = Attention(model_args)
        self.feed_forward = FeedForward(
            dim=model_args.dim,
            hidden_dim=4 * model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.num_layers = model_args.n_layers

        self.attention_norm = RMSNorm(dim=model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = RMSNorm(dim=model_args.dim, eps=model_args.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        residual = hidden_states
        # print(f'before input_layernorm {hidden_states.shape}')
        hidden_states = self.attention_norm(hidden_states)
        # print(f'before self_attn {hidden_states.shape}')
        hidden_states = self.attention(hidden_states)
        # print(f'after self_attn {hidden_states.shape}')
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        # print(f'before post_attention_layernorm {hidden_states.shape}')
        hidden_states = self.ffn_norm(hidden_states)
        # print(f'before mlp {hidden_states.shape}')
        hidden_states = self.feed_forward(hidden_states)
        # print(f'after mlp {hidden_states.shape}')
        # print(f'final residual: {residual.shape}, hidden_states: {hidden_states.shape}')
        hidden_states = residual + hidden_states
        return hidden_states


class Transformer(nn.Module):
    """
    Transformer Module

    Args:
        model_args (ModelArgs): Model configuration arguments.

    Attributes:
        model_args (ModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (ColumnParallelLinear): Linear layer for final output.
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

    """

    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.model_dim = model_args.dim
        self.gradient_checkpointing = False

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(model_args.n_layers):
            self.layers.append(TransformerBlock(layer_id, model_args))

        self.norm = RMSNorm(dim=model_args.dim, eps=model_args.norm_eps)

        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        # print(f'tokens.shape: {tokens.shape}')
        h = self.tok_embeddings(tokens)
        # print(f'tok_embeddings.shape: {h.shape}')

        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                h = cp.checkpoint(layer, h)
            else:
                h = layer(h)

        h = self.norm(h)
        output = self.output(h)
        return output

    @classmethod
    def from_model_args(cls, model_args: ModelArgs) -> "Transformer":
        """
        Initialize a Transformer model from a ModelArgs object.

        Args:
            model_args (ModelArgs): Model configuration arguments.

        Returns:
            Transformer: Transformer model.

        """
        return cls(model_args)
