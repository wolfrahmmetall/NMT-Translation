import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from models.kvcache import KVCache, repeat_kv


class NMTSelfAttention(nn.Module):
    def __init__(self,
                 hidden_size: int = 256, # \equiv d_model
                 num_heads: int = 1,
                 num_kv_heads: int = 1,
                 dropout_rate: float=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * self.num_heads == self.hidden_size

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int = 0,
        kv_cache: Optional[KVCache] = None,
        masked: bool = True,
        key_padding_mask: Optional[torch.Tensor] = None
    ):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        past_len = 0 if kv_cache is None else kv_cache.cur_len

        if kv_cache is not None:
            full_k, full_v = kv_cache.append(layer_idx, k, v)
        else:
            full_k, full_v = k, v

        # GQA expand
        n_rep = self.num_heads // self.num_kv_heads
        k_rep = repeat_kv(full_k, n_rep)
        v_rep = repeat_kv(full_v, n_rep)

        scale = 1.0 / np.sqrt(self.head_dim)
        scores = torch.matmul(q, k_rep.transpose(-2, -1)) * scale
        K = scores.shape[-1]

        key_pos = torch.arange(K, device=x.device).view(1, K)
        q_pos = (past_len + torch.arange(T, device=x.device)).view(T, 1)
        if masked:
            causal = key_pos > q_pos
            scores = scores.masked_fill(causal.view(1, 1, T, K), float("-inf"))

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        attn_out = torch.matmul(probs, v_rep)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        out = self.o_proj(attn_out)
        return out
    
class NMTCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 1,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                x: torch.Tensor,
                memory: torch.Tensor,
                memory_key_padding_mask: Optional[torch.Tensor] = None
               ):
        B, Tq, _ = x.shape
        _, Tk, _ = memory.shape

        q = self.q_proj(x).view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / np.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if memory_key_padding_mask is not None:
            scores = scores.masked_fill(memory_key_padding_mask[:, None, None, :], float("-inf"))
        
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        attn_out = torch.matmul(probs, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Tq, self.num_heads * self.head_dim)
        out = self.o_proj(attn_out)
        return out
    
class NMTFeedForward(nn.Module):
    def __init__(self,
                 d_model: int,
                 ff_dim: int,
                 dropout_rate: float = 0.1
                ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, d_model)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)