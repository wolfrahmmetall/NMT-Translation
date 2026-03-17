import torch
import torch.nn as nn
from typing import Optional

from models.attentions import NMTCrossAttention, NMTSelfAttention, NMTFeedForward
from models.kvcache import KVCache

class NMTEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.self_attn = NMTSelfAttention(
            hidden_size=d_model,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        self.ff = NMTFeedForward(
            d_model=d_model,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, 
                x: torch.Tensor, 
                source_key_padding_mask: Optional[torch.Tensor] = None
               ):
        sa = self.self_attn(x=x,
                            kv_cache=None,
                            masked=False,
                            key_padding_mask=source_key_padding_mask)
        x = self.norm1(x + self.dropout(sa))
        ff = self.ff(x)
        x = self.norm2(x + self.dropout(ff))
        return x
    
class NMTDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        num_kv_heads: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.self_attn = NMTSelfAttention(
            hidden_size=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout_rate=dropout_rate,
        )
        self.cross_attn = NMTCrossAttention(
            hidden_size=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        self.ff = NMTFeedForward(
            d_model=d_model,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                x: torch.Tensor,
                memory: torch.Tensor,
                target_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None
               ):
        sa = self.self_attn(x=x, 
                            kv_cache=None, 
                            masked=True,
                            key_padding_mask=target_key_padding_mask)
        x = self.norm1(x + self.dropout(sa))

        ca = self.cross_attn(x=x, 
                             memory=memory, 
                             memory_key_padding_mask=memory_key_padding_mask)
        x = self.norm2(x + self.dropout(ca))

        ff = self.ff(x)
        x = self.norm3(x + self.dropout(ff))
        return x

    def incremental_forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        layer_idx: int,
        kv_cache: KVCache,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ):
        
        sa = self.self_attn(
            x=x,
            layer_idx=layer_idx,
            kv_cache=kv_cache,
            masked=True
        )
        x = self.norm1(x + self.dropout(sa))

        ca = self.cross_attn(x=x, 
                             memory=memory, 
                             memory_key_padding_mask = memory_key_padding_mask)
        x = self.norm2(x + self.dropout(ca))

        ff = self.ff(x)
        x = self.norm3(x + self.dropout(ff))
        return x