import torch
import torch.nn as nn
from typing import Optional

from models.kvcache import KVCache
from models.layers import NMTEncoderLayer, NMTDecoderLayer
from models.posenc import PositionalEncoding
from settings.config import PAD_ID


class NMTTransformer(nn.Module):
    def __init__(
            self,
            source_vocab_size: int,
            target_vocab_size: int,
            d_model: int,
            num_heads: int,
            ff_dim: int,
            max_len: int,
            n_layers: int = 6,
            dropout_rate: float = 0.1,
            num_kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.max_len = max_len
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = d_model // num_heads

        self.source_embed = nn.Embedding(source_vocab_size, d_model, padding_idx=PAD_ID)
        self.target_embed = nn.Embedding(target_vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc = PositionalEncoding(max_len=max_len, d_model=d_model, dropout_rate=dropout_rate)

        self.encoders = nn.ModuleList([
            NMTEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate,
            )
            for _ in range(n_layers)
        ])

        self.decoders = nn.ModuleList([
            NMTDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_kv_heads=self.num_kv_heads,
                dropout_rate=dropout_rate,
            )
            for _ in range(n_layers)
        ])

        self.out = nn.Linear(d_model, target_vocab_size)

    def encode(self, src: torch.Tensor):
        source_key_padding_mask = (src == PAD_ID)
        x = self.pos_enc(self.source_embed(src))
        for layer in self.encoders:
            x = layer(x=x, source_key_padding_mask=source_key_padding_mask)
        return x, source_key_padding_mask

    def decode(
            self,
            target_in: torch.Tensor,
            memory: torch.Tensor,
            memory_key_padding_mask: Optional[torch.Tensor] = None
    ):
        target_key_padding_mask = (target_in == PAD_ID)
        x = self.pos_enc(self.target_embed(target_in))
        for layer in self.decoders:
            x = layer(x=x, 
                      memory=memory, 
                      target_key_padding_mask=target_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask
                     )
        return self.out(x)

    def create_kv_cache(self, batch_size: int, device, dtype) -> KVCache:
        return KVCache(
            num_layers=self.n_layers,
            batch_size=batch_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_len,
            device=device,
            dtype=dtype,
        )

    def decode_step(
        self,
        target_last: torch.Tensor,
        memory: torch.Tensor,
        kv_cache: KVCache,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ):
        x = self.pos_enc(self.target_embed(target_last), start_pos=kv_cache.cur_len)

        for layer_idx, layer in enumerate(self.decoders):
            x = layer.incremental_forward(
                x=x,
                memory=memory,
                layer_idx=layer_idx,
                kv_cache=kv_cache,
                memory_key_padding_mask=memory_key_padding_mask
            )

        logits = self.out(x)
        kv_cache.advance(x.size(1))
        return logits

    def forward(self, src: torch.Tensor, target_in: torch.Tensor) -> torch.Tensor:
        memory, source_key_padding_mask = self.encode(src)
        logits = self.decode(
            target_in=target_in,
            memory=memory,
            memory_key_padding_mask=source_key_padding_mask
        )
        return logits