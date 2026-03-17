import torch
from typing import Tuple


class KVCache:
    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        device,
        dtype,
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        self.keys = [
            torch.empty(batch_size, num_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.values = [
            torch.empty(batch_size, num_kv_heads, max_seq_len, head_dim, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.cur_len = 0

    def reset(self):
        self.cur_len = 0

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, n_kv, T, Hd = k.shape
        assert B == self.batch_size and n_kv == self.num_kv_heads and Hd == self.head_dim
        end = self.cur_len + T
        if end > self.max_seq_len:
            raise RuntimeError(f"KVCache overflow: need {end}, max_seq_len={self.max_seq_len}")

        self.keys[layer_idx][:, :, self.cur_len:end, :] = k
        self.values[layer_idx][:, :, self.cur_len:end, :] = v

        full_k = self.keys[layer_idx][:, :, :end, :]
        full_v = self.values[layer_idx][:, :, :end, :]
        return full_k, full_v

    def advance(self, n: int):
        self.cur_len += n

    def reorder(self, beam_idx: torch.Tensor):
        for layer in range(self.num_layers):
            self.keys[layer] = self.keys[layer].index_select(0, beam_idx)
            self.values[layer] = self.values[layer].index_select(0, beam_idx)

        self.batch_size = beam_idx.numel()


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)