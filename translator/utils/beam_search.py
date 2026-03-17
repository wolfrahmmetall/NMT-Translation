import torch
import torch.nn.functional as F

from settings.config import SOS_ID, EOS_ID, PAD_ID
from settings.config import DEVICE as device


@torch.no_grad()
def beam_search_decode(
    model,
    source,
    max_len=60,
    beam_size=4,
    length_penalty=0.6,
):
    model.eval()
    source = source.to(device)

    memory, source_key_padding_mask = model.encode(source)
    B, S, D = memory.shape
    V = model.out.out_features

    ys = torch.full((B, 1), SOS_ID, dtype=torch.long, device=device)
    beam_scores = torch.zeros(B, 1, device=device)
    finished = torch.zeros(B, 1, dtype=torch.bool, device=device)

    kv_cache = model.create_kv_cache(
        batch_size=B,
        device=source.device,
        dtype=memory.dtype,
    )

    logits = model.decode_step(
        target_last=ys[:, -1:],
        memory=memory,
        kv_cache=kv_cache,
        memory_key_padding_mask=source_key_padding_mask
    )

    logprobs = F.log_softmax(logits[:, -1, :], dim=-1)

    topk_logprobs, topk_tokens = logprobs.topk(beam_size, dim=-1)
    ys = torch.cat(
        [
            ys.unsqueeze(1).expand(B, beam_size, 1),
            topk_tokens.unsqueeze(-1),
        ],
        dim=-1,
    )

    beam_scores = topk_logprobs
    finished = (topk_tokens == EOS_ID)

    memory = memory.unsqueeze(1).expand(B, beam_size, S, D).contiguous()
    memory = memory.view(B * beam_size, S, D)
    source_key_padding_mask = source_key_padding_mask.unsqueeze(1).expand(B, beam_size, S).contiguous()
    source_key_padding_mask = source_key_padding_mask.view(B * beam_size, S)

    # а кэш после первого шага тоже нужно размножить под бим, потому что
    # сейчас внутри кэша лежит состояние после обработки <sos>
    expand_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, beam_size).reshape(-1)
    kv_cache.reorder(expand_idx)

    for step in range(1, max_len - 1):
        flat_ys_last = ys[:, :, -1].reshape(B * beam_size, 1)

        logits = model.decode_step(
            target_last=flat_ys_last,
            memory=memory,
            kv_cache=kv_cache,
            memory_key_padding_mask=source_key_padding_mask
        )

        logprobs = F.log_softmax(logits[:, -1, :], dim=-1).view(B, beam_size, V)

        if finished.any():
            forced = torch.full_like(logprobs, float("-inf"))
            forced[..., EOS_ID] = 0.0
            logprobs = torch.where(finished.unsqueeze(-1), forced, logprobs)

        cand_scores = beam_scores.unsqueeze(-1) + logprobs

        cur_len = ys.size(2) + 1
        if length_penalty > 0.0:
            lp = ((5.0 + cur_len) / 6.0) ** length_penalty
            rank_scores = cand_scores / lp
        else:
            rank_scores = cand_scores

        topk_scores, topk_idx = rank_scores.view(B, beam_size * V).topk(beam_size, dim=-1)
        next_beam_idx = topk_idx // V
        next_token = topk_idx % V

        next_beam_scores = cand_scores.view(B, beam_size * V).gather(1, topk_idx)

        gather_idx = next_beam_idx.unsqueeze(-1).expand(B, beam_size, ys.size(-1))
        ys = ys.gather(1, gather_idx)
        ys = torch.cat([ys, next_token.unsqueeze(-1)], dim=-1)

        prev_finished = finished.gather(1, next_beam_idx)
        finished = prev_finished | (next_token == EOS_ID)

        beam_scores = next_beam_scores

        global_beam_idx = (
            next_beam_idx
            + beam_size * torch.arange(B, device=device).unsqueeze(1)
        ).reshape(-1)

        memory = memory.index_select(0, global_beam_idx)
        source_key_padding_mask = source_key_padding_mask.index_select(0, global_beam_idx)
        kv_cache.reorder(global_beam_idx)

        if torch.all(finished):
            break

    final_len = (ys != PAD_ID).sum(dim=-1)
    if length_penalty > 0.0:
        lp = ((5.0 + final_len.float()) / 6.0) ** length_penalty
        final_scores = beam_scores / lp
    else:
        final_scores = beam_scores

    best_idx = final_scores.argmax(dim=-1)
    best = ys[torch.arange(B, device=device), best_idx]

    return best