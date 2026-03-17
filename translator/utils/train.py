import torch
from tqdm import tqdm
import sacrebleu

from utils.beam_search import beam_search_decode
from utils.dataset import target_indextotokens
from utils.loaders import ids_to_tokens
from settings.config import DEVICE as device

def train_epoch(model, loader, optimizer, criterion, scheduler=None):
    model.train()
    total_loss = 0.0

    for source, target in tqdm(loader, desc="train", leave=False, disable=True):
        source = source.to(device)
        target = target.to(device)
        target_in = target[:, :-1]
        target_out = target[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        logits = model(source, target_in)
        vocab_size = logits.size(-1)
        loss = criterion(
            logits.reshape(-1, vocab_size),
            target_out.reshape(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate_bleu(model, loader, max_len=60):
    refs, hyps = [], []
    model.eval()
    with torch.no_grad():
        for source, target in tqdm(loader, desc="val", leave=False):
            source = source.to(device)
            target = target.to(device)
            for seq in target[:, 1:].tolist():
                ref_tokens = ids_to_tokens(seq, target_indextotokens)
                refs.append(" ".join(ref_tokens))
            hyp_ids_batch = beam_search_decode(model, source, max_len=max_len)
            for hyp_ids in hyp_ids_batch:
                hyp_tokens = ids_to_tokens(hyp_ids.tolist(), target_indextotokens)
                hyps.append(" ".join(hyp_tokens))

    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="none", force=True).score
    return bleu

def train(model, optimizer, criterion, train_loader, val_loader, scheduler, epochs: int, max_len: int, best_model_path: str = "best_model.pt"):
    best_bleu = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model=model, loader=train_loader, optimizer=optimizer, criterion=criterion, scheduler=scheduler)
        bleu = evaluate_bleu(model, val_loader, max_len=max_len)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}: loss={train_loss:.3f}, BLEU={bleu:.2f}, lr={current_lr:.2e}")
        if bleu > best_bleu:
            best_bleu = bleu
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

        if epoch - best_epoch >= 5:
            print("Улучшений не было 5 эпох, обучение остановлено")
            break

    print(f"Лучший BLEU: {best_bleu:.2f}")

def test(model, test_loader, max_len: int):
    model.load_state_dict(torch.load("best_model.pt", map_location=device))

    all_hyps = []
    for source in tqdm(test_loader, desc="test"):
        hyp_ids_batch = beam_search_decode(model, source, max_len=max_len)
        for hyp_ids in hyp_ids_batch:
            hyp_tokens = ids_to_tokens(hyp_ids.tolist(), target_indextotokens)
            all_hyps.append(" ".join(hyp_tokens))

    with open("test1.de-en.en", "w", encoding="utf8") as f:
        for line in all_hyps:
            f.write(line.strip() + "\n")

    print("Saved test1.de-en.en")