import os

from settings.config import DATA_DIR, SUBSET, PAD_ID, UNK_ID, SOS_ID, EOS_ID


def read_parallel(source_path: str, target_path: str):
    with open(source_path) as source_file, open(target_path) as target_file:
        source_lines = [l.strip().split() for l in source_file]
        target_lines = [l.strip().split() for l in target_file]
    return source_lines, target_lines

def read_mono(path: str):
    with open(path) as file:
        return [l.strip().split() for l in file]
    
train_source_tokens, train_target_tokens = read_parallel(
    os.path.join(DATA_DIR, "train.de-en.de"),
    os.path.join(DATA_DIR, "train.de-en.en"),
)
val_source_tokens, val_target_tokens = read_parallel(
    os.path.join(DATA_DIR, "val.de-en.de"),
    os.path.join(DATA_DIR, "val.de-en.en"),
)

test_tokens = read_mono(os.path.join(DATA_DIR, "test1.de-en.de"))

if SUBSET is not None:
    train_source_tokens = train_source_tokens[:SUBSET]
    train_target_tokens = train_target_tokens[:SUBSET]

print(f"train size: {len(train_source_tokens)}, val size: {len(val_source_tokens)}")

from collections import Counter

def build_vocab(token_seqs, min_freq: int = 1, max_size: int | None = None):
    counter = Counter()
    for sent in token_seqs:
        counter.update(sent)
    sorted_tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    if max_size is not None:
        sorted_tokens = sorted_tokens[:max_size]

    itos = ["<pad>", "<unk>", "<sos>", "<eos>"]
    stoi = {
        "<pad>": PAD_ID,
        "<unk>": UNK_ID,
        "<sos>": SOS_ID,
        "<eos>": EOS_ID,
    }
    for tok, freq in sorted_tokens:
        if freq < min_freq:
            continue
        if tok in stoi:
            continue
        stoi[tok] = len(itos)
        itos.append(tok)
    return stoi, itos

source_tokentoindex, source_indextotokens = build_vocab(train_source_tokens, min_freq=3)
target_tokentoindex, target_indextotokens = build_vocab(train_target_tokens, min_freq=3)

print("source vocab size: ", len(source_indextotokens), "\n", "target vocab size: ", len(target_indextotokens))