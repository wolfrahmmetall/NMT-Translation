from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from settings.config import BATCH_SIZE, PAD_ID, EOS_ID, SOS_ID
from utils.dataset import train_ds, val_ds, test_ds

def ids_to_tokens(ids, itos):
    tokens = []
    for idx in ids:
        if idx == PAD_ID:
            break
        if idx == EOS_ID:
            break
        if idx == SOS_ID:
            continue
        tokens.append(itos[idx])
    return tokens

def collate_fn(batch):
    source_tensors = [source_ids for source_ids, _ in batch]
    target_tensors = [target_ids for _, target_ids in batch]
    source_batch = pad_sequence(source_tensors, batch_first=True, padding_value=PAD_ID)
    target_batch = pad_sequence(target_tensors, batch_first=True, padding_value=PAD_ID)

    return source_batch, target_batch

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

def collate_test(batch):
    padded = pad_sequence(batch, batch_first=True, padding_value=PAD_ID)
    return padded

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_test)

