import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split, ConcatDataset

from settings.config import UNK_ID, SOS_ID, EOS_ID, MAX_LEN
from utils.prepare_data import (train_source_tokens,
                          train_target_tokens,
                          source_indextotokens,
                          source_tokentoindex,
                          val_source_tokens,
                          val_target_tokens,
                          target_indextotokens, 
                          target_tokentoindex,
                          test_tokens)


class TranslationDataset(Dataset):
    def __init__(self, source_tokens, target_tokens, source_tokentoindex, target_tokentoindex, max_len: int):
        self.source_tokens = source_tokens
        self.target_tokens = target_tokens
        self.source_tokentoindex = source_tokentoindex
        self.target_tokentoindex = target_tokentoindex
        self.max_len = max_len

    def __len__(self):
        return len(self.source_tokens)

    def encode_source(self, sent_tokens):
        ids = [self.source_tokentoindex.get(tok, UNK_ID) for tok in sent_tokens]
        return ids[: self.max_len]

    def encode_target(self, sent_tokens):
        ids = [self.target_tokentoindex.get(tok, UNK_ID) for tok in sent_tokens]
        ids = ids[: self.max_len - 2]
        return [SOS_ID] + ids + [EOS_ID]

    def __getitem__(self, idx):
        source_ids = self.encode_source(self.source_tokens[idx])
        target_ids = self.encode_target(self.target_tokens[idx])
        source_tensor = torch.tensor(source_ids, dtype=torch.long)
        target_tensor = torch.tensor(target_ids, dtype=torch.long)
        return source_tensor, target_tensor
    
train_ds = TranslationDataset(train_source_tokens, train_target_tokens, source_tokentoindex, target_tokentoindex, MAX_LEN)
val_ds = TranslationDataset(val_source_tokens, val_target_tokens, source_tokentoindex, target_tokentoindex, MAX_LEN)

train_ds, val_from_train_ds = random_split(
    dataset=train_ds,
    lengths=[len(train_ds) - 5000, 5000],
    generator=torch.Generator().manual_seed(1337)
)

val_ds = ConcatDataset([val_ds, val_from_train_ds])

class TestDataset(Dataset):
    def __init__(self, token_lists, source_tokentoindex, max_len: int):
        self.tokens = token_lists
        self.source_tokentoindex = source_tokentoindex
        self.max_len = max_len

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        sent = self.tokens[idx]
        ids = [self.source_tokentoindex.get(tok, UNK_ID) for tok in sent]
        ids = ids[: self.max_len]
        return torch.tensor(ids, dtype=torch.long)
    
test_ds = TestDataset(test_tokens, source_tokentoindex, MAX_LEN)