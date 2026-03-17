import numpy as np
import torch
import torch.nn as nn

from models.transformer import NMTTransformer
from settings.config import (
    PAD_ID,
    WARMUP,
    EPOCHS,
    EMB_DIM,
    N_HEADS,
    FF_DIM,
    N_LAYERS,
    DROPOUT_RATE,
    MAX_LEN,
    NUM_KV_HEADS,
    LR,
    DEVICE as device
)
from utils.dataset import source_indextotokens, target_indextotokens
from utils.loaders import train_loader, val_loader, test_loader
from utils.train import train

SEED = 3407

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
# чтобы вот прям уж вот вот вот уж наверняка
torch.backends.cudnn.deterministic = True

model = NMTTransformer(
    source_vocab_size=len(source_indextotokens),
    target_vocab_size=len(target_indextotokens),
    d_model=EMB_DIM,
    num_heads=N_HEADS,
    ff_dim=FF_DIM,
    n_layers=N_LAYERS,
    dropout_rate=DROPOUT_RATE,
    max_len=MAX_LEN,
    num_kv_heads=NUM_KV_HEADS,
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

def lrate(step: int):
    s = step + 1
    return (EMB_DIM ** -0.5) * min(s ** -0.5, s * (WARMUP ** -1.5))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lrate)

train(model, optimizer, criterion, train_loader, val_loader, scheduler, EPOCHS, MAX_LEN)

