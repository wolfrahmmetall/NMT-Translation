from dotenv import load_dotenv
import os
from pathlib import Path
import torch

load_dotenv()
PAD_ID = int(os.getenv('PAD_ID'))
UNK_ID = int(os.getenv('UNK_ID'))
SOS_ID = int(os.getenv('SOS_ID'))
EOS_ID = int(os.getenv('EOS_ID'))


BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
LR = float(os.getenv('LEARNING_RATE'))
SUBSET = os.getenv('USE_SUBSET')
SUBSET = int(SUBSET) if SUBSET != "None" else None
EPOCHS = int(os.getenv('EPOCHS'))
WARMUP = int(os.getenv('WARMUP'))

EMB_DIM = int(os.getenv('EMB_DIM'))
MAX_LEN = int(os.getenv('MAX_LEN'))
FF_DIM = int(os.getenv('FF_DIM'))
DROPOUT_RATE = float(os.getenv('DROPOUT_RATE'))
N_HEADS = int(os.getenv('N_HEADS'))
N_LAYERS = int(os.getenv('N_LAYERS'))
NUM_KV_HEADS = int(os.getenv('NUM_KV_HEADS'))
DATA_DIR = os.getenv("DATA_DIR")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

