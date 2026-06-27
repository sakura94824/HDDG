# utils/seed.py
import os
import random
import numpy as np
import torch

DEFAULT_SEED = 2025

def setup_seed(seed=None):
    """固定随机种子，保证可复现。"""
    if seed is None:
        seed = DEFAULT_SEED
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
