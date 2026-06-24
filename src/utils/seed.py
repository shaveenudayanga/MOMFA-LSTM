import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all relevant RNG seeds for reproducibility.

    Covers: Python, NumPy, PyTorch (CPU + CUDA).
    Call this once at the start of each of the 30 independent runs.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
