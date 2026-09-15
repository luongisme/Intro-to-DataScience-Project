from pathlib import Path
import random

import numpy as np


def ensure_directory(path):
    """Create a directory if it does not exist and return Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
