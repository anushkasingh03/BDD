"""Utility to seed Python, NumPy, TensorFlow, and PyTorch."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    tf = None  # type: ignore

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if tf is not None:
        tf.random.set_seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
