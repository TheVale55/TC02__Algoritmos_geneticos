from __future__ import annotations
import numpy as np
from .renderer import render_individual
from .chromosome import Individual

def mse(a: np.ndarray, b: np.ndarray) -> float:
    """
    MSE sobre (H,W,3) en [0,1].
    """
    assert a.shape == b.shape and a.ndim == 3 and a.dtype == np.float32 and b.dtype == np.float32
    diff = a - b
    return float(np.mean(diff * diff))

def fitness_mse(ind: Individual, target_rgb01: np.ndarray) -> float:
    """
    Devuelve un score para MAXIMIZAR.
    """
    pred = render_individual(ind)
    loss = mse(pred, target_rgb01)
    return -loss  
