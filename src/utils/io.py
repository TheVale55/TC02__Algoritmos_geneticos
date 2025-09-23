from __future__ import annotations
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple

def load_target_rgb01(path: str | Path, size: Tuple[int,int], bg=(255,255,255)) -> np.ndarray:
    """
    Carga PNG con posible alpha y lo compone sobre un fondo dado (RGB).
    Devuelve float32 en [0,1], shape (H,W,3).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe imagen objetivo: {p}")

    im = Image.open(p).convert("RGBA").resize(size, Image.BICUBIC)

    # Componer sobre fondo (blanco por defecto)
    bg_img = Image.new("RGB", im.size, bg)
    bg_img.paste(im, mask=im.split()[-1])  # usa alpha
    arr = np.asarray(bg_img).astype(np.float32) / 255.0
    return arr
 
def save_snapshot_rgb01(img: np.ndarray, path: str | Path) -> None:
    """
    Guarda imagen float32 en [0,1], shape (H,W,3) como PNG.
    """
    if img.dtype != np.float32:
        raise ValueError("img debe ser float32")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img debe tener shape (H,W,3)")
    if img.min() < 0 or img.max() > 1:
        raise ValueError("img debe estar en rango [0,1]")

    arr_255 = (img * 255).round().astype(np.uint8)
    im = Image.fromarray(arr_255, mode="RGB")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    im.save(p, format="PNG")