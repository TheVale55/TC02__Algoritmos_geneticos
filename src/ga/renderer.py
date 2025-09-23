from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw
from .chromosome import Individual

def _circle_bbox(cx_px: float, cy_px: float, r_px: float) -> tuple[int,int,int,int]:
    left = int(round(cx_px - r_px))
    top = int(round(cy_px - r_px))
    right = int(round(cx_px + r_px))
    bottom = int(round(cy_px + r_px))
    return left, top, right, bottom

def render_individual(ind: Individual) -> np.ndarray:
    """
    Renderiza un individuo a un ndarray float32 en [0,1], shape (H,W,3).
    """
    W, H = ind.width, ind.height
    # Canvas transparente RGBA
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    for c in ind.circles:
        cx_px = c.cx * (W - 1)
        cy_px = c.cy * (H - 1)
        r_px = c.r * float(min(W, H))
        bbox = _circle_bbox(cx_px, cy_px, r_px)

        # Capa de un solo círculo con su alpha
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer, "RGBA")

        col = (
            int(round(c.rC * 255)),
            int(round(c.gC * 255)),
            int(round(c.bC * 255)),
            int(round(c.a * 255)),
        )
        draw.ellipse(bbox, fill=col)
        canvas = Image.alpha_composite(canvas, layer)

    # Convertir a RGB sobre fondo blanco para evitar negros por alpha 0.
    bg = Image.new("RGB", (W, H), (255, 255, 255))
    bg.paste(canvas, mask=canvas.split()[-1])  # usa alpha de canvas
    arr = np.asarray(bg).astype(np.float32) / 255.0
    return arr
