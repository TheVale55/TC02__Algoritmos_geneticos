from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

@dataclass
class Circle:
    
    cx: float  # centro x
    cy: float  # centro y
    r: float   # radio relativo (0..1)
    rC: float  # color R (0..1)
    gC: float  # color G (0..1)
    bC: float  # color B (0..1)
    a: float   # alpha (0..1)

    def clamp_(self) -> None:
        self.cx = clamp01(self.cx)
        self.cy = clamp01(self.cy)
        # Limitar radio a [0.01, 0.5] para que no sea imperceptible ni cubra todo
        self.r = float(max(0.01, min(0.5, self.r)))
        self.rC = clamp01(self.rC)
        self.gC = clamp01(self.gC)
        self.bC = clamp01(self.bC)
        # Alpha útil: evitar 0 y 1 extremos
        self.a = float(max(0.05, min(0.85, self.a)))

@dataclass
class Individual:
    width: int
    height: int
    circles: List[Circle] = field(default_factory=list)

    def copy(self) -> "Individual":
        return Individual(
            width=self.width,
            height=self.height,
            circles=[Circle(**c.__dict__) for c in self.circles]
        )

    @staticmethod
    def random(width: int, height: int, n_circles: int, rng: np.random.Generator) -> "Individual":
        circles: List[Circle] = []
        for _ in range(n_circles):
            c = Circle(
                cx=float(rng.random()),
                cy=float(rng.random()),
                r=float(rng.uniform(0.02, 0.25)),
                rC=float(rng.random()),
                gC=float(rng.random()),
                bC=float(rng.random()),
                a=float(rng.uniform(0.1, 0.8)),
            )
            circles.append(c)
        return Individual(width=width, height=height, circles=circles)

    def clamp_(self) -> None:
        for c in self.circles:
            c.clamp_()

    def as_array(self) -> np.ndarray:
        """
        Devuelve un array 1D con los parámetros de todos los círculos.
        Orden: [cx,cy,r,rC,gC,bC,a] por círculo, concatenado.
        """
        arr = []
        for c in self.circles:
            arr.extend([c.cx, c.cy, c.r, c.rC, c.gC, c.bC, c.a])
        return np.array(arr, dtype=np.float32)

    @staticmethod
    def from_array(width: int, height: int, flat: np.ndarray) -> "Individual":
        assert flat.ndim == 1 and flat.size % 7 == 0
        circles = []
        for i in range(0, flat.size, 7):
            cx, cy, r, rC, gC, bC, a = flat[i:i+7].tolist()
            circles.append(Circle(cx, cy, r, rC, gC, bC, a))
        ind = Individual(width=width, height=height, circles=circles)
        ind.clamp_()
        return ind
