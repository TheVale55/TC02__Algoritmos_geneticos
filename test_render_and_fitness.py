
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent      
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


from pathlib import Path
import numpy as np
from utils.io import load_target_rgb01, save_snapshot_rgb01
from utils.math_utils import make_rng
from ga.chromosome import Individual
from ga.renderer import render_individual
from ga.fitness import fitness_mse


W = H = 128
target = load_target_rgb01("data/target/hoyuelo.png", (W, H))

rng = make_rng(42)
ind = Individual.random(width=W, height=H, n_circles=100, rng=rng)
img = render_individual(ind)
print("Imagen render shape:", img.shape, img.dtype, img.min(), img.max())
save_snapshot_rgb01(img, "reports/figures/smoke_random.png")

score = fitness_mse(ind, target)
print("Fitness (−MSE):", score)
