# test_operators.py

from src.ga.chromosome import Individual
from src.ga.operators import tournament_selection, uniform_crossover, mutate
import numpy as np

rng = np.random.default_rng(42)
ind1 = Individual.random(256, 256, 10, rng)
ind2 = Individual.random(256, 256, 10, rng)

# Probar cruce
child1, child2 = uniform_crossover(ind1, ind2, rng)
print("Crossover OK")

# Probar mutación
mutated = mutate(ind1, mutation_rate=0.5, sigma=0.1, rng=rng)
print("Mutación OK")

# Probar selección por torneo
population = [ind1, ind2, mutated]
fitnesses = [0.1, 0.2, 0.15]
winner = tournament_selection(population, fitnesses, k=2, rng=rng)
print("Selección por torneo OK")