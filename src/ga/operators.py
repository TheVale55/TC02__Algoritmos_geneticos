## src/ga/operators.py

import numpy as np
from typing import List, Tuple
from .chromosome import Individual

# ================
#? SELECCIÓN
# ================

def tournament_selection(
    population: List[Individual],
    fitnesses: List[float],
    k: int = 3,
    rng: np.random.Generator = None
) -> Individual:
    """
    Selección por torneo: elige 'k' individuos al azar y devuelve el de mejor fitness.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    selected_indices = rng.choice(len(population), size=k, replace=False)
    selected_fitnesses = [fitnesses[i] for i in selected_indices]
    best_index_in_tournament = np.argmax(selected_fitnesses)
    winner_index = selected_indices[best_index_in_tournament]
    
    return population[winner_index].copy()

# ================
#? CRUCE (Crossover)
# ================

def uniform_crossover(
    parent1: Individual,
    parent2: Individual,
    rng: np.random.Generator = None
) -> Tuple[Individual, Individual]:
    """
    Cruce uniforme a nivel de gen (círculo).
    Para cada círculo, el hijo toma el círculo completo del padre 1 o del padre 2 (p=0.5).
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_circles = len(parent1.circles)
    child1_circles = []
    child2_circles = []
    
    for i in range(n_circles):
        if rng.random() < 0.5:
            child1_circles.append(parent1.circles[i])
            child2_circles.append(parent2.circles[i])
        else:
            child1_circles.append(parent2.circles[i])
            child2_circles.append(parent1.circles[i])
    
    child1 = Individual(width=parent1.width, height=parent1.height, circles=child1_circles)
    child2 = Individual(width=parent1.width, height=parent1.height, circles=child2_circles)
    
    # Asegurar que están dentro de límites (clamp)
    child1.clamp_()
    child2.clamp_()
    
    return child1, child2

# ================
#? MUTACIÓN
# ================

def mutate(
    individual: Individual,
    mutation_rate: float = 0.2,
    sigma: float = 0.05,
    reset_rate: float = 0.1,  # !Probabilidad de hacer reset en vez de gaussiana
    rng: np.random.Generator = None
) -> Individual:
    """
    Mutación HÍBRIDA: 
    - Con probabilidad 'mutation_rate', se muta un atributo.
    - Si se muta, con probabilidad 'reset_rate' se RESETEA a un valor aleatorio.
    - Si no se resetea, se aplica ruido gaussiano N(0, sigma).
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Convertir a array para operar fácilmente
    arr = individual.as_array()  # shape: (n_circles * 7,)
    n_genes = len(arr)
    
    # Máscara de mutación: qué genes mutar
    mask_mutate = rng.random(size=n_genes) < mutation_rate
    
    for i in range(n_genes):
        if mask_mutate[i]:
            if rng.random() < reset_rate:
                #! RESET: Asignar valor aleatorio nuevo (respetando rangos lógicos)
                if i % 7 == 0 or i % 7 == 1:  # cx, cy → [0,1]
                    arr[i] = rng.random()
                elif i % 7 == 2:  # radio → [0.01, 0.5]
                    arr[i] = rng.uniform(0.01, 0.5)
                elif i % 7 in [3, 4, 5]:  # rC, gC, bC → [0,1]
                    arr[i] = rng.random()
                elif i % 7 == 6:  # alpha → [0.05, 0.85]
                    arr[i] = rng.uniform(0.05, 0.85)
            else:
                # Mutación gaussiana suave
                arr[i] += rng.normal(0, sigma)
    
    # Reconstruir individuo desde el array mutado
    mutated = Individual.from_array(individual.width, individual.height, arr)
    mutated.clamp_()  # Asegurar límites después de mutar
    
    return mutated