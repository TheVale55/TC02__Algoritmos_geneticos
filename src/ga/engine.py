## src/ga/engine.py

# src/ga/engine.py

import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
import time

from .chromosome import Individual
from .operators import tournament_selection, uniform_crossover, mutate
from .fitness import fitness_mse
from .renderer import render_individual
from ..utils.io import save_snapshot_rgb01

class EvolutionaryEngine:
    def __init__(
        self,
        target_image: np.ndarray,
        population_size: int,
        n_circles: int,
        generations: int,
        mutation_rate: float = 0.2,
        mutation_sigma: float = 0.05,
        reset_rate: float = 0.1,
        tournament_k: int = 3,
        elite_size: int = 5,
        snapshot_every: int = 20,
        seed: int = 42,
        width: int = 256,
        height: int = 256,
        output_dir: str = "reports/figures"
    ):
        """
        Inicializa el motor evolutivo.
        
        Args:
            target_image: Imagen objetivo (H, W, 3) float32 [0,1].
            population_size: Tamaño de la población.
            n_circles: Número de círculos por individuo.
            generations: Número total de generaciones.
            mutation_rate: Probabilidad de mutar cada atributo.
            mutation_sigma: Desviación estándar del ruido gaussiano.
            tournament_k: Tamaño del torneo para selección.
            elite_size: Número de individuos elite que pasan sin cambios.
            snapshot_every: Guardar snapshot cada X generaciones.
            seed: Semilla para reproducibilidad.
            width, height: Dimensiones del lienzo.
            output_dir: Carpeta para guardar snapshots.
        """
        self.target = target_image
        self.pop_size = population_size
        self.n_circles = n_circles
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.reset_rate = reset_rate
        self.tournament_k = tournament_k
        self.elite_size = elite_size
        self.snapshot_every = snapshot_every
        self.width = width
        self.height = height
        self.output_dir = Path(output_dir)
        
        # Inicializar generador de números aleatorios
        self.rng = np.random.default_rng(seed)
        
        # Historial de métricas
        self.history = {
            'generation': [],
            'max_fitness': [],
            'avg_fitness': [],
            'best_fitness': float('-inf'),
            'best_individual': None
        }
        
        # Inicializar población
        self.population = [
            Individual.random(width, height, n_circles, self.rng)
            for _ in range(population_size)
        ]
        
        # Evaluar población inicial
        self.fitnesses = self.evaluate_population(self.population)
        self.update_history(0)

    def evaluate_population(self, population: List[Individual]) -> List[float]:
        """
        Evalúa el fitness de toda la población.
        """
        return [fitness_mse(ind, self.target) for ind in population]

    def select_elite(self, population: List[Individual], fitnesses: List[float]) -> List[Individual]:
        """
        Selecciona los 'elite_size' mejores individuos.
        """
        # Obtener índices ordenados por fitness (de mayor a menor)
        sorted_indices = np.argsort(fitnesses)[::-1]  # Descendente
        elite_indices = sorted_indices[:self.elite_size]
        return [population[i].copy() for i in elite_indices]

    def run(self) -> Tuple[List[Individual], List[float], Dict[str, Any]]:
        """
        Ejecuta el algoritmo genético por el número de generaciones especificado.
        
        Returns:
            population, fitnesses, history
        """
        print(f"Iniciando evolución: {self.generations} generaciones...")
        start_time = time.time()
        
        for gen in range(1, self.generations + 1):
            # --- Elitismo ---
            elite = self.select_elite(self.population, self.fitnesses)
            new_population = elite[:]
            
            # --- Generar nueva población ---
            while len(new_population) < self.pop_size:
                # Seleccionar dos padres
                parent1 = tournament_selection(self.population, self.fitnesses, self.tournament_k, self.rng)
                parent2 = tournament_selection(self.population, self.fitnesses, self.tournament_k, self.rng)
                
                # Cruce
                child1, child2 = uniform_crossover(parent1, parent2, self.rng)
                
                # Mutación
                child1 = mutate(child1, self.mutation_rate, self.mutation_sigma, self.reset_rate, self.rng)
                child2 = mutate(child2, self.mutation_rate, self.mutation_sigma, self.reset_rate, self.rng)
                
                # Agregar hijos
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)
            
            # Recortar si excedió (por si el while agregó uno de más)
            self.population = new_population[:self.pop_size]
            
            # Evaluar nueva población
            self.fitnesses = self.evaluate_population(self.population)
            
            # Actualizar historial y mejores
            self.update_history(gen)
            
            # Guardar snapshot
            if gen % self.snapshot_every == 0 or gen == self.generations:
                self.save_snapshot(gen)
            
            # Mostrar progreso
            if gen % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Gen {gen}/{self.generations} | "
                      f"Max Fitness: {self.history['max_fitness'][-1]:.4f} | "
                      f"Avg: {self.history['avg_fitness'][-1]:.4f} | "
                      f"Tiempo: {elapsed:.1f}s")

        print("¡Evolución completada!")
        return self.population, self.fitnesses, self.history

    def update_history(self, generation: int):
        """
        Actualiza el historial de métricas.
        """
        max_fit = max(self.fitnesses)
        avg_fit = sum(self.fitnesses) / len(self.fitnesses)
        
        self.history['generation'].append(generation)
        self.history['max_fitness'].append(max_fit)
        self.history['avg_fitness'].append(avg_fit)
        
        # Actualizar mejor individuo global
        if max_fit > self.history['best_fitness']:
            best_idx = np.argmax(self.fitnesses)
            self.history['best_fitness'] = max_fit
            self.history['best_individual'] = self.population[best_idx].copy()

    def save_snapshot(self, generation: int):
        """
        Guarda la imagen del mejor individuo de la generación actual.
        """
        best_idx = np.argmax(self.fitnesses)
        best_individual = self.population[best_idx]
        
        # Renderizar
        rendered_image = render_individual(best_individual)  # Usa el renderer de tu pareja
        
        # Guardar
        filename = f"gen_{generation:04d}.png"
        save_snapshot_rgb01(rendered_image, self.output_dir / filename)
        
        # También guardar el mejor global
        if self.history['best_individual'] is not None:
            best_global_image = render_individual(self.history['best_individual'])
            save_snapshot_rgb01(best_global_image, self.output_dir / "best_global.png")