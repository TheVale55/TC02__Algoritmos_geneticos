## experiments/run_experiment.py

import argparse
import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src.utils.io import load_target_rgb01, save_snapshot_rgb01
from src.ga.engine import EvolutionaryEngine
from src.ga.chromosome import Individual
from src.ga.renderer import render_individual

def load_config(config_path: str) -> dict:
    #? Carga configuración desde archivo YAML.
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def plot_fitness_history(history: dict, config_name: str, output_dir: Path):
    #? Grafica fitness máximo y promedio por generación.
    
    plt.figure(figsize=(12, 6))
    plt.plot(history['generation'], history['max_fitness'], label='Max Fitness', linewidth=2)
    plt.plot(history['generation'], history['avg_fitness'], label='Avg Fitness', linewidth=2)
    plt.title(f'Evolución del Fitness - {config_name}')
    plt.xlabel('Generación')
    plt.ylabel('Fitness (1 / (1 + MSE))')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Guardar
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "fitness_history.png", dpi=150)
    plt.close()
    print(f"Gráfica guardada en: {output_dir / 'fitness_history.png'}")

def main():
    parser = argparse.ArgumentParser(description="Ejecuta un experimento de Arte Evolutivo")
    parser.add_argument("--config", type=str, required=True, help="Ruta al archivo de configuración YAML")
    args = parser.parse_args()

    # Cargar configuración
    config = load_config(args.config)
    print(f"\n Ejecutando experimento: {config['name']}\n")

    # Cargar imagen objetivo
    target_image = load_target_rgb01(
        path=config['target_image_path'],
        size=(config['width'], config['height'])
    )
    print(f"Imagen objetivo cargada: {target_image.shape}")

    # Crear y ejecutar motor evolutivo
    engine = EvolutionaryEngine(
        target_image=target_image,
        population_size=config['population_size'],
        n_circles=config['n_circles'],
        generations=config['generations'],
        mutation_rate=config['mutation_rate'],
        mutation_sigma=config['mutation_sigma'],
        tournament_k=config['tournament_k'],
        elite_size=config['elite_size'],
        snapshot_every=config['snapshot_every'],
        seed=config['seed'],
        width=config['width'],
        height=config['height'],
        output_dir=config['output_dir']
    )

    # ¡Ejecutar la evolución!
    final_population, final_fitnesses, history = engine.run()

    # Graficar historia de fitness
    plot_fitness_history(history, config['name'], Path(config['output_dir']))

    # Guardar mejor individuo final
    best_idx = np.argmax(final_fitnesses)
    best_individual = final_population[best_idx]
    best_render = render_individual(best_individual)
    save_snapshot_rgb01(best_render, Path(config['output_dir']) / "final_result.png")

    print(f"\n✅ Experimento completado.")
    print(f"Mejor fitness final: {history['best_fitness']:.6f}")
    print(f"Resultados guardados en: {config['output_dir']}")

if __name__ == "__main__":
    main()