"""
File which tries every GA permutation across 3 variables:
    - Selection method (tourney Vs roulette)
    - Crossover rate (range)
    - Mutation rate (range)

This is an experiment to try different combinations of values and guage what gets the best results

The testing takes the following shape:
                                    selection method
                            tourn. /                \ roulette
                                  /                  \
                              co.-rate             co.-rate
         range of co. values  / | | \                  =
                            /
                        mut.-rate
    range of mut. values / | | \
                        /
                    mut.-std
    range of values / | | \

Result summaries are stored in an dictionary for later analysis
The dictionary takes the form {(selection_method, co_rate, mut_rate): population_history}
where we use the strings 't' or 'r' to encode selection_method
"""
import numpy as np
import sys
import pickle

from datetime import datetime
from typing import Dict, Optional, TypedDict
from numpy.typing import NDArray
from GA import run_GA_training, tournament_selection, roulette_selection, evaluate_fitness

SELECTION_METHODS = [tournament_selection, roulette_selection]

CO_RATES = [0.1, 0.2, 0.4, 0.6, 0.8]

# Discrete values to save on computation time
MUT_RATES = [0.01, 0.02, 0.05, 0.1, 0.15]

MUT_STDS = [0.025, 0.05, 0.1, 0.2, 0.4]

ConfigKey = tuple[str, float, float, float]

class RunSummary(TypedDict):
    """
    Per-generation fitness statistics and the best gait from the entire run
    """
    best_fitness: NDArray[np.float64]
    average_fitenss: NDArray[np.float64]
    best_gait: NDArray[np.float64]
    best_generation: int

def summarise_generations(generations: NDArray[np.float64]) -> RunSummary:
    """
    Summarise the run, including generation 0, without retaining the whole 73GB array
    """
    fitness = np.array([evaluate_fitness(population) for population in generations])
    best_fitness = fitness.max(axis=1)
    best_generation = int(best_fitness.argmax())
    best_individual = int(fitness[best_generation].argmax())

    return {
        "best_fitness": best_fitness,
        "average_fitenss": fitness.mean(axis=1),
        "best_gait": generations[best_generation, best_individual].copy(),
        "best_generation": best_generation
    }


def try_all_permutations(seed: int) -> Dict[ConfigKey, RunSummary]:
        """
        Returns a dictionary where the keys are the used settings and the values are the history
        of generations -- shape =
        (num_permutations, num_generations+1, population_size, NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
        """
        all_permuations = {}
        for selection_method in SELECTION_METHODS:
                selection_key = 't' if selection_method.__name__ == 'tournament_selection' else 'r'
                for crossover_rate in CO_RATES:
                        for mutation_rate in MUT_RATES:
                            for mutation_std in MUT_STDS:
                                key = (selection_key, crossover_rate, mutation_rate, mutation_std)
                                print(f"Trying config: {key}")
                                generations = run_GA_training(
                                    population_size=100,
                                    num_generations=50,
                                    crossover_rate=crossover_rate,
                                    mutation_rate=mutation_rate,
                                    mutation_std=mutation_std,
                                    selection_method=selection_method,
                                    pop_seed=seed
                                )
                                summary = summarise_generations(generations)
                                del generations
                                all_permuations[key] = summary
                                print(f"Max final fitness: {summary['best_fitness'][-1]}")
        return all_permuations

def main(file_to_read: Optional[str] = None) -> int:

    # In write mode
    if file_to_read is None:
        seed = int(datetime.now().timestamp() * 1_000_000)
        data = try_all_permutations(seed)

        with open(str(seed) + "-all_permutations.pkl", "wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        return 0

    # In read mode
    with open(file_to_read, "rb") as file:
        data = pickle.load(file)

    best_config = max(
        data,
        key=lambda config: data[config]["best_fitness"].max(),
    )
    best_result = data[best_config]

    print("Best configuration:", best_config)
    print("Best fitness:", best_result["best_fitness"].max())

if __name__ == "__main__":
    sys.exit(main())
