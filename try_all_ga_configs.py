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

All results are stored in an dictionary for later analysis
The dictionary takes the form {(selection_method, co_rate, mut_rate): population_history}
where we use the strings 't' or 'r' to encode selection_method
"""
import numpy as np
import sys
import pickle

from typing import Dict, Tuple
from numpy.typing import NDArray
from GA import run_GA_training, tournament_selection, roulette_selection

SELECTION_METHODS = [tournament_selection, roulette_selection]

CO_RATES = [0.1, 0.2, 0.4, 0.6, 0.8]

# Discrete values to save on computation time
MUT_RATES = [0.01, 0.02, 0.05, 0.1, 0.15]


def try_all_permutations() -> Dict[Tuple[str, float, float], NDArray[np.float64]]:
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
                                generations = run_GA_training(
                                    population_size=100,
                                    num_generations=50,
                                    crossover_rate=crossover_rate,
                                    mutation_rate=mutation_rate,
                                    selection_method=selection_method
                                )
                                key = (selection_key, crossover_rate, mutation_rate)
                                all_permuations[key] = generations
        return all_permuations

def main() -> int:
    data = try_all_permutations()

    with open("all_permutations.pkl", "wb") as file:
        pickle.dump(data, file)

    return 0

if __name__ == "__main__":
    sys.exit(main())
