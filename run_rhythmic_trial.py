"""Reproduce the prior independent experimental GA within the team's repo.

Run from the repository root:
    python run_rhythmic_trial.py

Requires GA.py and the two accompanying files. Does NOT modify GA.py.
Outputs best_gait_rhythmic_no_elitism.npy, best_gait_rhythmic_no_elitism.csv,
        rhythmic_fitness_no_elitism.png. This is a NUMERICAL proxy, not Webots validation.
"""
import csv
from pathlib import Path
import numpy as np
from GA import NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS, MIN_JOINT_ANGLES, MAX_JOINT_ANGLES
from rhythmic_reference import STANDING, seeded_initial_population
from fitness_function_rhythmic import losses

SEED = 33174
POP_SIZE = 56
GENERATIONS = 100
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.80
MUTATION_RATE = 0.018
MUTATION_SD = 0.018
LEG_NAMES = ('L1', 'L2', 'L3', 'L4', 'R4', 'R3', 'R2', 'R1')


def tournament(rng, fitness_loss):
    choices = rng.integers(0, len(fitness_loss), size=TOURNAMENT_SIZE)
    return int(choices[np.argmin(fitness_loss[choices])])


def run_ga():
    rng = np.random.default_rng(SEED)
    pop = seeded_initial_population(POP_SIZE, rng)
    history_best, history_mean = [], []
    best_ever_loss = float("inf")
    best_ever_gait = None
    for gen in range(GENERATIONS + 1):
        fit = losses(pop)
        generation_best_index = int(np.argmin(fit))
        generation_best_loss = float(fit[generation_best_index])
        if generation_best_loss < best_ever_loss:
            best_ever_loss = generation_best_loss
            best_ever_gait = pop[generation_best_index].copy()
        history_best.append(generation_best_loss)
        history_mean.append(float(np.mean(fit)))
        if gen == GENERATIONS:
            break
        # Full generational replacement: no elites are copied unchanged.
        nxt = np.empty_like(pop)
        for j in range(0, POP_SIZE, 2):
            p1 = pop[tournament(rng, fit)]
            p2 = pop[tournament(rng, fit)]
            if rng.random() < CROSSOVER_RATE:
                # One decision per leg/joint, preserved across ALL 300 frames.
                mask = rng.random((1, NUM_LEGS, NUM_LEG_SEGS)) < 0.5
                c1 = np.where(mask, p1, p2)
                c2 = np.where(mask, p2, p1)
            else:
                c1, c2 = p1.copy(), p2.copy()
            for k, child in enumerate((c1, c2)):
                idx = j + k
                if idx >= POP_SIZE:
                    break
                mutation_mask = rng.random(child.shape) < MUTATION_RATE
                child = child + mutation_mask * rng.normal(0., MUTATION_SD, child.shape)
                for _ in range(3):
                    smoothed = child.copy()
                    smoothed[1:-1] = (child[:-2] + 2 * child[1:-1] + child[2:]) / 4.0
                    child = smoothed
                child = np.clip(child, MIN_JOINT_ANGLES, MAX_JOINT_ANGLES)
                child[0] = STANDING
                child[-1] = STANDING
                nxt[idx] = child
        pop = nxt
    # The saved best-ever gait is never fed back into future generations.
    return best_ever_gait, np.array(history_best), np.array(history_mean)


def main():
    best, best_hist, mean_hist = run_ga()
    out = Path(__file__).resolve().parent
    assert best.shape == (NUM_FRAMES, NUM_LEGS, NUM_LEG_SEGS)
    assert np.isfinite(best).all()
    assert np.all(best >= MIN_JOINT_ANGLES) and np.all(best <= MAX_JOINT_ANGLES)
    np.save(out / 'best_gait_rhythmic_no_elitism.npy', best)
    with (out / 'best_gait_rhythmic_no_elitism.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame'] + [f'{leg}_{j}' for leg in LEG_NAMES for j in ('hip', 'thigh', 'knee')])
        for i, pose in enumerate(best):
            writer.writerow([i] + [f'{v:.7f}' for v in pose.reshape(-1)])
    print('Initial best loss:', best_hist[0])
    print('Final generation best loss:', best_hist[-1])
    print('Best-ever loss:', best_hist.min())
    print('Best-ever loss reduction %:', 100 * (best_hist[0] - best_hist.min()) / best_hist[0])
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(best_hist, label='Best loss')
        ax.plot(mean_hist, label='Mean loss')
        ax.set(xlabel='Generation', ylabel='Proxy loss (lower is better)', title='Rhythmic reference GA')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / 'rhythmic_fitness_no_elitism.png', dpi=160)
        plt.close(fig)
    except ImportError:
        print('matplotlib unavailable: skipping plot')
    print('Saved', out / 'best_gait_rhythmic_no_elitism.csv')


if __name__ == '__main__':
    main()
