# Evolving a walking gait for an eight-legged robot with a genetic algorithm

<!-- TODO: module code, group number, member names and student numbers. -->

This project uses a genetic algorithm (GA) to evolve a 300-frame walking gait for an eight-legged spider robot. Each leg has three joints, so every frame is a pose of 24 joint angles.

This README explains each design choice and why we made it, with citations. Comments in the code describe what each part does; the reasons for the design are given here.

## Contents

1. [How to run](#1-how-to-run)
2. [Encoding](#2-encoding)
3. [Constraint handling](#3-constraint-handling)
4. [Initial population](#4-initial-population)
5. [Fitness function](#5-fitness-function)
6. [Selection](#6-selection)
7. [Crossover](#7-crossover)
8. [Mutation](#8-mutation)
9. [Parameters and experiment design](#9-parameters-and-experiment-design)
10. [Results](#10-results)
11. [Alternatives we considered and rejected](#11-alternatives-we-considered-and-rejected)
12. [References](#12-references)

## 1. How to run

Python only. The MATLAB renderer was translated into `GA/kinematics.py` / `GA/plot_spider_pose.py`.

```bash
./init.sh
source venv/bin/activate
python -m GA.GA                                 # train; writes recent_run.pkl and maybe NN/data/
python -m GA.render_gait recent_run.pkl         # animate the best gait
python -m GA.try_all_ga_configs                 # parameter grid
```

| File | Role |
| --- | --- |
| `GA/GA.py` | Encoding constants, task keyframe poses, seeding, selection, crossover, mutation, the training loop |
| `GA/fitness_function.py` | All penalties and rewards, and the combined fitness |
| `GA/kinematics.py` | Shared body/leg geometry and forward kinematics (plot + overlap/step scoring) |
| `GA/try_all_ga_configs.py` | Grid experiment over selection method, crossover rate, mutation rate and step size |
| `GA/plot_spider_pose.py` | 3D pose renderer (uses `kinematics.py`) |
| `GA/render_gait.py` | Animates a saved gait, interactively or to a file |
| `test/` | Unit tests for the operators and every fitness term |

### Run a GA
- Edit the control varables in the main function of GA.py to the desired value.
- Run `python -m GA.GA` to run the genetic algorithm with these control variables.
  - This will save the gait with the best fitness from the last generation to `recent_run.pkl`

### Render a gait
Any saved gait can be rendered using the rendering script.
Just run `python -m GA.render_gait path/to/saved/gait.pkl`

For example, if you want to render the saved gait from the last GA.py run, just run `python -m GA.render_gait recent_run.pkl`

### Run the experiment script
For testing, we used a script called `./GA/try_all_ga_configs.py` to experiment with different permutations of the control variables ("all GA configs").
If you would like to run this yourself, just run `python -m GA.try_all_ga_configs`

### Train the next-pose neural network

The GA script's `main()` saves every distinct individual with fitness strictly
above `FITNESS_SAVE_THRESHOLD` (set in `GA/GA.py`) to `NN/data/`. This includes
all generations, from the initial population through the final generation.
Repeated elite gaits are saved once. Archives are written after all generations
finish. The script immediately prints the configured threshold and absolute
destination, then reports each generation's fitness and qualifying count as it
runs. If no individual exceeds the threshold, it prints that no NN data was
saved; this is an ordinary outcome, not an error. The configuration experiment
also explicitly exports each run's qualifying gaits. `run_GA()` itself only
runs evolution and returns the history; library callers choose whether to export.
Its optional `on_generation` callback lets a caller collect fitness and display
live progress without putting logging or file handling inside the algorithm.

Each `run_<id>.npz` archive contains `gaits` with shape `(N, 300, 8, 3)` in
radians, `fitness` and `generation_indices` with shape `(N,)`, and the threshold.
`GAIT_DATA_DIRECTORY` and `FITNESS_SAVE_THRESHOLD` are defined in `GA/GA.py`.
No copying of `recent_run.pkl` is needed; that file remains available for rendering.

Run the GA independently several times, then train the network from the
repository root with the environment activated:

```bash
python -m GA.GA
# Repeat GA runs to collect at least two independent runs with distinct gaits.
python -m NN.nn_template --output-dir nn_results --epochs 100
```

Training prints the training/test gait counts and number of transitions per
epoch before starting, then reports `Epoch 1/100: training MSE=...` after each
completed epoch. MSE is averaged over the scaled joint angles and training
examples as they are processed. An epoch visits every training transition, so
larger datasets take longer before the first epoch report appears. Progress
printing belongs to `main()`; library callers can supply an `on_epoch`
callback to `train_network()` or `NeuralNetwork.train_nn()`.

The NN reads `NN/data/` by default; an optional positional directory overrides
it. It removes exact duplicates across archives and shuffles/splits complete
runs 80:20. All remaining individuals from a run stay in one split, so the
proportions of individual gaits can differ from 80:20. Use substantially more
than two independent runs for meaningful evaluation. Runs sharing an initial
population or random seed should not be treated as independent experiments.

The default network is **25 → 40 → 24**. Inputs are 24 current joint angles,
followed by the current frame divided by 299; targets are the next 24 angles.
Each 300-frame gait supplies 299 transitions. Sigmoid outputs are between zero
and one, so the network uses the reversible scale
`(angle - joint_minimum) / (joint_maximum - joint_minimum)`. Stored and generated
angles stay in radians. Out-of-range training angles are rejected when loaded,
rather than silently clipped or repeatedly checked by each processing function.
If the physical limits change, update the GA limits and retrain.

The script reports next-pose error and generates complete held-out sequences by
feeding each prediction back in, starting from the reference's first pose.
It compares against a baseline that keeps the current pose: for single-step
prediction it copies each actual input pose; for complete-sequence generation it
holds the starting pose for all 300 frames. GA fitness, angle errors, and motion
metrics assess quality; joint-limit compliance alone does not demonstrate walking.
Reserve validation runs from the training portion before tuning; do not tune on
the test results. A different seed pose does not guarantee novel, effective motion.

Results are saved in `nn_results/`:

- `model.pkl`: trained parameters and preprocessing metadata.
- `training_loss.png`: mean training squared error by epoch.
- `metrics.json`: next-pose errors, baselines, per-gait metrics, and source archive names.
- `generated_gaits.npy`: generated radians, shape `(test_gaits, 300, 8, 3)`.

To animate the best generated gait, run:

```bash
python -m GA.render_gait nn_results/generated_gaits.npy
```

The renderer scores every gait using the current GA fitness function and opens
the highest-scoring gait in a Matplotlib window at 30 frames per second. It
prints the selected gait's zero-based index and fitness. That index also selects
its entry in `metrics.json` under `rollouts` and `testing_sources`. If you change
the fitness function after training, the newly computed scores may differ from
the recorded metrics. Close the window to stop playback. This command also
accepts `recent_run.pkl` for the existing GA summary workflow.

Use `--help` for learning-rate, hidden-layer, and random-seed options. To generate
from another legal starting pose in Python:

```python
from pathlib import Path
from GA.GA import SPREAD
from NN.nn_template import generate_gait, load_model

network = load_model(Path("nn_results/model.pkl"))
gait = generate_gait(network, SPREAD)  # (300, 8, 3), radians
```

Only load trusted pickle model files. The training archives use NumPy's array
format with pickle disabled. Old manual gait-summary files are not NN inputs.

## 2. Encoding

An individual is a complete gait, not a single pose. The encoding, from smallest to largest:

| Term | In this project | Size |
| --- | --- | --- |
| Allele | The value a gene holds: one angle in radians. | one number |
| Gene | One joint's angle at one frame. | 1 |
| Leg | The three genes for one leg's joints at one frame. | 3 genes |
| Frame (pose) | All eight legs at one moment. | 8 × 3 = 24 genes |
| Chromosome | Every frame in sequence: the full encoded solution the operators act on. | 300 × 24 = 7,200 genes |
| Individual | One member of the population: one chromosome and its fitness. | 1 chromosome |
| Gait | What the chromosome decodes to: the 300-frame motion that is rendered and scored. | 300 poses |

The array layout is `chromosome[frame][leg][joint]` with shape `(300, 8, 3)` and dtype `float64`. Legs are stored in the renderer's order L1, L2, L3, L4, R4, R3, R2, R1; joints are hip (yaw), thigh (pitch), knee (pitch); all angles are radians.

**Why the whole gait.** The task is to find the best walking gait over 300 frames. A gait can only be judged as a sequence, because smoothness and rhythm depend on how poses follow one another. A GA that optimised a single pose could not evaluate either.

**Why real-valued.** Joint angles are continuous (Heinen and Osório, 2006a; Vice, Sukthankar and Douglas, 2022; Oliveira et al., 2011).

**Angle convention.** Target poses come from the Webots controller, mapped into the renderer: mirrored coxae swing opposite ways; femur and tibia share a sign on left/right; tibia is negated relative to the controller so a standing pose plants the feet.

## 4. GA Approach

### Fitness function

The task standing photos are SPREAD (feet fanned), CLOSE (feet drawn in), SPREAD again, at frames 0, 150 and 299. Fitness only soft-nudges those three frames (`target_pose`); frames in between are not scored against a designed path.

Crossover and mutation do not fix bad angles. If a gait leaves the joint limits or crosses its legs, we still score it — the more wrong it is, the lower the fitness (e.g. 1° over a limit costs less than 50° over). That gives selection something to climb. Throwing invalid children away would discard almost everything early on in a 7,200-gene search. Soft penalties of this kind are standard (Raheem and Khaleel, 2015, for stability).

$F = 1 + 0.3R_{\text{move}} + R_{\text{step}} - \sum_i \frac{w_i}{s_i} P_i$

Nominal ceiling is **2.3** (`MAX_FITNESS`); scores may go negative. `fitness_breakdown` reports each weighted term. A raw error equal to scale s_i costs weight w_i. Detail for each penalty/reward is in `GA/fitness_function.py`.
| Term              | What it measures                                                                                                         | Weight / scale |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------ | -------------- |
| `step_reward`     | FK of the feet: visible lift, tetrapod antiphase, hip forward while up and back while planted. A hip-only fan scores ~0. | +1.0 max       |
| `movement_reward` | Mean over 24 joints of \min(\sigma_j / 0.15, 1). Stuck joints are visible in the number.                                 | +0.3 max       |
| `joint_limit`     | RMS range-normalised overflow. Replaces clamping.                                                                        | 1.0 / 0.02     |
| `leg_overlap`     | Socket heading + hip yaw stay in order around the body.                                                                  | 0.8 / 0.05     |
| `target_pose`     | RMS error at frames 0, 150, 299 vs SPREAD/CLOSE/SPREAD. Weak on purpose.                                                 | 0.05 / 0.05    |
| `acceleration`    | RMS second difference.                                                                                                   | 1.0 / 0.01     |
| `symmetry`        | Left/right mirror.                                                                                                       | 0.3 / 0.10     |
| `couple`          | Thigh/knee match on ipsilateral pairs. Hips excluded (front vs rear yaw to hit CLOSE).                                   | 0.5 / 0.08     |
| `support`         | Shortage of planted feet below 3 per frame.                                                                              | 0.6 / 0.25     |


Step and support use the same foot heights as the renderer (Heinen and Osório, 2006, also score feet on the ground rather than trusting joint motion). We do **not** reward closeness to the sinusoidal seed.

### Mutation and crossover

**Selection.** Roulette (fitness shifted because it can be negative; Heinen and Osório, 2006a) and tournament. Tournament size is a real knob (default 2 in the function; `main` uses **7**). Oliveira et al. (2011) used tournament.

**Elite 1.** Best individual copied unchanged (Heinen and Osório, 2006a; Wang et al., 2021). With a *noisy* seed this keeps progress without freezing gen 0.

**Crossover.** Uniform **whole-leg** blocks: with probability p_c a pair crosses, each of eight legs swapped independently, taking that leg’s 300-frame hip/thigh/knee path (Manglik, Gupta and Bhanot, 2016). Nothing downstream will smooth a cut through a leg, so the block is the unit.

**Mutation.** Additive Gaussian: gene flips with p_m, then $\mathcal{N}(0, \sigma^2)$ (Heinen and Osório, 2006b; Vice et al., 2022). No clamp, no neighbour-mean, no protected frames. Low p_m (0.002) is on purpose: each child has on the order of 14 spikes. p_m = 0.01 drowned children in acceleration cost.

### Generation 0

Every individual is `sinusoidal_seed_gait() + N(0, initial_std²)`. The current `initial_std` is in Parameters below. Seeding only chooses where search *begins* (Vice, Sukthankar and Douglas, 2022, start from mutations of a hand-tuned gait).

The seed is:

1. a coxa path through SPREAD → CLOSE → SPREAD, **phased with the lift** (forward in the air, back on the ground), `NUM_STEPS` cycles;
2. half-rectified sine femur/tibia lifts, tetrapod antiphase, same `NUM_STEPS`;
3. no 20-frame fade back to standing — endpoints stay evolvable;
4. independent Gaussian noise on every gene.

After generation 0, every frame is open to crossover and mutation.

### Parameters

Current defaults in `GA.GA.main`:

| Parameter     | `python -m GA.GA`  | Grid (`try_all_ga_configs`) |
| ------------- | ------------------ | --------------------------- |
| Population    | 100                | 100                         |
| Generations   | 500                | 50                          |
| Selection     | tournament, size 7 | tournament and roulette     |
| p_c           | 0.9                | 0.3, 0.6, 0.9               |
| p_m           | 0.002              | 0.001, 0.002, 0.005         |
| \sigma        | 0.005 rad          | 0.005, 0.01, 0.02           |
| `initial_std` | 0.005 rad          | 0.005 (function default)    |


The grid is 2 × 3 × 3 × 3 = 54 configurations, not one-factor-at-a-time. Population sizes sit in the range Heinen and Osório (2006a) and Wang et al. (2021) used. No paper justified a single p_c, p_m pair, which is why we still sweep them.

`run_GA` returns only the last population. Whatever invokes it can pass an `on_generation` hook each generation to plot fitness and save good gaits for Part 2.

## 5. What we had

### First pass (repair the children)

The first GA treated “make it look like a walk” as something the GA should guarantee:

- force (clamp) angles back into range after crossover and mutation;
- smooth each mutation into nearby frames;
- lock frames 0 and 299 to SPREAD;
- score the whole gait against an essentially hand-drawn 300-frame walk.

That reference interpolated SPREAD -> CLOSE -> SPREAD across 300 frames but only moved the **coxa**; femur and tibia stayed fixed. So the GA was never searching for a walk — it was copying a hips-only design. Bad children were fixed before fitness saw them, so selection never had to choose.

### Improvements in second iteration

- **no clamping** after mutation or crossover;
- **no mutation blending**;
- **do not** put a designed walk in fitness as the thing to copy;
- joint constraints belong in fitness as **graded** penalties and rewards;
- whole-leg chromosome blocks are allowed;
- seeding generation 0 is allowed;
- every joint must be **free to evolve** (not locked to a hips-only reference);
- method over appearance.

### Second pass (legal operators, too-good a seed)

We seeded **half** the population with a sine walk and left the rest uniform random. Gen 0’s **average** looked bad because of the random half, but the **best** individuals were already near the fitness ceiling. Selection kept those elites; fitness jumped in the first few generations then plateaued (typical best **1.75 -> 1.76**). An unseeded control (`seeded_fraction = 0`) stayed around **−42**. A hip-only slide scored ~1.08 with step ≈ 0 — kinematics were working; the GA was polishing a seed, not discovering a walk.

### Third pass (where we're at now)

- **every** individual starts as the seed walk plus random jitter on each angle (`initial_std`; see Parameters), not a 50/50 mix of perfect and junk;
- coxa and femur/tibia share `NUM_STEPS` (one swing cycle = one lift) — the previous desync looked like tap-dancing;
- coxa seed is forward while airborne and backward while planted, with no fade back to standing at the ends;
- `step_reward` credits that same forward/back split, not just “the coxa moved while the foot was up”;
- `target_pose` **weight 1.0 -> 0.05** so the last frames are not yanked onto CLOSE;
- `acceleration` **weight 0.25 -> 1.0**;
- shared numbers (limits, SPREAD/CLOSE, …) live in `gait_constants.py` so GA and fitness don’t import each other in a loop;
- good enough gaits (fitness above a threshold) are saved to `NN/data/` for Part 2.

The seed’s lift (sine) and coxa swing are only a start. They are not a fitness target.

## 6. What the runs show

**Current approach (noisy seed of everyone).** With every individual starting near the walk plus noise, longer runs show real improvement rather than an instant ceiling: Charlie’s ~5k-gen check went about **−6.87 -> −1.38** (still shaky); Alae’s shorter check (seed 42, pop 100, 300 gens, current-style settings) was about **1.37 -> 1.45**, ΔF ≈ 0.08 and still climbing, which is why `main` now runs 500 gens. The half-seed control that motivated this change is in §5 (best stuck ~1.75→1.76; unseeded ~−42).

The honest claim: fitness rewards stepping and punishes illegal motion; we start near a walk *with noise*; selection and mutation have room to move — a head start, not a designed walk in the fitness, and not an elite parked on an un-noised sine.

Render: `python -m GA.render_gait recent_run.pkl`. `main` also draws the fitness-history plot.


## 7. References

Heinen, M. R. and Osório, F. S. (2006a). Applying genetic algorithms to control gait of physically based simulated robots. In: *2006 IEEE International Conference on Evolutionary Computation*. IEEE. [https://doi.org/10.1109/CEC.2006.1688528](https://doi.org/10.1109/CEC.2006.1688528)

Heinen, M. R. and Osório, F. S. (2006b). Gait control generation for physically based simulated robots using genetic algorithms. In: *Advances in Artificial Intelligence – IBERAMIA-SBIA 2006*, Lecture Notes in Artificial Intelligence, vol. 4140. Berlin: Springer, pp. 562–571. [https://doi.org/10.1007/11874850_60](https://doi.org/10.1007/11874850_60)

Manglik, A., Gupta, K. and Bhanot, S. (2016). Adaptive gait generation for hexapod robot using genetic algorithm. In: *2016 IEEE 1st International Conference on Power Electronics, Intelligent Control and Energy Systems (ICPEICES)*. IEEE, pp. 1–6. [https://doi.org/10.1109/ICPEICES.2016.7853681](https://doi.org/10.1109/ICPEICES.2016.7853681)

Oliveira, M., Costa, L., Rocha, A., Santos, C. and Ferreira, M. (2011). Multiobjective optimization of a quadruped robot locomotion using a genetic algorithm. In: *Soft Computing in Industrial Applications*, Advances in Intelligent and Soft Computing, vol. 96. Berlin: Springer, pp. 427–436. [https://doi.org/10.1007/978-3-642-20505-7_38](https://doi.org/10.1007/978-3-642-20505-7_38)

Raheem, F. A. and Khaleel, H. Z. (2015). Hexapod robot static stability enhancement using genetic algorithm. *Al-Khwarizmi Engineering Journal*, 11(4), pp. 44–59.

Vice, J., Sukthankar, G. and Douglas, P. K. (2022). Leveraging evolutionary algorithms for feasible hexapod locomotion across uneven terrain. In: *Proceedings of the 35th International FLAIRS Conference*. [https://doi.org/10.48550/arXiv.2203.15948](https://doi.org/10.48550/arXiv.2203.15948)

Wang, B., Cui, X., Sun, J. and Gao, Y. (2021). Parameters optimization of central pattern generators for hexapod robot based on multi-objective genetic algorithm. *International Journal of Advanced Robotic Systems*. https://doi.org/10.1177/17298814211044934

### GA training-data export

The GA script's `main()` saves every distinct individual with fitness strictly
above `FITNESS_SAVE_THRESHOLD` (set in `GA/GA.py`) to `NN/data/`. This includes
all generations, from the initial population through the final generation.
Repeated elite gaits are saved once. Archives are written after all generations
finish. The script immediately prints the configured threshold and absolute
destination, then reports each generation's fitness and qualifying count as it
runs. If no individual exceeds the threshold, it prints that no NN data was
saved; this is an ordinary outcome, not an error. The configuration experiment
also explicitly exports each run's qualifying gaits. `run_GA()` itself only
runs evolution and returns the final population; earlier populations are discarded.
Its optional `on_generation(generation, fitness, population)` callback collects
qualifying gaits and mean/worst/best fitness for plotting, and displays live progress.

Each `run_<id>.npz` archive contains `gaits` with shape `(N, 300, 8, 3)` in
radians, `fitness` and `generation_indices` with shape `(N,)`, and the threshold.
`GAIT_DATA_DIRECTORY` and `FITNESS_SAVE_THRESHOLD` are defined in `GA/GA.py`.
No copying of `recent_run.pkl` is needed; that file remains available for rendering.

