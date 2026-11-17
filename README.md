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

The project is written in Python. The supplied MATLAB renderer was translated almost line for line, so MATLAB is not needed.

To set up an environment with the correct requirements, please use the provided `./init.sh` script.
This script will create a virtual environment and install all required packages into said environment automatically.
Before running please ensure that you have python3 and pip installed.
```bash
./init.sh
source venv/bin/activate
python -m GA.GA                                              # one training run -> recent_run.pkl
python -m GA.render_gait recent_run.pkl                      # animate the best gait
python -m GA.render_gait recent_run.pkl --save out.gif --stride 3
    # --stride 3 = every 3rd frame (faster gif)
python -m GA.try_all_ga_configs                              # parameter grid
python -m pytest test/                                       # unit tests
```

| File | Contents |
| --- | --- |
| `GA/GA.py` | Encoding constants, task keyframe poses, seeding, selection, crossover, mutation, the training loop |
| `GA/fitness_function.py` | All penalties and rewards, and the combined fitness |
| `GA/kinematics.py` | Shared body/leg geometry and forward kinematics (plot + overlap/step scoring) |
| `GA/try_all_ga_configs.py` | Grid experiment over selection method, crossover rate, mutation rate and step size |
| `GA/plot_spider_pose.py` | 3D pose renderer (uses `kinematics.py`) |
| `GA/render_gait.py` | Animates a saved gait, interactively or to a file |
| `test/` | Unit tests for the operators and every fitness term |

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

**Why real-valued genes.** Joint angles are continuous, so each gene holds an angle directly rather than a binary code. Heinen and Osório (2006a), Vice, Sukthankar and Douglas (2022) and Oliveira et al. (2011) all use real-valued genes for gait optimisation.

**Angle conventions.** The target poses come from the supplied Webots controller, converted into the renderer's convention: mirrored legs have opposite hip yaw and equal thigh pitch, and the knee sign is negated, because the renderer's positive tibia pitch rotates the foot up where the controller's bends it down. (With the controller's sign the rendered feet floated at body height; with the negated sign they plant about 0.9 body units below the body.)

**Closest precedent.** Manglik, Gupta and Bhanot (2016) also encode a gait as a sequence of steps, each holding three angles for every leg: 18 values per step for a six-legged robot. Our encoding takes the same approach to 8 legs and 300 frames, with continuous rather than quantised angles.

## 3. Constraint handling

The single most important design decision: **the genetic operators never repair their output.** No clamping to joint limits, no smoothing of mutations into neighbouring frames, no pinning of the first and last poses. Crossover and mutation produce whatever they produce — including invalid angles — and the fitness function prices every form of misbehaviour:

- an out-of-range angle costs in proportion to *how far* out of range it is;
- crossed (overlapping) legs cost in proportion to how deeply they cross;
- jitter costs through the acceleration penalty;
- missing the required start, middle and end poses costs through the keyframe penalty.

Selection then removes bad candidates. This keeps a fitness gradient alive that repair would destroy: a candidate 1° outside a limit is measurably better than one 50° outside, so evolution can walk back toward validity instead of being silently teleported onto the boundary. It also means the GA itself — not hand-written correction code — is responsible for every property of the result. Manglik, Gupta and Bhanot (2016) reject invalid offspring outright; we score them instead, because in a 7,200-gene search space early rejection would discard almost every candidate and all the information in them.

## 4. Initial population

Half of the initial population (`seeded_fraction = 0.5`) is **seeded** with sensible starting candidates; the other half is uniform random within the joint limits. Seeding the start is legitimate because it only chooses where the search begins — everything after generation 0 is decided by the operators, the fitness function and selection. Vice, Sukthankar and Douglas (2022) seed the same way, starting from mutations of a hand-tuned gait.

A seed gait (`sinusoidal_seed_gait`) is built from:

- a smoothstep hip sweep through the three required keyframe poses (SPREAD → CLOSE → SPREAD);
- half-rectified sinusoidal thigh/knee lifts, \( \max(\sin \varphi, 0)^2 \), with adjacent legs in opposite phase (a tetrapod rhythm), a random whole number of step cycles (3–6), random per-leg amplitudes and phase jitter;
- small Gaussian noise over every gene so no two seeds are identical.

The sinusoid appears **only here**. It is not a fitness target: evolution is free to keep, distort or abandon the rhythm, and the uniform random half of the population preserves unbiased diversity (it scores around −25 initially and is culled within a few generations, which also demonstrates that selection works).

## 5. Fitness function

Fitness is a base score plus a movement reward minus weighted penalties; each helper in `fitness_function.py` is unit-testable on its own and `fitness_breakdown` reports every term's contribution.

\[ F = 1 + 0.3\,R_{\text{move}} + R_{\text{step}} - \sum_i \frac{w_i}{s_i} P_i \]

Every raw measure is normalised by the joint ranges (or the legs' angular spacing), and each term's *scale* \(s_i\) states how much raw error costs its full weight \(w_i\):

| Term | Measures | Weight / scale |
| --- | --- | --- |
| `step_reward` | Foot kinematics of the renderer. Credits (i) every leg lifting a visible height, (ii) the two tetrapod groups stepping in antiphase, (iii) hip yaw while the foot is airborne. A hip-only spread/close scores ~0 because the feet never leave the ground. | +1.0 max |
| `movement_reward` | Mean per-joint credit \( \min(\sigma_j / 0.15, 1) \). Makes a stuck joint directly measurable. | +0.3 max |
| `joint_limit_penalty` | RMS range-normalised violation beyond the joint limits. Replaces clamping. | 1.0 / 0.02 |
| `leg_overlap_penalty` | Legs' headings (socket heading + hip yaw) must stay in order around the body. | 0.8 / 0.05 |
| `target_pose_penalty` | RMS range-normalised error at frames 0, 150, 299 against SPREAD, CLOSE, SPREAD. | 1.0 / 0.05 |
| `acceleration_penalty` | RMS range-normalised second difference. Smooth 4-cycle stepping is cheap; mutation spikes are not. | 0.25 / 0.01 |
| `symmetry_penalty` | RMS range-normalised error between each left leg and its mirrored right partner. | 0.3 / 0.10 |
| `couple_penalty` | RMS range-normalised thigh/knee error between ipsilateral couples (L1-L3, L2-L4, R1-R3, R2-R4). Hips are excluded: front and rear must yaw inward from opposite sides to hit CLOSE. | 0.5 / 0.08 |
| `support_penalty` | Shortage of planted feet below 3 per frame (a pose that would tip over). | 0.6 / 0.25 |

The best possible score is 2.3; scores can go negative for badly misbehaving gaits. Penalty-based constraint handling follows the standard GA practice of converting constraints into graded costs rather than hard rejections (Raheem and Khaleel, 2015, note the same squared-penalty approach for stability margins). Heinen and Osório (2006a) likewise score how many feet are on the ground rather than assuming a gait from joint motion alone.

## 6. Selection

We compare two stochastic methods, roulette-wheel selection and tournament selection, and keep the better one on the evidence of the grid experiment. In both, every individual has some chance of being chosen, which preserves diversity better than truncation.

**Roulette wheel** chooses with probability proportional to fitness. Because penalty-based fitness can be negative, the weights are shifted by the minimum before normalising (Heinen and Osório (2006a) rescaled similarly). **Tournament selection** picks the fittest of `tournament_size` random candidates (default 2); the size tunes selection pressure and negative fitness needs no special handling. Oliveira et al. (2011) used tournament selection.

One **elite** (the best individual of each generation) is copied unchanged into the next, so crossover and mutation cannot lose the best discovered gait; Heinen and Osório (2006a) and Wang et al. (2021) keep an elite the same way.

## 7. Crossover

We use **uniform whole-leg crossover**: with probability \(p_c\) a parent pair crosses, and each of the 8 legs is independently swapped (p = 0.5) between the two children, taking that leg's entire 300-frame trajectory — all three joints — from one parent.

Treating a whole leg as the chromosome block follows Manglik, Gupta and Bhanot (2016), who swap complete legs so children inherit coordinated movements rather than scattered angles. It also introduces no frame-to-frame discontinuities inside a leg, which matters because nothing downstream will smooth them away. The offspring are not clamped or repaired.

## 8. Mutation

Mutation is **pure additive Gaussian**: each gene independently mutates with probability \(p_m\), and a mutated gene has a step drawn from \( \mathcal{N}(0, \sigma^2) \) added to it. This is the standard real-valued operator (Heinen and Osório, 2006b; Vice, Sukthankar and Douglas, 2022).

Deliberately absent:

- **no clamping** — a step may leave the valid range; the joint-limit penalty prices it;
- **no smoothing or blending** — a mutation is not averaged with neighbouring frames and earlier poses are never adjusted to ease it in; the acceleration penalty and selection decide whether the spike survives;
- **no protected frames** — the first and last frames mutate like any other; the keyframe penalty holds them near the required poses.

Because every mutation lands as a raw spike, the useful mutation rate is low: at \(p_m = 0.002\) a child carries ~14 spikes, few enough that lucky beneficial ones can outweigh their jitter cost. Larger rates (we measured \(p_m = 0.01\), ~72 spikes per child) drown every child in acceleration penalties and the best fitness never improves past the best seed.

## 9. Parameters and experiment design

| Parameter | Value(s) | Basis |
| --- | --- | --- |
| Population size | 60 (single runs), 100 (grid) | Range used by Heinen and Osório (2006a) and Wang et al. (2021) |
| Generations | 150 (single runs), 50 (grid) | Best fitness plateaus well before 150 at these settings |
| Selection | roulette wheel, tournament | Section 6 |
| Crossover rate \(p_c\) | 0.3, 0.6, 0.9 | Around 0.6, Heinen and Osório (2006a) |
| Mutation rate \(p_m\) | 0.001, 0.002, 0.005 | Brackets the hand-found operating point (section 8) |
| Mutation std \(\sigma\) | 0.005, 0.01, 0.02 rad | Same |
| Seeded fraction | 0.5 | Half sensible seeds, half unbiased random |

`try_all_ga_configs.py` runs the full grid (2 selection methods × 3 × 3 × 3 = 54 configurations) rather than varying one parameter at a time, so interactions between parameters are visible. Each run records best and average fitness per generation and the best gait.

## 10. Results

A representative single run (pop 60, tournament, \(p_c = 0.6\), \(p_m = 0.002\), \(\sigma = 0.01\)):

- The uniform random half of generation 0 scores ≈ −25 and is culled within ~3 generations — selection against jitter, crossed legs and missed keyframes does what clamping used to do, visibly.
- Best fitness improves from 1.004 to **1.046** before plateauing around generation 30–60; the average converges to within 0.005 of the best.
- Every one of the 24 joints moves, with per-joint standard deviations of 0.06–0.25 rad over the gait — no knee sits in a frozen band.
- The best gait holds SPREAD at frame 0, draws the legs into the CLOSE pairs by frame 150 and returns to SPREAD by frame 299, with rhythmic leg lifts in between, and contains no limit violations or leg crossings (`fitness_breakdown` shows both penalties at exactly 0).

<!-- TODO: grid experiment table, fitness-over-generations plots for the best and worst configurations, and renders of frames 0/150/299. -->

## 11. Alternatives we considered and rejected

**Clamping offspring to the joint limits.** Our first implementation clipped every angle after crossover and mutation. Rejected: repair hides constraint handling from the GA, destroys the gradient back toward validity, and silently flattens any gene that drifts to a boundary. Replaced by the joint-limit penalty.

**Smoothing mutations into the gait.** We previously centred mutations on the mean of the neighbouring frames (and, in an experiment, low-pass filtered every child). Rejected: blending a mutation into the movement by adjusting the trajectory is manual correction of what selection should decide; the acceleration penalty now plays that role.

**Pinning the first and last frames to the required poses.** Rejected for the same reason: a hard overwrite is repair. The keyframe penalty holds the endpoints instead, and the endpoint frames remain evolvable genes.

**Tracking a dense 300-frame reference gait in the fitness.** An earlier fitness compared every frame against a hand-designed interpolated trajectory. Rejected: it reduces the GA to copying our design, and the 297 intermediate frames are exactly what evolution should discover. Only the three task keyframes are scored; the hand-designed waveform survives solely as an initial-population seed.

**Training half the gait and mirroring it in time.** Would halve the genome but penalises slower individuals disproportionately before the midpoint, collapsing diversity. We train all 300 frames.

**Copying a single published crossover and mutation rate.** No paper we read justified its values, so we test a range (section 9).

## 12. References

Heinen, M. R. and Osório, F. S. (2006a). Applying genetic algorithms to control gait of physically based simulated robots. In: *2006 IEEE International Conference on Evolutionary Computation*. IEEE. https://doi.org/10.1109/CEC.2006.1688528

Heinen, M. R. and Osório, F. S. (2006b). Gait control generation for physically based simulated robots using genetic algorithms. In: *Advances in Artificial Intelligence – IBERAMIA-SBIA 2006*, Lecture Notes in Artificial Intelligence, vol. 4140. Berlin: Springer, pp. 562–571. https://doi.org/10.1007/11874850_60

Manglik, A., Gupta, K. and Bhanot, S. (2016). Adaptive gait generation for hexapod robot using genetic algorithm. In: *2016 IEEE 1st International Conference on Power Electronics, Intelligent Control and Energy Systems (ICPEICES)*. IEEE, pp. 1–6. https://doi.org/10.1109/ICPEICES.2016.7853681

Oliveira, M., Costa, L., Rocha, A., Santos, C. and Ferreira, M. (2011). Multiobjective optimization of a quadruped robot locomotion using a genetic algorithm. In: *Soft Computing in Industrial Applications*, Advances in Intelligent and Soft Computing, vol. 96. Berlin: Springer, pp. 427–436. https://doi.org/10.1007/978-3-642-20505-7_38

Raheem, F. A. and Khaleel, H. Z. (2015). Hexapod robot static stability enhancement using genetic algorithm. *Al-Khwarizmi Engineering Journal*, 11(4), pp. 44–59.

Vice, J., Sukthankar, G. and Douglas, P. K. (2022). Leveraging evolutionary algorithms for feasible hexapod locomotion across uneven terrain. In: *Proceedings of the 35th International FLAIRS Conference*. https://doi.org/10.48550/arXiv.2203.15948

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
runs evolution and returns the history; library callers choose whether to export.
Its optional `on_generation` callback lets a caller collect fitness and display
live progress without putting logging or file handling inside the algorithm.

Each `run_<id>.npz` archive contains `gaits` with shape `(N, 300, 8, 3)` in
radians, `fitness` and `generation_indices` with shape `(N,)`, and the threshold.
`GAIT_DATA_DIRECTORY` and `FITNESS_SAVE_THRESHOLD` are defined in `GA/GA.py`.
No copying of `recent_run.pkl` is needed; that file remains available for rendering.

