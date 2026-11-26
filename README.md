# GA pt1

## Contents

1. [How to run](#1-how-to-run)
2. [Intro](#2-intro)
3. [Encoding](#3-encoding)
4. [GA Approach](#4-ga-approach)
   - [Fitness function](#fitness-function)
   - [Mutation and crossover](#mutation-and-crossover)
   - [Generation 0](#generation-0)
   - [Parameters](#parameters)
5. [What we had](#5-what-we-had)
6. [What the runs show](#6-what-the-runs-show)
7. [References](#7-references)
8. [Neural network: how it works](#13-neural-network-how-it-works)
9. [Neural network: how we got here](#14-neural-network-how-we-got-here)



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
| `GA/gait_constants.py` | Joint limits, SPREAD/CLOSE, keyframes, couples (shared by GA and fitness) |
| `GA/GA.py` | Seeding, operators, `run_GA`, export of qualifying gaits |
| `GA/fitness_function.py` | Rewards, penalties, combined fitness |
| `GA/kinematics.py` | Body/leg geometry and forward kinematics (from the supplied MATLAB; same robot as the plot) |
| `GA/plot_spider_pose.py` | One 3D pose (from the supplied MATLAB) |
| `GA/render_gait.py` | Animate a saved gait |
| `GA/try_all_ga_configs.py` | Experiment script: tries combinations of selection, crossover rate, mutation rate and step size |

### Train the phase-to-pose neural network

The single training and generation entry point is `NN/NN.py`. It learns one
GA-produced gait from `ga_training_check.npy`, using your coworker's phase workflow.
From the repository root, first run the GA and select a saved individual:

```bash
python -m GA.GA
python - <<'PYTHON'
from pathlib import Path
import numpy as np

# Choose an archive from NN/data; use its highest-fitness saved individual.
archive = sorted(Path("NN/data").glob("run_*.npz"))[-1]
with np.load(archive, allow_pickle=False) as run:
    gait = run["gaits"][np.argmax(run["fitness"])]
np.save("ga_training_check.npy", gait)
PYTHON
python -m NN.NN --train
```

The GA must have exported at least one qualifying gait. You can instead save any
chosen GA gait as `ga_training_check.npy`, shape `(300, 8, 3)`, in radians.
Training runs for 300 epochs, then scores, saves and animates the predicted gait.
To load the saved model and animate without retraining, run `python -m NN.NN`.

The model and its design choices are explained in
[NN: how it works](#13-neural-network-how-it-works). The earlier recursive design
and the measurements behind replacing it are recorded in
[NN: how we got here](#14-neural-network-how-we-got-here).

## 2. Intro

This README is the method write-up: what we tried first, what we ruled out, what that first GA actually did, and what we have now. Doc strings in the source describe *how* a function works.

## 3. Encoding

The brief asks for a **gait**, so smoothness and stepping only make sense across many frames. One individual is therefore the whole 300-frame sequence, not a single pose.


| Term       | Here                               | Size                 |
| ---------- | ---------------------------------- | -------------------- |
| Allele     | One joint angle, radians           | 1 number             |
| Gene       | That joint at one frame            | 1                    |
| Leg        | Coxa, femur, tibia at one frame    | 3                    |
| Frame      | All eight legs                     | 24                   |
| Chromosome | The full gait the operators act on | 300 \times 24 = 7200 |
| Individual | One chromosome plus its fitness    | 1                    |


Layout: `chromosome[frame][leg][joint]`, shape `(300, 8, 3)`, `float64`. One frame is the brief’s 1×24 pose (`[L1a, L1b, L1c, …, R1c]`), stacked over ~300 frames for a gait. Leg order matches the brief/MATLAB model: L1, L2, L3, L4, R4, R3, R2, R1. Joints are innermost → outermost: **coxa** (swings the leg around the body), **femur** (raises the upper leg), **tibia** (bends the lower leg). Angles are radians. (In code comments these are often called hip / thigh / knee — same three joints.)

**Why the whole gait.** A pose-only GA cannot score stepping or the standing photos as a cycle. Manglik, Gupta and Bhanot (2016) encode a sequence of steps, each holding every leg’s three angles; we take that to eight legs and 300 frames.

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

Wang, B., Cui, X., Sun, J., and Gao, Y. (2021). Parameters optimization of central pattern generators for hexapod robot based on multi-objective genetic algorithm. *International Journal of Advanced Robotic Systems*. [https://doi.org/10.1177/17298814211044934](https://doi.org/10.1177/17298814211044934)

## 13. Neural network: how it works

### What the five inputs mean

The network answers: **“At this point in the animation, what should the spider's
24 joint angles be?”** It predicts every pose directly, including frame zero.
It needs no starting pose, and one prediction is never fed into the next.

For frame `f` in `0..299`, let `t = f / 299`. Its five inputs are:

| Input | What it tells the network |
| --- | --- |
| `t` | Progress through the full animation, from 0 to 1 |
| `sin(2πt)`, `cos(2πt)` | Position around one complete cycle |
| `sin(4πt)`, `cos(4πt)` | Position around a cycle that repeats twice |

All five inputs are calculated from the frame number. They are different ways of
describing time, much like elapsed time together with the positions of clock hands.
A sine/cosine pair distinguishes positions that a sine value alone cannot distinguish.
These features offer a useful representation of periodic patterns; their benefit
for this dataset still needs a controlled comparison against time alone.

**Why this encoding?** Tancik et al. (2020) study sinusoidal input mappings for
learning functions of low-dimensional coordinates with MLPs. Their work supports
using Fourier features as input representations; it does not establish our exact
encoding as optimal for gaits. We use the first two harmonics as a compact basis:
one cycle expresses the whole animation's rhythm, while two cycles can describe
variation that repeats twice within it. This is our modelling choice, not a claim
that every joint must move at either frequency. Adding raw `t` retains elapsed
progress and distinguishes the start from the end. Kazemi et al. (2019) similarly
combine non-periodic and periodic time components in Time2Vec, although they learn
the periodic parameters; our encoding keeps them fixed.
([Tancik et al., 2020](https://arxiv.org/abs/2006.10739);
[Kazemi et al., 2019](https://arxiv.org/abs/1907.05321).)

Both circular pairs return to the same position at the endpoints. Their inclusion
does not supply new information beyond `t`; it changes how that information is
presented to the network. The appropriate follow-up is a validation comparison of
`t` alone, `t` plus one circular pair, and our five-feature encoding.

**The sine and cosine values are not the target joint angles.** The target for
frame 75 is the GA gait's actual pose at frame 75. The target for frame 76 is its
actual pose at frame 76. The input features supply useful periodic building blocks;
the learned weights determine how those features map to the evolved gait.

```text
GA evolves a complete 300-frame gait
                 ↓
For every frame: encode time → use the actual GA pose as the target
                 ↓
Train the NN using prediction error against those GA poses
                 ↓
Generate all 300 poses independently from their time features
```

### Architecture and learning

The default network is **5 → 64 → 24**: five inputs, one hidden layer with 64 neurons,
and one output for each joint in leg-major order L1, L2, L3, L4, R4, R3, R2, R1.
It has 1,944 trainable weights and biases. The 64-neuron width is the starting choice
from the phase experiment, not a claim that this is the optimal width.

The original phase implementation in `NN/NN.py` uses sigmoid in both trainable layers. Its output is restricted to 0–1,
whereas the targets contain negative and positive radians. We therefore change
units using the GA model's joint-limit constants:

```text
scaled_angle = (angle - minimum) / (maximum - minimum)
angle = minimum + scaled_angle * (maximum - minimum)
```

For a hip range of -1.57 to +1.57 radians, -1.57 becomes 0, zero becomes 0.5,
and +1.57 becomes 1. This conversion is reversible. It does not fix a joint at
one angle or tell the network which pose to produce; the GA data supplies that.
The limits are assumptions inherited from `GA/gait_constants.py`, not quantities
the network has discovered or independent proof of the robot's physical limits.

**Does this constrain the network? Yes.** Sigmoid followed by inverse scaling
restricts each predicted angle to the configured interval. The generator applies
the inverse scale directly, without an extra clipping step.
Loading rejects training examples outside the interval. Thus this workflow is
supervised approximation of GA gaits within a chosen output range, rather than an
unrestricted search over all angles. This differs deliberately from the GA's
penalty-based search, which permits invalid candidates and scores them.

We retain this representation to use the existing sigmoid network and express the
chosen valid-angle domain directly. It is not compulsory for neural regression.
A linear output layer with MSE can instead predict unrestricted real-valued angles,
with optional standardisation and a penalty or evaluation metric for violations.
Goodfellow, Bengio and Courville (2016), section 6.2.2.1, describe linear outputs
for regression. That alternative would require changing the output activation and
its backpropagation derivative, then comparing results; simply removing scaling
would leave sigmoid unable to represent negative radians.
([Goodfellow et al., 2016](https://www.deeplearningbook.org/contents/mlp.html).)
Files and rendered gaits remain in radians throughout.

Training uses the team's generic `NeuralNetwork` class in `NN/NN.py`: mean squared
error across the 24 scaled outputs, backpropagation, and stochastic gradient descent
with an update after each frame example. The default learning rate is 0.1 and the
default run is 300 epochs. An epoch presents the selected gait in frame order;
reported loss is the mean measured during those updates. Xavier-uniform initialisation
centres the weights with the aim of reducing initial saturation, following
[Glorot and Bengio (2010)](https://proceedings.mlr.press/v9/glorot10a.html).
This does not establish sigmoid as the best hidden activation. Biases start at zero.
Each gait contributes 300 same-frame examples, including both endpoints.

### Evaluation and limitations

This workflow trains on one selected GA gait: its loss measures reconstruction
of that gait, rather than held-out generalisation. It prints the generated gait's
GA fitness and the mean per-joint range of motion over its last 50 frames.
Inspect the loss curve and animation together; low angle error or legal joint
limits alone does not prove effective walking, and GA fitness is not a substitute
for simulator measurements.

A time-only network generates **one trajectory**. It cannot choose between gait
styles or adapt to disturbances. If extended to train on multiple gaits with the
same time inputs, MSE would encourage their mean pose at each frame, which might
be ineffective if their timing or styles differ. Separate models or additional
conditioning would be needed for distinct gait families. Squared-error regression
and its conditional-mean interpretation are discussed by
[Goodfellow et al. (2016)](https://www.deeplearningbook.org/contents/mlp.html).

The periodic features do not force the first and last poses to match, because raw
`t` distinguishes the endpoints. A seamless loop needs explicit evaluation and,
if necessary, a suitable training objective. Before tuning or making claims about
generalisation, reserve independent GA runs for validation and final testing.

### Saved outputs and generation

The workflow writes these files under `nn_results/phase_experiment/`:

- `phase_model.pkl`: architecture, weights, biases, joint limits and phase-format identity.
- `training_loss.png`: mean scaled-angle training MSE by epoch.
- `generated_gaits.npy`: one generated gait, shape `(1, 300, 8, 3)`, in radians.

The existing Python interface uses the same functions as the command-line workflow:

```python
from pathlib import Path
from NN.NN import generate_phase_gait, load_phase_network

network = load_phase_network(Path("nn_results/phase_experiment/phase_model.pkl"))
gait = generate_phase_gait(network)  # (300, 8, 3), radians; no seed pose needed
```

Only load trusted pickle models. Original phase models remain compatible;
25-input recursive models must be retrained. The old recursive template and
standalone experiment scripts have been removed so there is one gait workflow.
The generic `NeuralNetwork` class and the phase functions retain their existing APIs.

To render the saved output separately:

```bash
python -m GA.render_gait nn_results/phase_experiment/generated_gaits.npy
```

The renderer opens a looping animation at a target 30 FPS. Close its window to stop.
It recomputes fitness using the current GA implementation; this may differ from
previously recorded scores. Interactive rendering needs a GUI Matplotlib backend.
`MPLBACKEND=Agg` is suitable for headless training and tests only.

## 14. Neural network: how we got here

### The previous idea: predict the next pose

Our first design used **25 → 40 → 24**: the current 24 joint angles plus normalised
frame number as inputs, and the next pose as the target. It was trained on true
GA poses. To generate 300 frames, we supplied a starting pose and repeatedly fed
each predicted pose back into the network. This seemed useful because it could
condition the next prediction on the current configuration.

The difficulty was the change in inputs at generation time: the network then saw
its own imperfect predictions rather than the true poses used during training.
Errors could influence later predictions. This training/inference mismatch is
also discussed by [Bengio et al. (2015)](https://arxiv.org/abs/1506.03099) for
sequence models. Our network was a feedforward model called recursively, not an
RNN with a learned hidden state, but it encountered the same type of mismatch.

### Why we replaced it for this task

We tested both complete-run train/test assignments using the two supplied GA
archives (3,469 and 2,419 individuals). In each direction we sampled 16 training
gaits and 32 testing gaits, keeping runs separate. Training used learning rate
0.1 and seeds 7, 42 and 123 for 100 epochs; seed 42 continued to 300 epochs.
These are sampled exploratory results, not full-dataset training or independent
confidence intervals. At 300 epochs, the results were:

| Train/test direction | One-step RMSE using true previous poses | Recursive generation RMSE |
| --- | ---: | ---: |
| Run `0314dcd3…` → run `3c48185f…` | 0.01211 rad | 0.13763 rad |
| Run `3c48185f…` → run `0314dcd3…` | 0.01193 rad | 0.13507 rad |

Errors cover frames 1–299 and all joints, excluding the supplied seed pose.
At 100 epochs, recursive error was already 4.4–4.7 times the one-step error across
the six fits; at 300 epochs it was about 11 times larger. Longer training improved
one-step predictions but worsened generation. Supplying a true pose every ten
predictions reduced the 300-epoch generation error by about 81%, supporting
feedback as a substantial source of error. Those true-pose resets were a diagnostic,
not a solution available during independent generation.

We therefore rejected recursive pose prediction **for this fixed-trajectory task**
and chose direct phase-to-pose prediction. A frame prediction no longer depends
on earlier predicted angles, removing that route for cumulative error. The
single-gait phase experiment gave held-out pose RMSE of 0.02784–0.02809 radians.
Its training budget and architecture differed, so this is supporting evidence,
not a controlled proof that phase models always outperform transition models.

We did not conclude that every divergent trajectory was ineffective: some
100-epoch recursive gaits scored above the GA references, while the 300-epoch
ones scored below. Those fitness values used main's scoring function at the time,
which differs from this branch's current scoring function. Nor do these results demonstrate broad generalisation: the
mean training trajectory baseline reached about 0.01046 radians, suggesting the
two runs cover a narrow aligned gait family. More independent runs, a validation
set and simulator measurements are needed before claiming a general controller.

These measurements describe the earlier comparison experiment; the current
phase entry point does not implement the rejected predictor. Removing feedback
does not remove approximation
error, guarantee cyclic continuity or provide response to physical disturbances.

### Sources for the NN design

- Tancik, M. et al. (2020). *Fourier Features Let Networks Learn High Frequency
  Functions in Low Dimensional Domains*. NeurIPS 33.
  [Paper](https://arxiv.org/abs/2006.10739).
- Kazemi, S. M. et al. (2019). *Time2Vec: Learning a Vector Representation of Time*.
  arXiv:1907.05321. [Paper](https://arxiv.org/abs/1907.05321).
- Goodfellow, I., Bengio, Y. and Courville, A. (2016). *Deep Learning*, chapter 6.
  MIT Press. [Chapter](https://www.deeplearningbook.org/contents/mlp.html).
- Glorot, X. and Bengio, Y. (2010). *Understanding the difficulty of training deep
  feedforward neural networks*. AISTATS, PMLR 9, 249–256.
  [Paper](https://proceedings.mlr.press/v9/glorot10a.html).
- Bengio, S., Vinyals, O., Jaitly, N. and Shazeer, N. (2015). *Scheduled Sampling
  for Sequence Prediction with Recurrent Neural Networks*. NeurIPS 28.
  [Paper](https://arxiv.org/abs/1506.03099).
