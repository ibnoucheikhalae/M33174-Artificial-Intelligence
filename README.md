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
