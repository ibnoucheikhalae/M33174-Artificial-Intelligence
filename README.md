# Evolving a walking gait for an eight-legged robot with a genetic algorithm

<!-- TODO: module code, group number, member names and student numbers. -->

This project uses a genetic algorithm (GA) to evolve a 300-frame walking gait for an eight-legged spider robot. Each leg has three joints, so every frame is a pose of 24 joint angles.

This README explains each design choice and why we made it, with citations. Comments in the code describe what each part does; the reasons for the design are given here.

## Contents

1. [How to run](#1-how-to-run)
2. [Encoding](#2-encoding)
3. [Initial population](#3-initial-population)
4. [Fitness function](#4-fitness-function)
5. [Selection](#5-selection)
6. [Crossover](#6-crossover)
7. [Mutation](#7-mutation)
8. [Parameters and experiment design](#8-parameters-and-experiment-design)
9. [Results](#9-results)
10. [Alternatives we considered and rejected](#10-alternatives-we-considered-and-rejected)
11. [References](#11-references)

## 1. How to run

The project is written in Python. The supplied MATLAB code was translated almost line for line, so MATLAB is not needed.

<!-- TODO: Python version, dependencies (requirements.txt), and the exact commands to run the experiment and produce the graphs and renders. -->

To set up an environment with the correct requirements, please use the provided `./init.sh` script.
This script will create a virtual environment and install all required packages into said environment automatically.
Before running please ensure that you have python3 and pip installed.
```bash
./init.sh
source venv/bin/activate
```

The code is split into separate functions for initialisation, fitness, selection, crossover, mutation and the experiment loop. This keeps the nested experiment in section 8 manageable and lets each operator be tested on its own.

<!-- TODO: list the modules/files and what each contains. -->

### Run a GA
- Edit the control varables in the main function of GA.py to the desired value.
- Run `python ./GA/GA.py` to run the genetic algorithm with these control variables.
- - This will save the gait with the best fitness from the last generation to `recent_run.pkl`

### Render a gait
Any saved gait can be rendered using the rendering script.
Just run `python ./GA/render_gait.py path/to/saved/gait.pkl`

For example, if you want to render the saved gait from the last GA.py run, just run `python ./GA/render_gait.py recent_run.pkl`

### Run the experiment script
For testing, we used a script called `./GA/try_all_ga_configs.py` to experiment with different permutations of the control variables ("all GA configs").
If you would like to run this yourself, just run `python ./GA/try_all_ga_configs.py`

## 2. Encoding

An individual is a complete gait, not a single pose. The encoding, from smallest to largest:

| Term | In this project | Size |
| --- | --- | --- |
| Allele | The value a gene holds: one angle, e.g. 42.5°. The possible alleles for a gene are its joint's valid range. | one number |
| Gene | One joint's angle at one frame, e.g. the fifth leg's elbow at frame 37. | 1 |
| Leg | The three genes for one leg's joints at one frame. | 3 genes |
| Frame (pose) | All eight legs at one moment. | 8 × 3 = 24 genes |
| Chromosome | Every frame in sequence: the full encoded solution the operators act on. | 300 × 24 = 7,200 genes |
| Individual | One member of the population: one chromosome and its fitness. | 1 chromosome |
| Gait | What the chromosome decodes to: the 300-frame walking motion that is simulated and scored. | 300 poses |

Each individual carries a single chromosome, so here chromosome and genotype mean the same thing. The chromosome is the genotype, what crossover and mutation change. The gait is the phenotype, what fitness judges. Crossover recombines genes between two parents' chromosomes. Mutation changes a gene's allele to a new value within the same range.

```python
# chromosome[frame][leg][joint], shape 300 × 8 × 3
# joints per leg: 0 = shoulder, 1 = elbow, 2 = wrist; indices start at 0
# chromosome[37][4][1] is the gene for the fifth leg's elbow at frame 37; the number stored there is its allele
```

<!-- TODO: confirm the array layout and dtype used in the code, and whether angles are stored in degrees or radians. -->

**Why the whole gait.** The task is to find the best walking gait over 300 frames. A gait can only be judged as a sequence, because smoothness, stability and progress all depend on how poses follow one another. A GA that optimised a single pose could not evaluate any of these.

**Why real-valued genes.** Joint angles are continuous, so each gene holds an angle directly rather than a binary code. Heinen and Osório (2006a), Vice, Sukthankar and Douglas (2022) and Oliveira et al. (2011) all use real-valued genes for gait optimisation. Each gene is limited to its joint's valid range, following Heinen and Osório (2006a), who restricted every gene to the minimum and maximum angle of its joint.

<!-- TODO: the valid range for each of the three joints, and its source. Heinen and Osório (2006a) used hip [−60°, 15°], knee [0°, 120°] and ankle [−90°, 30°] for a dog-like robot; a spider's ranges will differ. -->

**Closest precedent.** Manglik, Gupta and Bhanot (2016) also encode a gait as a sequence of steps, each holding three angles for every leg: 18 values per step for a six-legged robot. They use 12 steps, 216 genes in all. Our encoding takes the same approach to 8 legs and 300 frames. The difference is that Manglik, Gupta and Bhanot limit each leg to five fixed states "to simplify the process of gait generation and minimize the possibility of generation of unstable states", whereas our angles are continuous.

**Why angles rather than gait parameters.** Several papers do not evolve angles at all. They evolve a few parameters of a periodic function and let that function produce the angles. Vice, Sukthankar and Douglas (2022) evolve the amplitude, phase and vertical shift of a cosine for each joint. Oliveira et al. (2011) and Wang et al. (2021) evolve the parameters of oscillator networks. That gives a far smaller search space. We encode the angles directly because the task asks for an encoding of the 24 joint angles, and this lets crossover and mutation act on the angles themselves. The cost is a much larger search space (7,200 genes), which the fitness function and the angle limits must keep in check.

## 3. Initial population

The default population is seeded from the reference gait, with independent Gaussian noise
added to each angle. `initial_std=0.01` means a noise standard deviation of 0.01 radians.
Angles are clamped to their joint limits, and the first and last poses are fixed to SPREAD.

This gives evolution a head start: it refines a noisy reference rather than discovering the
walking cycle from scratch. Crossover averages selected parents, and mutation adds
Gaussian variation around neighbouring frames. These operators use the existing population;
the reference is used for initialization and scoring.

`init_population`, `run_GA_training`, and `try_all_permutations` accept `initial_std`.
Increase it to investigate harder starting populations, or pass `None` for independent
uniform initialization. Use the same initial standard deviation when comparing configurations.

## 4. Fitness function

Fitness is a reward for walking, minus weighted penalties for undesirable behaviour. Each weight reflects how much that criterion matters.

\[ F = R - w_s P_{\text{smooth}} - w_c P_{\text{stability}} - w_f P_{\text{feet}} - w_t P_{\text{target}} \]

<!-- TODO: final formula, the weight values and their justification, and whether terms are normalised before weighting. Smoothness was agreed to weigh more than feet on the ground. -->

### 4.1 Reward: progress in the walking direction

<!-- TODO: define R. Options discussed:
     - signed distance along the walking axis, D = x_final − x_start (Heinen and Osório, 2006a). It is negative when the robot walks backwards, so the penalty for reversing is built in;
     - forward displacement minus sideways displacement, both divided by the number of steps (Manglik, Gupta and Bhanot, 2016), which also rewards walking straight;
     - how closely and how early the gait reaches the frame-150 target pose. -->

### 4.2 Smoothness

A natural gait changes its joint angles gradually. For each joint we take the step sizes, the change in angle between consecutive frames, and penalise how much they vary. The measure is their population standard deviation:

```python
from statistics import pstdev

steps = [b - a for a, b in zip(angles, angles[1:])]
roughness = pstdev(steps)
```

We chose the standard deviation over the mean step size because the mean cannot tell an even gait from a jerky one. A trajectory that stays still and then jumps 300° in one frame can have the same mean step as one that moves in small, even steps. Its standard deviation is much higher. A median-based measure was also considered and dropped.

A standard deviation of zero only means every step is the same size, not that the steps are small. A joint that moved a large, constant amount every frame would score as perfectly smooth. Clamping genes to each joint's valid range bounds this.

<!-- TODO: how roughness is combined across the 24 joints (sum or mean), whether joints are weighted (shoulder > elbow > wrist was proposed), and whether a maximum step size is also enforced. -->

### 4.3 Static stability

The robot is statically stable when its centre of mass lies inside the support polygon formed by the feet on the ground. Heinen and Osório (2006b) and Manglik, Gupta and Bhanot (2016) use the same condition. We penalise a pose whose stability margin \(s\) falls below a required margin \(m\):

\[ P_{\text{stability}} = \max(0,\; m - s)^2 \]

Here \(s\) is positive when the centre of mass is inside the polygon and negative outside it. Squaring the term makes a large violation cost much more than a small one. Raheem and Khaleel (2015) use the stability margin directly as their fitness, and note that other work squares it. We penalise rather than reject unstable poses. Manglik, Gupta and Bhanot (2016) reject unstable offspring outright, but in a population of random 7,200-gene individuals almost every early gait would be rejected.

<!-- TODO: the value of m and how s is computed; whether the penalty is summed or averaged over frames. -->

### 4.4 Feet on the ground

Following Heinen and Osório (2006a), we measure the average number of feet in contact with the ground over the gait:

\[ \mu_P = \frac{1}{N} \sum_{i=1}^{N} b_i, \qquad P_{\text{feet}} = \left(\mu_P - \frac{L}{2}\right)^2 \]

Here \(b_i\) is the number of feet touching the ground at frame \(i\), \(N\) is the number of frames, and \(L = 8\) is the number of legs. The penalty is lowest when half the feet are down on average, which Heinen and Osório (2006a) identify as suiting a trotting gait. It penalises both extremes: gaits that keep every foot planted and only lean forward, and gaits that leave the ground.

<!-- TODO: check that the reference gait keeps about four feet down; if not, change the target from L/2. Define when a foot counts as touching the ground. -->

### 4.5 Target poses

The gait is rewarded for being close to target poses at key frames: the start, the middle (frame 150) and the end.

<!-- TODO: how the targets are defined (from the reference video or the angle formula) and the distance measure. -->

## 5. Selection

We compare two stochastic methods, roulette-wheel selection and tournament selection, and keep the better one on the evidence of the experiment in section 8. In both, every individual has some chance of being chosen. Purely keeping the top few would give everyone else no chance and collapse diversity early.

| | Roulette wheel | Tournament |
| --- | --- | --- |
| Control | Fixed procedure, no settings | Tunable, e.g. tournament size |
| Role here | A consistent baseline | Can be refined to beat the baseline |
| Negative fitness | Needs rescaling first | Works directly |
| Risk | Less control over selection pressure | Tuning can introduce too much variation |

**Roulette wheel** chooses individuals with probability proportional to fitness. It has no settings, which makes it a consistent anchor to compare against. It needs non-negative fitness, and ours can be negative because it is a reward minus penalties. Heinen and Osório (2006a) used roulette-wheel selection with fitness that could be negative, and handled it by rescaling with sigma truncation; we do the same.

<!-- TODO: confirm the rescaling used in the code (sigma truncation or shifting by the minimum). -->

**Tournament selection** picks the fittest of a random subset. Its size controls selection pressure, so it can be tuned; that same freedom can introduce too much variation. It needs no rescaling for negative fitness. Oliveira et al. (2011) used tournament selection, and Vice, Sukthankar and Douglas (2022) used a variant that takes the fittest individual from each half of the population.

<!-- TODO: tournament size and its justification. -->
<!-- TODO: elitism — whether the best individuals are copied unchanged into the next generation. Heinen and Osório (2006a) and Wang et al. (2021) keep an elite; Vice, Sukthankar and Douglas (2022) do not. -->

## 6. Crossover

<!-- TODO: confirm the crossover operator and the definition of the rate below against the code. -->

We use arithmetic crossover: with probability \(p_c\), a parent pair produces two weighted averages, using one random weight \(a\) between 0 and 1. The children are \(A+a(B-A)\) and \(B-a(B-A)\). The same weight applies across the gait, so crossover does not create abrupt switches between parents at adjacent frames. This is a standard operator for real-valued GAs; see [MATLAB's arithmetic crossover documentation](https://www.mathworks.com/help/gads/genetic-algorithm-options.html). Uncrossed pairs and an unpaired final individual are copied. Joint limits and common fixed endpoints are preserved. The rate now applies to a parent pair, rather than to each angle independently.

Heinen and Osório (2006a) used \(p_c = 0.6\) but gave no justification, and none of the other papers we read justified theirs. We therefore test \(p_c \in \{0.2, 0.4, 0.6, 0.8\}\) rather than copying one value (section 8).

**Alternative: crossover by leg.** Manglik, Gupta and Bhanot (2016) swap whole legs: randomly chosen legs take their movement, across the entire gait, from the other parent. Children inherit complete leg trajectories rather than scattered angles, which suits our 300 × 8 × 3 layout.

<!-- TODO: say whether leg-wise crossover was implemented or compared, and why it was or was not chosen. -->

## 7. Mutation

Mutation selects individual angles with probability \(p_m\). For a selected interior angle, it uses \(\theta'_t=(\theta_{t-1}+\theta_{t+1})/2+\epsilon\), where \(\epsilon\) is Gaussian noise with standard deviation `mutation_std` in radians. Both neighbours are read from the original individual, then results are clamped to the joint limits. Endpoint poses remain fixed.

This is a deliberate local continuity bias: nearby frames usually need similar angles. It uses the current gait, and does not prescribe a cycle or use the target to choose a mutation. The Gaussian variation is independent for each selected angle; fitness selects among the resulting individuals. `mutation_std` controls variation around the neighbours' mean, so the total change can be larger than that value when a pose is far from its neighbours. With `mutation_std=0`, selected angles still take that mean; `mutation_rate=0` disables mutation.

The mutation rate \(p_m\) is the probability that each gene is mutated. Heinen and Osório (2006a) used \(p_m = 0.05\), Wang et al. (2021) used 0.1, and Vice, Sukthankar and Douglas (2022) used 0.4 per parameter on a much smaller genome. Since the published values disagree and none is justified, we test \(p_m \in \{0.01, 0.02, 0.05, 0.10\}\).

<!-- TODO: the standard deviation of the Gaussian step and its justification. -->

## 8. Parameters and experiment design

| Parameter | Value(s) | Basis |
| --- | --- | --- |
| Population size | 50, 100 | 50: Heinen and Osório (2006a), Oliveira et al. (2011), Manglik, Gupta and Bhanot (2016); 100: Wang et al. (2021) |
| Generations | <!-- TODO --> | Heinen and Osório (2006a): up to 100; Manglik, Gupta and Bhanot (2016): 40 |
| Selection | roulette wheel, tournament | Section 5 |
| Crossover rate \(p_c\) | 0.2, 0.4, 0.6, 0.8 | Around 0.6, Heinen and Osório (2006a) |
| Mutation rate \(p_m\) | 0.01, 0.02, 0.05, 0.10 | Around 0.05, Heinen and Osório (2006a) |
| Frames per individual | 300 | Task specification |

**Population size.** The papers report sizes from 50 (Heinen and Osório, 2006a) to 200 (Vice, Sukthankar and Douglas, 2022), and none explains its choice. Manglik, Gupta and Bhanot (2016) tested "a number of combinations for population size and number of generations", and found 50 over 40 generations gave "promising results". We follow the same approach and test 50 and 100. A larger population keeps more variety and lowers selection pressure, at the cost of more fitness evaluations per generation.

**A full grid, not one parameter at a time.** For each selection method, every crossover rate is combined with every mutation rate:

\[ 2 \text{ selection methods} \times 4 \text{ crossover rates} \times 4 \text{ mutation rates} = 32 \text{ configurations} \]

Changing two parameters at once makes it impossible to tell which caused a change in fitness. Varying one while holding the others fixed misses interactions between them. The full grid avoids both. For each configuration we record the average and best fitness, pick the best configuration from that table, and plot fitness over generations for both the best and the worst.

<!-- TODO: whether the grid runs at both population sizes (64 runs) or the population is fixed first; the number of repeat runs per configuration with different random seeds, since a single GA run is noisy. -->

## 9. Results

<!-- TODO: fill in from the experiment. -->

| Selection | \(p_c\) | \(p_m\) | Average fitness | Best fitness |
| --- | --- | --- | --- | --- |
| Roulette | 0.2 | 0.01 | | |
| … | | | | |

<!-- TODO: fitness-over-generations graphs for the best and worst configurations. -->
<!-- TODO: plots or 3D renders of spider poses from the best gait, e.g. frames 0, 150 and 300. -->
<!-- TODO: a short discussion — which parameter mattered most, whether tournament beat roulette, and the limitations. -->

## 10. Alternatives we considered and rejected

**Training half the gait and mirroring it.** We considered evolving only frames 0–150 and playing them back in reverse for frames 150–300. The reference gait appears to retrace its own movement, and this would halve the genome. We rejected it because the reference only reaches the middle of its step at about frame 230. Cutting training at frame 150 would penalise slower individuals far more than fast ones, removing them before they could reach the midpoint. That loses diversity in the population, which the GA depends on. We train all 300 frames.

**Generating gaits from the sinusoidal formula.** Rejected because it hard-codes the gait the GA is meant to discover (section 3).

**Copying a single published crossover and mutation rate.** Rejected because no paper justified its values. We test a range instead (section 8).

**A two-stage fitness function,** one stage for a standing pose and one for walking. Raised but not adopted.

<!-- TODO: keep or drop this one depending on the final fitness design. -->

## 11. References

Heinen, M. R. and Osório, F. S. (2006a). Applying genetic algorithms to control gait of physically based simulated robots. In: *2006 IEEE International Conference on Evolutionary Computation*. IEEE. https://doi.org/10.1109/CEC.2006.1688528

Heinen, M. R. and Osório, F. S. (2006b). Gait control generation for physically based simulated robots using genetic algorithms. In: *Advances in Artificial Intelligence – IBERAMIA-SBIA 2006*, Lecture Notes in Artificial Intelligence, vol. 4140. Berlin: Springer, pp. 562–571. https://doi.org/10.1007/11874850_60

Manglik, A., Gupta, K. and Bhanot, S. (2016). Adaptive gait generation for hexapod robot using genetic algorithm. In: *2016 IEEE 1st International Conference on Power Electronics, Intelligent Control and Energy Systems (ICPEICES)*. IEEE, pp. 1–6. https://doi.org/10.1109/ICPEICES.2016.7853681

Oliveira, M., Costa, L., Rocha, A., Santos, C. and Ferreira, M. (2011). Multiobjective optimization of a quadruped robot locomotion using a genetic algorithm. In: *Soft Computing in Industrial Applications*, Advances in Intelligent and Soft Computing, vol. 96. Berlin: Springer, pp. 427–436. https://doi.org/10.1007/978-3-642-20505-7_38

Raheem, F. A. and Khaleel, H. Z. (2015). Hexapod robot static stability enhancement using genetic algorithm. *Al-Khwarizmi Engineering Journal*, 11(4), pp. 44–59.

Vice, J., Sukthankar, G. and Douglas, P. K. (2022). Leveraging evolutionary algorithms for feasible hexapod locomotion across uneven terrain. In: *Proceedings of the 35th International FLAIRS Conference*. https://doi.org/10.48550/arXiv.2203.15948

Wang, B., Cui, X., Sun, J. and Gao, Y. (2021). Parameters optimization of central pattern generators for hexapod robot based on multi-objective genetic algorithm. *International Journal of Advanced Robotic Systems*. https://doi.org/10.1177/17298814211044934
