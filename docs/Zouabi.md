# Bouncing under correlated random excitation

Implements [Zouabi, Scheibert and Perret-Liaudet 2016][paper], *Memory in random bouncing ball dynamics*.
A ball follows ballistic flights above a randomly vibrating plate, with instantaneous restitution, sticking and acceleration-driven release.
Metal runs independent realizations; the same contact equations also execute in double precision on the CPU.
The implemented equations produce correlated bouncing, but the four published velocity distributions remain unmatched.

## Run

```sh
brew install ghostscript
python3 tools/Reproduce.py --method zouabi
```

The workflow recovers Figures 1–2 from pinned author EPS sources and compares both published sticking conventions.
Each curve retains 900,000 flights from 1,000 independent streams after discarding the first 100 flights of each stream.
The paper instead discards the first 100,000 flights of one million successive flights.
Outputs include `Excitation.png`, `Comparison.png`, `Metrics.json` and four plate/ball sonification pairs under `outputs/reproduction/zouabi/`.
Open [the listening page](Listen.html) after generation.
Use `--offline` with cached references and `--keep-diagnostics` to retain trajectories and per-stream histograms.

```sh
build/contactStatistics zouabi OUTPUT [lambda correlation streams flights trace_samples dt relative_sticking max_steps]
```

For the paper's single-stream million-flight sampling, use `streams=1`, `flights=1000000` and a sufficient `max_steps` budget.
For example, a 20-billion-step budget covers the typical high-excitation cases but has much lower GPU occupancy than the default ensemble.
The comparisons below distinguish parallel ensembles from a separate CPU check with the paper's successive-flight sampling.

`rough/Bouncing.h` exposes restitution, correlation time, excitation level, timestep, sticking threshold, seed and ensemble limits.
`AdvanceBouncing` accepts sampled plate motion for CPU stepping; `SimulateBouncing` generates correlated excitation on Metal.
Result counts report whether each stream reached the requested flight count before `MaxSteps`.
Trace channels contain plate height, plate velocity, ball height and ball velocity.

## Equations and source conflicts

The plate uses a second-order filter with unit natural angular frequency, damping `1/τcorr` and an additional first-order cutoff at 10.
A precomputed transition applies RK4 with one independent Gaussian forcing value per interval.
The timestep is 0.005, as in the letter.
Height has unit theoretical variance; gravity follows `Λ = σw² / (g σh)` using the filtered velocity variance.
The extra low-pass filter changes `σw/σh` to `sqrt(10 / (10 + 2/τcorr))`.
The letter's nondimensional equations assume that ratio equals one.
Writing `r = σw / (Ωc σh)`, the definitions instead give `Vnext = -e V + e r T/Λ + (1+e) Wnext`.
The height equation becomes `-r² T²/(2Λ) + r V T + H - Hnext = 0`.
The implementation follows dimensional Equations 2–4, preserving these scale factors.
The thesis profile pseudocode also contains inconsistent output and derivative assignments.
The implementation preserves a displacement signal and its mathematical derivatives.

Impact time solves the intersection of a ballistic parabola with linearly interpolated plate height.
Restitution uses linearly interpolated plate velocity and the incoming ball velocity, following Equations 2–4.
The finite sticking threshold is `5 g dt`.
The default thesis rule uses `abs(v_after - w) < 5 g dt`; the letter rule uses signed laboratory-frame velocity, `v_after < 5 g dt`.
Here `w` is plate velocity at impact.
The letter rule includes every negative post-impact velocity; it does not compare `abs(v_after)`.
Both conventions appear separately in the published comparison.

The [thesis][thesis] contains additional conflicts that prevent a unique literal implementation.

| Operation | Letter | Thesis | Implementation |
| --- | --- | --- | --- |
| Impact time | Parabola/plate intersection, Eq. 4 | Linear ball and plate segments, Eqs. 5-27–29 | Letter |
| Sticking | Signed post-impact velocity | Relative post-impact speed, p. 132; pre-impact speed, Algorithm 4, p. 134 | Both prose rules selectable |
| Release | Acceleration `> -g` | Prose `> -g`; Algorithm 4 `< -g` | Algorithm 4 |
| Velocity statistics | Initial velocity of each flight, including release | Called post-impact velocity; sticking assigns plate velocity | Letter |

At the published restitution of 0.8, Algorithm 4's pre-impact cutoff is equivalent to relative post-impact sticking with `StickFactor=4`.
Thesis Equation 5-28 repeats the same height in both numerator terms and therefore cannot locate the intersection as printed.
Algorithm 4 restores the previous sampled state on release without specifying the corresponding flight time origin.
It also uses the ball-velocity symbol when calculating plate acceleration.
The [2017 MEDYNA follow-up][medyna], proceedings pp. 14–17, repeats the flight equations without resolving these event rules.

A stuck ball releases when plate acceleration is below `-g`, as required by unilateral contact and thesis Algorithm 4.
The prose in both sources prints the opposite inequality.
The contact reaction is `m (g + acceleration)`; release below `-g` prevents a tensile normal reaction.
Release is evaluated on the integration grid.
The published finite sticking threshold approximates inelastic collapse; exact infinite-bounce collapse is outside this method.
Author solver code, excitation realizations and audio examples were not recovered.

## Validation and scope

The consolidated `RoughContactTest` checks ballistic impact time, restitution impulse, nonpenetration, acceleration-driven release and CPU/Metal trajectories.
It also compares the numerical excitation impulse response with the continuous two-stage filter at angular frequencies 0.1, 1 and 10.
The relative magnitude tolerance is 0.02% at the published timestep.
Figure 1's analytical curves match the RK4 filter magnitude to within 0.08% RMS relative error.
The comparison evaluates the FP32 transition coefficients uploaded to Metal.
Those axes say PSD, but the plotted curves equal the square root of the power response.
`Excitation.png` uses the recovered magnitude convention with unit DC gain.
Reproduction checks retain complete flight counts, extracted probability mass and finite trajectories within interpolation and float roundoff bounds.
Per-case metrics report distribution distance, mean velocity, negative-velocity fraction, successive take-off heights, sticking time and plate moments.
Descending pairs compare consecutive take-off heights, including release after sticking.
The first departure has no preceding height and is excluded from the pair count.
Agreement between CPU and GPU validates their numerical correspondence; it does not establish agreement with the authors' unpublished implementation.

Figure 2 uses restitution 0.8 with `(Λ, τcorr)` equal to `(5, 1.11)`, `(0.3, 1.11)`, `(5, 10)` and `(0.3, 10)`.
Integrating the Figure 2 bars gives mean normalized velocities of 3.844, 1.931, 3.224 and 0.850.
The caption instead gives 3.65, 1.93, 3.12 and 0.85.
Metrics preserve both references and compare distributions against the plotted bars.
The figure files are identical in arXiv versions 1 and 2.
Separate FP64 CPU checks use seed 2016 and retain the final 900,000 of one million successive flights, matching the paper's sampling.

| Figure 2 | Published mean | Thesis rule, ensemble | Thesis rule, serial | Letter rule, serial |
| --- | ---: | ---: | ---: | ---: |
| (a) | 3.844 | 3.543 | 3.535 | 3.170 |
| (b) | 1.931 | 2.222 | 2.215 | 1.583 |
| (c) | 3.224 | 3.477 | 3.470 | 2.386 |
| (d) | 0.850 | −0.141 | −0.141 | 0.710 |

The serial checks confirm that the parallel sampling does not explain the discrepancy.
In panel (d), negative take-offs comprise 56.20% under the thesis rule and 1.99% under the letter rule, versus 29.39% in the published bars.
The letter rule has a closer mean and a substantial distribution-shape error.
Counting raw post-impact velocities instead of actual take-offs also fails to reproduce the published distribution.
These checks exclude specific explanations; the cause of the remaining mismatch is unresolved.
A separate timestep comparison retains 1,843,200 flights per case using 1,024 streams of 2,000 flights with the thesis rule.
Halving the timestep changes mean velocities from 3.539, 2.225, 3.475 and −0.142 to 3.538, 2.186, 3.465 and −0.219.
The first and third means are stable; low-excitation cases remain sensitive to the timestep-dependent sticking threshold.
Refinement does not resolve the published distribution mismatch.
Each 900,000-flight ensemble took approximately 0.1–3.8 seconds on M5 Max; regeneration records current timings.

Audio compares our driving-plate and ball velocities under the thesis sticking convention, with equal gain and natural frequency 500 radians/s.
These are direct sonifications of a mechanical model with no acoustic-radiation or object-response filtering.
Published statistical curves are the author reference; neither player contains author audio.

[paper]: https://arxiv.org/abs/1609.07313
[thesis]: https://bibliotheque.ec-lyon.fr/documents/TH_T2534_czouabi.pdf
[medyna]: https://medyna2017.sciencesconf.org/data/Medyna_2017_Proceedings.pdf
