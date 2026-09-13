# Rough sliding between flexible beams

Implements the modal beam dynamics and unilateral normal contact in [Dang et al. 2013][paper].
Two rough steel profiles interact during prescribed horizontal sliding.
The CPU supports penalty and forward-increment multiplier contact; Metal supports penalty contact.
Outputs are surface vibration velocity, with airborne radiation outside the model's scope.
Tangential friction, adhesion and plastic deformation are also excluded.

The example combines journal parameters, settings from a separate thesis example and local initialization choices.
Author surface arrays, solver source and WAVs were not recovered.
Numerical checks pass, but the published vibration levels, spectra and scaling remain unmatched.

## Run

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target roughReproduce -j8
python3 tools/rough_reproduce.py --method dang
python3 tools/BuildListeningReport.py
```

The example renders Ra5 at 0.7 m/s for 0.45 seconds, within full overlap.
Open [the listening page](Listen.html) after generation.
Use `build/roughReproduce dang OUTPUT SECONDS SPEED RA_LABEL SEED` for other cases.
Roughness labels 3, 5, 8, 10, 20 and 30 select Table 8 RMS heights and correlation lengths.
The 5 µm spatial grid and 0.1 µs integration step require offline rendering.

`vibration.wav` contains midpoint velocity, filtered to 44.1 kHz sampling and peak normalized.
`Metrics.json` records physical RMS, contact statistics, gain, inputs and render time.
`Comparison.json` and `Comparison.png` compare complete contact events with the approximate Figure 12 thresholds.
Regenerate comparisons from retained results with `--analyze-only --offline`.

Use `--sweep` for six roughness levels, seven speeds and three surface realizations.
It writes one WAV, scalar results in `Sweep.csv` and `Sweep.json`, and the comparison in `Scaling.json` and `Scaling.png`.
The ensemble comparison requires complete records and exact repetition of the canonical trajectory within the GPU batch.

## Inputs and limits

| Setting | Value | Source |
| --- | --- | --- |
| Resonator | Supported beam, 450 × 2 mm | Journal Table 6 |
| Slider | Free beam, 20 × 5 mm | Journal Table 6 |
| Material | 7,800 kg/m³; 210 GPa; 2% modal damping | Journal Table 6 |
| Discretization | 5 µm; 0.1 µs; 2.1 × 10¹² Pa penalty | Journal Table 7 and Section 4 |
| Modes | 40 resonator bending; 2 rigid and 2 bending slider modes | Journal Table 7; thesis Figure 4.2 |
| Width | 1 m | Thesis Figure 4.2 |
| Gravity | Slider only, 9.81 m/s² | Thesis Figure 4.2 |
| Initial offset | 0.1 m | Thesis Figure 4.2 |
| Initial motion | Zero displacement and velocity | Journal Equations 6 and 23 |
| Initial separation | Highest overlapping pair just touching | Local choice |
| Recording | 100 kHz contact snapshots | Thesis Section 5.3.2 |

The journal includes gravity on both bodies and aligns their origins at zero in its general formulation.
Its 40-mode count does not distinguish the two bodies.
The separate thesis example specifies a 400 mm resonator, 20 resonator modes, 1% damping and a different penalty coefficient.
The 1 m width implies 7.6518 N slider weight; the journal's event discussion cites 0.78 N.
Figure 13's resonance markers support the 450 mm beam within 0.34%, while Table 7's modal periods match 400 mm.
The journal gives one-second duration in prose and five seconds in Table 7.
These inconsistencies prevent establishing a complete author-run configuration.

Gaussian surfaces follow [Bergström's supplied MATLAB filter][generator] on the exact requested periodic grid.
Indexed PCG noise and seed 2013 replace the unavailable author random realization.
Individual profiles retain their sample mean and variance.
[Source metadata](../repros/rough/Dang.json) pins the figures, calibrations and transcribed thesis inputs.

Table 8 specifies 450 µm correlation for Ra5; Figure 10's printed 2 mm axis implies approximately 5 µm.
The thesis repeats this discrepancy.
Plotted coordinates cannot establish the original sampling grid or correct axis scale.
The reproduction retains Table 8; shorter-correlation renders are sensitivity cases.
Their large surface slopes also weaken the paper's small-angle justification for neglecting vertical friction.

Journal and thesis Chapter 5 results use different damping and surface settings.
Chapter 5 specifies loss factor 0.0006, corresponding to damping ratio 0.0003, versus the journal's 0.02.
Keep their level curves and parameter sets separate.

## Numerical conventions

The implementation follows the governing equations where printed formulas conflict with them.

- Free beam shapes satisfy the stated end conditions and use exact characteristic roots; printed Equations 15–16 fail those conditions.
- The force coefficient is `dt² / (mass * (1 + dt * zeta * omega))`, derived from Equations 22 and 24.
- Multiplier compliance uses the same coefficient; Equations 25 and 38 omit its damping denominator.
- Slave quadrature and transpose interpolation preserve force, moment and virtual work.

The reaction projection is a local discretization change at endpoints and on unequal grids.
The paper redistributes force density before trapezoidal integration, which can produce unequal integrated reactions.
An inferred density convention closely matches a printed thesis moving-mass trajectory; the conservative projection differs substantially there.
Agreement with the governing mechanics therefore does not establish agreement with the unavailable author implementation.

Events use total nodal force from both contact passes, with resonator and slider populations reported separately.
Snapshots occur every 100 integration steps; sub-10 µs contacts can be missed.
Start/end-censored events are excluded from complete-event histograms, while total work includes them.
Sampled event work approximates native-step work.
The journal leaves the event population and recording phase unspecified.

## Validation and reproduction gaps

`ContactMechanicsTest` checks modal boundaries, normalization, energy balance and unilateral damping.
`RoughContactTest` checks force and moment conservation, potential gradients, impact momentum and failed-step recovery.
It also checks CPU/Metal trajectories, screening, batching, streaming state and direct surface convolution.
The missing ABAQUS geometry include prevents replaying the paper's finite-element comparison.

Run `build/roughReproduce dang-calibrate` for the paper's one-millisecond penalty/multiplier comparison on the local Ra5 fixture.
Displacement differs by 3.37% on the resonator and 1.41% on the slider; impulse differs by 0.23% and force-waveform RMS by 90.82%.
The paper's 10% criterion leaves the comparison quantity unspecified.
Displacement agreement alone does not establish equivalent contact excitation.

The Table 8 Figure 13 fixture underestimates 1–8 kHz receiver power by approximately 20–24 dB.
A 5 µm correlation sensitivity adds texture but retains weak high-frequency resonances.
Full 5, 2.5 and 1.25 µm contact-grid records remain approximately 6–8 dB low at 4–8 kHz and 10 dB low at 8–10 kHz.
Successive band-power changes reach 1.62 and 1.87 dB with opposite signs; spatial convergence remains unestablished.
The source PSD estimator and actual contact-force spectra are unavailable, so these discrepancies do not identify a replacement damping law.
All comparisons retain the journal's 2% damping.

## Performance

Metal uses conservative displacement bounds to skip separated contacts and fixed node order for reproducible force projection.
Run `build/roughReproduce dang-benchmark 8192 21` for screened versus full evaluation at the same resolution and event cadence.
The command requires identical traces, modal states and event statistics after each of three consecutive segments.

| M5 Max, Homebrew Clang 23.1 Release | Full evaluation | Screened | Speedup |
| --- | ---: | ---: | ---: |
| 21 trajectories × 24,576 steps | 2.586 s | 1.183 s | 2.19× |
| 4 trajectories × 150,000 steps | 9.247 s | 5.678 s | 1.63× |

Use `dang-benchmark 50000 4` for the second workload; an optional final argument sets the cache interval.
Timings include dispatch, synchronization and readback, and exclude preparation and audio conversion.
They measure bounded workloads rather than complete-record throughput.

[paper]: https://doi.org/10.1007/s00466-013-0870-7
[thesis]: https://bibliotheque.ec-lyon.fr/documents/TH_T2349_vdang.pdf
[generator]: https://www.mysimlabs.com/matlab/surfgen/rsgeng1D.m
