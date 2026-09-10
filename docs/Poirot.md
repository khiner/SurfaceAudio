# Poirot collision signal model

This method implements Poirot et al., [“A Perceptually Evaluated Signal Model: Collisions Between a Vibrating Object and an Obstacle”][paper].
The paper appeared in TASLP 31, 2338–2350, 2023.
It models modal power transfer, collision sidebands, and the return to ordinary string decay.
The implementation includes streaming C++, Metal synthesis, and an independent FP64 stiff-string/obstacle reference.
Comparisons use author stimuli, with substantial remaining differences in some collision trajectories.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target PoirotTest poirotReproduce -j8
build/PoirotTest
python3 tools/poirot_reproduce.py --binary build/poirotReproduce --offline
```

Omit `--offline` to fetch missing pinned WAVs.
The runner verifies source hashes and finite output, then writes eleven comparison WAVs, `cases.json`, and `manifest.json`.
Outputs are saved in `outputs/reproduction/poirot/` for the shared listening report.
Comparisons use one gain estimated before contact, with fixed interaction timing and gain.
The manifest records level, windowed waveform/spectral errors, sideband trajectories, and texture metrics.

## Sources

The [companion site][author] redirects to a login, but the [April 2024 archive][archive] retains public audio links and experiment labels.
Twenty-nine recovered WAVs include physical M1 and signal M2/M3 stimuli, pick interaction, and fretboard buzz.
Author code and an audio redistribution license remain unavailable.
String comparisons cover the collision model, while the site's tanpura and membrane extensions remain unimplemented.
[Source provenance](../repros/poirot/sources.json) records URLs, timestamps, hashes, and rights information.
Author files remain under ignored `references/poirot/`.

The local [HAL manuscript][hal] has a CC BY-SA cover and an IEEE rights statement.
The [Edinburgh PDF][edinburgh] identifies itself as a peer-reviewed manuscript.
The IEEE typeset version remains unverified.
HAL corrects the excitation sign, but both manuscripts retain the equation discrepancies below.

## Equations and discrepancies

The signal model uses Equation 8 modes below Nyquist, Equations 11–13 shapes and thresholds, and Equation 15 phase increments.
Each sample applies power transfer and exponential loss before advancing the phases.
Nodal modes have zero redistribution weights, and state persists across blocks.
Immutable GPU dispatch headers support differently sized encodes in one batch.
Metal parallelizes modes and voices through the shared SIMD reduction.
The FP64 physical reference uses a simply supported stiff string and Table I's downward half-raised-cosine excitation.
Linear contact interpolation and an implicit discrete gradient preserve the contact potential/work identity.

Printed Equation 10 uses donor weight `theta_j`, violating power conservation and nodal invariance.
The implementation uses recipient weighting:

```text
T_i = -lambda max(P_i-p_i,0) + theta_i sum_j lambda max(P_j-p_j,0)
```

An independent literal-equation counterexample remains in `PoirotTest`.
Printed Equation 16 uses normalized `theta_i` for splitting, limiting the first-mode upper shift to about 6.5 Hz for 41 midpoint modes.
The upper frequency shift in Figure 7 and the recording approaches `f1/3`.
The calibrated option uses absolute modal shape for splitting and normalized weights for power transfer.
A literal Equation 16 comparison WAV retains the printed behavior.

Author WAVs reset all modal phases at 0.5 s, including nodal modes, with a one-sample phase increment after reset.
`ResetPhaseAtActivation` reproduces this behavior, while the default preserves phase continuity.
The paper mentions redistribution every 800 samples but gives a per-sample recurrence and no interpolation rule.
The implementation uses that recurrence with `lambda=1/800`.
`PowerScale` calibrates the empirical split threshold while preserving modal power.

## Calibration and remaining differences

[Calibration data](../repros/poirot/calibration.json) records measured modes and training stimuli.
Signal amplitudes, phases, dispersion, and damping are fitted on 0.02–0.48 s of M2_x1_y5 using separable nonlinear least squares.
Equation 8 supplies frequency and damping, and linear least squares supplies damped sine/cosine coefficients.
The precontact relative waveform error is 4.81e-5 before float synthesis.

| Signal parameter | Fitted | Paper Table I |
|---|---:|---:|
| Wave speed, m/s | 410.010757 | 404.02 |
| Stiffness, m²/s | 1.33592428 | 1.297 |
| Loss0, /s | 0.0250017841 | 0.05 |
| Loss1, m²/s | 0.00291608153 | 0.002 |

Other signal stimuli retain this modal shape with a precontact scalar gain estimate.
Only M2_x1_y1 and M2_x1_y5 determine the split-power scale, with the paper's lambda and Table II height ratios fixed.
Heights use first-mode amplitude at activation multiplied by its shape at the obstacle.
M3, other positions, and weak M2_x1_y9 are held out from collision fitting.
These settings are inferred from audio, and the unpublished author implementation remains unknown.

The physical reference fits precontact M1_x1_y9 on 46 segments with wave speed 410.634047 and stiffness 1.3399786.
Its fitted loss0 is 0.0500082708 and loss1 is 0.00206600956, in the units above.
The inferred readout `x/L=0.3` is nonunique, and excitation and contact parameters retain Table I values.
Obstacle height uses maximum absolute free displacement over one fundamental period centered on activation.
Literal Table I parameters can activate an already penetrated barrier and introduce substantial potential energy.
The inferred precontact parameters avoid this activation impulse in the rendered examples.

Measurements below use Apple M5, Homebrew LLVM Release, September 9, 2026.

| Interaction | Waveform relative error | Spectrogram relative error |
|---|---:|---:|
| M2 midpoint, strong, calibrated | 0.184 | 0.0617 |
| Same stimulus, literal Equation 16 | 0.723 | 0.570 |
| M2 midpoint, weak, held out | 0.0005 | 0.0004 |
| M2 one third, strong, held out | 0.314 | 0.138 |
| M2 five twelfths, strong, held out | 0.603 | 0.268 |
| Physical midpoint, strong | 0.455 | 0.182 |
| Physical midpoint, moderate | 0.447 | 0.187 |
| Physical midpoint, weak | 0.165 | 0.060 |

At 0.51–0.59 s, strong midpoint first-mode peaks are 273.37/539.00 Hz for the author and 273.37/539.51 Hz for the reconstruction.
Literal Equation 16 produces a dominant upper component at 416.03 Hz.
Some moderate cases exceed relative waveform error 1 because frequency-shift phase accumulates differently despite similar spectra.
The five-twelfths case also retains a substantial spectral mismatch.
Individual errors remain in the manifest and listening pairs, and perceptual equivalence remains untested.

## Validation and performance

`PoirotTest` checks modal equations, power conservation/loss, nodal invariance, block continuity, dispatch headers, reset timing, and frame bounds.
Physical checks cover contact potential/work, repeated contacts, a free-string eigenmode, and energy through 5000 undamped contact steps.
The 8192-frame CPU/Metal relative waveform error was 1.18e-4, and CPU streaming was bit-identical to single-block rendering.

Run `build/poirotReproduce --benchmark` for 41-mode, 44100-frame comparisons with a warmup and three measured runs.
Creation and compilation are excluded, and GPU time includes dispatch, waiting, and output copying.
Three fresh-process M5 Max Release runs measured these median ranges:

| Voices | CPU | GPU |
|---|---:|---:|
| 1 | 27.75–29.40 ms | 29.95–40.48 ms |
| 64 | 1881.87–1959.14 ms | 44.94–46.26 ms |

CPU was faster for one voice, and GPU was faster for 64 voices.
Both workloads had relative waveform error 4.66e-4.
These timings cover offline one-second rendering, with audio-device latency unmeasured.

[paper]: https://doi.org/10.1109/TASLP.2023.3284515
[author]: https://www.prism.cnrs.fr/publications-media/IEEEPoirot/
[archive]: https://web.archive.org/web/20240415080231/https://www.prism.cnrs.fr/publications-media/IEEEPoirot/
[hal]: https://hal.science/hal-04355439v1
[edinburgh]: https://www.pure.ed.ac.uk/ws/portalfiles/portal/338872061/Bilbao2023IEEEPerceptually.pdf
