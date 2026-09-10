# Poirot collision signal model

This implementation reproduces the modal power redistribution, rough sidebands, and return to ordinary string decay described by Poirot, Bilbao, Aramaki, Ystad and Kronland-Martinet, [“A Perceptually Evaluated Signal Model: Collisions Between a Vibrating Object and an Obstacle”](https://doi.org/10.1109/TASLP.2023.3284515), TASLP 31, 2338–2350 (2023).
It includes a streaming C++ reference, Metal 4 synthesis across modes and independent voices, and an independent stiff-string/obstacle finite-difference reference.
The reconstruction is compared with actual author stimuli, including sustained collisions rather than isolated impacts.
Some collision trajectories still differ substantially, so numerical agreement and spectral similarity are reported separately from perceptual equivalence.

## Sources and reproduction

The [author companion site](https://www.prism.cnrs.fr/publications-media/IEEEPoirot/) currently redirects to a PRISM Cloud login.
The [archived April 2024 page](https://web.archive.org/web/20240415080231/https://www.prism.cnrs.fr/publications-media/IEEEPoirot/) retains the experiment mapping and public audio links.
Twenty-nine WAVs were recovered, including physical M1 and signal M2/M3 stimuli, a pick interaction and recorded fretboard buzz.
One additional attempted stimulus, M2_x3_y9, returned HTTP 429 and is excluded.
No author implementation or software repository was found in the companion site, paper references, or public code search.
The archived page lists extensions such as tanpura and membrane synthesis, but those are not asserted to be reproduced by the string comparisons here.

[Source provenance](../repros/poirot/sources.json) records original URLs, archive timestamps, SHA256 hashes, and local rights information.
Author audio and papers stay under ignored `references/poirot/`; no audio redistribution license was identified.
The local paper is HAL [hal-04355439v1](https://hal.science/hal-04355439v1), submitted December 20, 2023, with a HAL CC BY-SA cover followed by an IEEE rights statement.
The separate [Edinburgh PDF](https://www.pure.ed.ac.uk/ws/portalfiles/portal/338872061/Bilbao2023IEEEPerceptually.pdf) explicitly identifies itself as a peer-reviewed manuscript.
Neither has been verified as the IEEE typeset version of record.
The HAL version corrects the excitation sign and numbers the companion site as reference 46; the Edinburgh version uses reference 44.
Both have the equation inconsistencies discussed below.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target PoirotTest poirotReproduce -j 8
build/PoirotTest
python3 tools/poirot_reproduce.py --binary build/poirotReproduce --offline
```

Omit `--offline` to fetch missing pinned WAVs from their archived URLs.
The runner checks source hashes and rejects nonfinite synthesis.
It writes eleven full-length comparison WAVs, `cases.json`, and `manifest.json` under `outputs/reproduction/poirot/`.
Temporary renderer configuration files are deleted after execution.
The shared listening report consumes `cases.json` and preserves originals separately from audition gain.
The manifest includes full-record level measurements, matched precontact/interaction/decay windows, first-mode sideband trajectories, and shared texture measurements.
A single gain is estimated before contact for comparison; interaction gain and time alignment are not fitted.

## Equations and source discrepancies

`StringModes` implements Eq. 8 with all modal frequencies below Nyquist.
`MakeSignal` computes nodal shapes, thresholds and redistribution weights from Eqs. 11–13.
Each rendering sample applies modal power transfer and exponential loss, then advances the two phase accumulators from Eq. 15.
State survives arbitrary block boundaries.
Each queued GPU dispatch receives an immutable header from the shared batch upload arena, so several differently sized encodes can share a batch.
A bank with only nodal modes remains uncoupled with zero redistribution weights.
The physical reference uses the simply supported stiff-string equation, the Table I half-raised-cosine downward force, linear point interpolation, the shared stable grid-spacing bound, and an implicit discrete gradient of the unilateral contact potential.
The scalar contact solve preserves the potential/work identity without explicit penalty-force stepping.
It uses double precision on the CPU as an independent physical reference; production signal summation and power reductions run on Metal through the shared `SumThreadgroup` SIMD reduction.

Three discrepancies must be made explicit:

* Printed Eq. 10 uses a donor weight `theta_j` inside a sum identical for every recipient.
  This fails both the stated conservation law and the requirement that nodal modes remain unchanged.
  Production uses `T_i = -lambda max(P_i-p_i,0) + theta_i sum_j lambda max(P_j-p_j,0)`.
  The test retains an independent literal equation counterexample: powers `[10,2,1]`, thresholds `[1,1,infinity]`, weights `[.4,.6,0]` and lambda `.1` create spurious net power `.26` with the printed index.
* Printed Eq. 16 multiplies splitting by normalized `theta_i`.
  With 41 modes at the midpoint, this limits the upper first-mode shift to about 6.5 Hz, inconsistent with Fig. 7 and the actual author recording.
  The source-calibrated option uses the unnormalized absolute modal shape for splitting, while keeping normalized weights for power transfer.
  A literal Eq. 16 comparison WAV isolates this choice.
* The author WAVs reset every modal phase at 0.5 seconds, including unaffected nodal modes.
  The first post-reset phase is one sample increment.
  For example, mode 2 has measured phase -2.109180 radians after contact, matching `-2*pi*f2*.5 + 2*pi*f2/fs` modulo a turn.
  `ResetPhaseAtActivation` reproduces this measured behavior; ordinary library construction retains continuous phase unless requested.

The paper mentions redistribution every Nd=800 samples alongside lambda=1/800, but its summary recurrence is per sample and does not specify an interpolation rule.
This implementation evaluates that per-sample recurrence with lambda=1/800.
It does not silently hold a transfer for 800 steps or introduce an undocumented interpolation scheme.
`PowerScale` makes the missing normalization of the empirical split threshold explicit.
It affects the split detector only and does not change conserved modal power.

## Calibration and remaining differences

[Calibration data](../repros/poirot/calibration.json) contains measured modes and the two training stimuli.
Initial amplitudes, phases, dispersion and damping were fitted jointly on 0.02–0.48 seconds of M2_x1_y5 using separable nonlinear least squares: Eq. 8 determines frequency and damping, and a linear least-squares step fits damped sine/cosine coefficients.
The fitted waveform error in that window was 4.81e-5 before float synthesis.
The inferred signal parameters are wave speed 410.010757 m/s, stiffness 1.33592428 m²/s, loss0 0.0250017841 s^-1, and loss1 0.00291608153 m²/s.
These differ from Table I's 404.02, 1.297, 0.05 and 0.002, so using Table I literally cannot match the published precontact waveform.
Other signal stimuli use the same initial modal shape with a precontact scalar level estimate.

Only M2_x1_y1 and M2_x1_y5 determine the empirical split-power scale.
The selected calibration keeps the paper's lambda and Table II height ratios fixed; a trial that also varied lambda and height was rejected.
Heights are interpreted relative to the first-mode amplitude at activation and multiplied by the first-mode shape at the obstacle.
M3, other obstacle positions and the weak M2_x1_y9 case are held out from collision fitting.
This is source calibration, not identification of the unpublished author code.

The physical reference has a separate precontact modal fit to M1_x1_y9, giving wave speed 410.634047, stiffness 1.3399786, loss0 0.0500082708 and loss1 0.00206600956 on 46 grid segments.
Its readout at x/L=0.3 is inferred from modal signs and is not uniquely identified.
The physical excitation and contact parameters retain Table I values.
Obstacle height uses the maximum absolute free displacement in one fundamental period centered on activation.
An exactly literal Table I run can activate the stiff barrier while it is already penetrated, introducing substantial potential energy at the switching instant; changing normalization would conceal this physical difference.
The separate author-derived precontact parameters avoid that spurious activation impulse for the rendered examples.

Validated on Apple M5 with Homebrew LLVM, Release, September 9, 2026:

| Comparison | Interaction waveform relative error | Interaction spectrogram relative error |
|---|---:|---:|
| M2 midpoint, strong, calibrated | 0.184 | 0.0617 |
| Same stimulus, literal Eq. 16 | 0.723 | 0.570 |
| M2 midpoint, weak, held out | 0.0005 | 0.0004 |
| M2 one third, strong, held out | 0.314 | 0.138 |
| M2 five twelfths, strong, held out | 0.603 | 0.268 |
| Physical midpoint, strong | 0.455 | 0.182 |
| Physical midpoint, moderate | 0.447 | 0.187 |
| Physical midpoint, weak | 0.165 | 0.060 |

At 0.51–0.59 seconds, the strong midpoint signal has author first-mode peaks at 273.37 and 539.00 Hz, versus 273.37 and 539.51 Hz in the reconstruction.
The literal Eq. 16 version instead has its dominant upper component at 416.03 Hz.
Some moderate cases have waveform errors above 1 despite similar spectra because the accumulated frequency-shift phase differs.
The five-twelfths case also has a material remaining spectral mismatch.
These failures are retained in the manifest and listening pairs rather than hidden by a global average or postcontact phase alignment.
No human listening equivalence test has been completed.

`PoirotTest` checks Eq. 8, the literal conservation counterexample, passive power loss, nodal invariance, exact CPU block continuity, queued GPU dispatch headers across activation/reset, all-nodal CPU/Metal synthesis, frame bounds, Metal/CPU rendering, discrete potential/work equality, repeated contacts, an independent free-string eigenmode, and full discrete energy conservation through 5000 undamped contact steps.
The 8192-frame CPU/Metal waveform relative error was 1.18e-4; the streaming result is bit-identical to a single CPU block.

For a bounded throughput comparison, run `build/poirotReproduce --benchmark`.
The benchmark uses identical 41-mode banks, 44100 frames per voice, a warmup and three measured runs, with median wall time reported.
Creation and compilation are excluded; GPU time includes every dispatch, wait and output copy.
Three fresh-process M5 Max Release runs measured CPU/GPU median ranges of 27.75–29.40 / 29.95–40.48 ms for one voice and 1881.87–1959.14 / 44.94–46.26 ms for 64 voices.
The 64-voice GPU advantage persists across these runs; one voice favors CPU.
Both workloads had CPU/GPU waveform relative error 4.66e-4.
These are one-second rendering measurements, not callback-latency or real-time audio-device tests.
