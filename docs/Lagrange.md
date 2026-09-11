# Lagrange 2010 sustained contact analysis/synthesis

Implements modal analysis, sampled impact estimation, uncompressed contact resynthesis, and a moving comb with supplied motion.
The [paper](https://doi.org/10.1109/TASL.2009.2038670) and [author manuscript](https://hal.science/hal-01106568/document) define the source/filter model.
[Settings](../repros/lagrange/cases.json) and [source hashes](../repros/lagrange/sources.json) pin the reproduction inputs.
The FoleyAutomatic corpus named in section VI-B supplies the recordings; original CIQS settings, author code, and Lagrange synthesis WAVs remain unavailable.
The first two materials use a joint excitation/audio estimator that extends the paper and preserves changing spectral structure.
Statistical trigger compression is outside the implemented scope.

## Run

The Python runner retains main WAVs, parameters, metrics, and plots after successful checks.
Use `--keep-diagnostics` to retain intermediate audio.
Failed runs preserve generated diagnostics.

```sh
python3 tools/Reproduce.py --method lagrange --offline
build/lagrangeReproduce input.wav settings.txt output
build/lagrangeReproduce benchmark
build/LagrangeTest
build/LagrangeRecoveryTest
build/SparseConvolutionTest
python3 tests/RenderMetricsTest.py
```

The renderer requires mono audio and writes raw float WAVs, fitted modes, contact coefficients, and timings.
The [listening page](../outputs/reproduction/listening/index.html) compares identified input recordings with contact reconstructions.
Inverse-filter round trips are separate numerical diagnostics and retain information such as speech from the input.
The [manifest](../outputs/reproduction/lagrange/manifest.json) records complete-render timings, equation checks, and reconstruction metrics.
Timings include allocations and transfers and exclude device creation and WAV I/O.

The settings file contains one line of space-separated fields:

```text
modal_begin modal_end impact_begin impact_end modal_gains trigger_fit center_impact audio_weight
```

Intervals use sample indices with exclusive ends; the impact interval addresses the inverse-filtered excitation.
The last four fields default to `0 0 0 .01`.
Modal gains are `0` fitted complex gains, `1` unit gains, `2` magnitudes, `3` causal magnitudes, and `4` incremental phase.
Source estimators are `0` regularized envelope, `1` excitation waveform, `2` published two-pass inverse, and `3` joint excitation/audio.
Impact centering removes the sampled impact's arithmetic mean before source fitting and synthesis.

## Method and reconstruction choices

Modal analysis uses eight frequency subbands, decimation by eight, eighth-order whitening, and twenty ESPRIT poles per band.
Stable poles are ranked by fitted amplitude, limited to eighty, and refitted against the real analysis interval.
CPU FP64 FFTs and Accelerate provide SVD, least-squares, eigenvalue, and prediction calculations.
Metal evaluates the modal response, inverse filtering, convolution, and nonnegative source optimization.
The modal inverse has an absolute magnitude floor of `1e-8`.

Section III-A assigns amplitude and phase to the source and frequency and damping to the resonant filter.
Reproduction gains and representative impact intervals are calibrated on each input.
Causal magnitudes use `a * exp(-d*n/fs) * sin(theta*(n+1))`, where `theta = 2*pi*f/fs`.
This follows the available JASS 1.25 modal-filter phase convention, with its radius factor absorbed into amplitude.
The SDK establishes a convention rather than the exact 2001 demonstration settings.

Incremental phase implements [Lagrange, Whetsell and Depalle, DAFx 2008](https://www.dafx.de/paper-archive/2008/papers/dafx08_10.pdf).
It preserves frequencies, damping, and gain magnitudes and puts adjacent one-pole responses in quadrature at equal-magnitude crossings.
FP64 bisection evaluates equation 13; when no crossing exists, a sampled search and local refinement minimize equation 10, including endpoints.

The excitation envelope is a natural cubic spline through absolute-value maxima.
A delayed, stretched, and scaled Meixner window uses equations 6–8 with 200 samples, beta 10, and gamma 0.89.
The regularized envelope estimator uses `conj(W)/(abs(W)^2+lambda^2)`, with `lambda=0.03*sum(w)`.
Equation 9 detects peaks with attack 0.999 and release 0.3; nonnegative least squares fits amplitudes including overlap and recording truncation.
The CPU SIMD solver uses relative KKT tolerance `1e-7` and at most 1,500 FISTA iterations.

The published two-pass estimator in section V-C instead inverts the decay forward and the reversed attack backward, with zero boundary states.
Their cascade inverts the product of the sections, which differs from the concatenated Meixner window.
The impact origin is shifted by `peak_index-1` samples.
Iterative detection removes dominant peaks between adjacent positive local minima because the paper omits the removal boundaries.
Each pass precedes the stopping check; original peak amplitudes are retained once.
This configuration passes equation checks but fails rolling reconstruction quality gates:

```sh
python3 tools/lagrange_reproduce.py --offline --case trough_pass1 --case trough_pass2 \
  --modal-gains incremental_phase --trigger-fit split_envelope --no-center-impact \
  --audit --output /tmp/lagrange-published-audit
```

`--audit` records every quality failure and exits unsuccessfully; equation failures remain immediate errors.

Overlapping oscillatory impacts can cancel, so the observed envelope differs from a sum of individual envelopes.
The waveform extension minimizes `0.5*||I*d-e||² + lambda*sum(d)` subject to `d>=0`.
`I` convolves the sampled impact with coefficients, and `e` is the inverse-filtered excitation.
The target is zero beyond the recording; the penalty is `0.001*max(I.T*e)` and the solver runs 1,200 iterations.
Impact centering avoids accumulation of its DC component and is an explicit preprocessing extension.
Coefficients are source basis weights, with physical contact identification unverified.

The joint extension fits one vector to excitation and audio:
`0.5*||I*d-e||²/||e||² + 0.5*eta*||S*I*d-x||²/||x||² + lambda*sum(d)`, with `d>=0`.
`S` is a recording-length modal response, `x` is the recording, and both convolution targets extend with zeros.
The penalty is `0.001` times the maximum positive weighted adjoint correlation; Metal runs 4,800 FISTA iterations.
Both rolling materials use `eta=0.01`, exposed as `--audio-weight`.
Synthesis evaluates causal `d*i*s` over `2*N+L-2` samples for recording length `N` and impact length `L`.
The modal response spans the entire output horizon, preserving slow decay after the input ends.
The fitting response remains a separate finite approximation.

`MovingCombGpu` implements Stoelinga 2007 equations 4.6 and 4.8 with causal fractional delays `l/c` and `(L-l)/c`.
Callers supply position, plate length, wave speed, and reflection gains; these are absent from the FoleyAutomatic metadata.
The model leaves dispersive wave speed and distance-dependent attenuation to callers.

## Validation and limits

Tests cover modal and Meixner equations, planted poles, JASS phase, independent phase adaptation, and both envelope inverses.
A recursive FP64 oracle checks modal rendering, including cancellation of a slowly decaying 3 kHz mode after the input ends.
Dense NNLS fixtures check single and joint source fits, tails, scale invariance, silence, overlapping impacts, and optimality.
Controlled recovery tests check known contact sources, spectral sweeps, supplied moving filters, and modal inversion.
Python independently checks the full recordings with SciPy filtering and optimization.
Joint-fit gates bound coefficient differences by 1%, weighted output differences by 0.1%, and relative objective differences by `2e-5`.
Independent KKT residuals must be below `0.001`.

| Material | Spectral motion | Short cepstra | Envelope correlation | RMS-matched STFT error |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.889 | 0.922 | 0.914 | 0.248 |
| 2 | 0.772 | 0.668 | 0.941 | 0.223 |

Motion measures compare time-centered log spectra and cepstra, with separate tests rejecting stationary and reversed spectral motion.
Both cases limit post-input peak 10 ms RMS to 1% of reference RMS and tail energy to `0.0001` of output energy.
Material 2 also limits excess power around 2,998.45 Hz to 3 dB after energy normalization.
[Case settings](../repros/lagrange/cases.json) contain all acceptance thresholds.
These checks establish bounded reconstruction errors; perceptual equivalence and recovered physical contacts remain unverified.
The remaining materials retain audible reconstruction differences.

The three rolling crops span video seconds 17–20, 20.5–23.5, and 24.25–27.5.
The recorded and simulated wok crops span 56–64 and 72–84 seconds.
All retain the 44.1 kHz mono MP2 decode without resampling or gain changes; the recorded wok has silence at both ends.
No lossless source, exact CIQS excerpt, empirical parameter distributions, or impact database was recovered.
The paper's thousand-trial table and listening study remain unreproduced.
