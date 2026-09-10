# Traer 2019

Implements [Traer, Cusimano and McDermott's rigid-body impacts and scrapes][paper].
The model includes cosine resonances, noise transients, joint Gaussian sampling, spring impacts and spatially varying responses.
Comparisons execute later TDW author code; the original 2019 listening experiments remain unreproduced.

```sh
cmake --build build --target TraerTest traerReproduce -j8
build/TraerTest
python3 tools/traer_reproduce.py --offline
```

Omit `--offline` to download missing pinned inputs.
Outputs are in `outputs/reproduction/traer/`, with listening pairs in `cases.json` and numerical results in `manifest.json`.
`build/traerReproduce benchmark` measures 32 independent one-second responses with 15 modes and 30 transient bands.

## Model and assumptions

Equations 2–3 use `10^((a-b*t)/20)*cos(2*pi*f*t)` modes and `10^((alpha-beta*t)/20)*noise(t)` transients.
Onsets are dB and decay rates are dB per second; RT60 `T` corresponds to `60/T`.
The paper uses 15 modes and 30 ERB bands, with cosine modes starting at nonzero amplitude.
ERB noise uses centered Hamming-sinc filters with unit L2 norm and one Gaussian realization.
FIR coefficients, tap count, normalization and seed are explicit implementation choices left unspecified by the paper.

`MakeGaussian` preserves the full covariance and accepts singular positive semidefinite matrices.
`SampleModes` takes contiguous frequency, onset and decay planes and rejects complete draws outside the 10% mean-frequency-spacing tolerance.
Angular-frequency inputs require conversion to Hz, including covariance rows and columns.
Additional bounds require positive decays and frequencies strictly between zero and Nyquist.
Repeated contacts preserve frequency and decay, perturbing onset dB by a standard deviation of 20% of its absolute mean.

Impacts use nonnegative half-sine forces with explicit duration and peak force.
`SpringContactDuration` uses `pi*sqrt(m/k)`, consistent with the spring frequency in Equation 5.
The printed duration bound and Section 3.3's calibrated 10.9 mass-scaling ratio disagree with that square-root dependence.
The fitted stiffnesses and reference recordings needed to reproduce that calibration are unavailable.

Scrape force follows Equation 8: `m*S''*v^2+A*(v*S')^gamma`, including its omission of the `S'*dv/dt` term.
Integer exponents use literal powers; negative bases with nonintegral exponents use a signed real extension.
The demonstration uses `gamma=1`.
Depth derivatives use centered finite differences and linear interpolation with caller-supplied spacing.
Profile quilting joins measured rows using overlap-error selection and a linear crossfade; overlap length is an explicit implementation choice.

Equation 9 interpolates response amplitudes between neighboring centerpoints while preserving modal frequencies and decays.
Current output-time position selects the response for every convolution lag, with endpoint clamping and the final position held through the tail.
Shared Metal FIR convolution and signal interpolation implement this linear combination without a time-varying FIR matrix.

## Source comparisons and limitations

[sources.json](../repros/traer/sources.json) pins downloaded source, data and license hashes under `references/traer/`.
TDW uses revision `e28687dac79ef7a2aa25cc41569154b550e3fb84`; Clatter uses `79cac6cbe3f7c452ba28b56c7da4a0124ad04806`.
These descendants use ten independently perturbed modes and omit the paper's transients, joint covariance and spacing rejection.
Their modes truncate at onset-dependent -80 dB cutoffs.
TDW uses duration `min(0.001*mass,0.002)`, samples half-sine endpoints and normalizes by the positive peak.
At the paper's 0.7 g pellet mass this produces one zero sample and nonfinite normalization; comparisons use 0.2 kg and 1 kg.

The runner executes the original TDW `Modes`, `_get_object_modes` and `_synth_impact_modes` bodies with isolated imports and decorators.
Four materials and two masses produce eight full-waveform comparisons with original sampled parameters and known normalization.
Clatter source and one material file support provenance comparison; its C# program is not executed.
TDW has a BSD-style two-clause license, Clatter uses Hippocratic License 3.0, and the paper is CC BY 3.0.

The scrape demonstration uses a later TDW basswood profile, wood-medium modal means and prescribed back-and-forth motion.
It compares constant and three-location responses; both clips are native demonstrations.
The original recordings, joint statistics, microscope maps and OptiTrack trajectories are unavailable.
Original listener-study results and calibrated recreation of the 2019 experiment remain unverified.

## Validation and timing

Tests cover friction excitation, half-sine area, mass scaling, covariance, singular distributions, spacing rejection and deterministic quilting.
Full GPU responses are compared with FP64 equations, including finite cutoffs, repeated dispatches and a 19 kHz ten-second tail.
Spatial tests compare every sample against direct convolution with the current position and complete tail.
The reproduction also checks NumPy scrape derivatives and SciPy spatial convolution independently.

The eight TDW comparisons have maximum relative waveform error `8.98e-6` and IR error `9.99e-6`.
Scrape force error is `2.55e-8` and complete spatial-waveform error is `3.30e-7`.
The 15-mode, 30-band GPU test differs from FP64 by `2.81e-7` relative RMS.
The 19 kHz test has `3.55e-7` late-tail relative error and `6.47e-7` maximum absolute error.

The benchmark compares FP32 single-thread CPU and Metal on the same 32-voice response workload.
Timing excludes noise preparation and allocation and includes GPU submission and completion.
Three fresh-process M5 Max Release runs give CPU medians of 182.5–194.8 ms and Metal medians of 5.04–5.18 ms.
The matching FP32 waveform error is `1.06e-7` relative RMS.
Complete scene/force generation, optimized SIMD CPU throughput and audio-device deadlines remain unmeasured.

[paper]: https://www.dafx.de/paper-archive/2019/DAFx2019_paper_57.pdf
