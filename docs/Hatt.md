# Penn Haptic Texture Toolkit

Implements the texture renderer in [Culbertson et al. 2014](https://doi.org/10.1109/HAPTICS.2014.6775475).
The [Penn release][release] supplies 100 original textures at 10 kHz and downsampled versions at 1 kHz.
Outputs are tool acceleration in m/s²; listening to that signal is an audition of haptic vibration.
Airborne radiation and haptic-device integration are outside this implementation.

## Run

```sh
python3 tools/Reproduce.py --method hatt
open outputs/reproduction/listening/index.html
```

Use `--offline` after the first download.
The runner executes all 100 models at both rates and retains ten 10 kHz author/implementation pairs, one texture per material category.
Use `--keep-diagnostics` to retain every rendered texture, prepared input and coefficient trace.
Source archives and their SHA-256 hashes are recorded in `repros/hatt/cases.json`.
The original code is compiled with Homebrew Clang as a separate reference executable.
Penn's non-profit research license accompanies the downloaded data and source under `references/hatt/`.

## Model and API

`hatt::Texture` contains the published force/speed points, triangulation, AR/MA line frequencies, innovation variance and gain.
Call `Validate` when loading a texture, then `Interpolate` for each control sample.
Controls use normal force in N and tangential speed in mm/s.
Barycentric weights interpolate line frequencies, variance and gain inside the published triangulation.
The shared `LineSpectrumPolynomial` converts odd or even orders to monic polynomials.
`Tick` applies the resulting ARMA filter to a unit-variance Gaussian innovation while preserving excitation and output histories.
Polynomial coefficients use the denominator convention `1 + a1*z^-1 + ...`; the feedback subtracts their weighted output history.
AR models use a unit numerator; ARMA models include the interpolated numerator gain.

Default control saturation includes exact zero-force and zero-speed boundaries.
`Interpolate(..., true)` reproduces the original renderer's small positive lower bounds and 0.01 inward upper-bound offsets.
The listening comparisons use those author bounds and identical Gaussian innovations for both implementations.
The listening page uses the original 10 kHz models, whose Nyquist limit is 5 kHz.
The original Boost random stream is replaced by a supplied common stream to isolate interpolation and filtering differences.

`FrictionForce` implements the paper's continuous viscous-to-Coulomb transition with slope 0.004 s/mm and threshold `mu/0.004`.
Its signed scalar output opposes tangential motion.
`FrictionForce(..., true)` preserves the original code's additional factor of mu in the viscous region.
That source convention has a discontinuity at the stated threshold when mu differs from one.
A haptic caller can multiply texture acceleration by the paper's effective mass of 0.05 kg and apply it along the surface binormal.

`RenderGpu` interpolates controls and constructs ARMA coefficients in parallel on Metal, then filters independent voices with separate histories.
Metal uses two float components for interpolation, polynomial construction and filter accumulation.
The CPU path retains double precision as an independent reference.
The GPU call starts each voice from zero history and includes preparation, transfers, dispatch, synchronization and readback.
CPU `State` supports continuous block rendering.

## Verification and limits

The reproduction compares every GPU sample with a separate NumPy implementation at both published rates.
The original C++ interpolation runs for every trajectory sample at both rates, with separate synthesis from its coefficients.
The largest NumPy/GPU relative L2 error across 200 renders was 2.71e-8.
At 1 kHz, the maximum difference from the executed author renderer was 1.26%; the aggregate coefficient difference was 0.0124%.
Our double-precision real-polynomial calculation differs from its float complex-polynomial construction and float control arithmetic.
At 10 kHz, all ten listening pairs differ from the executed author renderer by at most 0.407% relative L2.
The author triangle search produces negative weights at three controls in the full 100-texture sweep.
For Floortile 5, one invalid selection raises the full-record audio difference to 6.31%; its declared comparison limit in `cases.json` is 7%.
The remaining 99 textures retain the 2% limit and differ by at most 0.907%.
Our interpolation uses a containing triangle; the author output remains unmodified, with its diagnostic messages retained in `metrics.json`.
Tests separately evaluate polynomials against complex root products, analytic impulse responses, boundary controls and friction laws.
Ordered line frequencies ensure stable frozen filters; arbitrary rapid parameter changes can still produce large time-varying responses.

For 100 simultaneous three-second renders, the complete GPU calls took 0.036 s at 1 kHz and 0.152 s at 10 kHz on the M5 Max.
These are offline batch timings.
The runner records CPU interpolation, filtering, comparison and coefficient/WAV writing separately as `reference_seconds`.
`outputs/reproduction/hatt/metrics.json` contains per-texture numerical errors and measured timings.

The importer uses the original 10 kHz AR models and their downsampled 1 kHz ARMA counterparts.
New Auto-PARM model identification and arbitrary-rate model conversion remain outside this renderer.
The retained author MATLAB resampling source documents the release's zero-order-hold conversion, variance scaling and zero adjustments.
Replay uses published model parameters at their supplied decimal precision.

[release]: https://repository.upenn.edu/items/c1b8168b-b8a7-4f22-9e38-6430c233a600
