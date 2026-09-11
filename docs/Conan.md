# Conan rolling synthesis

Implements [Conan et al., IEEE/ACM TASLP 22(8), 2014](https://doi.org/10.1109/TASLP.2014.2327297), using original C++23 and Metal code.
This is the rolling-specific model; the broader CMJ continuous-interaction model remains queued.

## Run

Analysis retains the WAVs selected in `cases.json` and metrics for all trials.
Use `--keep-diagnostics` with `conan_analyze.py` and `conan_velocity.py` to retain all trial WAVs.
Regenerating discarded trials requires rerunning synthesis before analysis.

```sh
python3 tools/Reproduce.py --method conan
open outputs/reproduction/listening/index.html
```

The harness fetches and verifies original author assets, decodes the recordings, calibrates force and resonator data, renders independent random realizations, and evaluates the published velocity examples.
The native `conanReproduce` executable also accepts `OUTPUT_DIR --parameters FILE`.
Rows contain a name, amplitude mean/sigma/a1/b1, interval mean/sigma/a1/b1, and duration scale, using seconds for intervals and duration.

## Synthesis


`MakeParameters` implements the published perceptual preset, with linear interpolation over roughness ρ in [0,1].
These are the authors' published calibrated controls, not parameters fitted to a new recording or a physical material in this repository.

| Process parameter | Smooth, ρ = 0 | Rough, ρ = 1 |
|---|---:|---:|
| Amplitude innovation standard deviation | 0.04 | 0.04 |
| Amplitude denominator a₁ | −0.97 | −0.93 |
| Amplitude numerator b₁ | 0.07 | 0.32 |
| Amplitude mean | 0.43 | 0.27 |
| Interval innovation standard deviation | 0.19 ms | 0.85 ms |
| Interval denominator a₁ | −0.97 | −0.93 |
| Interval numerator b₁ | −0.34 | 0.35 |
| Interval mean | 3.1 ms | 6.4 ms |

Both centered event sequences use the same unit Gaussian innovation, as justified by the residual correlation in Section IV-B and the Gaussian approximation in IV-E:

```
y[n] = σ w[n] + b₁ σ w[n−1] − a₁ y[n−1]
A[n] = μA + yA[n]
ΔT[n] = μT + yT[n]
```

This implements H(z) = (1 + b₁ z⁻¹)/(1 + a₁ z⁻¹), Equation (13).
An amplitude belongs to the following interval, ΔT[n] = T[n+1] − T[n].
`NextEvent` exposes the unmodified process for statistical analysis.
`FitProcess` can instead supply empirical innovation quantiles; both inverse CDFs receive the same uniform variable, obtained from the common Gaussian through its CDF.
The stored 65-point inverse CDF uses linear interpolation and includes its observed endpoints.
The normal CDF approximation has measured absolute error below 3e−7 across [−6,6].

Each impact has the centered raised-cosine shape in Equation (9), peak one and integral t₀/2.
Equations (15)–(17) give duration t₀ = 7.88e−4 S A⁻⁰·²⁹ seconds and modulation frequency 3V/S Hz.
`Size` and `Velocity` are the paper's normalized perceptual controls in [0.1,1], not SI radius or translational speed.
Equation (7) multiplies the force by 1 + m sin(2πνt); `Asymmetry` controls m in [0,1], default 0.3.
Changing velocity changes the modulation frequency, and preserves the event statistics as in the paper's intuitive mapping.
The output is an excitation force signal; the shared modal filter supplies the independently chosen resonant object.
The published source/filter repro uses the separately calibrated author responses.

`State` contains the complete PCG stream, ARMA memories, modulation phase, scheduler and active pulses.
Pulse fields use separate arrays.
`Render` and the allocation-free `Step` preserve sample-exact block continuity.
`ConanSynthesize` runs independent voices in parallel on Metal 4, with the same stepping equations and state layout as C++.
Buffers are Parameters[voice], State[voice], planar float output[voice * frame_count + frame], and uint frame_count at bindings 0–3 respectively.
Dispatch exactly the number of voices.

### Positive support and latency

The paper's Gaussian approximation has unbounded tails although physical amplitudes and intervals are positive.
The rendering layer floors amplitudes at 1e−4 and intervals at 0.25 ms, and caps duration at 8 ms by default.
These are explicit numerical extensions, not claims about the fitted distribution.
Counters report amplitude, interval and duration clamps and pulse capacity overflow.
A ten-second midpoint preset run with seed 1 at 44.1 kHz scheduled 2046 events, with 19 amplitude clamps, 7 interval clamps, no duration clamps and no overflow.
Consequently the rendered event distribution differs in its tails from the unconstrained analytic ARMA process.

A fixed 4 ms lookahead preserves centered pulse peak times and overlapping impacts.
Event and pulse clocks use sample units, avoiding accumulated error from repeated FP32 second increments.
Modulation uses compensated phase accumulation and the same delay as the force, so removing the lookahead recovers Equation (7) at the original source time.
The 64-pulse state can hold every live event under validated interval/duration bounds.
Changing `MaximumDuration` on an active stream changes the lookahead of newly scheduled events, so use a new state when changing that numerical setting.
The oscillator phase and random/filter histories remain continuous when perceptual controls change at block boundaries.
State begins with zero filter memories; applications that need a stationary onset can warm up with `NextEvent` before rendering.

## Calibration and physical reference

`FitProcess` removes the empirical mean and fits an invertible ARMA(1,1) using residual-energy minimization with analytic Gauss–Newton derivatives and backtracking.
It reports residual variance and lag-one correlation, and stores Gaussian deviation and empirical inverse-CDF quantiles.
`ExtractImpacts` detects local maxima, estimates subsample peak time and amplitude by parabolic interpolation, and estimates raised-cosine duration from twice its full width at half maximum.
That duration estimate assumes isolated, baseline-corrected peaks; overlapping impacts or lossy encoding can distort it.
The search stops at neighboring minima, so it never measures across multiple peaks.
Every local maximum above threshold remains an event even when no isolated half-height crossing exists; its `Duration` is then zero.
`Calibrate` fits amplitudes and their following intervals from every event, then least-squares fits the duration scale with a fixed exponent using only measured durations.
At least two measured durations are required; missing durations are not fabricated.
For example, a two-second default physical reference with β = −1 and surface seed 9 contains 866 peaks, of which only 144 have measurable isolated durations.
Discarding the other peaks would incorrectly calibrate a different amplitude and interval process.

`Reference.h` preserves the predictive calibration path from Sections II–IV rather than replacing it with generic filtered noise:

- `FractalSurface` generates a periodic spatial Gaussian profile with power spectrum proportional to frequency raised to β, using an inverse radix-2 FFT and a requested maximum absolute height.
- `RollingCurve` applies the sphere's upper contact envelope, using the spherical sag across neighboring surface points.
- `ContactStiffness` evaluates Equation (4) from radius and the two materials' Young's moduli and Poisson ratios.
- `RenderReference` integrates the rigid-surface specialization of Equations (2)–(3) with double-precision RK4 and explicit substeps, including gravity, Hunt–Crossley dissipation, prescribed lateral velocity, spatial interpolation and its velocity contribution.
- `MeasureContact` measures isolated impact duration, peak force and outgoing velocity. `FitDurationLaw` fits the amplitude-duration exponent on these isolated contacts; that exponent can then be held fixed when calling `Calibrate` on a rolling trace, following Section IV-E2.

The reference defaults use κ = k/m = 10¹⁰/3, μ = 1 s/m, speed 0.2 m/s, spatial resolution 0.1 mm and sample rate 44.1 kHz.
Mass 0.01 kg is an explicit demonstration choice; the published κ determines its corresponding stiffness.
The rigid surface is the reference used for the paper's excitation analysis, with the resonator decoupled as described in Section III.
It does not implement a position-dependent compliant plate or claim a physically measured surface.
The periodic grid and integration substeps are explicit numerical choices.

For a frictionless elastic isolated Hertz contact, the exact duration scales as impact velocity⁻¹/⁵ and peak force as impact velocity⁶/⁵.
The fitted amplitude-duration exponent therefore approaches 1/6, which the tests independently verify.
This differs from the paper's empirically selected 0.29 perceptual mapping and is why a newly physically calibrated model is kept distinct from the published preset.

## Author comparisons and limits

The [author companion site](https://kronland.fr/publications/a-synthesis-model-with-intuitive-control-capabilities-for-rolling-sounds/) supplies force, impulse-response and sound examples.
The source/filter comparisons identify a scalar response gain from the first half of the supplied force/convolution pair and check the second half.
The force model uses disjoint training/held-out intervals; explicitly labelled whole-record cases fit the complete recording.
No original random seed or exact source implementation was found, and many inputs are lossy MP3 recordings.
Decoded amplitudes do not establish physical force units or calibrated sound pressure.

Velocity coefficients come from the author's vector EPS plots, retaining their original axes and coordinates.
The default innovation scale is estimated by whitening decoded training forces.
The separate EPS-variance option assumes an unpublished correlation normalization; it is labelled separately.
Eight new seeds per velocity are generated for distribution comparisons, not waveform identity.
The published controls and physical rolling reference are distinct models with different velocity semantics.

The source/filter examples are close in audible character to their author references.
Independent pulse sums, analytic ARMA statistics, positive-support behavior, RK4 refinement, and CPU/GPU streaming checks remain in `ConanTest`.
See [Validation.md](Validation.md) for measured numerical errors and performance scope.
