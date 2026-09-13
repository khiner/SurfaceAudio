# Conan rolling synthesis

Implements [Conan et al., IEEE/ACM TASLP 22(8), 2014](https://doi.org/10.1109/TASLP.2014.2327297), using original C++23 and Metal code.
The [CMJ continuous-interaction model](Continuous.md) provides rubbing, scratching and transitions.

## Run

Analysis retains the WAVs selected in `cases.json` and metrics for all trials.
Use `--keep-diagnostics` with `conan_analyze.py` and `conan_velocity.py` to retain all trial WAVs.
Regenerating discarded trials requires rerunning synthesis before analysis.

```sh
python3 tools/Reproduce.py --method conan
open outputs/reproduction/listening/index.html
```

The runner downloads pinned author assets and calibrates force and resonator inputs.
It compares independent random realizations with the published examples.
The native `conanReproduce` executable also accepts `OUTPUT_DIR --parameters FILE`.
Rows contain a name, amplitude mean/sigma/a1/b1, interval mean/sigma/a1/b1, and duration scale, using seconds for intervals and duration.

## Synthesis

`MakeParameters` implements the published perceptual preset, with linear interpolation over roughness ρ in [0,1].
These controls use the authors' published calibration.

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

Both centered event sequences use one unit Gaussian innovation, following Sections IV-B and IV-E:

```
y[n] = σ w[n] + b₁ σ w[n−1] − a₁ y[n−1]
A[n] = μA + yA[n]
ΔT[n] = μT + yT[n]
```

This implements H(z) = (1 + b₁ z⁻¹)/(1 + a₁ z⁻¹), Equation (13).
An amplitude belongs to the following interval, ΔT[n] = T[n+1] − T[n].
`NextEvent` exposes the unmodified process for statistical analysis.
`FitProcess` supports empirical innovation quantiles.
Both inverse CDFs receive the same uniform variable derived from the common Gaussian.
The stored 65-point inverse CDF uses linear interpolation and includes its observed endpoints.
The normal CDF approximation has measured absolute error below 3e−7 across [−6,6].

Each impact has the centered raised-cosine shape in Equation (9), peak one and integral t₀/2.
Equations (15)–(17) give duration t₀ = 7.88e−4 S A⁻⁰·²⁹ seconds and modulation frequency 3V/S Hz.
`Size` and `Velocity` are normalized perceptual controls in [0.1,1].
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
These bounds extend the published distribution.
Counters report amplitude, interval and duration clamps and pulse capacity overflow.

A fixed 4 ms lookahead preserves centered pulse peak times and overlapping impacts.
Event and pulse clocks use sample units, avoiding accumulated error from repeated FP32 second increments.
Modulation uses compensated phase accumulation and the force delay, preserving Equation (7) after latency removal.
The 64-pulse state accommodates all live events within the validated interval/duration bounds.
Create a new state when changing `MaximumDuration`, which determines lookahead.
The oscillator phase and random/filter histories remain continuous when perceptual controls change at block boundaries.
State begins with zero filter memories; applications that need a stationary onset can warm up with `NextEvent` before rendering.

## Calibration and physical reference

`FitProcess` removes the empirical mean and fits an invertible ARMA(1,1) by minimizing residual energy.
It reports residual variance and lag-one correlation, and stores Gaussian deviation and empirical inverse-CDF quantiles.
`ExtractImpacts` estimates peak time and amplitude by parabolic interpolation.
Raised-cosine duration is twice the full width at half maximum.
That duration estimate assumes isolated, baseline-corrected peaks; overlapping impacts or lossy encoding can distort it.
Duration searches stop at neighboring minima.
Peaks above threshold have `Duration = 0` when isolated half-height crossings are absent.
`Calibrate` fits amplitudes and following intervals from every event.
It fits duration scale with a fixed exponent using at least two measured durations.

`Reference.h` implements the predictive calibration from Sections II–IV:

- `FractalSurface` generates a periodic Gaussian profile with spectrum proportional to frequency raised to β and bounded absolute height.
- `RollingCurve` applies the sphere's upper contact envelope, using the spherical sag across neighboring surface points.
- `ContactStiffness` evaluates Equation (4) from radius and the two materials' Young's moduli and Poisson ratios.
- `RenderReference` integrates rigid-surface Equations (2)–(3) with FP64 RK4 substeps.
  Forces include gravity and Hunt–Crossley dissipation, with prescribed lateral motion and interpolated surface height and velocity.
- `MeasureContact` returns isolated impact duration, peak force and outgoing velocity.
- `FitDurationLaw` fits the isolated-contact amplitude-duration exponent for use in `Calibrate`, following Section IV-E2.

The reference defaults use κ = k/m = 10¹⁰/3, μ = 1 s/m, speed 0.2 m/s, spatial resolution 0.1 mm and sample rate 44.1 kHz.
The demonstration mass is 0.01 kg, with stiffness determined by the published κ.
The rigid surface is the reference used for the paper's excitation analysis, with the resonator decoupled as described in Section III.
The generated periodic surface is rigid; grid spacing and integration substeps are numerical choices.

For a frictionless elastic isolated Hertz contact, the exact duration scales as impact velocity⁻¹/⁵ and peak force as impact velocity⁶/⁵.
Tests verify the resulting amplitude-duration exponent of 1/6.
Physical calibration and the paper's empirical 0.29 preset remain separate.

## Author comparisons and limits

The [author companion site][author] supplies force, impulse-response and sound examples.
The source/filter comparisons identify a scalar response gain from the first half of the supplied force/convolution pair and check the second half.
The force model uses disjoint training/held-out intervals; explicitly labelled whole-record cases fit the complete recording.
No original random seed or exact source implementation was found, and many inputs are lossy MP3 recordings.
Decoded amplitudes have uncalibrated force and acoustic units.

Velocity coefficients come from the author's vector EPS plots, retaining their original axes and coordinates.
The default innovation scale is estimated by whitening decoded training forces.
The separate EPS-variance option assumes an unpublished correlation normalization; it is labelled separately.
Eight seeds per velocity provide independent distribution samples.
The published controls and physical rolling reference are distinct models with different velocity semantics.

Independent pulse sums, analytic ARMA statistics, positive-support behavior, RK4 refinement, and CPU/GPU streaming checks remain in `ConanTest`.
See [Validation.md](Validation.md) for measured numerical errors and performance scope.

[author]: https://kronland.fr/publications/a-synthesis-model-with-intuitive-control-capabilities-for-rolling-sounds/
