# Agarwal scraping and rolling

Implements [Agarwal et al., DAFx 2021](https://mcdermottlab.mit.edu/papers/Agarwal_etal_2021_scraping_rolling_synthesis_DAFx.pdf), including the full horizontal, vertical and rolling force terms and the complete position-dependent impulse response.
[Author audio and ablations](https://mcdermottlab.mit.edu/scraping_rolling.html) provide the comparison recordings.
The [arXiv v1](https://arxiv.org/html/2112.08984v1) supplies additional numerical defaults.

## Run

```sh
python3 tools/Reproduce.py --method agarwal
open outputs/reproduction/listening/index.html
```

The workflow recalibrates scraping and rolling examples from author-associated profiles, renders complete spatial surface/object responses, checks the provided PyImpact source against Metal convolution, and replays four retained rolling reconstructions.
The retained inputs can be rendered directly after building, without another parameter search:

```sh
python3 tools/agarwal_reproduce.py --retained
```

`repros/agarwal/cases.json` records motion, sampling, source scope and expected force/WAV hashes.
The shared NPZ contains FP64 profiles and motion and the exact FP32 modal values uploaded by the renderer.
Forces are regenerated from these inputs by C++/Metal, not loaded from recorded author audio.
The four retained cases reproduce their original WAVs byte for byte on the verified toolchain.
They use vertical excitation only, with 64x sampling for wood and 128x for glass, sigma ratio 0.1 and anti-alias filtering.
Cell and pointwise refer to different orders of curvature interpolation and nonlinear constraint evaluation.
These are fitted reconstructions with explicit omissions, not complete physical reproductions of the paper.

For fresh parameter inference, run `agarwal_reproduce.py --temporal --whole-record`, then `agarwal_spatial_fit.py --case roll --whole-record` or `--case roll-glass`.
The low-level tools accept supplied inputs:

```text
agarwalReproduce prepare-temporal CASE_DIRECTORY [OVERSAMPLING [SIGMA_RATIO]] [--bandlimit]
agarwalReproduce render CASE_DIRECTORY OUTPUT_PREFIX
agarwalSpatialReproduce CASE_DIRECTORY OUTPUT_PREFIX [--basis]
agarwalContactFit FORCE.f32 MODES.txt TARGET.wav OUTPUT STEPS TAPS [LEARNING_RATE [linear|ln|db]]
agarwalEndpointFit BASIS.f32 LOCATION.f32 INITIAL.txt TARGET.wav OUTPUT STEPS [LEARNING_RATE [linear|ln|db [POOLED_WEIGHT]]]
```

Contact mode rows are frequency Hz, exponential decay seconds and amplitude.
The contact fitter holds force and frequencies fixed and fits log amplitude/decay through the complete finite tail.
Endpoint fitting holds force, frequencies and envelopes fixed and fits endpoint log amplitudes.
The optional pooled spectral term is our inference objective, not an author-specified loss.

## Equations and inputs


`ConstraintSettings` implements equations 4–7: each axial curvature becomes `tanh(alpha * curvature) / alpha`, with `alpha` interpolated between the maximum and minimum according to the normalized normal force raised to `Exponent`.
The defaults are the paper's `AlphaMax = 0.05`, `AlphaMin = 0.01`, and `Exponent = 0.95`.
Equal normal-force bounds select `ConstantAlpha` explicitly.
Normal forces outside the stated range saturate the interpolation at the corresponding endpoint.
Setting alpha to zero gives the unconstrained curvature limit.

The constrained curvatures are smoothed with a normalized symmetric Gaussian before integration.
The half-width is `GaussianHalfWidth * alpha / ReferenceAlpha` grid samples, rounded up for finite support, with sigma equal to `GaussianSigmaRatio` times the unrounded half-width.
At a grid boundary, the available weights are renormalized.
The default half-width is five samples at alpha 0.03.
This low-level spatial operator does not by itself implement the paper's audio-time window.
The reproduction executable also provides temporal smoothing of translated height, slope and curvature fields.
The paper specifies an average half-width of five samples at 44.1 kHz and proportionality to alpha, but does not give a spatial-to-temporal window conversion, Gaussian sigma, boundary extension, or exact integration scheme.
Here the caller supplies the spatial sampling and Gaussian calibration explicitly; changing the audio sample rate does not change the physical texture.
This is a documented discretization choice, not a claim to recover the unpublished reference smoothing settings.

`SurfaceGrid` uses meters for coordinates, spacing, and heights.
The grid is finite and at least three samples wide in each axis.
Paths outside it are rejected.
Second differences estimate curvature, with the nearest three-point stencil used at endpoints.
For each row and column, the raw edge height and caller-supplied edge slope anchor integration of the piecewise-linear constrained curvature.
`LeftSlopes` requires one x slope per row, and `TopSlopes` requires one y slope per column.
They describe the constrained trajectory's integration constants and are not inferred from a coarse texture stencil.
The polynomial primitive is integrated exactly inside each grid interval.
Sampling between rows or columns uses linear interpolation.
Preparation evaluates each supplied normal force directly, without quantizing it into a force atlas.
An inappropriate slope anchor can produce a large linear height drift even while the curvature equations remain satisfied.
Boundary conditions must therefore be chosen and validated for the supplied texture or replaced by caller-prepared trajectories.
No curvature mean is removed and no detrending is applied, since those operations would change equation 4 or its integration constants.

The paper independently prescribes two nonlinear axial curvatures, which need not be the Hessian of a single height field for a general nonseparable 2D texture.
It does not provide a compatibility reconstruction or boundary conditions.
This implementation uses independently integrated row and column derivatives for the two scraping terms, and the row-integrated height for the x-axis rolling equations.
That choice is exact for the specified discrete axial construction, but is not a general 2D constrained-surface solver.
Callers with a separately constrained surface may supply `Trajectory` values directly to bypass this preparation choice.

`ScrapingForce` implements equations 2–3, using the scalar sum of horizontal and vertical components:

```
horizontal = Beta1 * abs(vx * Sx + vy * Sy)^Beta2
vertical   = Mass * (Sxx * vx^2 + Syy * vy^2)
```

The mass, beta coefficients, signed velocities, and surface units are used directly.
The normal force controls curvature; it is not applied again as an extra amplitude multiplier.
Exactly zero applied normal force releases the contact and produces zero force.
As in the paper's equation 2, the implementation omits mixed curvature and tangential-acceleration terms.
`ScrapingNormalRange` and `ScrapingNormalForce` implement the harmonic-motion normal envelope from equations 8–9.
The position is measured relative to the trajectory center and increases away from the torso.
Parameter combinations that lose contact or produce a nonpositive force denominator are rejected.

`RollingForce` adds equations 15–19 to the scraping force:

```
rho     = Radius - Eccentricity * cos(x / Radius) + S
rho_dot = Eccentricity / Radius * vx * sin(x / Radius) + vx * Sx
rolling = rho^(3/2) * (Stiffness + Dissipation * rho_dot)
```

The default `Dissipation = 0.1` follows the value selected by ear in [arXiv v1, section 4.1](https://arxiv.org/html/2112.08984v1#S4.SS1).
The source does not specify its units, so this numerical default is not an SI-calibrated material property.
The default `Stiffness = 1000` is a demonstration value, not a recovered author parameter.

The rolling equations are defined along x; nonzero y velocity is rejected.
The radius must exceed the nonnegative center-of-mass offset.
The supplied height datum must make rho nonnegative.
The force is not rectified, DC-subtracted, normalized, or clipped after evaluation.
In particular, sufficiently negative penetration velocity can make the dissipative sum negative, as in the published algebra.
Within `rho >= 0` and `Stiffness + Dissipation * rho_dot >= 0`, the rolling term is nonattractive.
Its elastic potential is `(2/5) * Stiffness * rho^(5/2)` and its dissipated power is `Dissipation * rho^(3/2) * rho_dot^2`.
Those statements apply to the rolling term and prescribed penetration coordinate, not to the paper's scalar sum of differently directed force components or to the radiated audio energy.
The source model does not solve mechanical back reaction or impose global passivity.

The paper calls rho a deviation from mean vertical position, but equation 16 includes the radius as a positive baseline and supplies no mean-height subtraction convention.
Consequently, a stationary ball on a flat zero-height profile has the nonzero elastic force `Stiffness * (Radius - Eccentricity)^(3/2)`.
That is retained as printed; it must not be interpreted as a calibrated Hertz deformation or automatically forced to zero when velocity vanishes.
`RollingPosition` and `RollingVelocity` implement equation 18 and its derivative.
The paper uses `x / Radius` in equations 16–17 even though equation 18 makes this distinct from the angular displacement for an eccentric ball; this implementation retains that distinction.

The paper describes a periodic rolling normal force without specifying its equation.
`RollingNormalForce` supplies an explicit mechanics-based envelope, `Mass * (g + r * (cos(theta) * omega^2 + sin(theta) * angular_acceleration))`, derived from the acceleration of `R - r cos(theta)`.
It rejects loss of contact.
Caller-provided `Motion.NormalForce` can replace this envelope.
It must not be treated as a recovered parameterization of the authors' stimuli.

## Impulse responses and convolution

`ImpulseResponses` accepts matched endpoint mode frequencies and positive, mode-major sampled amplitude envelopes.
Mode matching is the caller's responsibility.
The paper used the strongest 50 modes; the implementation accepts any supplied count.
At each requested output location, both frequency and each lag's amplitude are interpolated logarithmically between corresponding surface modes.
Their decaying sinusoids are summed, then the object's position-independent response is added with the explicit `ObjectGain` weight.
Zero or negative amplitudes cannot be logarithmically interpolated and are rejected.
`ExponentialEnvelopes` is a convenience for explicit exponential modal inputs; measured envelopes are accepted without imposing exponential decay.

`BuildImpulseResponseBlock` constructs frame-major coefficients for `core/Convolution.h`.
For a block with K taps and F frames, excitation contains K−1 preceding force samples followed by F current samples:

```
y[n] = sum(k = 0 .. K-1) h(location[n], k) * force[n-k]
```

Every past excitation is therefore evaluated using the current output location's response, as in equation 13.
A recursive resonator whose coefficients change over time would not have the same impulse history.
Retaining K−1 force samples across blocks and appending K−1 zeros at the end renders the complete finite response tail.
The caller must also supply output locations for that tail, normally holding the final location.
The implementation performs no implicit fade or gain normalization.
The envelopes supplied here are digital FIR amplitudes for the specified sample rate.
When sampling a continuous-time impulse response, the caller includes the integration step `1 / sample_rate` in those amplitudes.
When resampling a response calibrated at a reference sample rate, its corresponding factor is `reference_rate / sample_rate`.
Without this explicit calibration, equation 13's discrete sum changes gain when sample rate changes.

`AgarwalBuildImpulseResponses` constructs the same coefficient matrix on Metal with independent work for every frame and lag.
`AgarwalGpu.h` defines the six-field, 24-byte parameter ABI and packs immutable endpoint data into contiguous frequency and envelope arrays.
The coefficient matrix is passed to the common FIR convolution kernel.
Use bounded blocks so storage is proportional to block size times IR length.
`AgarwalPrepareForces` performs texture sampling, force-dependent curvature clipping, Gaussian smoothing, axial integration, and scraping or rolling force evaluation on Metal, with one independent thread per trajectory sample.
`PrepareGpuTrajectoryData` validates and converts the caller's surface and motion inputs without evaluating the CPU texture solver.
Its motion buffer contains five contiguous planes: x, y, x velocity, y velocity, and normal force.
Buffer 6 contains the left-edge slopes followed by the top-edge slopes.
The kernel writes five trajectory planes, a force plane, and one status integer per frame.
The trajectory planes contain height, x slope, y slope, x curvature, and y curvature.
`ValidateGpuTrajectoryStatus` reports negative rolling penetration or nonfinite arithmetic after GPU completion.
GPU errors are not replaced with a CPU render or silent output.

CPU trajectory preparation uses double precision as the numerical reference.
GPU trajectory preparation, IR construction, and common convolution use float precision.
Small height variations must remain representable relative to the supplied height datum: placing a nanometer texture on a large numerical height offset can erase it during float conversion.
Second differences also amplify height quantization as grid spacing decreases.
The integer Gaussian support radius uses a ceiling operation, so float rounding near an integer half-width can select an adjacent support size.
The caller should validate its actual surface scale and force range against the double-precision path when using measured textures.
Independent samples permit parallel GPU preparation even when normal force changes every sample; the implementation makes no unmeasured real-time throughput claim.

## Reproduction limits

The published recordings do not come with the exact depth maps, motion/load schedules, integration anchors, paired surface responses, object responses, mode correspondences or fitted envelopes.
The checked public source and archive releases did not identify that package.
No author contact or data request is part of this workflow.

Public TDW profiles are author-associated synthesized textures, not verified physical scan lines or identified input arrays for these recordings.
Their quilting/preprocessing is unresolved.
Motion, radius, eccentricity, Gaussian shape, integration constants and response parameters remain inferred.
The paper's alpha coordinate units are unknown: for coordinate units ux and uz, alpha_SI = alpha_numeric * ux^2 / uz.
Normalized load and alpha alone cannot identify absolute force in newtons.
The paper gives beta1=0.05, beta2=1 and arXiv dissipation=0.1, but not calibrated coefficient units or all event-specific weights.

The full force implementation preserves the printed radius baseline and plus sign in penetration.
It does not solve a coupled force balance or obtain deformation from the normal load.
Enabling horizontal and rolling terms under retained inferred geometry can produce invalid penetration or implausible forces.
Adding those terms did not resolve the missing scraping texture.
Importing a Hertz equilibrium, subtracting the radius, or adding a separation switch would change this model.

The retained wood render still lacks the author's rapid, correlated broadband texture, despite containing related resonances.
Spectral fit alone does not certify texture or timing; cross-channel modulation and local grain measures expose the remaining difference.
Glass also retains response/motion ambiguities.
No other paper's synthesis is used as an Agarwal comparison target.

The extracted [PyImpact implementation](https://github.com/vinayakagarwal4/tdw/tree/465c379fd0833296a5372edecc51c7aba1999649) uses forward blocks, generic material modes and cosine responses.
Its upstream comparison is retained in `--upstream`; it is distinct from the complete 2021 construction.
The 2022 poster and [2025 thesis](https://hdl.handle.net/1721.1/158825) add finite micro-impact filtering; that extension is queued, not implemented here.
See the [method queue](../ContactAudioMethodRanking.md) for followups and [validation](Validation.md) for numerical limits.
