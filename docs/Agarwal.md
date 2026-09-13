# Agarwal scraping and rolling

Implements [Agarwal et al., DAFx 2021][paper].
The model includes horizontal, vertical and rolling forces with position-dependent impulse responses.
[Author audio and ablations](https://mcdermottlab.mit.edu/scraping_rolling.html) provide the comparison recordings.
The [arXiv v1](https://arxiv.org/html/2112.08984v1) supplies additional numerical defaults.

## Run

```sh
python3 tools/Reproduce.py --method agarwal
open outputs/reproduction/listening/index.html
```

The workflow calibrates motion and spatial responses from author-associated inputs and compares PyImpact source with Metal convolution.
To replay the four fitted rolling reconstructions:

```sh
python3 tools/agarwal_reproduce.py --retained
```

`repros/agarwal/cases.json` records motion, sampling, source scope and expected force/WAV hashes.
The shared NPZ contains FP64 profiles and motion and the exact FP32 modal values uploaded by the renderer.
C++/Metal regenerates forces from the supplied profiles and motion.
They use vertical excitation only, with 64x sampling for wood and 128x for glass, sigma ratio 0.1 and anti-alias filtering.
Cell and pointwise refer to different orders of curvature interpolation and nonlinear constraint evaluation.
The omitted force terms limit these fitted reconstructions.

Refit with `agarwal_reproduce.py --temporal --whole-record`.
Then run `agarwal_spatial_fit.py --case roll --whole-record` or select `--case roll-glass`.
The low-level tools accept supplied inputs:

```text
agarwalReproduce prepare-temporal CASE_DIRECTORY [OVERSAMPLING [SIGMA_RATIO]] [--bandlimit]
agarwalReproduce render CASE_DIRECTORY OUTPUT_PREFIX
agarwalSpatialReproduce CASE_DIRECTORY OUTPUT_PREFIX [--basis]
agarwalContactFit FORCE.f32 MODES.txt TARGET.wav OUTPUT STEPS TAPS [LEARNING_RATE [linear|ln|db]]
agarwalEndpointFit BASIS.f32 LOCATION.f32 INITIAL.txt TARGET.wav OUTPUT STEPS [LEARNING_RATE [linear|ln|db [POOLED_WEIGHT]]]
```

Contact mode rows are frequency Hz, exponential decay seconds and amplitude.
Contact fitting varies log amplitude and decay with fixed force and frequencies, including the complete tail.
Endpoint fitting varies endpoint log amplitudes with fixed force, frequencies and envelopes.
The pooled spectral term is a local inference extension over time/frequency regions.
It adds `Weight * mean(0.5 * ((R - T) / D)^2)`, with equal weights per region.
R and T are spectral RMS values with `MagnitudeFloor^2` added before the square root.
D is the maximum of the target's whole-event band RMS, `RelativeFloor` times the largest band RMS, and `MagnitudeFloor`.
Band RMS uses equal weights per frame and bin.

## Equations and inputs

`ConstraintSettings` implements equations 4–7 with axial curvature `tanh(alpha * curvature) / alpha`.
Alpha interpolates between its bounds using normalized normal force raised to `Exponent`.
The defaults are the paper's `AlphaMax = 0.05`, `AlphaMin = 0.01`, and `Exponent = 0.95`.
Equal normal-force bounds select `ConstantAlpha`.
Normal forces outside the stated range saturate the interpolation at the corresponding endpoint.
Setting alpha to zero gives the unconstrained curvature limit.

The constrained curvatures are smoothed with a normalized symmetric Gaussian before integration.
The half-width is `GaussianHalfWidth * alpha / ReferenceAlpha` grid samples, rounded up for finite support.
Sigma is `GaussianSigmaRatio` times the unrounded half-width.
At a grid boundary, the available weights are renormalized.
The default half-width is five samples at alpha 0.03.
The reproduction executable separately smooths translated height, slope and curvature in audio time.
The paper specifies an average half-width of five samples at 44.1 kHz, proportional to alpha.
Spatial-to-temporal conversion, sigma, boundary extension and integration are unspecified.
Callers supply spatial sampling and Gaussian calibration independently of the audio sample rate.

`SurfaceGrid` uses meters for coordinates, spacing, and heights.
The grid is finite and at least three samples wide in each axis.
Paths outside it are rejected.
Second differences estimate curvature, with the nearest three-point stencil used at endpoints.
For each row and column, the raw edge height and caller-supplied edge slope anchor integration of the piecewise-linear constrained curvature.
`LeftSlopes` requires one x slope per row, and `TopSlopes` requires one y slope per column.
These slopes set the integration constants.
The polynomial primitive is integrated exactly inside each grid interval.
Sampling between rows or columns uses linear interpolation.
Preparation evaluates each supplied normal force directly.
Slope anchors determine linear height drift and require validation for the supplied texture.
Preparation preserves the curvature mean and integration constants.

The prescribed axial curvatures can be incompatible with a single 2D height field.
The paper omits compatibility reconstruction and boundary conditions.
Scraping uses separately integrated row and column derivatives; x-axis rolling uses row-integrated height.
This construction supports the discrete axial model only.
Supply `Trajectory` values directly when using a separately constrained surface.

`ScrapingForce` implements equations 2–3, using the scalar sum of horizontal and vertical components:

```
horizontal = Beta1 * abs(vx * Sx + vy * Sy)^Beta2
vertical   = Mass * (Sxx * vx^2 + Syy * vy^2)
```

The mass, beta coefficients, signed velocities, and surface units are used directly.
Normal force controls curvature and releases contact at zero load.
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
Its units are unspecified.
The default `Stiffness = 1000` is an uncalibrated demonstration value.

The rolling equations are defined along x; nonzero y velocity is rejected.
The radius must exceed the nonnegative center-of-mass offset.
The supplied height datum must make rho nonnegative.
Force values preserve the signed, unscaled result of the published algebra.
Sufficiently negative penetration velocity can make the dissipative sum negative.
Within `rho >= 0` and `Stiffness + Dissipation * rho_dot >= 0`, the rolling term is nonattractive.
Its elastic potential is `(2/5) * Stiffness * rho^(5/2)` and its dissipated power is `Dissipation * rho^(3/2) * rho_dot^2`.
These energy expressions apply only to the rolling term and prescribed penetration.
Mechanical back reaction and global passivity are outside this model.

Equation 16 includes a positive radius baseline despite describing rho as deviation from mean vertical position.
A stationary ball on a flat zero-height profile therefore has elastic force `Stiffness * (Radius - Eccentricity)^(3/2)`.
This printed baseline has no calibrated Hertz interpretation.
`RollingPosition` and `RollingVelocity` implement equation 18 and its derivative.
Equations 16–17 use `x / Radius`, which differs from the eccentric ball angle in equation 18.
The implementation preserves that distinction.

The periodic rolling normal-force equation is unspecified.
`RollingNormalForce` derives the following load from the acceleration of `R - r cos(theta)`:

```text
Mass * (g + r * (cos(theta) * omega^2 + sin(theta) * angular_acceleration))
```

This inferred envelope rejects loss of contact and can be replaced through `Motion.NormalForce`.

## Impulse responses and convolution

`ImpulseResponses` accepts matched endpoint mode frequencies and positive, mode-major sampled amplitude envelopes.
Supply corresponding endpoint modes; the paper used the strongest 50.
At each requested output location, both frequency and each lag's amplitude are interpolated logarithmically between corresponding surface modes.
Their decaying sinusoids are summed, then the object's position-independent response is added with the explicit `ObjectGain` weight.
Zero or negative amplitudes cannot be logarithmically interpolated and are rejected.
`ExponentialEnvelopes` is a convenience for explicit exponential modal inputs; measured envelopes are accepted without imposing exponential decay.

`BuildImpulseResponseBlock` constructs frame-major coefficients for `core/Convolution.h`.
For a block with K taps and F frames, excitation contains K−1 preceding force samples followed by F current samples:

```
y[n] = sum(k = 0 .. K-1) h(location[n], k) * force[n-k]
```

Equation 13 evaluates every past excitation with the current output location's response.
Retaining K−1 force samples across blocks and appending K−1 zeros at the end renders the complete finite response tail.
Supply tail locations, normally repeating the final location.
Rendering preserves the supplied gain and decay.
The envelopes supplied here are digital FIR amplitudes for the specified sample rate.
When sampling a continuous-time impulse response, the caller includes the integration step `1 / sample_rate` in those amplitudes.
When resampling a response calibrated at a reference sample rate, its corresponding factor is `reference_rate / sample_rate`.
Without this explicit calibration, equation 13's discrete sum changes gain when sample rate changes.

Metal constructs spatial impulse responses and prepares force-dependent trajectories in parallel.
Use bounded convolution blocks to limit response storage.
`ValidateGpuTrajectoryStatus` reports negative rolling penetration or nonfinite arithmetic after GPU completion.

CPU trajectory preparation uses double precision as the numerical reference.
GPU trajectory preparation, IR construction, and common convolution use float precision.
A large height offset can erase nanometer texture variations during float conversion.
Second differences also amplify height quantization as grid spacing decreases.
The integer Gaussian support radius uses a ceiling operation, so float rounding near an integer half-width can select an adjacent support size.
The caller should validate its actual surface scale and force range against the double-precision path when using measured textures.

## Reproduction limits

Exact depth maps, motion/load schedules, integration anchors and paired object/surface responses are unavailable.
Mode correspondences and fitted envelopes are also missing.

Public TDW profiles are author-associated synthesized textures with an unverified connection to these recordings.
Their quilting/preprocessing is unresolved.
Motion, radius, eccentricity, Gaussian shape, integration constants and response parameters remain inferred.
The paper's alpha coordinate units are unknown: for coordinate units ux and uz, alpha_SI = alpha_numeric * ux^2 / uz.
Normalized load and alpha alone cannot identify absolute force in newtons.
The paper gives beta1=0.05, beta2=1 and arXiv dissipation=0.1, but not calibrated coefficient units or all event-specific weights.

The full force implementation preserves the printed radius baseline and plus sign in penetration.
It does not solve a coupled force balance or obtain deformation from the normal load.
Enabling horizontal and rolling terms under retained inferred geometry can produce invalid penetration or implausible forces.
Importing a Hertz equilibrium, subtracting the radius, or adding a separation switch would change this model.

The retained wood render still lacks the author's rapid, correlated broadband texture, despite containing related resonances.
Spectral fit alone does not certify texture or timing; cross-channel modulation and local grain measures expose the remaining difference.
Glass also retains response/motion ambiguities.

The extracted [PyImpact implementation][pyimpact] uses forward blocks, generic material modes and cosine responses.
Its upstream comparison is retained in `--upstream`; it is distinct from the complete 2021 construction.
The [2022/2025 extension](Agarwal2025.md) adds finite micro-impact filtering.
See [validation](Validation.md) for numerical limits.

[paper]: https://mcdermottlab.mit.edu/papers/Agarwal_etal_2021_scraping_rolling_synthesis_DAFx.pdf
[pyimpact]: https://github.com/vinayakagarwal4/tdw/tree/465c379fd0833296a5372edecc51c7aba1999649
