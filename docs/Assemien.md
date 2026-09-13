# Rough sliding on a plate

Implements the plate/contact formulation in Chapter 3 of [Assemien's 2023 thesis][thesis].
The model combines simply supported Kirchhoff plate modes, rigid-body translation and rotations, and dissipative normal contact.
Surface grids can cover a patch within a larger vibrating body.
Modal shapes use the body's coordinates and normalization.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target roughReproduce ContactMechanicsTest RoughContactTest -j8
ctest --test-dir build -R '^(ContactMechanicsTest|RoughContactTest)$' --output-on-failure
python3 tools/rough_reproduce.py --method assemien
python3 tools/BuildListeningReport.py
```

The reproduction requires PyMuPDF and Pillow to extract published spectral plots from a pinned thesis PDF.
Use `--offline` with cached inputs or `--paper PATH` with a local copy of that PDF.
Use `--analyze-only` to regenerate comparisons from existing WAVs and metrics.
Use `build/roughReproduce assemien OUTPUT SECONDS SPEED SEED` for other configurations.
The command writes a 44.1 kHz `vibration.wav` and `Metrics.json` with physical velocity RMS, contact force, solver residuals, inputs and rendering time.
The WAV contains peak-normalized surface velocity at the printed receiver position, `(0.465, 0.195)` m.
The audio gain is recorded; airborne pressure requires an additional radiation model.
`Comparison.png` and `Comparison.json` compare the receiver level and spectral shape with the thesis figures.

## Published comparison

[Assemien.json](../repros/rough/Assemien.json) pins the PDF, embedded figure images and pixel-axis calibration.
Figures 3.7 and 3.13 provide measured and simulated spectra; Table 3.5 reports the simulated vibration level.
Extraction retains each image column's median blue-curve position and vertical extent.
The plotted spectra subtract each curve's mean dB value over a logarithmic 100–1,000 Hz grid.
Absolute vibration levels are evaluated separately using Equation 4.1, with reference velocity 1 nm/s.
The exact author stationary interval and simulated spectral-estimator settings are unavailable.
Our spectral analysis uses the 0.5–2 second interval and shows Hann windows of 256 and 8,192 samples.
`--keep-diagnostics` retains the extracted spectral coordinates.

The two-second example has receiver level 126.70 dB over the whole record and 127.25 dB over the spectral-analysis interval.
The thesis reports 127.19 dB for its simulation and 126.11–127.48 dB for the measurements.
The contact coefficients were calibrated against these targets by the author; agreement is a reproduction check, not independent prediction.
The 1–10 kHz spectral slopes are approximately −27 dB/decade measured, −42 published simulation and −45 to −47 for our simulation.
Our spectrum drops sharply above the retained modal range near 11 kHz.
The complete simply supported basis gives the third mode at 274.05 Hz and the 200th at 11,208.61 Hz.
Table 3.9 instead lists 308.31 Hz and 13,510.62 Hz for those indices.
Reaching its stated upper frequency requires 240 modes with the printed geometry and material properties.
The comparison records the discrepancy; the solver retains all 200 lowest modes, including repeated eigenfrequencies.
Surface realizations, estimator resolution and boundary conditions limit pointwise spectral comparisons.
Ensemble statistics and predictions across speeds remain unverified.
The two-second listening example takes 228.17 seconds on M5 Max, excluding surface preparation and audio conversion.

## Model

`MakePlateSurface` constructs separable mode shapes and modal dynamics for a specified body and contact patch.
`MakeRigidPlateSurface` takes the body's mass and both moments of inertia explicitly.
`MakePlateContact` selects one-pass or two-pass contact; the default uses the upper surface as slave.
`MakePlateState` initializes modal displacement and velocity.
`StepPlateContact` detects contact and advances the coupled state.
Callers supply the horizontal offsets and reference-plane separation at each step.

Heights protrude toward contact, and displacements use a common upward-positive axis.
Bilinear interpolation follows Equations 3.24–3.33.
Slave quadrature integrates pressure once; transpose interpolation distributes the reaction to preserve force, moments and virtual work.
The reaction is a force in newtons after area integration.

The pressure law is `K * penetration^n + chi * penetration^m * rate`, following Equation 3.27.
Pressure is clamped to zero when the expression becomes attractive during release.
This clamp is an explicit local constraint on the printed expression.
The rate includes relative vertical material velocities, following Equation 3.29.
It excludes the convective derivative of roughness or the moving interpolation coordinates.

Equation 3.34 defines centered modal velocities using the next displacement.
The velocity-dependent pressure therefore couples the next modal displacements through contact.
`SolveContactDamping` solves that system with fixed current penetration and nonnegative forces.
Each Newton step factors the smaller of the active-contact and modal systems in FP64.
The contact system follows algebraic elimination of modal increments and preserves the constitutive equations and residual tolerance.
It reports convergence and checks the returned force against the velocity update.
An unconverged plate step preserves the current and previous displacements.
This numerical interpretation satisfies the centered equations; the unavailable author implementation's damping evaluation order remains unknown.

## GPU and surface generation

`MakePlateContactGpu` uploads immutable surface grids and separable modal shapes.
`PreparePlateContactGpu` uses Metal to select possible contact nodes over bounded horizontal motion and modal displacement.
Each step validates those bounds and refreshes the candidate set when required.
The shared CPU contact evaluator computes the selected nodes in FP64; dense candidate sets use complete GPU detection in extended-float arithmetic.
Contact rows use a fixed material-node order before the CPU solve, independent of GPU scheduling.
Pass `reuse_candidates=false` to `MakePlateContactGpu` for complete GPU detection at every step.
This changes rounding relative to the GPU detector while preserving the contact equations, timestep, modes and solver tolerance.
GPU contact storage grows and detection repeats when required.
`AdvancePlateContact` then performs the shared coupled solve using Accelerate on the CPU.
The current GPU interface supports up to 256 total modes.

`GaussianSurface` implements the separable two-dimensional Gaussian convolution printed in Appendix 4.
It uses the requested grid dimensions, mixed random seeds and circular boundary conditions.
Filtering runs on Metal; individual surfaces retain their sample mean and variance.
The exact author surfaces and random seeds were not located.

## Performance

Run `build/roughReproduce assemien-benchmark 65536` for a bounded comparison without writing audio or diagnostics.
It uses a 32 × 30 mm contact patch, the 1,001 × 1,001-node slider, 203 modes and the unchanged 1 µs integration step.
The full reproduction retains its 460 × 30 mm track.
The benchmark alternates complete GPU detection, fresh candidate selection with FP64 evaluation, and cached candidate selection with FP64 evaluation.
It requires exact FP64 contact records and trajectories between fresh and cached candidate scans at every step.
Differences from the extended-float GPU trajectory are reported separately.

On M5 Max, 65,536 steps took 23.62 s with complete GPU detection, 18.62 s with fresh candidates and 4.68 s with cached candidates.
Candidate reuse gave 5.05× speedup against complete GPU detection and 3.98× against fresh candidate scans.
The run required 2,521 candidate refreshes, with up to 77 simultaneous contacts and maximum solver residual 1.33 × 10⁻¹⁴.
Relative RMS differences from the GPU trajectory were 3.31 × 10⁻⁸ for modal displacement and 1.06 × 10⁻⁷ for normal force.
The complete two-second render took 228.17 s, compared with 728.43 s for the earlier GPU implementation: a 3.19× reduction.
Both use the same surfaces, 203 modes, 1 µs timestep and 10⁻¹⁰ solver tolerance.
Maximum residual was 5.88 × 10⁻¹²; mean normal force was 0.306137 N against the 0.306072 N slider weight.

Rounding differences grow through repeated contact transitions: waveform relative RMS difference reaches 1.10 over two seconds.
It is 4.06 × 10⁻⁸ over the first 65 ms.
The stationary receiver level changes from 127.12 to 127.25 dB, and the 1–10 kHz spectral slopes change by at most 0.23 dB per decade.
These comparisons establish preserved equations and tolerances; exact cache equivalence is checked separately against fresh FP64 evaluation.

## Thesis configuration

The printed example specifies a 20 × 20 × 10 mm slider and a 600 × 400 × 4 mm plate.
The plate's rough track covers 460 × 30 mm, beginning at body coordinates `(0.070, 0.185)` m.
Its 20 µm grid contains 23,001 × 1,501 nodes; the slider contains 1,001 × 1,001 nodes.
The Gaussian surface parameters are 17 µm RMS height and 80 µm correlation length.
The example uses 200 plate modes, rigid slider motion, 100 mm/s sliding and a 1 µs integration step over two seconds.
The nonlinear pressure coefficients are `K = 1e16`, `chi = 3e17`, `n = m = 1.5`.
The printed recording rate is 44.1 kHz.
The reproduction begins at rest vertically with the highest overlapping profile pair just touching.
This initial clearance and the choice of one-pass contact are local settings because the example does not specify them.
Slider inertias use the stated dimensions and a homogeneous rectangular body.

The numerical plate has simply supported boundaries; the experimental plate is freely suspended.
The thesis explicitly identifies this boundary-condition approximation.
The model also omits tangential friction, adhesion and plastic deformation.

## Verification

`ContactMechanicsTest` checks modal bases, discrete energy balance and damping against independent solutions.
`RoughContactTest` checks force and moment conservation, CPU/Metal contact assembly, candidate reuse and direct surface convolution.
Run the reproduction command above to compare the published plate modes and spectra.

[thesis]: https://bibliotheque.ec-lyon.fr/documents/TH_2023ECDL0044.pdf
