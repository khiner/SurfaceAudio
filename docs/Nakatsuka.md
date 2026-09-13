# Nakatsuka adhesion-based friction

Implements the microrectangle and radiation model in [Nakatsuka and Morishima, DAFx 2017][paper].
GPU scene reconstruction covers a deformable sheet pulled across a sphere and a sheet pulled across a floor.
Exact scene reproduction requires missing author settings.
The [2016 predecessor supplement and demonstration][supplement] provide author audio and the same material table.

## Run

```sh
python3 tools/Reproduce.py --method nakatsuka
open outputs/reproduction/listening/index.html
```

Use `--offline` after downloading the pinned supplement and video.
The workflow retains three main WAVs and three corresponding author-video excerpts.
`repros/nakatsuka/cases.json` records all case settings, excerpt times, resource URLs and hashes.
The motion windows approximate the author excerpt timing; the copper excerpt includes the complete first audible pass.

## Equations and conventions

`AngularFrequency` implements Eq. 18 with flexural rigidity `E*h^3 / (12*(1-nu^2))` and caller-supplied areal density.
Each patch has independent normally distributed widths at a declared reference speed.
Dimensions at another speed obey Eq. 8, producing frequencies proportional to speed squared.
`MakePatches` rejects nonpositive sampled dimensions; the paper leaves that boundary unspecified.
Odd mode pairs follow Eqs. 14–15, with equal displacement amplitude per mode from Eq. 13.
Amplitude uses vertex displacement magnitude because the paper leaves its scalar projection unspecified.

`Render` evaluates the pressure sum from Eqs. 19–22 in double precision.
`RenderGpu` evaluates patches independently on Metal and sums their pressure contributions with the shared GPU mixer.
Mode phases integrate instantaneous frequency at sample endpoints; propagation uses the printed instantaneous monopole phase offset.
Modes above Nyquist contribute zero pressure while their phase continues advancing.
The printed `exp(-R)` attenuation is a constant amplitude factor, matching Eq. 17 literally.
The implementation retains the paper's radiation gain convention, whose surface-velocity/volume-velocity distinction is ambiguous.
Output gain is uncalibrated.

`AdhesionPotential` implements the printed Eq. 12: `Z/r` for r greater than d, and `Z` otherwise.
Using this residual in a standard PBD projection produces an outward correction for r greater than d.
`AdhesionCorrection(..., false)` implements the printed behavior and is the default scene convention.
Reconstructed scenes use `Z/r - Z/d`, with zero at the adhesion boundary.
The correction acts toward a persistent point on the obstacle, resisting tangential sliding as well as separation.
Contact points attach within one adhesion distance and release beyond four adhesion distances; sliding vertices can then attach at new points.
Collision projection prevents penetration after the adhesion correction.
The shifted residual and attachment rules are reconstruction choices.

`SimulateGpu` uses Jacobi distance projections, collision projection, gravity and a driven row on a rectangular mesh.
The row moves along negative Y, perpendicular to its edge.
`SettlingTime` advances the stationary scene before recording.
`MotionStart`, `MotionDuration`, `MotionRise` and `MotionFall` define a velocity pulse with integrated smoothstep ramps.
Zero duration selects continuous motion; the rise and fall times must otherwise fit within the duration.
The reproduction uses inferred motion windows and a half-second settling interval.
Equation 8 uses the tangential component of vertex velocity; Eq. 13 retains the magnitude of its displacement.
Vertices execute concurrently within a Metal threadgroup, with barriers between constraint iterations.
Uniform free-vertex inverse mass, relaxation, damping, contact range, gravity, geometry and motion are explicit scene settings.
The paper does not provide executable scene code, mesh trajectories, iteration settings, thickness, alpha/beta, adhesion values or mode truncation.
Its Eq. 10 also omits the squared gradient norm used by standard PBD; the implementation uses the standard scalar denominator.

The cloth scenes retain the table's 900 vertices, material numbers, 0.5 mm mean dimensions and 1e-12 m² variance.
The copper scene retains its 242 vertices, material numbers, 1e-8 m mean dimensions and 1e-7 m² variance.
All three use an inferred thickness of 1 micrometre and nine odd mode pairs.
The table's modulus labels, unusually high cloth Poisson ratio and density units conflict with other descriptions in the paper.
These table values are uncalibrated inputs.
The rectangular scene helper covers the demonstrations' basic contact arrangements; arbitrary meshes require externally prepared contact trajectories.

## Verification and performance

Tests check adhesion derivatives, tangential resistance, motion windows, plate frequencies, Gaussian dimensions and analytic radiation.
An independent double-precision PBD calculation checks the evolving contact constraints.
A separate test checks modes immediately below and above Nyquist; Metal compares that boundary in extended precision.
Each reproduction checks sixteen central patches over its full duration against the independent double-precision renderer.
The maximum relative L2 radiation error was 8.68e-6 with extended-precision phase accumulation on Metal.
The reproduction rejects nonfinite or silent output and records spectral and temporal comparisons.
Envelope RMSE compares peak-normalized 10 ms RMS envelopes without temporal alignment.
Spectral RMSE compares normalized power in 31 logarithmic bands from 80 Hz to 15 kHz after resampling both signals to 44.1 kHz.

| Scene | Envelope RMSE | Spectral RMSE |
| --- | ---: | ---: |
| Cloth | 0.111 | 0.86 dB |
| Cloth, half speed | 0.190 | 0.82 dB |
| Copper | 0.135 | 1.13 dB |

Declared limits in `cases.json` reject excessive envelope or spectral mismatch.
Perceptual equivalence remains unvalidated.
The author video uses AAC audio with reduced high-frequency bandwidth; the reconstructed WAVs retain their full synthesis bandwidth.

On the M5 Max, two-second cloth scenes took 2.05–2.07 s for simulation and 0.37–0.39 s for synthesis.
The copper scene took 0.71 s for simulation and 0.26 s for synthesis.
Simulation timings include the half-second settling interval.
Timings include each call's allocation, dispatch, synchronization and readback; WAV writing and numerical comparison are excluded.
The offline simulation retains patch-major trajectories and uses about 1.27 GB per two-second 900-vertex trajectory buffer.
`outputs/reproduction/nakatsuka/metrics.json` contains timings, numerical errors and audio descriptors.

[paper]: https://www.dafx.de/paper-archive/2017/papers/DAFx17_paper_58.pdf
[supplement]: https://diglib.eg.org/items/14a16c0b-7868-494b-af41-1f9079222cde
