# Matusiak bowed-string friction

This method implements [“Numerical modelling of elasto-plastic friction in bow–string interaction with guaranteed passivity”][paper].
The paper is by Matusiak, Chatziioannou and Van Walstijn (Frontiers in Signal Processing, 2025).
It includes a finite-width bow, transverse stiff-string motion, torsion, compliant bow hair, and implicit elasto-plastic friction.
FP64 C++ is compared with the authors' MATLAB computation executed in GNU Octave.
Metal 4 renders independent string trajectories and the paper's lumped bowed-mass model.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target MatusiakTest matusiakReproduce -j8
build/MatusiakTest
python3 tools/matusiak_reproduce.py --binary build/matusiakReproduce
```

Use `--offline` with cached references and `--author-reference` to regenerate both author computations with `octave-cli`.
The committed fixture supports native checks without MATLAB or Octave.
Use `--gpu-voices 0` for CPU-only reproduction or change the default 128-voice batch.
WAVs, a manifest, and listening cases are saved in `outputs/reproduction/matusiak/`.
`bridge_force.wav` stores newtons, and `bridge_listening.wav` applies one recorded gain.

## Author reference and conventions

The paper links [Zenodo 15341818][zenodo], with versioned release [15341819][release].
The [author repository][source] is pinned to `0e2d39d0d62460370f1de3945a4f1a2c984b038c`.
Its GPL-3.0 license is retained in [src/matusiak/LICENSE](../src/matusiak/LICENSE).
[Reference.json](../repros/matusiak/Reference.json) records source and fixture hashes.
The fixture contains complete 22050-sample original and corrected outputs from GNU Octave 11.3.0, with plotting excluded.

The reproduction uses code defaults, which differ from rounded paper Table 2 values:

| Quantity | Retained author convention |
|---|---|
| Duration and rate | 0.5 s, 44100 Hz, initial zero sample |
| String | Length 0.7 m, radius 0.0005 m, fundamental 98 Hz, tension 149.7415 N, Young modulus 1.37e10 Pa |
| Transverse loss | 1.53714 /s and 0.0087 m²/s |
| Torsion | Stiffness 3.0312e-4 N m², polar inertia 4.2e-10 kg m, damping 1/58 /s |
| Bow | 2.3433 N, 0.3439 m/s, acceleration 0.8722 m/s², width 0.01 m, position 0.0786 L |
| Bow hair | 0.0045 kg, 48297 N/m, 5.7674 kg/s, each divided by bow width |
| Friction | Stiffness 318600, damping 0.0027, Stribeck velocity 0.228 m/s, dynamic/static coefficients 0.5071/1.0207 |
| Contacts | Four subdivisions produce five inclusive endpoints |
| Breakaway | 0.7 times dynamic-friction steady bristle deflection |

String density is derived from tension and `c=2Lf0`.
The source divides total bow force in newtons across contacts, despite the paper table's N/m label.
Cubic interpolation retains the source's first-interior-node coordinate zero and its approximately one-cell location offset.
The grids have 158 transverse and 36 torsional intervals at 44100 Hz.
The velocity ramp includes both endpoints over `ceil(v/a * fs)` samples.

## Source correction

The final bow-hair term in the author's `IJ_mat` adds a scalar to every entry, while Equation 59 requires that scalar times the identity.
The native solver follows Equation 59, and both original and corrected author outputs are retained.
The corrected author run changes only this term to `1/M * 1/O3 * eye(M)`.
Final assumed and reconstructed slip velocities differ by 4.38e-5 m/s in the original and 1.31e-15 m/s in the corrected run.
Both runs satisfy their reported energy balance and have nonnegative bristle dissipation in this example.
The discrepancy does not establish a passivity violation in the original run.
Against the unchanged author output, native relative waveform L2 error is 0.0276.
Active-bin short-time spectral RMSE is 0.401 dB, and relative RMS-envelope error is 0.000495.

## Equations and validation

Friction uses the adhesion and steady-bristle relations in Equations 6–9, damping in Equation 14, and bristle evolution in Equation 24.
Damping is `muC * normal_force / sqrt(v² + epsilon²)`, with `epsilon = muC * normal_force / nominal_damping`.
Dissipation is evaluated independently as `v * friction_force - stiffness * midpoint_bristle * bristle_rate`.
Zero nominal damping is supported, while normal force must remain positive.
Bow lifting requires an additional contact/separation model.

Equations 52–61 use simply supported transverse ends, fixed torsional ends, central string differences, averaged hair stiffness, and staggered bristles.
The nonlinear system uses an analytic Jacobian, pivoted elimination, and the author's 1.1 Jacobian scaling after iteration 50.
FP64 convergence requires residual below 2e-13, with failure reported at 100 updates.
Independent stored energies, dissipation, and external work follow Section 6.3, including nonzero initial states.
Frequency-dependent string loss follows the paper's discrete energy accounting, with the local passivity guarantee applying to bristle dissipation.

Tests cover CFL bounds, derivatives, elastic/steady-slip limits, high-damping passivity, full author traces, energy balance, and CPU/Metal trajectories.
The complete corrected author bridge matches FP64 C++ with maximum error 7.88e-11 N and relative L2 error 3.73e-12.
Energy error remains below 4.4e-14 J over 0.5 s.
The comparison accounts for author energy sampled before each update and native energy sampled after it.

## Metal performance and accuracy

`RenderStringsGpu` supports eight contacts per string, with one sequential trajectory per GPU thread.
It eliminates the diagonal bristle block before solving the coupled slip system and evaluates four nonzero cubic interpolation coefficients.
Outputs include failed-step counts, maximum residual, and final state.
FP32 passivity at machine precision remains uncertified.
Clang 23.1.0 confirms two-wide FP64 SIMD in CPU interpolation, elimination, force spreading, and updates.
`RenderLumpedGpu` implements Equations 25–35 using Table 1 parameters.

The 128-voice workload used three fresh-process M5 Max Release runs, 22050 frames per voice, and forces `2.3433f * (1 + 0.0001f * voice)`.

| Voices | FP64 CPU | FP32 GPU |
|---|---:|---:|
| 1 | 0.0566 s | 2.85 s |
| 128 | 7.40–7.89 s | 4.04–7.95 s |

The single-voice row is one additional measurement.
Both paths use identical float-rounded controls.
CPU timing includes state construction, trajectory generation, and full energy accounting, with comparison reductions excluded.
GPU timing includes kernel creation, allocation, upload, dispatch, waiting, and readback, with global energy accounting excluded.
A CPU/GPU speedup comparison requires matching precision and diagnostics.

Aggregate waveform relative L2 error is 2.51e-4, worst-voice error is 5.61e-4, and maximum force error is 0.00953 N.
Final-state relative L2 errors are 9.07e-5 for transverse displacement, 8.59e-4 for torsion, 6.30e-4 for bristles, and 0.00507 for slip velocity.
All steps converged below the FP32 residual threshold of 2e-6.
Other parameters and chaotic bowing regimes require separate validation.
Each reproduction manifest records its timings and errors.

## Audio scope

The official [fast sautillé example][audio] is downloaded and decoded as `author_fast_sautille.wav` when FFmpeg is available.
It is a separate listening reference requiring periodic force/velocity drives and a measured cello impulse response.
Those inputs are absent from the archived code and the inspected later main revision `3c29a71bcd3b35891710b4e4642c31ec7a1be11d`.
The matched reproduction covers the supplied 0.5 s bridge-force computation, while the sautillé recording remains unreproduced.
The manifest compares waveform, short-time spectrum, RMS envelope, and texture for matching computations.

[paper]: https://doi.org/10.3389/frsip.2025.1525044
[zenodo]: https://doi.org/10.5281/zenodo.15341818
[release]: https://zenodo.org/records/15341819
[source]: https://github.com/wasilakis/the_bowed_string/tree/1.0
[audio]: https://www.mdw.ac.at/upload/MDWeb/iwk/mp3/FastSautille.mp3
