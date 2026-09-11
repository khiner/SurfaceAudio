# Matusiak and Chatziioannou bowed-string transients

Implements the finite-width bowed-string model in [JASA 156, 1135–1147 (2024)][paper].
It couples transverse motion, torsion, massless compliant bow hair and elasto-plastic friction.
The S-model and epsilon-model use the four published parameter sets in Tables III–V.

```sh
cmake --build build --target Matusiak2024Test matusiak2024Reproduce -j8
build/Matusiak2024Test
python3 tools/matusiak2024_reproduce.py --offline --check-author --gpu-voices 64
```

Outputs are in `outputs/reproduction/matusiak2024/`.
Raw WAVs contain bridge force in newtons; listening copies apply constant gain and DC removal.
The [listening reference](../outputs/reproduction/matusiak2024/listening/index.html) contains eight author-versus-native comparisons.
Use `--gpu-voices 0` for CPU rendering and `--convention paper` for the printed 2024 discretization.
The optional `--check-author` requires GNU Octave and the pinned author source under `references/matusiak/`.

## References and numerical conventions

[Reference.json](../repros/matusiak2024/Reference.json) records the paper, extraction coordinates and fixture hashes.
The CC BY 4.0 fixtures contain measured and author-simulated force traces recovered from Figures 3, 4 and 9.
PDF coordinates are linearly resampled to 44100 Hz without fitted alignment or gain.
The original acquisition samples and original 2024 simulation code are unavailable.
Coordinate rounding contributes approximately 2.6 microseconds and 0.00034 N of uncertainty, with additional polyline interpolation error.
Independent extraction of a repeated measured trace differs by relative L2 0.000107.
Regenerate the fixtures with PyMuPDF and `--extract-paper /path/to/matusiak2024_bowed_string_transients.pdf`.

The [authors' synthesis page][synthesis] links a 2025 MATLAB implementation alongside both publications.
The runner defaults to `archive`, which adapts its numerical conventions to the 2024 interaction laws.
The library defaults to `NumericalConvention::Paper2024`.

| Choice | `paper` | `archive` |
|---|---|---|
| Torsional velocity feedback | Radius times angular velocity, Equation 24 | Additional archived `h/hT` factor |
| Cubic interpolation | Physical bow coordinates | Archived interior-node indexing, offset by each grid spacing |
| Material density | Published 10059 kg/m3 | Derived from tension and the stated 137.2 m/s wave speed |
| Torsional energy and damping work | Physical weighting | Archived `h/hT` weighting |

The archive adaptation changes the numerical model and its energy weighting; its energy checks apply to that selected system.
Both presets use five inclusive contacts over a 1 cm bow, simply supported transverse endpoints and fixed torsional endpoints.
Integration defaults to 44100 Hz; `--sample-rate` selects another rate and comparison WAVs are resampled with polyphase filtering.
The paper leaves integration rate, contact count and interpolation order unspecified; its plotted traces exhibit a 50 kHz sample lattice.
Both presets use the archived torsional damping rate `1/(2*29)` per second, which is distinct from a physical modal-Q conversion.
Bow velocity is the published acceleration times elapsed time, without fitted terminal speed.

The hair law is `f=-K*eta-D*eta_dot`, with diagonal contact coupling `1/(M*(D+K*dt))`.
Bristle damping is constant, as in the 2024 paper; the 2025 hair inertia and passivity refinement are separate implementations.
The 2024 law permits negative instantaneous bristle dissipation outside the reproduced parameter regime.
The implicit solve uses an analytic Jacobian, pivoted elimination, backtracking and bounded sliding-branch initializations.
Both friction and bristle residuals must fall below `2e-13`.
Derived solver code retains GPL-3.0-only notices.

## Accuracy and limits

The optional author check executes the pinned MATLAB sparse matrices and Newton solve with explicit 2024 hair, friction and control substitutions.
All eight complete CPU renders agree with that adapted execution within `1.05e-7` relative L2.
This comparison covers the adapted archive; the original 2024 script remains unavailable.
Independent tests cover interpolation, reconstructed slip, hair balance, nonzero initial energy and sustained oscillation.
The eight reconstructions have cumulative energy errors below `3e-14 J`.

Spectral error is relative L2 between AC STFT magnitudes without fitted gain.
The amplitude ratio uses standard deviation over the final 40% of each record.
AC waveform relative L2 ranges from 0.279 to 1.288, so the published trajectories are not reproduced sample-exactly.

| Case | Spectral error | Sustained AC amplitude / author |
|---|---:|---:|
| Circle S | 0.081 | 1.002 |
| Circle epsilon | 0.179 | 1.002 |
| Red S | 0.110 | 0.993 |
| Red epsilon | 0.094 | 0.993 |
| Green S | 0.317 | 0.970 |
| Green epsilon | 0.531 | 0.949 |
| Blue S | 0.451 | 0.940 |
| Blue epsilon | 0.536 | 0.994 |

Metal 4 evaluates independent FP32 trajectories with complete energy and loss accounting.
The 64-voice archive-convention circle ensemble has full-record waveform relative L2 `2.36e-4` against FP64 over 17387 frames.
Stored-energy relative L2 is `4.21e-5`, maximum energy-accounting error is `8.41e-6 J`, and all nonlinear steps converge.
The GPU test covers that complete transient; the printed convention has a 60 ms acceptance test and fails full-record precision agreement.
One M5 Max Release run takes 2.77 s on FP64 CPU and 3.38 s on FP32 Metal for that complete batch.
Both timings include mechanics and energy accounting; Metal includes kernel setup, allocation, dispatch and readback.
These checks establish numerical agreement for the tested workload; audio-device deadlines remain unverified.

[paper]: https://doi.org/10.1121/10.0028228
[synthesis]: https://www.mdw.ac.at/iwk/?PageId=206
