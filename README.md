# SurfaceAudio

C++23 contact-sound synthesis for Apple Silicon and Metal 4.

| Method | Implementation and caveats |
|---|---|
| Agarwal 2021 | [Constrained scraping/rolling forces and position-dependent surface/object responses](docs/Agarwal.md) |
| SDT | [Coupled modal bodies, impact, elasto-plastic friction, rolling and scraping](docs/Sdt.md) |
| Conan TASLP 2014 | [Correlated micro-impact rolling synthesis and physical reference](docs/Conan.md) |
| Agarwal 2026 | [Statistical object responses and nonlinear impact-force coupling](docs/Agarwal2026.md) |
| Agarwal 2022/2025 | [Finite micro-impact filtering with complete scraping and rolling forces](docs/Agarwal2025.md) |
| Agarwal ICML 2023 | [Differentiable modal/noise responses, fitting and material sampling](docs/AgarwalResponseReproduction.md) |
| Matusiak 2025 | [Passive bow-string friction, torsion and compliant bow hair](docs/Matusiak.md) |
| Poirot TASLP 2023 | [String-obstacle collisions and perceptual signal synthesis](docs/Poirot.md) |
| Conan CMJ 2014 | [Rubbing, scratching, rolling and continuous transitions](docs/Continuous.md) |
| Matusiak 2024 | [Finite-width bow friction and published robot-transient comparisons](docs/Matusiak2024.md) |
| Falaize–Roze 2024 | [Modal/FEM string interactions, energy balance and figure reconstructions](docs/Falaize.md) |
| Traer 2019 | [Statistical responses, impacts, spatial scrapes and later TDW code comparisons](docs/Traer.md) |
| Willemsen 2019 | [Stiff-string friction, published figure traces and executed author source](docs/Willemsen.md) |
| Lagrange 2010 | [Modal analysis, excitation extraction and uncompressed contact resynthesis](docs/Lagrange.md) |
| Lee 2010 | [Contact detection, position-dependent notch/LPC analysis and rolling resynthesis](docs/Lee.md) |
| HaTT 2014 | [Published texture models, force/speed interpolation and haptic vibration rendering](docs/Hatt.md) |
| Nakatsuka 2017 | [Deformable contact, microrectangle vibration and point-source radiation](docs/Nakatsuka.md) |

The library uses free functions, explicit state structs and contiguous arrays.
Independent contacts, texture samples, modal work and convolution run on the GPU.
CPU references use double precision where required and ARM SIMD for filtering.
The library supports offline rendering, with audio-device integration left to callers.

## Build

Requires macOS 26+, Apple Silicon with Metal 4, full Xcode with the Metal toolchain, CMake, Ninja and Homebrew LLVM at `/opt/homebrew/opt/llvm`.
CMake selects Homebrew Clang for C, C++ and Objective-C++.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
ctest --test-dir build --output-on-failure
```

## Reproduce and listen

```sh
python3 tools/Reproduce.py
open outputs/reproduction/listening/index.html
```

The workflow requires Python, NumPy, SciPy, Matplotlib, FFmpeg and network access for the initial author-input downloads.
Use `--offline` with cached inputs and `--method NAME [NAME ...]` to select methods.
The `--help` output lists the available method names.
Outputs include WAVs, a listening page and run/case manifests.
The [combined listening page](outputs/reproduction/listening/index.html) contains the methods selected by the latest reproduction run.
Each comparison identifies its reference: executed author code, published audio or figure traces, or a reconstruction with inferred inputs.
Per-method documentation records the comparison scope and remaining differences.
Willemsen's `--author-oracle` comparison requires Octave and pinned MATLAB sources.

Replay the fitted Agarwal inputs for four rolling reconstructions and twenty object responses:

```sh
python3 tools/agarwal_reproduce.py --retained
python3 tools/agarwal_response_reproduce.py --retained --output outputs/reproduction/retained-responses
```

The [object-response workflow](docs/AgarwalResponseReproduction.md) covers fitting, material cohorts and rendering with measured/fitted resonators.
Author audio and source inputs are stored under `references/`, with hashes in `docs/ReferenceInputs.json`.
HaTT inputs retain Penn’s non-profit research license; their terms are separate from this library’s license.
Reproduction fixtures and inferred calibration inputs are stored under `repros/`.
Generated outputs are Git-ignored.
Successful runs retain main WAVs, fitted parameters, provenance and metrics.
Use `--keep-diagnostics` to retain intermediate audio and trajectories; failed checks preserve them automatically.
Listening pages link to canonical raw WAVs and share identical level-matched audio under `outputs/playback/`.
Rebuilding a page removes playback files that ordinary pages no longer reference.
Use `BuildListeningReport.py --freeze` for a self-contained review snapshot.

## Performance and scope

[Timings and numerical limits](docs/Validation.md) gives the measured workloads, errors and verification commands.
The shared core provides Metal execution, FFT/FIR convolution, resampling, modal filtering, ESPRIT/LPC analysis, random streams and WAV I/O.
Real-time deadlines, arbitrary-material accuracy and perceptual equivalence remain unverified.
Raw WAVs retain gain and can exceed [-1, 1].
Listening copies apply stated constant gain/DC processing.

SDT-derived code retains its notices in [src/sdt/NOTICE.md](src/sdt/NOTICE.md).
The [Matusiak documentation](docs/Matusiak.md) identifies its GPL author reference.
The project uses [GNU GPL v3](LICENSE).
