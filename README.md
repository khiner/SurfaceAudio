# SurfaceAudio

C++23 contact-sound synthesis for Apple Silicon and Metal 4.

| Method | Implementation and caveats |
|---|---|
| Agarwal 2021 | [Constrained scraping/rolling forces and position-dependent surface/object responses](docs/Agarwal.md) |
| SDT | [Coupled modal bodies, impact, elasto-plastic friction, rolling and scraping](docs/Sdt.md) |
| Conan TASLP 2014 | [Correlated micro-impact rolling synthesis and physical reference](docs/Conan.md) |
| Agarwal ICML 2023 | [Differentiable modal/noise responses, fitting and material sampling](docs/AgarwalResponseReproduction.md) |
| Matusiak 2025 | [Passive bow-string friction, torsion and compliant bow hair](docs/Matusiak.md) |
| Poirot TASLP 2023 | [String-obstacle collisions and perceptual signal synthesis](docs/Poirot.md) |
| Conan CMJ 2014 | [Rubbing, scratching, rolling and continuous transitions](docs/Continuous.md) |
| Matusiak 2024 | [Finite-width bow friction and published robot-transient comparisons](docs/Matusiak2024.md) |

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
Use `--offline` with cached inputs and `--method NAME` to run one method.
Names are `agarwal`, `sdt`, `conan`, `matusiak`, `poirot`, `continuous`, `matusiak2024`.
Outputs include WAVs, a listening page and run/case manifests.
Temporary force and basis arrays are deleted after use.
Per-method pages record author inputs, inferred settings, differences and texture/spectral comparisons.
Matusiak 2025 compares executed author code, while Poirot and Conan CMJ compare author media using inferred parameters.
Matusiak 2024 compares published force traces.
Each method records the limits of those comparisons.
The [Matusiak listening reference](outputs/reproduction/matusiak2024-revised-2026-09-10/index.html) uses archived numerical conventions.

The committed Agarwal input packages replay four retained rolling reconstructions and twenty fitted object responses without optimization:

```sh
python3 tools/agarwal_reproduce.py --retained
python3 tools/agarwal_response_reproduce.py --retained --output outputs/reproduction/retained-responses
```

The [object-response workflow](docs/AgarwalResponseReproduction.md) covers fitting, material cohorts and rendering with measured/fitted resonators.
Original author audio and source inputs remain under `references/`, with hashes in `docs/ReferenceInputs.json`.
Reproduction fixtures and inferred calibration inputs are stored under `repros/`.

## Performance and scope

[Timings and numerical limits](docs/Validation.md) gives the measured workloads, errors and verification commands.
[ContactAudioMethodRanking.md](ContactAudioMethodRanking.md) records the remaining paper queue.
The shared core provides Metal execution, random streams, modal filtering, convolution, spectral optimization and WAV I/O.
Real-time deadlines, arbitrary-material accuracy and perceptual equivalence remain unverified.
Raw WAVs retain gain and can exceed [-1, 1].
Listening copies apply stated constant gain/DC processing.

SDT-derived code retains its notices in [src/sdt/NOTICE.md](src/sdt/NOTICE.md).
The [Matusiak documentation](docs/Matusiak.md) identifies its GPL author reference.
The project uses [GNU GPL v3](LICENSE).
