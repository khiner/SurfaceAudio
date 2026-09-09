# SurfaceAudio

C++23 contact-sound synthesis for Apple Silicon and Metal 4.

| Method | Implementation and caveats |
|---|---|
| Agarwal 2021 | [Constrained scraping/rolling forces and position-dependent surface/object responses](docs/Agarwal.md) |
| SDT | [Coupled modal bodies, impact, elasto-plastic friction, rolling and scraping](docs/Sdt.md) |
| Conan TASLP 2014 | [Correlated micro-impact rolling synthesis and physical reference](docs/Conan.md) |
| Agarwal ICML 2023 | [Differentiable modal/noise responses, fitting and material sampling](docs/AgarwalResponseReproduction.md) |

The library uses free functions, explicit state structs and contiguous arrays.
Independent contacts, texture samples, modal work and convolution run on the GPU.
CPU references use double precision where required and ARM SIMD for filtering.
This is a library and offline renderer; it has no audio-device callback.

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
Use `--offline` with cached inputs or `--method agarwal`, `sdt` or `conan` for one method.
It retains output WAVs, a listening page and compact run/case information, and removes large disposable force/basis arrays after use.
Per-method pages identify author inputs, inferred settings and remaining differences.
Comparisons use the matching author's result from each paper.

The committed Agarwal input packages replay four retained rolling reconstructions and twenty fitted object responses without optimization:

```sh
python3 tools/agarwal_reproduce.py --retained
python3 tools/agarwal_response_reproduce.py --retained --output outputs/reproduction/retained-responses
```

The [object-response page](docs/AgarwalResponseReproduction.md) also documents fresh fitting, material cohorts and sustained-contact rendering with measured/fitted resonators.
Original author audio and source inputs remain under `references/`, with hashes in `docs/ReferenceInputs.json`.
Our calibration inputs under `repros/` are explicitly distinct from author data.

Existing historical WAVs are preserved without sample changes in the local [searchable audio archive](outputs/audio/index.html).
Its index retains their original paths, including old listening snapshots; duplicate WAV bytes are stored once.
The archive includes experiments and force traces, so its files are not all paper reproductions.
Identical current WAVs share storage through APFS clones; regenerating a file leaves archived audio unchanged.

## Performance and scope

[Timings and numerical limits](docs/Validation.md) gives the measured workloads, errors and verification commands.
[ContactAudioMethodRanking.md](ContactAudioMethodRanking.md) records the remaining paper queue.
The shared core provides Metal execution, deterministic random streams, modal filtering, finite convolution, spectral losses/gradients, optimization and WAV I/O.
No universal real-time deadline, arbitrary-material accuracy or perceptual equivalence is claimed.
Raw WAVs retain gain and can exceed [-1, 1]; listening copies apply only stated constant gain/DC processing.

SDT-derived code retains its notices in [src/sdt/NOTICE.md](src/sdt/NOTICE.md).
The project uses [GNU GPL v3](LICENSE).
