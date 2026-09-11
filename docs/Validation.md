# Timings and numerical limits

Verified September 10, 2026 on Apple M5 Max, macOS 26.5.2, Homebrew Clang 23.1.0 and Metal 4.
Release uses C++23, `-O3 -mcpu=native` and Metal `-fno-fast-math`.
Measurements cover offline workloads, with audio-device deadlines unverified.

## Run

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
ctest --test-dir build --output-on-failure
./build/SurfaceAudioBenchmark

cmake -S . -B build-sanitize -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_OBJCXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'
cmake --build build-sanitize -j8
UBSAN_OPTIONS=halt_on_error=1 ctest --test-dir build-sanitize --output-on-failure
```

Tests cover independent equations, gradients, statistical moments, source traces, finite tails, CPU/GPU agreement and streaming state.
Tests require a Metal device and shaders, with sanitizer coverage limited to host code.
Run paper reproductions with `python3 tools/Reproduce.py --offline`.
See [object-response reproduction](AgarwalResponseReproduction.md) for fitting and replay commands.

## CPU/GPU benchmark

Conan synthesis, a 32-mode resonator and mono mixing at 48 kHz in 128-frame blocks.
Both paths retain state through four warmup and 64 measured blocks, alternating execution order.
CPU timing includes synthesis, ARM SIMD filtering and mixing.
GPU timing includes synthesis, fused modal filtering/mixing, queue submission, completion waiting and mono readback.
Both exclude allocation, compilation and file I/O.

These measurements span three fresh-process runs, each with its own p50/p99 calculation.

| Voices | CPU p50 ms | CPU p99 ms | GPU p50 ms | GPU p99 ms |
|---|---:|---:|---:|---:|
| 1 | 0.00213–0.00221 | 0.00337–0.00354 | 0.19479–0.58683 | 0.23221–0.75496 |
| 64 | 0.10542–0.11204 | 0.11800–0.16996 | 0.24192–0.78125 | 0.61917–1.74925 |
| 256 | 0.42417–0.43462 | 0.44642–0.63883 | 0.35063–0.67100 | 0.65371–1.34400 |
| 1024 | 1.70079–1.79662 | 1.91650–2.73679 | 0.62175–1.25233 | 1.19279–2.52712 |

GPU median time is lower in all three runs at 1024 voices and one run at 256 voices.
Smaller batches favor CPU execution.
Spatial Agarwal reconstruction, optimizer throughput and complete SDT sustained-contact dynamics are outside this benchmark's scope.
A 128-frame block spans 2.667 ms at 48 kHz.
System load, compilation caches and scheduling affect observed tails.

## Numerical accuracy

The following representative full-record measurements use relative L2 unless labelled otherwise.
The test source contains the exact fixtures and acceptance limits.

| Comparison | Observed result |
|---|---:|
| Agarwal GPU texture/force preparation versus FP64 | Maximum force error 1.20e-7 |
| Agarwal continuous Gaussian reconstruction, successively doubled spatial grids | 0.00442 → 0.00105 → 0.000268 → 0.0000741 |
| Agarwal temporal full-profile force preparation versus FP64 with matching atlas storage | Component relative L2 below 3.2e-7 |
| Complete spatial convolution versus independent direct sum | Relative L2 3.95e-7; maximum sample error 5.16e-8 |
| Spatial convolution with long, high-frequency lags | Relative L2 0.00178; separate 0.2% precision bound |
| Finite contact response with 11,025 taps | Relative RMS 1.97e-4; log-parameter gradient discrepancy 1.71e-3 |
| SDT retained evaluator versus actual upstream C fixture | Maximum error 5.55e-17 |
| SDT GPU one-second near-sticking friction | Force/output relative L2 0.001076 / 2.88e-6 |
| Conan three-second waveform versus independent pulse sum | Relative L2 2.10e-5 |
| Shared low-frequency modal response, 80 Hz full two-second tail | Maximum error 7.35e-6 for amplitude 0.1 |
| Shared finite modal response, two-second high-Q record | Maximum error 3.53e-6; post-support residual 2.60e-8 |
| Shared arbitrary-response GPU convolution versus direct double sum | Maximum error below 2e-6 |
| Shared random streams | Exact CPU/GPU PCG integers |

Independent gradient checks cover response synthesis, contact fitting, endpoint interpolation and both spectral objectives.
The pooled spectral objective is an optional loss introduced here for inference.
A fitted contact-interface example differs from ideal FP64 by 0.00523%.

## Additional paper reproductions

[Matusiak](Matusiak.md) records author-code comparisons, energy checks and CPU/Metal timings.
[Poirot](Poirot.md) records equation corrections, calibrated stimuli, held-out errors and CPU/Metal timings.
[Conan CMJ](Continuous.md) records inferred controls/responses, held-out texture comparisons and CPU/Metal timings.
[Matusiak 2024](Matusiak2024.md) records recovered paper traces, numerical conventions and remaining waveform differences.
Its archived-convention GPU check covers the full circle transient; the printed convention retains a 60 ms check and full-record precision failure.
[Falaize–Roze](Falaize.md) records independent dense solves, figure constraints, inferred hammer stiffness and matched CPU/Metal workloads.
[Traer](Traer.md) records later TDW code comparisons, independent spatial-scrape checks and matched CPU/Metal workloads.
[Willemsen](Willemsen.md) records exact paper-era source execution, published EPS comparisons and complete CPU/Metal trajectory checks.
These method timings were measured on September 10.
Texture metrics cover narrowband concentration, modulation, envelope fluctuation and amplitude statistics.

## Scope

Physical-input identification and perceptual equivalence require separate validation.
[Agarwal](Agarwal.md) documents unresolved geometry, units, response data and texture differences.
[SDT](Sdt.md) documents deliberate energy-limiter refinement, source chatter and sustained-impact timing sensitivity.
[Conan](Conan.md) documents statistical approximations, positive-support bounds and missing author seeds.
[Object responses](AgarwalResponseReproduction.md) documents filter, optimizer, alignment and distribution assumptions.
WAV amplitudes use digital or pickup gains, with sound-pressure calibration unavailable.
