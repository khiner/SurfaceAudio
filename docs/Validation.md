# Timings and numerical limits

Verified September 9, 2026 on Apple M5 Max, macOS 26.5.2, Homebrew Clang 23.1.0 and Metal 4.
Release uses C++23, `-O3 -mcpu=native` and Metal `-fno-fast-math`.
These are offline measurements of stated workloads, not audio-device deadline guarantees.

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

All 15 tests passed in Release and with host AddressSanitizer/UndefinedBehaviorSanitizer.
They cover independent equations, gradients, statistical moments, source traces, finite tails, CPU/GPU agreement and streaming state.
The Conan executable test exercises published parameters and rejects nonfinite output.
Tests require the actual Metal device and shaders.
Sanitizers instrument host code, not GPU kernels.
Paper reproductions use `python3 tools/Reproduce.py --offline`; response fitting and replay commands are in [AgarwalResponseReproduction.md](AgarwalResponseReproduction.md).

## CPU/GPU benchmark

Conan synthesis, a 32-mode resonator and mono mixing at 48 kHz in 128-frame blocks.
Both paths retain state through four warmup and 64 measured blocks, alternating execution order.
CPU timing includes synthesis, ARM SIMD filtering and mixing.
GPU timing includes synthesis, fused modal filtering/mixing, queue submission, completion waiting and mono readback.
Both exclude allocation, compilation and file I/O.

Ranges below span three fresh-process runs; each run retains its own p50/p99 calculation.

| Voices | CPU p50 ms | CPU p99 ms | GPU p50 ms | GPU p99 ms |
|---|---:|---:|---:|---:|
| 1 | 0.00208–0.00225 | 0.00267–0.00333 | 0.14783–0.19554 | 0.64283–3.00937 |
| 64 | 0.10437–0.11167 | 0.11563–0.14333 | 0.23700–0.30421 | 0.61942–5.59837 |
| 256 | 0.41950–0.47892 | 0.43062–0.93275 | 0.32796–0.49750 | 0.64642–5.68904 |
| 1024 | 1.65779–1.83600 | 1.75246–2.34696 | 0.61621–0.88246 | 1.22962–7.57558 |

The GPU median is consistently lower at 1,024 voices; the 256-voice comparison varies between runs and small batches favor CPU execution.
These measurements do not benchmark 50-mode spatial Agarwal reconstruction, offline optimizer throughput, or SDT's complete sustained-contact dynamics.
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
The pooled spectral objective is an optional inference extension, not an author-specified loss.
A fitted contact-interface example differs from ideal FP64 by 0.00523%.

## Additional paper reproductions

[Matusiak](Matusiak.md) records full-waveform comparisons with executed author MATLAB code, discrete energy checks, Metal precision and scoped batch timings.
[Poirot](Poirot.md) records recovered collision stimuli, the printed-equation discrepancies, pre-contact calibration, held-out collision errors and CPU/Metal timings.
The common texture descriptors include narrowband concentration, modulation, envelope fluctuation and amplitude statistics; matching them does not prove perceptual equivalence.

## Scope

Accurate evaluation of supplied parameters does not identify missing physical inputs or establish perceptual equivalence.
[Agarwal](Agarwal.md) documents unresolved geometry, units, response data and texture differences.
[SDT](Sdt.md) documents deliberate energy-limiter refinement, source chatter and sustained-impact timing sensitivity.
[Conan](Conan.md) documents statistical approximations, positive-support bounds and missing author seeds.
[Object responses](AgarwalResponseReproduction.md) documents filter, optimizer, alignment and distribution assumptions.
WAV amplitudes use digital or pickup gains and are not calibrated sound pressure levels.
