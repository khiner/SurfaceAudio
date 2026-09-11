# Willemsen, Bilbao and Serafin 2019

Reproduces the [DAFx paper][paper] bowed-string waveforms and hysteresis loop with C++23 and Metal 4.
An elasto-plastic bristle law couples to a finite-difference stiff string.
Output is dry string displacement; instrument-body filtering and radiation are outside the model.

The authors' [MATLAB source][source] at commit `12898a254e69cfbd68705ab028dd8894e4f05dfe` reproduces Figures 8 and 9.
Controls are 44,100 Hz sampling, nominal 440 Hz pitch, 5 N normal force, 0.1 m/s bow velocity, and bow position 0.25.
Comparisons use original physical samples without gain, time, pitch, or parameter fitting.

## Run

The Python runner retains main WAVs, parameters, metrics, and plots after successful checks.
Use `--keep-diagnostics` to retain full author and native trajectories.
Failed runs preserve generated diagnostics.

```sh
cmake --build build --target WillemsenTest willemsenReproduce -j8
build/WillemsenTest
python3 tools/willemsen_reproduce.py --binary build/willemsenReproduce --offline
python3 tools/willemsen_reproduce.py --binary build/willemsenReproduce --author-oracle --offline
build/willemsenReproduce outputs/reproduction/willemsen 44100 64
```

The Python workflow requires NumPy, SciPy, and Matplotlib.
Normal reproduction uses checked-in figure fixtures and needs neither network access nor Octave.
`--author-oracle` additionally executes the pinned MATLAB source in Octave; omit `--offline` to download uncached files.
The adapter changes figure controls, duration, plotting, and string-comparison syntax while preserving the numerical update.
Source hashes are verified before execution.
The final command benchmarks 64 CPU/Metal trajectories of 44,100 samples each.

| Output under `outputs/reproduction/willemsen` | Content |
| --- | --- |
| `figure_listen.wav`, `author_listen.wav` | Native and optional Octave renders, peak normalized to 0.8 |
| `figure_displacement_m.wav`, `author_displacement_m.wav` | Corresponding raw displacement WAVs |
| `paper_listen.wav` | Native literal-equation convention |
| `four_open_strings_listen.wav` | Additional eight-second G3, D4, A4, E5 demonstration with attack, bow noise, and release |
| `comparison.png`, `reference.json` | Published figure and optional author-source comparisons |
| `cases.json`, `diagnostics.json` | Author/native listening comparison and separate native-convention diagnostic |
| `cpu.json`, `gpu.json` | Solver residuals, failures, CPU/Metal errors, and benchmark timings |

## Published equations and source conventions

Both conventions use the same model and stepping kernel.

| Setting | `Scheme::Paper`, API default | `Scheme::AuthorFigure`, reproduction default |
| --- | --- | --- |
| Grid | N+1 samples, simply supported | N samples, two zero cells at each end |
| Bow | Cubic interpolation and adjoint spreading | Single point at the source's MATLAB index |
| Bristle damping | 0.1 kg/s, Table 1 | 1 kg/s, pinned source |
| Initial bristle rate | Zero | +0.1 m/s |
| Initial relative velocity | Zero | -0.1 m/s |
| Contact losses | Published normalized coefficients | Coefficients multiplied by linear density |
| Figure noise | Explicitly disabled | Disabled in source |

The literal path implements equations 7–9, 12–15, and the trapezoidal Newton solve in equations 19–22, including cubic self-coupling.
Separate contact-loss scaling in the source convention produces a contact/grid velocity discrepancy, reported independently of the Newton residual.
The nominal A4 figure repeats at approximately 464.21 Hz under its shorter-grid convention.
The source repository's additional `elastoPlastic.wav` has unspecified recording controls.

## Validation and limits

[The provenance manifest](../repros/willemsen/source.json) pins source commits, hashes, axis transformations, and figure sample counts.
CSV fixtures contain the original vector paths from `newPaperWaveform.eps` and `hysteresis3.eps`, with displacement, velocity, and force in SI units.
Credit: Silvin Willemsen, Stefan Bilbao, and Stefania Serafin, 2019, CC BY 3.0.
Figure 8 retains 442 points per waveform; Figure 9 retains 500 points, limited by three-decimal EPS coordinate precision.
Figure 9 uses nearest-sample curve distance without changing velocity or force scales.

| Comparison on Apple M5 Max, Homebrew Clang 23, Release | Result |
| --- | ---: |
| One-second C++/Octave displacement relative L2 | 9.69e-13 |
| Largest relative L2 across six author trajectory fields | 4.17e-12 |
| Figure 8 pickup / bow displacement relative L2 | 1.35e-5 / 1.90e-5 |
| Figure 9 nearest-sample RMS in velocity/force coordinates | 4.36e-7 |
| Maximum contact/grid velocity discrepancy | 1.24e-5 m/s |
| Source-convention Newton failures | Zero |
| Maximum FP64 Newton iterations / scaled residual | Four / 1.15e-13 m/s |

Author-source checks require finite relative L2 errors below `1e-9` for all six trajectory fields.
Tests check friction derivatives, sign symmetry, boundary stencils, and an independent trapezoidal bristle identity.
CPU/Metal checks compare every waveform sample and all final displacement, bristle, rate, and velocity states.
Metal runs one independent trajectory per thread; LLVM vectorizes CPU grid updates.
Benchmark timings include allocation, kernel setup, dispatch, synchronization, and readback and exclude device creation.

| Batch | CPU FP32 seconds | Metal FP32 seconds | Waveform relative L2 |
| --- | ---: | ---: | ---: |
| 64 × 44100 | 0.375 | 0.466 | 8.86e-05 |
| 1024 × 4096 | 0.565 | 0.109 | 6.97e-05 |

Small batches can favor CPU execution.
Audio-device deadlines and perceptual evaluation remain unverified.
The nonlinear solve caps Newton iterations at 50 and reports failures; unconditional stability is unproved.

[paper]: https://dafx.de/paper-archive/2019/DAFx2019_paper_18.pdf
[source]: https://github.com/SilvinWillemsen/ElastoPlastic/tree/12898a254e69cfbd68705ab028dd8894e4f05dfe
