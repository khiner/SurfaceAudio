# Geometric filtering by rough sliding surfaces

Implements [Ponthus et al. 2019][paper], *Statistics of the separation between sliding rigid rough surfaces*.
A rigid square slider translates over a longer rigid track with fixed orientation.
At each whole-grid displacement, the maximum overlapping height difference gives the smallest nonpenetrating separation.
This is quasistatic geometry; deformation, inertia, tangential forces and acoustic radiation require additional models.

## Run

```sh
brew install ghostscript
python3 tools/Reproduce.py --method ponthus
```

The workflow downloads pinned author figure sources and compares fifteen independent surface pairs at each of five slider sizes.
Outputs are `outputs/reproduction/ponthus/Comparison.png`, `Metrics.json` and two separation sonifications.
Open [the listening page](Listen.html) after generation.
Use `build/contactStatistics ponthus OUTPUT [nodes draws exponent ratio]` for other ensembles.
`--offline` requires cached references; `--keep-diagnostics` retains surface arrays and separation records.

## Model and conventions

`rough/GeometricContact.h` provides self-affine surface generation, sliding separation, Gaussian-maximum statistics and spectral approximations.
Metal generates Hermitian random-phase spectra, performs the two-dimensional inverse FFT and reduces the contact bound across each overlap.
The track is exactly five slider lengths; independent surface pairs use deterministic PCG seeds starting at 2019.
The reproductions use exponent 4, low/high cutoff ratio 0.46, unit theoretical RMS height and twelve samples per shortest wavelength.
Slider widths range from 64 to 1,024 nodes, giving approximately 5.5–84.1 central wavelengths.

Frequencies use cycles per unit length with the Fourier factor `exp(2πikx)`.
The paper mixes cycle and angular-frequency notation in its sampling and Fourier formulas.
Our lower cutoff is `1/L`, and central wavelength is `0.5 sqrt(M0/M2)` with the radial spectral Jacobian included.
Individual realizations retain their discrete mean and variance.

Separation is normalized by `sqrt(2) Rq`; the equivalent independent patch count is `4 (L/λ₀)² / π`.
The Gaussian-maximum density and its integrated moments follow Equations 12–15.
Preumont's approximation follows Equations 16–19, including its patch-count domain restriction.
The empirical spectral parameters follow Equations 9–11.

## Published comparison

Figure 7 markers are recovered from the author's EPS files, with axis calibration and marker-count checks.
Local means follow the published size dependence; standard deviation and skewness include variation across fifteen surface pairs.
`Metrics.json` records deviations from interpolation between published markers and measured high-frequency PSD slopes.
Mean separation differs from that interpolation by −0.035 to +0.117 in normalized units.
Measured high-frequency PSD exponents range from −4.30 to −4.07, compared with the published approximate −4 decay.
The author random seeds and surface arrays are unavailable, and the largest published slider sizes are outside the default ensemble.

A fifteen-pair ensemble took approximately 0.06–1.05 seconds over the default size range on M5 Max.
Regeneration records current timings, including surface synthesis and transfers.
The consolidated `RoughContactTest` checks the contact bound against exhaustive search and both FFT dimensions against direct Fourier sums.
It also checks the exact Gaussian-maximum moments for one and two patches.

The WAVs directly sonify mean-removed separation at 800 spatial samples per second with gain 0.2.
The paper provides statistical results; these WAVs are local illustrations without author reference audio or an object resonator.

[paper]: https://arxiv.org/abs/1904.12502
