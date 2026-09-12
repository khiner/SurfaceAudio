# Agarwal 2026 object acoustics

Implements the generative model in [Agarwal et al. 2026](https://doi.org/10.64898/2026.01.28.702236).
The 1,502-response survey, object labels, fitted category distributions and author source remain unavailable.
Our reproduction uses the twenty measured inputs from the [2023 author supplement](https://mcdermottlab.mit.edu/ICML2023/sound_website.html).
Those inputs cover four materials with unknown sizes; they are not a verified subset of the 2026 survey.

## Run

After the normal CMake build, replay the retained response fits and generate impacts and statistical samples:

```sh
python3 tools/Reproduce.py --method agarwal2026 --offline
open outputs/reproduction/agarwal2026/listening/index.html
```

The workflow requires the pinned 2023 recordings under `references/agarwal/icml2023/`.
The direct runner can download these response inputs when `--offline` is omitted.
To refit instead of replaying:

```sh
python3 tools/agarwal2026_reproduce.py --fit --offline --steps 5000
```

`--paper-optimizer` selects physical parameter coordinates and the published Adam learning rate of 2e-6.
The default fit uses log-decay coordinates and learning rate .001, retaining the best complete objective value.
`repros/agarwal2026/parameters.json` records our fitted parameters, noise seeds, alignment, source hashes and expected response hashes.
Replay checks exact causal response WAV hashes.
Use `--keep-diagnostics` to retain intermediate parameters, loss traces and force WAVs.
Failed checks preserve their diagnostics.

## Equations and conventions

Ten exponentially decaying sinusoids and twenty exponentially decaying ERB noise bands form each response.
A mode is `10^(amplitude_dB/20 - 3*t/RT60) * sin(2*pi*frequency*t)`.
Noise uses the same amplitude envelope and one shared Gaussian stream filtered into twenty bands.
The retained 2023 FIR conventions remain explicit choices: centered Hamming-windowed sinc filters with unit L2 norm.
Seventy float32 parameters contain ten frequencies, ten mode amplitudes, ten mode RT60s, twenty noise amplitudes and twenty noise RT60s.

Metal evaluates synthesis, analytic gradients, four STFT losses at sizes 4096/1024/256/64, and the additional full-record spectral loss.
Each loss is a mean Huber loss with delta 3; the five losses are summed.
STFTs use periodic Hann windows, quarter-window hops and centered zero padding.
The full spectrum uses a rectangular record padded to the next power of two.
Linear, unnormalized, one-sided magnitudes with a 1e-6 floor are the reproduction convention.
Windowing, normalization and FIR design were not fully specified by the paper.
Input alignment follows the disclosed [2023 preprocessing](AgarwalResponseReproduction.md).

Each material/size category pools eligible modal frequency/amplitude/RT60 triples into one 3D Gaussian.
Each of the twenty noise bands has a separate amplitude/RT60 2D Gaussian.
`FitObjectResponseDistribution` fits one caller-selected category; size conditioning requires externally supplied category labels.
Covariance uses 1/(n-1), with zero-variance dimensions preserved and no added regularization.
Modes below 20 Hz or 10 ms RT60 and noise bands below 5 ms RT60 are excluded from distribution estimation.
Whole-vector rejection enforces 20–20,000 Hz, amplitudes at most 0 dB, mode RT60 in [.01,1] s and noise RT60 in [.005,1] s.
Small covariance fits and random parameter draws run on the CPU; waveform evaluation and convolution run on Metal.

The series stiffness is `k = kA*kB/(kA+kB)` and pulse duration is `T = pi*sqrt(m/k)`.
Within [0,T], the force is `cm*tanh(A*m*v*sin(sqrt(k/m)*t)/cm)`; it is zero outside this interval.
An infinite `cm` selects the linear limit.
Convolving this pulse with the sum of both object responses produces the impact sound.
`RenderImpactForcesGpu` evaluates independent pulses in parallel and returns voice-major samples.
Its default sample-interval averages preserve pulses shorter than one audio sample; point sampling is also available.
Linear integration is analytic; nonlinear integration uses eight-point Gauss–Legendre quadrature per interval.
Discrete convolution preserves digital gain without an additional sample-period multiplier.
The numerical gain A and material clipping limits remain uncalibrated; reproduction values are stated on each comparison.

`PlateStiffness` implements the printed square-plate expression using SI inputs.
`PublishedStiffness` exposes all seven materials and four size categories from Table 3 separately.
Some Table 3 entries disagree with the printed expression and Table 2 constants; neither source is silently substituted for the other.

## Comparisons and limits

The page contains twenty measured-input fits, three unpaired statistical cohorts, four mass controls and four nonlinear-force comparisons.
Measured-input fits compare author recordings with our reconstructions; statistical and control comparisons contain our model outputs.
Glass has only one eligible observation in its highest noise band after the published decay exclusion, so its statistical cohort is omitted.

Tests check equations, all seventy parameter derivatives, covariance and truncation against independent numerical references.
Spectral checks include independent DFT and long-record analytic loss/adjoint references.
The reproduction additionally checks covariance against NumPy and pulse integration/convolution against SciPy.
`metrics.json` retains fit losses, response features, numerical errors and executable/shader hashes.
Impact `render.json` reports elapsed synthesis time including GPU setup within rendering, allocation, transfers and synchronization, excluding WAV I/O.
On the M5 Max, the twenty 5,000-step fits took 1.83–6.71 s each and reduced their objectives by 54–97%.
The twelve impact renders took 3.87–7.91 ms each, with maximum quadrature/convolution relative L2 error 2.41e-6.
These offline measurements do not establish audio callback deadlines or calibrated sound pressure.
