# Conan continuous interactions

This method implements Conan et al., [“An Intuitive Synthesizer of Continuous-Interaction Sounds: Rubbing, Scratching, and Rolling”][paper].
The paper appeared in Computer Music Journal 38(4), 24–37, 2014.
It covers rubbing, scratching, rolling, and continuous transitions through their probability distributions.
It shares rolling calibration and stochastic primitives with the separate [TASLP rolling model](Conan.md).

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ContinuousTest continuousReproduce -j8
build/ContinuousTest
python3 tools/continuous_reproduce.py --binary build/continuousReproduce
```

The reproduction requires NumPy, SciPy, Matplotlib, and FFmpeg.
Use `--offline` with cached inputs and `--output PATH` to change the output directory.
Author excerpts, native and force WAVs, inferred modes, controls, metrics, and the A/B page are saved in `outputs/reproduction/continuous/`.
The shared runner accepts `python3 tools/Reproduce.py --method continuous --offline`.

## Model and conventions

Equation 2 sums signed, symmetric raised-cosine impacts with Equation 6 duration `zeta * abs(amplitude)^(-theta)`.
One Gaussian innovation drives both inverse distributions through its normal CDF, followed by the ARMA filters and offsets in Equation 3.
The recurrence is `x[n] = e[n] + b1*e[n-1] - a1*x[n-1]`.
Equations 4–5 add rolling amplitude modulation at frequency `3*velocity/size`.

| Prototype angle | Amplitude and interval | Pulse |
|---|---|---|
| Rubbing, 0 | Independent signed Gaussian amplitudes, one event per sample | One sample |
| Scratching, 2 pi / 3 | Independent signed Gaussian amplitudes, exponential intervals | One sample |
| Rolling, 4 pi / 3 | Correlated Gaussians with positive offsets | Amplitude-dependent duration, exponent 0.29 |

Equations 8–11 interpolate PDFs and source parameters with triangular angular weights and radial interpolation toward equal thirds at the center.
Analytic mixture CDFs are inverted into 65 nonuniform quantiles, with exact Gaussian and deterministic forms at the endpoints.
Exponential and mixture probabilities are clipped to `[1e-7, 1-1e-7]`.

The paper leaves discrete and mixed-interaction cases partly unspecified.
Amplitudes retain their sign, while duration uses absolute amplitude floored at `1e-4`.
Intervals below one sample and durations above 8 ms are clamped and counted.
A 1024-sample ring uses fixed integer lookahead to preserve symmetric pulse centers and one-sample rubbing pulses.
Lookahead is 177 samples at 44100 Hz, and phase compensation persists across blocks.
Rolling calibration uses a minimum normalized velocity of 0.1, its fitted lower bound.
Modulation and gesture filtering use the requested velocity.
Supported size is `[0.1, 1]` and sample rate is `[8000, 96000]` Hz.

The gesture filter uses pole `exp(-2*pi*cutoff*velocity/sample_rate)`.
The paper specifies velocity scaling, with filter order and numerical calibration left unspecified.
Cutoff zero bypasses filtering, and velocity zero mutes force while retaining state.
The object uses the shared exponentially decaying sinusoidal modal bank described on page 32.
Source, filter, and resonator states persist across controls and blocks.
At the requested end, the renderer discards pending source pulses and retains modal decay to 60 dB attenuation.
CPU and Metal share the source recurrence, with sequential time steps per voice and independent voices parallelized on Metal.

## Author inputs and inferred settings

The [author page][author] provides the [companion video][video].
The original is retained at `references/continuous/ConanEtAlCMJ2014.mp4`, with hashes and provenance in [sources.json](../repros/continuous/sources.json).
Author code, numerical presets, gesture records, random seeds, and a redistribution license remain unavailable.

A 240 ms plastic-plate hit near 47.22 s supplies candidate modal frequencies and half-power bandwidths.
Decay times use inverse pi bandwidth, with unknown recorded excitation.
Of 40 candidate modes, 29 exceed the `1e-12` amplitude threshold after fitting the first two seconds of rubbing.
The fit uses discrete damped-sine transfer magnitudes and nonnegative amplitudes.
The later action intervals and transition are held out from this object fit.

The orange action cursor is measured at 20 Hz from video.
The green/yellow indicator supplies inferred relative velocity, independently of the audio envelope.
Fixed clips use their prototype positions, and the transition uses the measured cursor path.
Size and roughness are 0.5, asymmetry is zero, and friction pulse duration is one sample.
Calibration uses each action's first two seconds to select a cutoff and one constant RMS gain.
Cutoff candidates are bypass, 1000, 3000, 8000, and 20000 Hz.
Scratching also selects density from 100, 300, and 1000 events/s.
Selection minimizes normalized PSD RMS error in dB plus three times the absolute log ratio of 10 ms envelope coefficients of variation.
Candidates and selected controls are recorded in `metrics.json` and the control files.
The transition interpolates these gains and cutoffs, using Nyquist for bypass interpolation, and retains the selected scratch density.
Its audio is held out from calibration.

## Validation and remaining differences

`ContinuousTest` checks mixture CDFs, 150000-event statistics, ARMA variance, an FP64 signed-pulse reference, filter response, and invalid controls.
It also checks exact CPU block partitioning and retained CPU/Metal state through action transitions.
The 4-voice, 40-block, 127-frame comparison had relative L2 error 2.94481e-5 and maximum error 7.31409e-4.
The respective limits are 0.003 and 0.025.

The following comparisons use new stochastic realizations at 44100 Hz.
PSD error uses normalized Welch spectra from 80 to 12000 Hz with a reference-relative floor of -60 dB.
The final four seconds of each fixed action and the entire ten-second transition are held out.
The listed video intervals include the two-second calibration portions of fixed actions.
Envelope measures use 10 ms windows.

| Action and video interval | Held-out PSD RMS error | Envelope correlation | RMS native/author | Envelope CV author/native |
|---|---:|---:|---:|---:|
| Rubbing, 59–65 s | 5.20 dB | 0.734 | 0.964 | 0.764 / 0.717 |
| Rolling, 68–74 s | 2.49 dB | 0.658 | 0.993 | 0.382 / 0.327 |
| Scratching, 78–84 s | 3.72 dB | 0.633 | 0.886 | 0.897 / 1.312 |
| Transition, 107–117 s | 9.66 dB | 0.352 | 1.195 | 0.509 / 0.563 |

The transition differs substantially in upper-band energy and timing.
Scratching is too intermittent, with local envelope fluctuation 0.880 versus 0.599 and kurtosis 22.18 versus 14.64.
Rolling has less broadband energy and a higher median per-frame top-three-bin fraction, 0.420 versus 0.350.
Rubbing has higher spectral flatness despite similar envelope CV.
Texture metrics record these differences, and perceptual equivalence remains unestablished.
Selected clips have zero duration clamps, with 146/5968 short intervals clamped for scratching and 174/10245 for transition.
The A/B page retains raw examples and labels listening gain.

## Performance

Run `build/continuousReproduce --benchmark` for source-only CPU/Metal comparisons over 44100 frames per voice.
Both paths use identical FP32 parameters, initial states, and recurrence.
Timing excludes construction, allocation, and one GPU warmup, and includes GPU dispatch, completion, and output copying.
CPU timing covers sequential voice rendering.
One voice uses the disk center, while 64 voices cycle through center, rubbing, scratching, and rolling with seeds `7000 + voice`.
Three fresh-process M5 Max Release runs with Homebrew Clang 23.1.0 measured these ranges:

| Voices | CPU | GPU | Relative waveform L2 | Maximum absolute error |
|---|---:|---:|---:|---:|
| 1 | 0.345–0.363 ms | 6.487–20.135 ms | 9.94e-4 | 0.00356 |
| 64 | 29.780–32.282 ms | 39.014–90.572 ms | 1.75e-4 | 0.01573 |

All runs passed the 0.003 relative and 0.025 absolute waveform limits.
CPU execution was faster at both workloads.
Modal filtering, live audio transport, and block scheduling are outside this timing scope.
Measurements are retained in `outputs/reproduction/continuous/benchmark.json`.

[paper]: https://doi.org/10.1162/COMJ_a_00266
[author]: https://kronland.fr/publications/an-intuitive-synthesizer-of-continuous-interaction-sounds-rubbing-scratching-and-rolling/
[video]: https://kronland.fr/wp-content/uploads/2015/05/ConanEtAlCMJ2014.mp4
