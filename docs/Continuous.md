# Conan continuous interactions

This method implements Conan et al., [“An Intuitive Synthesizer of Continuous-Interaction Sounds: Rubbing, Scratching, and Rolling”](https://doi.org/10.1162/COMJ_a_00266), Computer Music Journal 38(4), 24–37, 2014.
It provides signed rubbing and scratching, correlated rolling, continuous navigation through their probability distributions, gesture filtering, and a streamed modal object.
The CMJ action space is separate from the TASLP rolling model in [Conan.md](Conan.md); only their shared rolling calibration and stochastic primitives are reused.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ContinuousTest continuousReproduce -j8
build/ContinuousTest
python3 tools/continuous_reproduce.py --binary build/continuousReproduce
```

The reproduction requires NumPy, SciPy, Matplotlib, and FFmpeg.
Use `--offline` after downloading the pinned companion video, and `--output PATH` to change the output directory.
Raw author excerpts, native WAVs, force WAVs, inferred modes, controls, provenance, metrics, and the A/B page are written under `outputs/reproduction/continuous/`.
The shared runner also accepts `python3 tools/Reproduce.py --method continuous --offline`.

## Model and numerical conventions

Equation 2 sums signed, symmetric raised-cosine impacts from Equation 6, with duration `zeta * abs(amplitude)^(-theta)`.
Each event draws one shared Gaussian innovation; its normal CDF drives both inverse distributions before the two ARMA(1,1) filters and mean offsets in Equation 3.
The recurrence is `x[n] = e[n] + b1*e[n-1] - a1*x[n-1]`.
Equations 4–5 add rolling amplitude modulation, with the prototype frequency `3*velocity/size`.

| Prototype | Amplitude and interval | Pulse and correlation |
|---|---|---|
| Rubbing, angle 0 | Signed zero-mean Gaussian; one event per sample | One-sample pulse; independent events |
| Scratching, angle 2 pi / 3 | Same signed Gaussian; exponential intervals | Same pulse; independent events |
| Rolling, angle 4 pi / 3 | Correlated Gaussian amplitudes and intervals with positive mean offsets | Amplitude-dependent duration, exponent 0.29; TASLP calibration |

Equations 8–11 interpolate the prototype PDFs, ARMA coefficients, offsets, pulse parameters, and modulation depth using triangular angular weights and radial interpolation toward equal thirds at the center.
The implementation evaluates analytic mixture CDFs and inverts them into 65 nonuniform quantiles; it does not interpolate prototype quantiles or crossfade three rendered sounds.
The endpoints use the exact Gaussian and deterministic-interval forms where available.
Exponential quantiles and mixtures use probabilities clipped to `[1e-7, 1-1e-7]`.

The paper leaves some discrete and hybrid cases unspecified.
Negative friction amplitudes retain their sign; only duration uses the absolute amplitude, floored at `1e-4`, to define fractional powers in mixed interactions.
Intervals below one sample are clamped to one sample and counted.
Durations above 8 ms are clamped and counted.
A 1024-sample ring scatter-adds each pulse with fixed integer lookahead, 177 samples at 44100 Hz, preserving symmetric pulse centers and the exact one-sample rubbing anchor.
The phase accumulator retains compensation across blocks.
The rolling calibration uses a minimum normalized velocity of 0.1, matching its supported fitted range; modulation and gesture filtering still use the requested velocity.
Normalized size is supported on `[0.1, 1]` and sample rates on `[8000, 96000]` Hz.

The gesture filter is a one-pole low-pass with pole `exp(-2*pi*cutoff*velocity/sample_rate)`.
The paper specifies the velocity relationship but does not publish its numerical calibration or filter order.
Cutoff zero explicitly bypasses this filter; velocity zero mutes the force while retaining its state.
The object uses the shared exponentially decaying sinusoidal modal bank, as described on page 32.
Source, filter, and resonator states persist across control changes and audio blocks.
The command-line renderer stops excitation at the requested end and retains the modal decay until 60 dB attenuation; it does not extend the gesture to drain pending source pulses.

CPU and Metal share the source recurrence and POD parameters/state.
The GPU advances independent voices in parallel, retaining sequential event and time dependencies within each voice, and uses shared GPU modal filtering.

The bounded source benchmark runs with `build/continuousReproduce --benchmark`.
It measures 44100 frames per voice with identical FP32 parameters, initial states, and recurrence on CPU and Metal, excluding construction, allocation, and one GPU warmup.
GPU elapsed time includes dispatch, completion, and copying all output samples; CPU elapsed time covers sequential voice rendering.
The single voice uses the disk center; the 64 voices cycle through center, rubbing, scratching, and rolling, using defaults and seeds `7000 + voice`.
The complete waveforms are compared after timing.
Three fresh-process Apple M5 Max Release runs with Homebrew Clang 23.1.0 measured the following elapsed-time ranges:

| Voices | CPU | GPU | Relative waveform L2 | Maximum absolute error |
|---|---:|---:|---:|---:|
| 1 | 0.345–0.363 ms | 6.487–20.135 ms | 9.94e-4 | 0.00356 |
| 64 | 29.780–32.282 ms | 39.014–90.572 ms | 1.75e-4 | 0.01573 |

All runs passed the same 0.003 relative and 0.025 absolute waveform gates.
Metal did not outperform CPU at either workload in these measurements.
These timings cover the source only, excluding modal filtering, live audio transport, and block scheduling; they do not establish real-time deadline behavior.
The measured output is retained in `outputs/reproduction/continuous/benchmark.json`.

## Author inputs and inferred settings

The [author publication page](https://kronland.fr/publications/an-intuitive-synthesizer-of-continuous-interaction-sounds-rubbing-scratching-and-rolling/) provides the [original companion video](https://kronland.fr/wp-content/uploads/2015/05/ConanEtAlCMJ2014.mp4).
Its SHA-256 is `0256b0f30f88e261553f6bfa84d6e3adcaf52790d8d6d5570e8306b4798e4bcb`.
The untouched file is retained at `references/continuous/ConanEtAlCMJ2014.mp4`, with source metadata in `references/continuous/manifest.json` and `repros/continuous/sources.json`.
No source code, numeric modal/material presets, gesture records, random seeds, or redistribution license were found on the author pages.
The old LMA CMJ2014 page returns 404; the current `/CMJ2014/` page is an image attachment.

The video contains isolated object hits before continuous gestures on the plastic plate.
A 240 ms hit near 47.22 seconds supplies candidate modal frequencies and half-power bandwidths; inverse pi bandwidth estimates decay times.
Because the spacebar excitation is unknown, the recorded hit is preserved for inspection and is not assumed to be a unit impulse response.
Forty candidate modes are fit to the first two seconds of the rubbing excerpt using the exact discrete damped-sine transfer magnitude and nonnegative amplitudes; 29 modes remain above the `1e-12` amplitude threshold.
This object fit uses no later held-out or transition waveform.

The orange action cursor is measured from video frames at 20 Hz.
Occupancy of the green/yellow action-panel indicator supplies an inferred relative gesture drive, not an authenticated velocity record; no author audio envelope is used as synthesis input.
Fixed clips use their labeled prototype anchors, while the transition uses the visible cursor path.
Size and roughness are 0.5, asymmetry is zero, and friction pulse duration is one sample.
For each fixed action, its first two seconds select one cutoff from bypass, 1000, 3000, 8000, and 20000 Hz and one constant RMS gain.
Scratching additionally selects density from 100, 300, and 1000 events/s.
The selection score is normalized PSD RMS error in dB plus three times the absolute log ratio of 10 ms envelope coefficients of variation.
All candidates and exact controls are recorded in `metrics.json` and the control files.
The selected cutoff/gain pairs are bypass/2.37029 for rubbing, 8000/10.0841 for rolling, and 1000/38.4307 for scratching, with scratching density 1000 events/s.
The transition inherits interpolated gains and cutoffs, treating bypass as Nyquist only for interpolation, and inherits the scratch density without fitting its audio.

## Validation and remaining mismatch

`ContinuousTest` independently checks disk anchors and convexity, analytic mixture CDFs, 150000-event amplitude/interval statistics, ARMA stationary variance, an FP64 signed pulse reference, unrectified rubbing, the low-pass step response, invalid controls, exact CPU block partitioning, and retained CPU/Metal state through an action transition.
The measured 4-voice, 40-block, 127-frame CPU/Metal source comparison had relative L2 error `2.94481e-5` and maximum absolute error `7.31409e-4`, below gates of 0.003 and 0.025.
These source tests establish numerical behavior independently of the fitted author examples.

The following results compare new stochastic realizations with matching author intervals at 44100 Hz.
PSD error uses normalized Welch spectra from 80 to 12000 Hz and a reference-relative floor of -60 dB.
The final four seconds of each fixed action are held out; the entire ten-second transition is held out.
No pointwise waveform match or perceptual equivalence is claimed.

| Action; video interval | Held-out PSD RMS error | 10 ms envelope correlation | RMS ratio, native/author | Envelope CV, author/native |
|---|---:|---:|---:|---:|
| Rubbing; 59–65 s | 5.20 dB | 0.734 | 0.964 | 0.764 / 0.717 |
| Rolling; 68–74 s | 2.49 dB | 0.658 | 0.993 | 0.382 / 0.327 |
| Scratching; 78–84 s | 3.72 dB | 0.633 | 0.886 | 0.897 / 1.312 |
| Transition; 107–117 s | 9.66 dB | 0.352 | 1.195 | 0.509 / 0.563 |

The transition remains a poor author match, especially in upper-band energy and temporal alignment.
Scratching remains too intermittent: held-out local envelope fluctuation is 0.880 versus the author's 0.599, and amplitude kurtosis is 22.18 versus 14.64.
Rolling has less broadband energy and a higher median per-frame top-three-bin fraction, 0.420 versus 0.350.
Rubbing has substantially higher spectral flatness despite similar envelope CV.
These differences are retained in the shared texture metrics, including PSD bands, peak concentration, amplitude quantiles/kurtosis, envelope autocorrelation, modulation bands, and local envelope fluctuations.
The matching spectrum and broad gesture envelope do not establish matching fine texture.
The selected native clips report no duration clamps; scratching reports 146 short-interval clamps in 5968 events, and transition 174 in 10245 events.
The generated A/B page preserves both raw examples for listening, alongside any clearly labeled listening gain applied by the common report tool.
