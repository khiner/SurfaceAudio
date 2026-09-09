# Agarwal object responses

Implements [Agarwal, Traer and McDermott, ICML workshop 2023](https://differentiable.xyz/papers-2023/paper_44.pdf): differentiable modal-plus-noise impulse responses, spectral fitting and joint material distributions.
The [author supplement](https://mcdermottlab.mit.edu/ICML2023/sound_website.html) provides five measured recordings and twenty generated examples for each of wood, plastic, metal and glass.
All 100 original WAVs are retained under `references/agarwal/icml2023/` with source URLs and SHA-256 pins.
They are mono 44.1 kHz PCM16; generated examples are one second long.

## Run

After the normal CMake build, render our twenty retained fitted responses without optimization:

```sh
python3 tools/agarwal_response_reproduce.py --retained --output outputs/reproduction/retained-responses
open outputs/reproduction/retained-responses/listening/index.html
```

`repros/agarwal-response/parameters.json` contains our fitted parameters, noise seeds, recording alignment and expected causal-WAV hashes.
The retained replay checks exact WAV identity on the verified toolchain.
These are our calibrated reconstructions, not author-provided parameters.

To refit the twenty author recordings and generate material cohorts:

```sh
python3 tools/agarwal_response_reproduce.py --binary build/agarwalResponseFit \
  --steps 5000 --learning-rate .001 --parameter-scale-mode log-decay \
  --loss-scale linear --align-onset --skip-loo \
  --output outputs/reproduction/agarwal-response-aligned
open outputs/reproduction/agarwal-response-aligned/listening/index.html
```

Omit `--skip-loo` for five leave-one-out folds per material, each using four training records and twenty generated samples.
Use `--case Wood_1` for one fit, `--prepare-only` for input preparation, `--sample-only` to reuse successful fits, or `--analyze-only` to rebuild existing cohort reports.
The latter operations verify source/preprocessing/parameter provenance before reuse.
`--self-test` checks joint covariance, deterministic sampling, held-out exclusion and recording alignment.

The paper-learning-rate experiment is explicit: `--steps 140000 --learning-rate 2e-6 --parameter-scale-mode physical`.
The native tools are:

```text
agarwalResponseFit fit INPUT.wav INITIAL.f32 OUTPUT_DIR STEPS SEED LEARNING_RATE PARAMETER_SCALE_MODE LOSS_SCALE
agarwalResponseFit sample PARAMETERS.f32 OUTPUT.wav FRAMES SEED
```

Parameters contain fifty little-endian float32 values: ten frequencies in Hz, ten mode amplitudes in dB, ten mode RT60 values in seconds, then ten noise amplitudes in dB and ten noise RT60 values in seconds.
The renderer preserves digital gain and finite duration.
Successful fits retain parameter files, initial/fitted WAVs, optimizer settings, alignment, input hashes and synthesis artifact hashes.

## Model and inference conventions

Ten decaying sinusoids are summed with ten exponentially decaying ERB noise bands.
The paper specifies ERB-spaced FIR cutoffs and shared Gaussian noise but leaves filter design unspecified.
Our design uses eleven ERB edges, odd-length symmetric Hamming-windowed sinc filters, unit filter L2 norm, and noise margins on both sides of centered convolution.
Noise remains fixed during a fit; band levels have unit expected RMS before their amplitude/decay parameters.

The GPU evaluates synthesis, analytic parameter gradients and multiresolution spectral loss at FFT sizes 4096, 1024, 256 and 64.
The chosen conventions are periodic Hann windows, quarter-window hops, centered zero padding, unnormalized one-sided FFTs, smoothed magnitudes and summed per-resolution mean Huber losses.
Spectrogram scaling is unspecified in the paper; `db`, `ln` and `linear` are separately recorded choices.
CPU Adam updates double master parameters.
The evaluated log-decay configuration rescales frequency/amplitude updates and optimizes log RT60 with its chain-rule gradient.
It changes optimizer coordinates, not the forward response equations.

Alignment subtracts whole-record median DC and trims samples before the first 0.1% cumulative centered-energy crossing.
The crossing sample is retained.
Full-record comparison WAVs restore the removed delay and median DC; causal `*-response.wav` files retain the complete remaining response and are used as convolution kernels.
This is an explicit inference convention, not an author-provided onset rule.

Material fitting pools all mode triples into one joint 3D Gaussian and all twenty noise parameters into one joint 20D Gaussian.
Maximum-likelihood covariance uses 1/n; centered-data factors retain deficient rank without invented regularization.
Sampling draws ten independent mode triples and one joint noise vector, rejecting whole vectors outside physical domains.
The C++ API exposes `FitResponseMaterial` and `SampleResponseMaterial`.

## Results and missing inputs

The evaluated sweep fits all twenty measured recordings and compares four generated material cohorts with the eighty author examples.
Author-generated audio is an unpaired cohort, not the exact target of each measured-response fit.
The measured peaks are approximately 0.9 and generated peaks approximately 0.8, consistent with separate author normalization.
Original floating-point recordings, measurement calibration, object identities, fitted parameters, seeds and synthesis code have not been recovered.
Raw RMS comparisons therefore do not establish physical acoustic levels.
Level-independent spectra, decay and distribution comparisons are more useful, but five measured examples per material do not establish broad generalization.
No perceptual equivalence claim follows from successful optimization.

These inputs are not identified as the response pairs for the 2021 contact examples or as a verified subset of the later database.
The [2025 thesis](https://hdl.handle.net/1721.1/158825) and [2026 preprint](https://doi.org/10.64898/2026.01.28.702236) report 1,502 retained responses from 410 objects, but no downloadable numerical database was located in the checked public sources.
The 2026 model uses twenty noise bands, separate per-band 2D distributions and size conditioning; it is queued, not implemented by this 2023 model.

## Sustained contact example

After the main paper repros and the aligned response fits above:

```sh
python3 tools/contact_response_render.py
open outputs/reproduction/contact-responses/listening/index.html
```

This drives measured and fitted responses with four seconds of Agarwal scraping or Conan rolling excitation, a 20 ms onset/release gate and the full convolution tail.
The material label identifies the resonator, not a separately measured contact surface.
The measured-versus-fitted players use the same synthesized excitation; neither is an original continuous-contact recording.
Responses are fixed, with no spatial variation or force feedback.
Every output is checked against independent full-tail FP64 convolution.
Raw WAVs preserve gain; audition copies apply one constant DC subtraction and gain with a peak ceiling.
