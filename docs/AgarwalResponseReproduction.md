# Agarwal object responses

Implements [Agarwal, Traer and McDermott, ICML workshop 2023][paper].
The model provides modal-plus-noise impulse responses, spectral fitting and joint material distributions.
The [author supplement][author] provides five measured and twenty generated examples per material: wood, plastic, metal and glass.
All 100 original WAVs are retained under `references/agarwal/icml2023/` with source URLs and SHA-256 pins.
They are mono 44.1 kHz PCM16; generated examples are one second long.

## Run

After building, replay the twenty fitted responses:

```sh
python3 tools/agarwal_response_reproduce.py --retained --output outputs/reproduction/retained-responses
open outputs/reproduction/retained-responses/listening/index.html
```

`repros/agarwal-response/parameters.json` contains our fitted parameters, noise seeds, recording alignment and expected causal-WAV hashes.
The retained replay checks exact WAV identity on the verified toolchain.
The replay parameters come from local fits.

To refit the twenty author recordings and generate material cohorts:

```sh
python3 tools/agarwal_response_reproduce.py --binary build/agarwalResponseFit \
  --steps 5000 --learning-rate .001 --parameter-scale-mode log-decay \
  --loss-scale linear --align-onset --skip-loo \
  --output outputs/reproduction/agarwal-response-aligned
open outputs/reproduction/agarwal-response-aligned/listening/index.html
```

Omit `--skip-loo` for five leave-one-out folds per material, each using four training records and twenty generated samples.
Use `--case Wood_1` for one fit or `--prepare-only` for input preparation.
Use `--sample-only` to reuse successful fits or `--analyze-only` to rebuild cohort reports.
The latter operations verify source/preprocessing/parameter provenance before reuse.
Initialization WAVs are regenerated from pinned parameters and checked against their original hashes when needed.
Use `--keep-diagnostics` to retain initialization audio and its listening players.
`--self-test` checks joint covariance, deterministic sampling, held-out exclusion and recording alignment.

Use the published learning rate and physical parameter coordinates with `--steps 140000 --learning-rate 2e-6 --parameter-scale-mode physical`.
The native tools are:

```text
agarwalResponseFit fit INPUT.wav INITIAL.f32 OUTPUT_DIR STEPS SEED LEARNING_RATE PARAMETER_SCALE_MODE LOSS_SCALE
agarwalResponseFit sample PARAMETERS.f32 OUTPUT.wav FRAMES SEED
```

Parameters are fifty little-endian float32 values in five contiguous groups of ten.
The groups contain frequency Hz, mode amplitude dB, mode RT60 seconds, noise amplitude dB and noise RT60 seconds.
The renderer preserves digital gain and finite duration.
Successful fits retain parameter files, fitted WAVs, optimizer settings, alignment, input hashes and synthesis artifact hashes.

## Model and inference conventions

Ten decaying sinusoids are summed with ten exponentially decaying ERB noise bands.
The paper specifies ERB-spaced FIR cutoffs and shared Gaussian noise but leaves filter design unspecified.
Filters use eleven ERB edges and odd-length symmetric Hamming-windowed sinc coefficients with unit L2 norm.
Gaussian noise extends through both margins of centered convolution.
Noise remains fixed during a fit; band levels have unit expected RMS before their amplitude/decay parameters.

The GPU evaluates synthesis, analytic parameter gradients and multiresolution spectral loss at FFT sizes 4096, 1024, 256 and 64.
STFTs use periodic Hann windows, quarter-window hops, centered zero padding and unnormalized one-sided FFTs.
The objective sums per-resolution mean Huber losses of smoothed magnitudes.
Spectrogram scaling is unspecified in the paper; `db`, `ln` and `linear` are separately recorded choices.
CPU Adam updates double master parameters.
The evaluated log-decay configuration rescales frequency/amplitude updates and optimizes log RT60 with its chain-rule gradient.
The forward response equations are unchanged.

Alignment subtracts whole-record median DC and trims samples before the first 0.1% cumulative centered-energy crossing.
The crossing sample is retained.
Full-record comparison WAVs restore the removed delay and median DC.
Causal `*-response.wav` files preserve the remaining response for use as convolution kernels.
This onset rule is a local inference choice.

Material fitting pools all mode triples into one joint 3D Gaussian and all twenty noise parameters into one joint 20D Gaussian.
Maximum-likelihood covariance uses 1/n and preserves deficient rank.
Sampling draws ten independent mode triples and one joint noise vector, rejecting whole vectors outside physical domains.
The C++ API exposes `FitResponseMaterial` and `SampleResponseMaterial`.

## Results and missing inputs

The evaluated sweep fits all twenty measured recordings and compares four generated material cohorts with the eighty author examples.
Author-generated audio supplies an unpaired cohort comparison.
Original floating-point recordings, measurement calibration, object identities, fitted parameters, seeds and synthesis code have not been recovered.
Acoustic levels are uncalibrated.
Level-independent spectra and decay characterize these five measured examples per material.
Generalization and perceptual equivalence are unverified.

These inputs are not identified as the response pairs for the 2021 contact examples or as a verified subset of the later database.
The [2025 thesis][thesis] and [2026 preprint][impact] report 1,502 responses from 410 objects.
The numerical database is unavailable in the checked public sources.
The [2026 implementation](Agarwal2026.md) uses twenty noise bands, separate per-band 2D distributions and size conditioning.

## Optional sustained contact example

After the main paper repros and the aligned response fits above:

```sh
python3 tools/contact_response_render.py
open outputs/reproduction/contact-responses/listening/index.html
```

The example drives measured and fitted responses with four seconds of Agarwal scraping or Conan rolling excitation.
It applies a 20 ms onset/release gate and renders the full convolution tail.
The material label identifies the resonator.
Both players use the same synthesized excitation.
The example uses fixed responses and prescribed excitation.
Every output is checked against independent full-tail FP64 convolution.
Raw WAVs preserve gain; audition copies apply one constant DC subtraction and gain with a peak ceiling.

[paper]: https://differentiable.xyz/papers-2023/paper_44.pdf
[author]: https://mcdermottlab.mit.edu/ICML2023/sound_website.html
[thesis]: https://hdl.handle.net/1721.1/158825
[impact]: https://doi.org/10.64898/2026.01.28.702236
