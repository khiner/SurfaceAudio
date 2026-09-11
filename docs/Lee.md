# Lee, Depalle and Scavone 2010

Implements the [DAFx 2010 rolling source/filter method](https://www.dafx.de/paper-archive/2010/DAFx10/LeeDepalleScavone_DAFx10_P104.pdf).
Original Lee source code, recordings, and synthesis examples remain unrecovered.
The FoleyAutomatic comparisons are failed audio reconstruction diagnostics; numerical agreement alone does not establish a successful paper reproduction.

## Run

The Python runner retains main WAVs, parameters, metrics, and plots after successful checks.
Use `--keep-diagnostics` to retain intermediate audio and no-notch outputs.
Failed runs preserve generated diagnostics.

```sh
cmake --build build --target leeReproduce LeeTest
build/LeeTest
python3 tools/lee_reproduce.py --binary build/leeReproduce --offline
```

```text
leeReproduce input.wav output [threshold=.03] [envelope=16] [tail=.1] [max_notches=12] [highpass=10000] [taps=129]
```

The renderer accepts mono or averaged multichannel WAVs, with a sample rate exceeding twice the selected detector cutoff.
Main outputs contain the input and CPU reconstruction at the input scale.
With `--keep-diagnostics`, outputs also include GPU, independent-reference, half-speed, reversed-trajectory, and no-notch WAVs.
Records expose contact times, LPC denominators, gains, and notch frequencies and bandwidths.
The [manifest](../outputs/reproduction/lee/manifest.json) records full-record errors, model comparisons, and interleaved timings.
[Controls](../repros/lee/controls.json) and [source hashes](../repros/lee/sources.json) pin settings and exact video crops.

## Method and reconstruction choices

Equations 1–5 detect contacts using a 10 kHz Hamming high-pass filter, centered squared envelope, and positive threshold crossings.
Only complete filter/envelope support contributes to detection, and the first valid sample cannot create an onset.
Four QMF bands use decimation 8/8/4/2 and LPC orders 25/10/5/5.
Each contact uses reciprocal-spectrum quadratic notch fits, radii `exp(-BW/2)`, pole-radius ratio 0.95, inverse-notch whitening, and LPC.
Synthesis restores notches, upsamples, compensates group delays, and overlap-adds truncated causal responses.
The final contact extends to the end of the recording.
Contact timing and filter order can be varied independently; reversing the trajectory preserves each filter's causal response.

The paper omits filter taps, envelope length, threshold, peak-selection settings, and truncation length.
The supplied 64-tap symmetric QMF satisfies equation 7 and gives near-perfect reconstruction, with relative error 0.000315 on independent noise.
Notch estimation fits three reciprocal-spectrum dB samples around each peak, with a 3 dB prominence threshold.
Bandwidth is the full width 3 dB below the quadratic peak.
Greedy prominence selection excludes overlapping pole bandwidths and features overlapping DC or Nyquist.
This prevents repeated inverse filters at unresolved valleys and QMF edges; it is an explicit reconstruction choice.
LPC gain is the square root of unnormalized prediction energy, which the paper leaves unspecified.

Metal 4 accelerates contact detection filtering, contact QMF convolutions, synthesis filtering, and overlap summation.
Small notch/LPC solves and recursive contact responses use CPU FP64.
CPU FIR resampling computes only retained phases with Accelerate dot products.
CPU/GPU timings include matching models, tails, allocations, and completion waits and exclude GPU-context creation and file I/O.

Measured on Apple M5 Max with Homebrew Clang 23, Release:

| Input | CPU analysis ms | GPU analysis ms | CPU synthesis ms | GPU synthesis ms |
| --- | ---: | ---: | ---: | ---: |
| real_wok | 90.11 | 80.63 | 49.54 | 51.63 |
| trough_pass1 | 60.95 | 40.68 | 51.60 | 21.14 |
| trough_pass2 | 47.22 | 28.81 | 58.24 | 23.12 |
| trough_pass3 | 66.53 | 48.24 | 12.64 | 9.61 |

## Input suitability and validation limits

The [2013 thesis](https://www.collectionscanada.gc.ca/obj/thesescanada/vol2/QMM/TC-QMM-116954.pdf), chapter 2, uses an accelerometer recording.
It requires distinguishable micro-contact transients and describes failures when continuous rolling cannot be segmented into meaningful contacts.
The linked DAFx and thesis audio pages are unavailable at their original locations.

Our recorded example is the 56–64 s rock-in-wok crop from FoleyAutomatic 2001, with camera audio during 57–63 s and silence around it.
The three CGI rolling inputs use video seconds 17–20, 20.5–23.5, and 24.25–27.5 and contain little useful energy above 10 kHz.
They use a 300 Hz cutoff, 513 taps, a 128-sample envelope, and a 0.03 relative threshold.
Those detections are signal features; physical contact times remain unidentified.
The four examples yield 89, 126, 145, and 13 events, respectively.
Shared FoleyAutomatic inputs establish neither the original accelerometer conditions nor the quality of Lee's published synthesis.
The MPEG codec further limits the references, and raw synthesis peaks can exceed unity.

Tests cover independent SciPy filtering, Toeplitz LPC, onset indices, analytic notch responses, QMF reconstruction, and CPU/GPU agreement.
They also cover crop boundaries, overlapping notches, finite output, and deterministic synthesis.
Separate no-notch ablations expose the contribution of notch estimation.
Full-record metrics report lost activity and spectral and envelope differences without treating these failed reconstructions as successful reproductions.
The author video has no explicit redistribution license identified on its download page.
