# Agarwal continuous-contact extension

Implements finite micro-impact filtering from Chapters 4–5 of the [2025 thesis](https://hdl.handle.net/1721.1/158825).
A finite pulse filters the complete scraping or rolling excitation before convolution with fixed object responses.
Mass and stiffness control pulse duration and the contact spectrum.

## Run

After the normal CMake build:

```sh
python3 tools/Reproduce.py --method agarwal2025 --offline
open outputs/reproduction/listening/index.html
```

The workflow requires the pinned TDW surface profiles and 2023 author response recordings already used by the Agarwal reproductions.
`repros/agarwal2025/cases.json` contains all construction settings.
The page contains six finite-pulse ablations, two mass/radius controls and one stiffness control.
Comparisons contain local renders because matching thesis stimuli are unavailable.
The 50 g/8 mm and 10 kg/50 mm cases use the published Experiment 1 mass/radius pairs.
The 100 g/8 mm hard/soft cases use the Experiment 2 stiffnesses of 5e8 and 5e4 N/m against a 1e8 N/m surface.
The stiffness comparison isolates continuous contact and omits the experiment's bouncing and restitution behavior.

## Model and reconstruction choices

`RenderMicroImpactGpu` produces a finite linear half-sine pulse with duration `pi*sqrt(m/k)` and amplitude `A*m*v`.
The thesis uses series stiffness; `StiffnessCombination::Sum` implements the 2022 poster's equation.
The poster is Agarwal et al., *Perceiving Physical Consistency in Object Interactions Using Physics-based Sound Synthesis*.
Micro-impact pulses use the linear limit even when a clipping limit is supplied.
`RenderContactGpu` convolves the contact force, linear pulse and sum of both responses, preserving complete tails.
Pulse samples are interval averages, preserving sub-sample pulses; discrete convolution retains digital gain.
The absolute force-to-sound calibration remains unspecified.

The workflow exercises the [2021 horizontal, vertical, elastic and dissipative force equations](Agarwal.md).
Scraping sums the horizontal and vertical terms; rolling adds both elastic and dissipative terms with their supplied coefficients.
Metal prepares the constrained trajectory, mixes the components and performs full convolution.
The fixed response sums the measured Wood_1 and Metal_1 recordings from the public 2023 corpus with equal gain.
These recordings substitute for the unavailable thesis responses.

TDW profiles use metre-scale heights after mean subtraction, with integrated trajectories anchored to both profile endpoints.
The default interprets the numerical curvature coefficient alpha in micrometres, with its metre conversion recorded as `alpha_scale_m = 1e-6`.
This unit convention and the integration boundaries are inferred.
Use `--alpha-scale` to select another interpretation.

Ramp acceleration, rolling eccentricity, Gaussian width, relative force coefficients and the 20 ms onset/release gate are declared input choices.
Normal load is mapped affinely to the shared kernel's [1,6] coordinate, preserving the paper's normalized-load interpolation.
Scraping uses a constant alpha; rolling uses the prescribed eccentric motion's varying normal load.
Rolling stiffness and dissipation are separate coefficients from the stiffness used to calculate the micro-impact duration.
These choices leave the original rolling-texture reproduction unresolved.

## Verification and outputs

The runner compares every force component against FP64, checking both full-signal and mean-subtracted relative L2 error below .005.
It independently integrates the micro-impact pulse with SciPy quadrature and checks complete convolution against a double-precision reference.
`metrics.json` retains component RMS, varying-component RMS, numerical errors, texture descriptors and executable/shader hashes.
Each `render.json` reports offline synthesis time excluding WAV I/O; it includes allocation, GPU kernel preparation and synchronization.
On the M5 Max, finite-pulse and response convolution took 25.7–31.3 ms per case, excluding trajectory preparation.
Maximum force-component error was 0.142%; maximum independently checked convolution relative L2 error was 4.02e-6.
Main outputs are the finite-pulse `synthesis.wav` files and instantaneous-pulse `unfiltered.wav` ablations.
Use `--keep-diagnostics` to retain prepared trajectories, component arrays, excitation WAVs and pulse WAVs.
Failed checks preserve those intermediates.
Real-time deadlines are unverified.
