# [Energy-stable mass–barrier collisions with contact damping and dry friction][paper]

Maarten van Walstijn, Vasileios Chatziioannou and Nikolaos Athanasopoulos, IFAC PapersOnLine 58(6), 214–219 (2024).

Both the explicit energy-quadratisation scheme and the implicit discrete-gradient scheme run on CPU and Metal. The port includes the two-dimensional oscillator, unilateral power-law contact, velocity-dependent contact damping, Coulomb sliding and sticking, modal correction, and the optional six-sample minimum contact-duration stiffness cap. Eight WAVs compare executed author MATLAB with Metal for repeated impacts and stick-slip motion.

## Run and listen

```sh
python3 tools/Reproduce.py --method vanwalstijn --offline
open 'file:///Users/khiner/Development/SurfaceAudio/outputs/reproduction/listening/index.html#vanwalstijn'
```

The reproduction requires NumPy, SciPy, Matplotlib, PyMuPDF and Octave. Omit `--offline` to fetch missing pinned inputs. The publisher PDF mirror can reject automated downloads; a matching cached copy is accepted and hash-checked. Author source is pinned to commit `ad1b11d4ff6a9149c93f0dd74ec94cdb6e3d3fb2` in [Sources.json](../repros/vanwalstijn/Sources.json); file URLs and hashes have one manifest in [ReferenceInputs.json](ReferenceInputs.json). The MATLAB functions execute unchanged in Octave with the same force arrays supplied to the native port.

[Experiments.json](../repros/vanwalstijn/Experiments.json) contains all reproduction and audio settings. Generated results are under `outputs/reproduction/vanwalstijn/`: `checks.json`, `convergence.json`, recovered vector traces, trajectory plots, source execution logs, and WAVs. `--keep-diagnostics` retains binary forces and state histories when using the shared reproduction runner.

The four audio pairs are inelastic collisions, more elastic collisions, explicit stick-slip, and implicit stick-slip. Each uses a two-second drive, audio `vx + vy`, a common gain for the author/Metal pair, and 10 ms endpoint fades. No acoustic radiation or added resonator is applied. The short paper experiments are also run at their original durations.

## Published results

The reproduction extracts vector paths directly from the pinned PDF, without image tracing or a trajectory fit.

| Check | Result |
| --- | --- |
| Figure 3 explicit displacement vs published paths | x: 0.0062%; y: 0.0119% relative L2 |
| Figure 4 explicit displacement vs published paths | x: 0.0062%; y: 0.0031% relative L2 |
| Figure 4 samples with zero tangential velocity | 27.56% |
| Figure 5 asymptotic explicit order | 1.04 |
| Figure 5 asymptotic implicit order | 1.98 |
| CPU/Metal velocity error, two-second examples | At most 0.063% relative L2 |
| CPU cumulative energy-balance residual, short and audio cases | Below 2.3e-12 relative to work/loss/peak stored energy |

Figure 4 omits the tangential drive amplitude and frequency and prints the barrier offset as `2e-5 mm`. Inverting the tangential equation on the published slipping segments recovers cosine/sine amplitudes of approximately `-0.2514/-0.0045 N` at 100 Hz. The reproduction uses `-0.25*cos(2*pi*100*t) N` and barrier `2e-5 m`; these reproduce both published displacement paths to the errors above. The code's excitation-2 preset independently supplies the 100 Hz cosine convention. The literal caption offset is also simulated and its error retained in `checks.json`. This is a documented input reconstruction, with no fitted solver correction.

Figure 5 runs all nine time steps from 44.1 kHz to 256×44.1 kHz, both schemes, and capped/uncapped stiffness, against a 1024× implicit reference on the identical time grid. The Hann envelope is evaluated on one continuous 50 ms interval. The expected first- and second-order convergence is reproduced. The plot also overlays the paper's extracted error values. Across all nine points, the capped explicit and implicit error curves agree to 0.22% and 0.34% relative L2, respectively. The uncapped explicit curve differs by 2.57%; the uncapped implicit curve differs by 29.9%, dominated by the coarsest time step. Agreement with the published convergence orders does not establish exact agreement with every Figure 5 ordinate.

## Equations and numerical choices

`src/barrier/Barrier.h` implements Equations 27–60 as shared CPU/Metal scalar operations. `MakeParameters<T>` computes the modal correction and stiffness cap in FP64 before conversion to the target precision. CPU supports FP64 and FP32; Metal uses FP32 with one independent oscillator per GPU thread. `MakeState` initializes contact energy consistently from displacement and the preceding half-step velocity. `Render` and `EncodeSource` retain state across blocks; input and output GPU buffers use contiguous voice-major blocks.

The implicit contact step solves the monotonic scalar equation with a bracketed root solve. It uses the factor of two from Equation 40 and the author implementation; Equation 41 omits that factor in print. A divided-difference series avoids cancellation near a zero step. Displacement increments and stable coefficient formulas avoid subtracting nearly equal position or modal terms at high sample rates.

The explicit step uses the paper's nonnegative auxiliary-energy constraint and the author's optional roundoff clamp. It rationalizes the positive-root expression when its numerator would subtract nearly equal values. The output distinguishes the actual normal-equation force from the normal force used for tangential friction: the two discretizations differ in Equations 48–49, and the author's exported explicit `Fc` is the latter.

The source evaluates `0/0` in the friction coefficient at unloaded rest. The port computes the sticking friction force directly, yielding finite zero force and loss. Source nonfinite diagnostic samples are counted, with comparison restricted to finite values; displacement, velocity and contact force must remain finite. The original source's Newton stopping criterion differs from the bracketed solve. The coarse uncapped Figure 5 implicit trajectory disagrees with the source (about 5.2% in x); that diagnostic is retained rather than hidden by adjusting the model. The same capped case agrees to approximately 1e-11 in force.

For Figures 3–4, source displacement errors are at most 9.2e-8 relative L2. Across all four two-second examples, the largest source comparison error is 2.7e-5 in implicit sticking friction; velocity errors remain below 1.2e-6. The source's auxiliary energy, displacement, velocity, contact force, friction, total energy and loss conventions are checked as applicable.

## Validation

`BarrierTest` checks the exact sampled linear oscillator, passive free impact and separation, discrete energy balance, Coulomb bounds, sticking and resumed slip, finite rest, CPU block continuity, and four simultaneous Metal voices spanning both schemes and contact regimes. The reproduction executes the pinned reference with both algorithm switches disabled as well as the paper's capped, modal-corrected cases. Failures in source convergence, state finiteness, passivity, matched source comparisons, GPU agreement, trajectory matching or convergence order stop the runner before publishing the listening manifest.

[paper]: https://doi.org/10.1016/j.ifacol.2024.08.283
[author]: https://github.com/mvanwalstijn/Mass-Barrier/tree/ad1b11d4ff6a9149c93f0dd74ec94cdb6e3d3fb2
