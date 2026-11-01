# Ducceschi, Bilbao, Willemsen & Serafin (2021)

[Linearly-implicit schemes for collisions in musical acoustics based on energy quadratisation][paper], JASA 149, 3502–3516.

[paper]: https://doi.org/10.1121/10.0005008
[author]: https://mdphys.org/collisions.html
[predecessor]: https://doi.org/10.5281/zenodo.3898741
[tromba-source]: https://github.com/SilvinWillemsen/TrombaMarina/tree/3e839ba187a89bb59da314aa0c0f901107db6e04

The port implements the paper's N-IT collision method for the six models below, including their complete mechanical systems.

```sh
python3 tools/Reproduce.py --method ducceschi --offline
open 'file:///Users/khiner/Development/SurfaceAudio/outputs/reproduction/listening/index.html#ducceschi'
```

Octave executes the three author MATLAB scripts. Omit `--offline` to fetch missing inputs. The author's server currently rejects direct HTTP requests for the MATLAB files with HTTP 406; the retained local inputs were recovered from complete web-cached source text. Their line-number prefixes were removed and line endings normalised to LF. These are source-text hashes, not claims of byte identity with the unavailable raw downloads. `docs/ReferenceInputs.json` pins each input, including both papers. No source equations are edited in the cached scripts.

## Solver

`src/collision/Collision.h` shares the sample update between C++ FP64/FP32 and Metal FP32. Mechanical coordinates have diagonal masses, sparse stiffness, centered diagonal damping and backward-difference spatial damping. Contact rows interpolate the relative displacement of both interacting bodies. Fixed barriers are offsets in those rows. All simultaneous contacts are solved together.

The auxiliary variable is signed, as in Eq. (27). The separation branch uses the free predicted penetration to release stored contact energy. The mass author script applies its additional outward-force condition during contact; the hammer and fret scripts omit that condition. Each model retains its source convention. Bilateral contacts use the derivative of the absolute-displacement potential in Eq. (57), without the auxiliary-sign factor, matching `Tromba::calculateCollisions()` in the [author implementation][tromba-source].

Writing `W = dt²/(M + dt D/2)`, `J` for relative-contact interpolation and `g` for the explicit gradient, elimination of the mechanical coordinates gives

```
[I + diag(g) J W Jᵀ diag(g)/4] λ = ψ_previous + diag(g) J(q_free − q_previous)/4
q_next = q_free − W Jᵀ diag(g) λ
ψ_next = ψ_previous + diag(g) J(q_next − q_previous)/2
```

Cholesky factorisation uses only nonzero-gradient rows and is algebraically equivalent to the author scripts' mechanical-coordinate solve. The tromba's Gaussian bow-friction equation uses Newton iteration initialized from the previous relative velocity, following `TrombaString::NRbow()` in the author implementation. Cubic bow interpolation follows the predecessor's Eq. (23) and `Global.h`; its support is disjoint from the bridge contact. Newton is capped at 100 iterations, as in the source; its tolerance is tightened to 1e-12, with FP32 also stopping when the iterate stops changing. The physical bow-force response includes the interpolation self-coupling required by the published PDE. The GUI source's approximate exponential and inconsistent normalization are not substituted into that equation.

The lossless discrete energy is

```
H = (q_next − q)ᵀ M (q_next − q)/(2 dt²) + q_nextᵀ K q/2 + ψ_nextᵀ ψ_next/2
```

Backward spatial damping `R` replaces `M` in this energy by `M − dt R/2`. Every run also checks the energy change against centered damping losses and bow work. Python verifies positivity of `M − dt R/2 − dt² K/4` before rendering.

The state stores displacement increments to avoid subtracting large adjacent positions when the hammer leaves the string. This is an algebraic rearrangement of the same recurrence.

Metal dispatches one threadgroup per voice, distributes sparse mechanical updates across its threads and retains state between blocks. Contact solves use the shared update; pickup and energy reductions use a different FP32 summation order. Large spatial systems render offline.

## Models and inputs

| Model | Parameters and comparison |
|---|---|
| Mass–barrier | Execute `massBarr.m`; compare displacement, force, auxiliary state and energy, retaining IT-1 and IT-2 reference outputs |
| Hammer–string | Execute `stringHammer.m` with both its defaults and Figure 3 settings: 10 g hammer, 0.5 m/s, 1 mm initial separation, stiffness 10¹², exponent 1.3, string density 6.3 g/m, tension 100 N and length 0.7 m |
| Frets | Execute `stringFret.m` for twelve semitone-spaced frets, in both finite-difference and modal form, with source defaults and Figure 6 parameters |
| Curved backboard | Execute the same source with `flagBarr=1`; Figure 5 uses `b(x)=−0.02x²−0.001x−0.0001` and a 3 mm first-mode initial displacement |
| Wire–membrane | Section V: fixed circular membrane, Cartesian grid, bilinear wire interpolation, 1 g/m wire, 10 N tension, 0.2 m length, 1 mm first mode and 0.4 mm rest gap; membrane radius 0.15 m, density 0.2 kg/m², tension 2000 N/m |
| Tromba marina | Section VI and the [2020 predecessor][predecessor], Table 1: simply supported stiff string, clamped plate, sprung/damped bridge, bilateral string–bridge connection and unilateral bridge–body contact; both exponents changed to 1.3 as prescribed in 2021 |

The snare's Figure 7 caption prints wire endpoints at `x=0.3 m`, outside its `R=0.15 m` membrane. This implementation explicitly uses `x=0.03 m`; the caption does not specify its contact stiffness or exponent, for which the retained configuration uses `10¹²` per unit length and `1.3`. The wire quadrature weights enter both contact potential and mechanical masses. Those omissions prevent claiming an exact Figure 7 trajectory match.

The tromba uses the predecessor's 5 µm rest offset, 1 cm minimum plate spacing and Gaussian bow friction with shape parameter 100. String and bridge coordinates are relative to their rest offset. The output combines plate displacement at the bridge with one tenth of the string pickup. The listening example uses a 0.1 N bow force and −0.2 m/s bow speed; Figure 10 instead uses 1 N for visibility. That 1 N case is checked at 352.8 kHz: the previous-velocity Newton iteration failed to converge for some steps at lower tested rates, while eightfold sampling converged without changing the equations or adding an alternate root solver. This is a numerical resolution choice, not a claimed Figure 10 sample-rate match. The independent 50 ms snare trajectory differs by about `3.1e-12` in relative L2. For the eightfold-rate 1 N tromba, displacement differs by `5.4e-6`, final coordinates by `2.4e-7`, and brief contact-force pulses by `2.7%`; this force sensitivity is reported separately from displacement accuracy.

An independent SciPy solve checks the snare and tromba in full mechanical coordinates. The tromba application uses different GUI grid indexing and bow-force normalization, so it is not a sample-level oracle for these PDE discretisations.

## Validation and audio

`CollisionTest` checks contact-space elimination against an independently assembled full mechanical solve, lossless energy, silent rest, block continuity and three independent Metal voices. `tools/ducceschi_reproduce.py` executes eleven short author comparisons. Successful runs retain listening assets, measurements, source controls, provenance and hash-checked author-output caches in `outputs/reproduction/ducceschi/`. Raw solver dumps, model inputs, generated scripts, logs and undisplayed comparison plots are removed; use `--keep-diagnostics` to retain them. Failed runs retain their diagnostics.

The run also reconstructs Figure 1 with author IT-1/IT-2 traces and checks Figure 2 against the exact linear-barrier solution. The measured convergence orders are approximately 2.00, 2.08, 1.12 and 1.00 for the four published stiffnesses. Figure 4 repeats the four hammer stiffness/sample-rate combinations. The fivefold-rate case lands exactly on a contact sample: its nominal C++/author waveform difference is about 1.17%. Perturbing only the author initial position by ±9.8e-17 m changes its waveform by up to 1.84%. The nominal case and the two roundoff probes are retained separately; neither the published inputs nor the production equations are adjusted to obtain a closer fit. The paper's stiff contact trajectories are more sensitive in FP32: a small energy error does not establish sample-identical trajectories. `metrics.json` reports waveform and energy errors separately.

The retained 1.5-second author/C++ waveform errors are `2.83e-9` (hammer), `5.00e-10` (frets), and `8.76e-12` (backboard). Long FP32 waveform differences are substantially larger for stiff contacts: approximately `1.05` relative L2 for the frets, `0.44` for the backboard, and `0.23` for the three-second tromba. The snare difference is `8.87e-5`. These are energy-balanced implementations of the same update, but FP32 is not a substitute for FP64 when matching the authors' sample trajectories.

The five listening pairs compare author MATLAB against C++ FP64 for hammer and fretboard models, and C++ FP64 against Metal FP32 for wire–membrane and tromba models. Each pair shares a playback gain, DC removal and 10 ms endpoint fades. These are mechanical pickup signals, without added resonators or calibrated sound-pressure units.
