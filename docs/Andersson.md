# Andersson–Kropp rough interfacial contact

Ports the normal-contact model in Andersson and Kropp (2008), *Time domain contact model for tyre/road interaction including nonlinear contact stiffness due to small-scale roughness* ([paper][paper]).
NumPy/SciPy prepares surface-dependent springs and rigid-backed elastic-layer responses; an FP64 C++/Accelerate solver computes coupled normal forces with structural memory.
The original tread-block curves remain unreproduced because the measured road heights, material curve and some author settings are unavailable.

## Run and listen

```sh
python3 tools/Reproduce.py --method andersson --offline
```

This builds the binaries, runs validation, renders `presses.wav` and `vibration.wav`, and updates `outputs/reproduction/listening/index.html#andersson`.
For individual stages, use `tools/andersson_reproduce.py --validate` or `--listening` after building `anderssonReproduce` and `PatchContactTest`.

The audio run simulates 4.5 seconds: eight increasing 8 ms presses, followed by band-limited normal vibration under preload.
It uses a 2 cm square patch with 10 × 10 elements, a 1 cm elastic layer, 256 response taps and 51.2 kHz sampling.
The generated surface has 100 micrometres RMS height, Gaussian smoothing sigma 0.25 mm, 38 micrometre scan spacing and RNG seed 2008.
The modulus is interpolated linearly between the paper's endpoints; density 1300 kg/m³ and Poisson ratio 0.499 are chosen inputs.
These settings and the generated arrays are saved in `listening_input/`; they are not recovered author inputs.

Playback is total normal force, with a causal second-order 20 Hz high-pass, polyphase resampling to 44.1 kHz and one common gain for both clips.
No resonator or radiation model is added; the paper supplies contact forces, not reference audio.
`audio.json` records gain per newton, segment durations, WAV hashes and solver diagnostics; `audio_run/force.f64` retains the force trace in newtons.
The additional figure compares the punch calculation with independent published BEM data, not Andersson's tread-block result.

## Measured inputs

```sh
python3 tools/andersson_reproduce.py --config /path/to/contact.json
python3 tools/andersson_reproduce.py --config /path/to/contact.json --equilibrium
```

The JSON configuration requires the following fields. File paths are relative to the configuration file; units are SI.

| Field | Input |
| --- | --- |
| `surface` | Finite 2D `.npy` height array, rows along y and columns along x; the patch starts at the first pixel |
| `scan_spacing` | Fine scan pixel width; 38 micrometres in the paper |
| `material` | CSV with header and columns `frequency_Hz,storage_Pa,loss_Pa`, strictly increasing from DC through Nyquist; DC loss is zero |
| `schedule` | CSV with header and columns `indentation_m,hold_samples`; indentation is relative to first contact at the highest road point |
| `source` | Description identifying the inputs |
| `poisson`, `density` | Bulk-layer Poisson ratio in [0, 0.5) and density in kg/m³ |
| `spring_bonded` | Boolean choosing bonded or frictionless support for the incompressible punch layer; its punch face is frictionless |
| `domain_elements` | Periodic lateral domain width in coarse elements; at least twice the larger patch dimension |
| `oversampling` | Positive integer setting the spatial Fourier cutoff; calculation spacing is element spacing divided by this number |

Defaults are `elements: [20,20]` (rows, columns), `element_spacing: 0.001`, `thickness: 0.01`, `sample_rate: 51200`, `taps: 256`, `spring_knots: 257` and `spring_fraction: 0.9`.
The spring fraction defines the prepared compression domain relative to each element's height range; exceeding it fails without advancing state.
Increase the prepared domain and verify convergence when necessary; no compression or stiffness is capped.
The scan must cover the patch. Boundary pixels contribute their overlap areas, so 38 micrometre pixels need not divide 1 mm elements evenly.
Exactly flat elements have rigid interfaces and are rejected by the roughness-spring path.

Outputs include `curve.csv`, `contact.png`, final element forces/compressions/displacements, and the resolved configuration plus input hashes in `run.json`.
Time-domain runs also write `force.f64` and `contact_force.wav`; `--equilibrium` solves using the DC response sum and has no transient audio.
For the paper's stepped calculation, increment indentation by **0.01 micrometres**, hold each level for more than 256 samples and check force settling.

## Surface and elastic-layer preparation

`tools/andersson_model.py` implements Equations 34–36 within each element:

```text
A(zeta) = sum of fine-pixel areas with height > zmax - zeta
radius(zeta) = sqrt(A(zeta) / pi)
layer_thickness(zeta) = zmax - zeta - zmin
```

The matching point's undeformed height is the element maximum, not an envelopment profile.
The incompressible flat-punch problem is cited through [Yang (1998)](https://doi.org/10.1016/S0167-6636(98)00035-0); explicit bonded and frictionless base kernels appear in Equations 29 and 31 of [Forsbach (2023)](https://doi.org/10.3390/machines11040474).
For effective modulus `E*=E/(1-nu²)`, confinement `R=a/h` and static kernel `chi(t)`, the numerical solve is

```text
u(x) + (2 R/pi) integral_0^1 u(y) integral_0^infinity
       (chi(t)-1) cos(R t x) cos(R t y) dt dy = 1
K = 2 E* a integral_0^1 u(x) dx
```

Gauss–Legendre orders double until stiffness converges; a logarithmic punch-factor table is refined against direct solves to relative tolerance 2e-5.
The resulting spring stiffness is piecewise linear with an exact quadratic force integral, replacing the paper's suggested degree-ten polynomial approximation.
There is no extrapolation, fitted effective modulus or replacement asperity law.

The bulk response solves the isotropic Navier equations in frequency/wavenumber space for a shear-free loaded surface and bonded rigid backing.
Loads are uniform over square elements; receivers are at element centres, as in Section 2.2.
Thin-slab dynamic stiffness matrices are repeatedly condensed to the requested thickness, avoiding growing evanescent exponentials and degenerate P/S potentials at DC.
The complex modulus is interpolated with the `exp(+i omega t)` convention and nonnegative loss.
The lateral domain is periodic; increase its width and spatial cutoff until the measured quantities converge.

The two-times-real transform folds an even inverse transform onto nonnegative times, with half weight at both endpoints.
For `taps` samples it uses `2*(taps-1)` transform samples; the response sum equals static compliance without a fitted correction.
Time resolution, response length and spatial discretization remain convergence requirements.

## Coupled contact

`rough/PatchContact.h` accepts the prepared springs and Green functions, including externally supplied responses.
Each spring stores compression, stiffness and integrated force together at each knot; construct models with the `Make` functions and leave their data unchanged while stepping.
For receiver `e`, force element `m` and time sample `n`, Equations 27–33 become

```text
F_m(zeta) = integral from 0 to max(zeta, 0) of k_m(x) dx
K_lag[e,m] = dt * g_m,e(lag)
w_e[n] = sum over m and lag of K_lag[e,m] * F_m[n-lag]
R_e = zeta_e + w_e - approach_e = 0
J_e,m = delta_e,m + K_0[e,m] * k_m(zeta_m)
```

`approach = z1 - z2` is positive toward contact. Green functions enter in m/(N s), ordered `[lag][receiver][force element]`; construction applies `dt` once.
For two deformable bodies, supply their summed opening response. Separated springs exert no tensile force but retain displacement from earlier loads.
The implementation follows the consistent geometry in Equations 31–33: scalar Equations 19–20 reverse the height signs, and Equation 22 omits the `g(1) dt` factor present in Equation 21.
Newton steps use pivoted LAPACK solves and residual-decreasing backtracking over all elements; tolerance bounds the largest geometry residual in metres.
Failed steps preserve the entire state and history. Adhesion, tangential friction and rotation are outside this normal-contact model.

## Validation and source gaps

`PatchContactTest` checks exact integration, scalar nonlinear solutions, tangent stiffness, coupled contact/release, causal history and transactional failure handling.
A 400-element test exercises 256 taps over 260 steps against an independent scalar-convolution oracle.
`tests/AnderssonTest.py` checks published punch data, static and dynamic layer solutions, an independent Navier boundary solve, transform normalization, scan-pixel geometry and the connected surface-to-force pipeline.
The punch comparison's maximum error is **0.216%** across 48 published BEM values; the reference has finite spatial resolution.
The tested cutoff/domain refinements change response by 0.54%/0.22%, and settled transient force differs from independent equilibrium by at most 2.1e-9 N; these are test-case results, not universal error bounds.
The fixture `tests/data/andersson_punch.csv` is extracted from Forsbach's [CC BY 4.0 supplement](https://depositonce.tu-berlin.de/bitstreams/8a3bba67-3bad-4b6b-b257-6a2805436bcc/download), with its source URL and archive SHA-256 recorded in the file.

Reproducing Andersson's Figures 8–12 additionally requires:

- The measured road array and exact positions A–D.
- The complete complex modulus curve, density and Poisson ratio for the 2008 run.
- The selected punch interface conditions and exact lateral-boundary/transform settings.

These are absent from the checked Chalmers [paper](https://research.chalmers.se/en/publication/73082) and [thesis](https://research.chalmers.se/en/publication/6388) records.
[Winroth's 2010 thesis](https://odr.chalmers.se/bitstreams/e7c8364e-ade9-447f-b431-0575351e74be/download) supplies further settings and MATLAB snippets, but not the required arrays; its later settings are not treated as the original 2008 experiment.

[paper]: https://doi.org/10.1016/j.jsv.2008.04.013
