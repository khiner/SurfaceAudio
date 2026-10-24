# Ren–Yeh–Lin textured contact sound

Ports *Synthesizing Contact Sounds Between Textured Models* (IEEE VR 2010): collision impulses, normal-map momentum changes and fractal microscopic friction excite position-dependent modal responses. The source runs on CPU and Metal; modal rendering reuses the shared CPU/Metal bank.

```sh
python3 tools/Reproduce.py --method ren --offline
open "file://$PWD/outputs/reproduction/listening/index.html#ren"
```

This runs the checks, renders ten WAVs, and updates the combined listening page. Four pairs compare collision plus microscopic roughness against the full three-level model: brick, tile, wood across the grain and wood along the grain. Two further clips demonstrate equal impacts at different positions and a faceted roller. Each clip retains at least 60 dB of modal decay after the last contact. No recordings are mixed into the synthesis.

## Contact model

`src/ren/Ren.h` exposes parameters, explicit per-contact state, normal maps, a roughness wavetable and sample-rate contact inputs. Coordinates and velocities use a local surface tangent frame with macro normal N = (0, 0, 1); transform world-space contacts into this frame before calling the source.

- Macro impulses are supplied by the caller's collision solver, in N s. They are applied once at the sample of collision, without another multiplication by the time step.
- Section 5.1 classifies overlapping, approaching contacts as contacting. Nonzero tangential velocity selects lasting contact; otherwise only the supplied macro impulse is used. `NormalVelocity` is the pre-resolution relative normal velocity, not the velocity after a contact constraint has stopped penetration.
- Section 5.2 evaluates normal momentum as `pN = -mass * dot(vT, n) / n.z`. The transition impulse is the printed `pN(previous) - pN(current)`. The first lasting sample initializes momentum; separation or transient contact clears that history. Maps repeat in U and V with nearest-texel sampling, preserving the paper's pixel-transition interpretation. Normals with nonpositive `n.z` are rejected, without a denominator clamp.
- The roughness wavetable advances by tangential speed divided by spatial sample spacing. Linear interpolation supplies the microscopic force, proportional to normal load and `RoughnessGain`; division by sample rate converts that force to an impulse. Stopped or separated contacts produce no roughness excitation and retain wavetable phase. `fractal_table` constructs a deterministic, zero-mean, unit-RMS spatial profile with power proportional to frequency raised to minus the chosen exponent.

The three terms are returned separately in N s. Source state persists across arbitrary block partitions. GPU voices share a texture and wavetable while maintaining independent contact state. GPU buffer callers must validate contact data and preserve the parameter and state domains documented by the CPU API. Choose wavetable spacing/bandwidth and map resolution for the maximum traversal speed; the source does not silently low-pass or clamp the paper's impulses.

## Mesh response

`tools/ren_model.py` constructs the 3D axial spring network referenced by Section 4, following [Raghuvanshi–Lin (2006)][modal], Section 3: one particle per vertex, one spring per unique mesh edge, spring stiffness `Young * thickness`, and particle masses from one third of the incident triangle areas times density and thickness. The renderer uses a closed 14-vertex, 24-triangle block with slightly raised face centers.

We solve `K phi = lambda M phi` with mass-normalized eigenvectors and Rayleigh damping `C = gamma M + eta K`. Each retained mode has decay rate `(gamma + eta*lambda)/2` and damped angular frequency `sqrt(lambda - decay_rate^2)`. Rigid, overdamped and out-of-band modes are omitted and counted in the checks. Contact impulses are distributed barycentrically to the three vertices of the contacted face and projected onto the mode shapes. A fixed top-center normal-displacement pickup supplies the output weight. Both the contact position and the mesh response therefore affect the audio.

This solves the paper's Equation 2 directly. Its subsequent printed derivation interchanges the damping symbols and treats a general diagonal mass matrix as commuting with the mode basis; Equation 5 also prints a sum of conjugate poles where the velocity-jump condition requires their difference. The implementation follows the original ODE and the impulse condition `Delta velocity = impulse / mass`. Tests compare against the coupled matrix dynamics with unequal vertex masses.

WAVs contain the synthesized normal-displacement response, retaining raw physical scale. The listening page applies its usual playback gain. No acoustic radiation or room filter is appended.

## Inputs and replay

`repros/ren/Scenes.json` records the material coefficients, mesh dimensions, load, slider mass, map size, wavetable spacing and seed. `tools/ren_reproduce.py` generates periodic brick, tile and wood height maps, derives their normals, and saves texture previews. The gestures occupy a three-second interval with a 0.25-second onset delay; the renderer extends the silent input through at least 60 dB of modal decay. Scraping uses a prescribed load, with the corresponding downward pre-resolution velocity from a 100 Hz contact step. The roller is an externally driven regular 24-sided cross-section: each facet transition supplies the normal momentum change `2 * mass * speed * sin(pi/24)`. This pure rolling example has zero local tangential slip and thus uses the macro branch.

The paper delegates object motion and collision response to a physics engine. This port accepts those contact results; its example gestures and roller impulses are generated deterministically. The original demonstration meshes, textures and manually chosen material presets have not been recovered. These supplied inputs exercise the published method rather than matching the original demo recordings or repeating its user study.

To render without rebuilding the full report:

```sh
python3 tools/ren_reproduce.py --binary build/renReproduce
python3 tools/BuildListeningReport.py
```

The native `renReproduce INPUT_DIRECTORY OUTPUT_DIRECTORY` interface accepts the arrays prepared by that script. Parameters, contacts and bindings use the native structs (`Parameters`, `Contact`, and three uint32 vertex indices followed by three float32 barycentric weights). `normals.f32` contains row-major XYZ normals, `roughness.f32` the wavetable, `modes.f32` frequency/decay/pickup-amplitude triples, and `shapes.f32` vertex-major normal-projected mode shapes. The latter two arrays can also come from a different modal analysis. All values are SI except normalized UV coordinates and dimensionless interpolation weights.

## Validation

```sh
ctest --test-dir build -R RenTest --output-on-failure
python3 tests/RenTest.py
```

The C++ tests cover the published momentum equation and sign, load/time-step scaling, periodic addressing, transient/lasting/separated states, state reset, exact block partitioning, invalid inputs and multiple CPU/Metal voices. Python tests check mesh mass, six rigid-motion null modes, unequal-mass impulse response against independent coupled dynamics, spatial spectrum slope and analytic normal-map derivatives.

Every rendered clip is checked against independent NumPy excitation and double-precision SciPy modal filtering, and also against the complete native CPU path. `outputs/reproduction/ren/checks.json` records component errors, eigen residuals, CPU/Metal audio errors, waveform statistics and WAV hashes. The A/B clips differ only in the normal-map contribution; no fitting to the reference output is performed.

[paper]: https://doi.org/10.1109/VR.2010.5444799
[modal]: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/12.pdf
