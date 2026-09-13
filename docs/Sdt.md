# SDT solid interactions

Implements coupled modal/inertial bodies, impact, elasto-plastic friction and rolling/scraping controllers from [SkAT-VG/SDT][source].
The source revision is `0509de418e7bebc8b37866b3b4458e0acc8cf1f4`.
The publication is Baldan, Delle Monache and Rocchesso, *The Sound Design Toolkit*, SoftwareX 6 (2017), 255–260.
Sánchez and Reiss, AES 146 (2019), provides related modal/contact equations.
Licenses and attribution remain in [NOTICE.md](../src/sdt/NOTICE.md).

## Run

```sh
python3 tools/Reproduce.py --method sdt
open outputs/reproduction/listening/index.html
```

The fetcher downloads SHA-256-pinned source archives for SDT and its two JSON dependencies, then builds an unmodified upstream C library with Homebrew Clang.
Cached packages support `python3 tools/sdt_fetch.py --offline`.
The native interface is `sdtReproduce LIBSDT.dylib OUTPUT_DIR [SECONDS [SAMPLE_RATE]]`.

Reproductions follow the supplied Pd/Max graphs, including message rounding, noise seeds, filtering, resonator loading and pickups.
The harness renders all four interactions for six seconds at 44.1 kHz and compares the actual upstream C implementation with our CPU/GPU implementation.
The historical RollSDT recording uses unidentified older settings and is only an additional listening reference.

## Discrete model

Each CPU body owns mode coefficients and state as separate arrays.
Modes have frequency, decay, mass, contact gain, and output gain.
The contact pickup and listening pickup are independent.
A mode with zero frequency and zero decay is an inertial point mass.

For sample interval `h`, fragment size `s`, angular frequency `w`, and modal decay `d`, coefficients follow upstream:

```text
u = sqrt(s)
theta = w h / u
g = d > 0 ? 2 / (d u) : 0
r = exp(-g h)
b1 = r sinc(theta) h² / (mass s)
a1 = -2 r cos(theta)
a2 = r²
b0v = cos(theta) / (h sinc(theta)) - g
b1v = -r / (h sinc(theta))
p_next = b1 force - a1 p - a2 p_previous
v_next = b0v p_next + b1v p
```

Contact force is distributed in proportion to nonnegative contact gains, normalized by their sum.
The listening output is the sum of modal displacement multiplied by output gain.
The energy estimate uses upstream's contact-gain-weighted modal kinetic and potential energy.
The interactor retains upstream's predicted-energy inequality, but solves its force bound analytically.
Contact-induced energy change is `A f² + B f` within the source's displacement-clipping boundary.
`A` sums `0.5 gain (stiffness dp² + mass dv²)` over both bodies, where `dp` and `dv` are their force response coefficients.
`B` sums `gain (stiffness p_free dp + mass v_free dv)` with opposite signs for the two bodies.
The allowed force satisfies `A f² + B f <= contact_energy`.
The implementation evaluates this polynomial directly and uses a cancellation-resistant quadratic root to reduce a candidate force only when necessary.
The contact law is unchanged; the force bound uses an analytic solution of the source inequality.
The retained reference evaluator uses the original 50-iteration bisection and 0.001 relative-total-energy stopping rule.
Impact carries the residual contact energy between samples and resets it at separation.
Friction resets this energy allowance each sample, matching upstream.
Apply the limiter inside the contact feedback loop.

For compression `x = position1 - position0` and velocity `v = velocity1 - velocity0`, impact evaluates `k x^shape (1 + dissipation v)` when compressed.
Positive force acts on body 0 and the opposite force acts on body 1.
This is the published and pinned-upstream expression, including its tensile result when `1 + dissipation v < 0`.
A nonadhesive use of this law requires the trajectory to stay in the domain `1 + dissipation v >= 0` while compressed.
The ordinary impact reference and collision tests stay in that domain.
`UnilateralImpactForce` provides a separate scalar extension that clamps negative force to zero.
The coupled CPU and GPU solvers use the source expression.

Friction uses Dupont mean bristle displacement, sinusoidal breakaway and Gaussian Stribeck steady-state displacement.
It includes bristle damping, viscosity and noise scaled by `sqrt(abs(v) normal_force)`.
The bristle force is evaluated before the explicit state update.
Zero normal force resets the bristle state.
At zero velocity the derivative is zero, while an existing bristle may retain elastic force.
Elastic release during reversal can make instantaneous `force * velocity` negative.
The dissipation-sign test applies to steady sliding.

Rolling tracks the upper envelope of a surface sample stream with a grain- and speed-dependent decay.
Rising surface samples generate bumps weighted by depth and kinetic energy, and the flight accumulator suppresses repeated bumps until it decays under gravity.
Scraping uses the same envelope and scales its upward increments by applied force and squared speed.
These controllers retain SDT's discrete, perceptually motivated scaling.
Surface and grain inputs use normalized SDT texture units.
The controller's `grain * velocity` decay and per-sample gravity decrement depend on sample index.
Their scaling follows the sample-dependent source equations.

## API and parameter contract

`MakeBody` constructs and validates a body outside the audio loop.
Sample rate must be positive, fragment size in `(0, 1]` and effective modal mass greater than `1e-6`.
Frequencies, decays and contact gains must be nonnegative; output gains must be finite.
Frequencies must remain below the upstream impulse-invariance cutoff `acos(-0.9995) / (2 pi h)` after fragment scaling.
`SetPosition` and `SetVelocity` set the contact pickup and update the recurrence history consistently.
`StepImpact` and `StepFriction` accept external forces and return the contact force, body pickup states, and separate listening outputs.
Each independently advanced contact pair requires exclusive bodies with the same sample rate.
Stepping is allocation-free.

Parameter structs use direct physical or SDT controller values.
Impact stiffness and dissipation are nonnegative and shape is at least one.
Friction stiffness and Stribeck velocity must be positive; normal force and damping coefficients must be nonnegative.
Static/dynamic coefficients and breakaway lie in `[0, 1]`, with static coefficient at least dynamic coefficient.
Noise inputs are finite and normally in `[-1, 1]`.
Grain lies in `[0, 1]`, depth, mass and scraping force are nonnegative, and speed can have either sign.
The explicit bristle update still requires a sufficiently small sample interval relative to slip speed and steady-state bristle displacement.
Extreme stiffness, speed, or vanishing positive normal force may require oversampling.

```cpp
using namespace surface_audio::sdt;
const std::array exciter_modes{ModeParameters{.Mass = 0.03}};
const std::array object_modes{ModeParameters{230, 0.3, 0.4, 0.9, 0.3}};
auto exciter = MakeBody(exciter_modes, 48000);
auto object = MakeBody(object_modes, 48000);
SetVelocity(exciter, -0.5);
ContactState contact;
const auto sample = StepImpact(exciter, object, contact, {});
// sample.Output1 is the vibrating object's listening pickup displacement.
```

## Metal 4

`AppendBody` packs a CPU body into `GpuMode` and `GpuModeState` arrays and returns its offset and count.
The GPU computes the same impulse-invariant recurrence in position/velocity coordinates.
The transition coefficients are derived in CPU double precision before upload.
Position/velocity coordinates reduce cancellation from subtracting float displacements.
The production CPU evaluates the same state transition in double precision.
The GPU accumulates position and velocity increments with compensated summation to avoid drift over sustained motion.
The independent source reference retains the original displacement-history recurrence.

`SdtContacts` assigns one thread to each independent body pair.
Each thread advances its samples in order and reduces its own modal states.
Mode ranges must be disjoint between contacts.
The CPU SoA mode loops vectorize to two-wide ARM FP64 SIMD with Homebrew clang 23 at `-O3 -mcpu=native`.

| Buffer | `SdtContacts` |
|---|---|
| 0 | `GpuDispatch {Contacts, Frames, TimeStep}` |
| 1 | `GpuContact[Contacts]` |
| 2 | Persistent `GpuContactState[Contacts]` |
| 3 | Packed `GpuMode[]` |
| 4 | Persistent `GpuModeState[]` |
| 5 | `GpuInput[Contacts * Frames]` |
| 6 | `GpuOutput[Contacts * Frames]` |

Input and output arrays are contact-major: `contact * Frames + frame`.
`GpuContactKind` selects impact or friction.
All configuration, input, and persistent state must be initialized before dispatch.
After completion, call `ValidateGpuContactStates` to reject any failed contact before using its results.
Update `Frames` and repack each block's input stride when changing block size.

`SdtSurfaceForces` generates the external force for a rolling or scraping exciter.
Its buffers are dispatch at 0, `GpuSurfaceParameters[]` at 1, persistent `GpuSurfaceState[]` at 2, contact-major float heights at 3, and `GpuInput[]` at 4.
It writes `External0` and preserves the other initialized input fields.
Dispatch this kernel before `SdtContacts` with an ordered GPU dependency.

## Numerical differences and limits

Production solves the source's quadratic energy inequality accurately; upstream stops a coarse bisection using 0.001 of total body energy.
Both can return feasible forces, but upstream may suppress more force than necessary.
Tightening its bisection converges to the direct solution.
For the frozen near-sticking trace, the refined CPU differs from upstream by relative L2 0.350649 in force and 0.00628529 in output.
The unchanged reference evaluator still matches the actual upstream fixture to 5.55e-17 maximum error.
The refined solver intentionally differs from the source waveform and provides no additional passivity guarantee.

The compensated GPU state update reduces the isolated one-second near-sticking CPU/GPU force discrepancy to about 0.108%, and output discrepancy to 0.000288%.
Repeated-impact timing can cause large GPU/FP64 waveform differences in sustained rolling and scraping.
RMS and block continuity can remain close.
The isolated first impact converges under timestep refinement; the sustained sequences do not establish strong timestep convergence.

Near-sticking force chatter can remain at Nyquist in the source model.
Outputs are model displacement and force, with acoustic calibration left to callers.

`SdtTest` retains the independent energy, force-bound, source-trace, CPU/GPU and streaming checks.
`src/sdt/reference/Generate.c` regenerates `Trace.f64` when compiled against the pinned upstream source/library; `Upstream.cpp` retains the original evaluator.
See [Validation.md](Validation.md) for verification commands and performance limits.

[source]: https://github.com/SkAT-VG/SDT/tree/0509de418e7bebc8b37866b3b4458e0acc8cf1f4
