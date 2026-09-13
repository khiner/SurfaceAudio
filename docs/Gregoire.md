# Instrumented rough slider

Implements the nine-asperity model accompanying [Grégoire et al. 2021][paper].
It provides vertical translation, two rotations, unilateral Hertz contact, nonlinear damping and static load redistribution.
The reproduction compares contact-force predictions with measured transients recovered from Figure 14.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target roughReproduce -j8
python3 tools/gregoire_reproduce.py
```

Outputs are [Comparison.png](../outputs/reproduction/rough/gregoire/Comparison.png) and `Metrics.json` in the same directory.
Use `--offline` after downloading the pinned manuscript source.
Use `--keep-diagnostics` to retain extracted figure vertices and native traces.
The comparison uses force in newtons and time in seconds.
It produces a force plot; the approximately 1 Hz sensor transients are not airborne audio.

## Model and inputs

`rough/Slider.h` provides `InstrumentedSlider`, `EquilibrateSlider` and the sensor step response.
`rough/SliderDynamics.h` provides explicit state, track and parameter structs with `EvaluateSlider`, `StepSlider` and `SliderEnergy`.
Track inputs contain height and vertical velocity at each asperity.
Callers supply the track samples at both ends of each integration step.

Asperity coordinates and heights follow supplementary Table S1.
The weight is 2.5 N; Hertz stiffness uses the stated 0.75 mm radius, 210 GPa steel modulus and Poisson ratio 0.3.
The static solve includes every active contact and balances both moments and total force.
The dynamic example approximates inertia using the supplementary model's homogeneous box and the drawing's 60 × 60 × 29 mm dimensions.
The API's default damping ratio of 0.1 is a local choice; the published transient comparison depends on static forces and sensor filtering.

The sensor filter follows Equation 8 with cutoff 0.74 Hz and quality factor 0.656.
Its analytical step response is evaluated in C++.
The returned-to-contact comparison applies the static load as a step; its amplitude is independent of sliding speed.

## Equation corrections

Equation S2 maps upward contact force through `[1, y, -x]` to force and torques.
The printed rotational terms in Equation S4 have the opposite indentation derivatives required by that map.
The implementation uses indentation `h + z_j - z - y_j*phi + x_j*psi`, preserving virtual work with Equation S2.
Tests check this relationship through an independent derivative of potential energy.

Equation S3 is clamped to zero when its damping term would produce attraction during release.
This enforces the paper's nonnegative contact-force condition and retains nonnegative dissipation.
Time integration uses velocity Verlet with an endpoint velocity prediction for the velocity-dependent damping force.
It requires positive mass, inertias and time step, with a step small enough to resolve the loaded contact frequencies.

## Published comparisons

[Gregoire.json](../repros/rough/Gregoire.json) pins the arXiv v2 source archive, EPS member, axis calibration and vertex counts.
The 15 extracted paths include four measured speeds and one fitted sensor-response curve in each of three panels.
No amplitude fitting or waveform alignment is applied.
The figure coordinates preserve plotting quantization and possible decimation; they are not the original sensor recordings.

| Quantity | Implementation | Paper |
|---|---:|---:|
| Flat-track contacts | 1, 7, 9 | 1, 7, 9 |
| Contacts with asperity 7 over the groove | 1, 4, 9 | 1, 4, 9 |
| Asperity 7 static force | 0.11597 N | Prediction about 0.11 N; measured drop about 0.12 N |
| Asperity 9 static force | 1.17090 N | Prediction about 1.15 N; measured drop about 0.96 N |

The published dashed curves fit the sensor transients and remain distinct from the contact-model predictions.
The approximately 22% difference between the predicted and measured asperity 9 force drop remains visible.
Measured return transients vary with speed; the paper attributes this to lateral impacts against the groove edge.
Those lateral impacts and the measured high-frequency track texture are outside this normal-contact comparison.
Metrics report both the full plotted bandwidth and a stated 20 Hz low-pass comparison.

The reproduction checks static load balance and the published active-contact identities.
`ContactMechanicsTest` checks the shared normal-contact law and its potential derivatives.

[paper]: https://arxiv.org/abs/2009.07062
