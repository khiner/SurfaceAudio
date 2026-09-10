# Falaize–Roze nonlinear interactions

Implements [Falaize and Roze's model][paper], published online in 2024 and in Nonlinear Dynamics 113 (2025).
It includes Dupont elasto-plastic friction, a nonlinear lossy hammer, modal projection and consistent linear finite elements.
C++23 FP64 and Metal 4 FP32 share the trajectory equations.

```sh
cmake --build build --target FalaizeTest falaizeReproduce -j8
build/FalaizeTest
python3 tools/falaize_reproduce.py --offline
```

Outputs are in `outputs/reproduction/falaize/`.
Velocity WAVs retain metres per second; listening comparisons identify native variants.
Use `--gpu-voices 0` for CPU rendering and `--regenerate-oracle` to rebuild the independent SciPy fixtures.
The default GPU workload contains 128 trajectories with slightly different drives.

## Sources and controls

The publisher states that the manuscript has no associated data.
Paper-specific author code and audio are unavailable; comparisons use independent equations and published figure constraints.
[Reference.json](../repros/falaize/Reference.json) records the paper hash, controls, constraints and fixture hashes.
The implementation copies no author code.

Interaction examples use 96000 Hz, ten modes, twenty finite elements and contact position `0.3 L`.
Physical and friction parameters follow Tables 1–3.
Bow speed `0.1 m/s` is inferred from the sticking velocity in Figure 19.
State-dependent dissipation is evaluated at `x+theta*delta_x`.
The default `theta=0` follows the cited [PyPHS framework][pyphs]; an additional `theta=0.5` render exercises the unspecified evaluation choice.

## Equations and passivity

Equations 3–17 give `r=alpha(x,v)*abs(v)/fss(v)`, `xdot=v-r*k*x` and `F=k*x+sigma_c*xdot+sigma_f*v`.
The adhesion map retains the elastic, transition, steady-slip and same-sign conditions in Equations 6–7.
Discrete elastic force `k*(x_next+x)/2` satisfies the quadratic energy chain rule.
The reconstructed system satisfies `delta_energy+dt*(dissipation-input_power)=0` to solver accuracy.
Nonnegative dissipation requires a separate condition.

The symmetric part of Equation 17 has determinant `r*(sigma_c+sigma_f)-sigma_c^2*r^2/4`.
Its power is nonnegative for all port efforts when `sigma_c^2*r <= 4*(sigma_c+sigma_f)` or `r=0`.
Tests retain a negative-power counterexample and independently check the passive `sigma_c=0` domain.
The [2025 successor][successor] refines the damping law; that model is implemented separately in [Matusiak.md](Matusiak.md).

The modal basis is `sqrt(2/L)*sin(n*pi*x/L)`, Equation 45.
Finite elements use the consistent mass and stiffness matrices in Equations 48–54.
Their simultaneous sine diagonalization retains all interior degrees of freedom.
Quadratic discrete gradients give implicit midpoint integration, including frequency warping.
The conservative pluck uses Equations 59–60 and the 100-mode analytic reference at 48000 Hz.
Linear string variables are eliminated before a two-variable bow solve or scalar hammer solve.
Velocity residual tolerances are `1e-12` for FP64 and `2e-6` for FP32, with failure reported after forty updates.

## Hammer equation and figure discrepancy

Appendix B gives `c=max(q/l,0)`, elastic force `k*l*c^beta`, energy `k*l^2*c^(beta+1)/(beta+1)` and damping `a*beta*c^(beta-1)*qdot`.
Table 2 gives `k=13.8`, `a=0.184`, `beta=2.5`, thickness `l=0.015 m` and mass `0.03 kg`.
The literal render starts at zero crush and `1 m/s`, retaining the printed unloading term that can become tensile before separation.
At `q=0.01 m`, the printed force is `0.0751 N`, while Figure 10 shows approximately `5 N`.
The separately labeled figure-inferred variant uses `k=13.8/0.015=920 N/m` in both force and energy.

| Quantity | Literal equations and table | Figure-inferred stiffness | Published plot |
|---|---:|---:|---:|
| Maximum crush | 0.027167 m | 0.009981 m | Approximately 0.010 m |
| Maximum force | 1.13675 N | 4.9877 N | Approximately 5 N |
| Energy at 0.1 s | 0.002739 J | 0.013590 J | Approximately 0.0136 J |

The 55/440 Hz variants retain the Figure 9 tension ratio of 64.
The source implementation behind the force discrepancy remains unknown.

## Validation and timing

The bow's late displacement peak is `6.0812e-5 m`, its period is `2.28125 ms`, and velocity spans `-0.1990` to `0.1114 m/s`.
It reaches 90% sustained amplitude near `0.25 s`, within the approximate Figure 19 constraints.
Those constraints reflect plot resolution and missing controls rather than waveform error bounds.

Independent NumPy/SciPy fixtures solve all ten modal midpoint velocities and the interaction rate with a numerical Jacobian.
They cover 4096 consecutive samples of displacement, velocity, force and interaction state for bow and both hammer variants.
Maximum relative field errors are `1.46e-13` for bow, `9.89e-10` for literal hammer and `2.53e-10` for figure-inferred hammer.
Tests also cover derivatives, dispersion, FEM spectra, nonzero initial energy and complete modal/FEM trajectories.
Cumulative FP64 energy errors are below `7e-15 J` for the tested parameters.
The reproduction exercises the paper's three residual tolerances `1e-9`, `1e-12` and `1e-15`.

Each Metal thread advances one causal trajectory; independent strings run concurrently, with up to 128 modes or 129 finite elements.
Both CPU precisions and Metal retain waveforms, final states, residuals, failure counts and energy diagnostics during timed rendering.
CPU timing includes model/state construction and allocation; GPU timing includes kernel creation, allocation, upload, dispatch, waiting and readback.
Controls are identically float-rounded, and comparisons and file I/O are excluded.
Three fresh-process M5 Max Release measurements used 128 voices per batch.

| Interaction and frames per voice | FP64 CPU | FP32 CPU | FP32 Metal |
|---|---:|---:|---:|
| Bow, 96000 | 1.548–1.550 s | 0.951–0.953 s | 0.3926–0.3938 s |
| Literal hammer, 9600 | 0.1399–0.1570 s | 0.0513–0.0518 s | 0.0606–0.0608 s |

FP64/Metal waveform relative L2 is `2.29e-4` for bow and `8.20e-4` for hammer; all steps converge.
The hammer's final modal velocity differs by 11.8%, or `5.64e-8 m/s` RMS against a `4.80e-7 m/s` residual field.
Other final-state relative errors are below `5.4e-4`; the manifest records each field and its physical scale.
Audio-device deadlines and FP32 passivity at machine precision remain unverified.

[paper]: https://doi.org/10.1007/s11071-024-10438-9
[pyphs]: https://github.com/pyphs/pyphs/tree/ff0922838f7dcaa6948e7a30352a006b3ba13527
[successor]: https://doi.org/10.3389/frsip.2025.1525044
