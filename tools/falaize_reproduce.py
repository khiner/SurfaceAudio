#!/usr/bin/env python3
"""Reproduce Falaize–Roze equations and compare published figure constraints."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.signal import find_peaks
from AnalyzeRenders import read_wave, metrics, texture_metrics

ROOT = Path(__file__).resolve().parents[1]
DOI = "https://doi.org/10.1007/s11071-024-10438-9"


def equation_oracle(hammer, frames=4096, figure=False):
    """Solve all modal midpoint velocities and the interaction rate with a numerical Jacobian."""
    dt, mu, damping, stiffness = 1 / 96000, .0245, .1, 1e4
    modes = np.arange(1, 11)
    omega2 = 61483 / mu * (modes * np.pi / 1.8) ** 2
    shape = np.sqrt(2 / 1.8) * np.sin(modes * np.pi * .3)
    q, v, z, vh = np.zeros(10), np.zeros(10), 0., 1. if hammer else 0.
    result, guess = np.zeros((frames, 4)), np.zeros(11)

    def potential(x):
        return (920 if figure else 13.8) * .015**2 / 3.5 * max(x / .015, 0)**3.5

    for i in range(frames):
        def residual(x):
            vm, rate = x[:10], x[10]
            if hammer:
                next_z = z + dt * rate
                gradient = (potential(next_z) - potential(z)) / (dt * rate) if abs(dt * rate) > 1e-15 else (920 if figure else 13.8) * .015 * max(z / .015, 0)**2.5
                force = gradient + .184 * 2.5 * max(z / .015, 0)**1.5 * rate
                constraint = rate - (vh - dt * force / .06 - shape @ vm)
            else:
                slip, mid = .1 - shape @ vm, z + dt * rate / 2
                steady = (.3 + .5 * np.exp(-(slip / .1)**2)) / stiffness
                alpha = 0.
                if slip * z > 0 and abs(z) > .7 * .3 / stiffness:
                    alpha = 1. if abs(z) >= steady else .5 * (1 + np.sin(np.pi * (abs(z) - .5 * (steady + .7 * .3 / stiffness)) / (steady - .7 * .3 / stiffness)))
                force = stiffness * mid + .1 * rate + .4 * slip
                constraint = rate - slip + alpha * abs(slip) / (stiffness * steady) * stiffness * mid
            linear = vm - v - dt / 2 * (-omega2 * (q + dt / 2 * vm) - damping / mu * vm + shape / mu * force)
            return np.r_[linear, constraint]

        solution = root(residual, guess, method="hybr", options={"xtol": 1e-10})
        if np.max(np.abs(residual(solution.x))) > 2e-10:
            raise RuntimeError(f"Independent dense oracle failed at sample {i}")
        guess = solution.x
        vm, rate = guess[:10], guess[10]
        if hammer:
            next_z = z + dt * rate
            force = (potential(next_z) - potential(z)) / (dt * rate) + .184 * 2.5 * max(z / .015, 0)**1.5 * rate
            vh -= dt * force / .03
        else:
            force = stiffness * (z + dt * rate / 2) + .1 * rate + .4 * (.1 - shape @ vm)
        q, v, z = q + dt * vm, 2 * vm - v, z + dt * rate
        result[i] = [shape @ q, shape @ v, force, z]
    return result


def figure_checks(trace):
    late = trace[trace["time"] > .9]
    peaks, _ = find_peaks(late["displacement"], distance=100)
    period = float(np.median(np.diff(late["time"][peaks])))
    amplitude = float(np.max(np.abs(late["displacement"])))
    blocks = np.max(np.abs(trace["displacement"].reshape(100, -1)), axis=1)
    onset = float(np.argmax(blocks > .9 * amplitude) * .01)
    return {"period_seconds": period, "late_displacement_peak_m": amplitude,
            "late_velocity_min_m_s": float(np.min(late["velocity"])), "late_velocity_max_m_s": float(np.max(late["velocity"])),
            "time_to_90_percent_amplitude_seconds": onset,
            "figure19_period_interval_seconds": [.0021, .0025], "figure19_peak_interval_m": [5e-5, 7e-5],
            "figure19_onset_interval_seconds": [.25, .5],
            "period_within_figure_interval": .0021 <= period <= .0025,
            "amplitude_within_figure_interval": 5e-5 <= amplitude <= 7e-5,
            "onset_within_figure_interval": .25 <= onset <= .5}


def plot_results(output):
    bow = np.genfromtxt(output / "bow_modal.csv", delimiter=",", names=True)
    literal = np.genfromtxt(output / "hammer_modal.csv", delimiter=",", names=True)
    inferred = np.genfromtxt(output / "hammer_figure.csv", delimiter=",", names=True)
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(bow["time"], bow["displacement"])
    axes[0, 0].set(ylabel="Bow contact displacement (m)", xlabel="Time (s)")
    late = bow["time"] > .98
    axes[1, 0].plot(bow["time"][late], bow["velocity"][late])
    axes[1, 0].set(ylabel="Bow contact velocity (m/s)", xlabel="Time (s)")
    axes[2, 0].plot(bow["time"], np.maximum(np.abs(bow["balance_error"]), 1e-25))
    axes[2, 0].set(yscale="log", ylabel="Bow step energy error (J)", xlabel="Time (s)")
    for trace, label in [(literal, "Literal equations and table"), (inferred, "Figure-inferred stiffness")]:
        axes[0, 1].plot(trace["time"], trace["displacement"], label=label)
        contact = trace["elastic"] >= 0
        axes[1, 1].plot(trace["elastic"][contact], trace["force"][contact], label=label)
        axes[2, 1].plot(trace["time"], trace["energy"], label=label)
    axes[0, 1].set(ylabel="Hammer contact displacement (m)", xlabel="Time (s)")
    axes[1, 1].set(ylabel="Hammer interaction force (N)", xlabel="Crush (m)")
    axes[2, 1].set(ylabel="Hammer total energy (J)", xlabel="Time (s)")
    axes[0, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=.2)
    fig.savefig(output / "validation.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/falaizeReproduce")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reproduction/falaize")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--gpu-voices", type=int, default=128)
    parser.add_argument("--regenerate-oracle", action="store_true")
    args = parser.parse_args()
    provenance = json.loads((ROOT / "repros/falaize/Reference.json").read_text())
    if not args.regenerate_oracle:
        for name, expected in provenance["oracle"]["files_sha256"].items():
            path = ROOT / "repros/falaize" / name
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Missing or changed pinned equation fixture: {path}")
    args.output.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(args.binary.resolve()), str(args.output.resolve()), str(args.gpu_voices)], check=True)
    records = {}
    for name in ["bow_modal", "hammer_modal", "bow_fem", "hammer_fem", "bow_midpoint", "hammer_figure", "hammer_figure55"]:
        trace = np.genfromtxt(args.output / f"{name}.csv", delimiter=",", names=True)
        record = json.loads((args.output / f"{name}.json").read_text())
        if name in ("bow_modal", "bow_midpoint"):
            record["figure19"] = figure_checks(trace)
        if name in ("bow_modal", "hammer_modal", "hammer_figure"):
            path = ROOT / "repros/falaize" / f"{name}_oracle.f64"
            if args.regenerate_oracle:
                equation_oracle(name.startswith("hammer"), figure=name == "hammer_figure").astype("<f8").tofile(path)
            if path.exists():
                oracle = np.fromfile(path, dtype="<f8").reshape(-1, 4)
                native = np.column_stack([trace[field][:len(oracle)] for field in ["displacement", "velocity", "force", "elastic"]])
                record["independent_equation_oracle"] = {"samples": len(oracle), "maximum_absolute_errors": np.max(np.abs(native - oracle), axis=0).tolist(), "relative_l2_by_column": (np.linalg.norm(native - oracle, axis=0) / np.maximum(np.linalg.norm(oracle, axis=0), 1e-30)).tolist(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                if np.max(record["independent_equation_oracle"]["relative_l2_by_column"]) > 1e-7:
                    raise RuntimeError("Independent equation oracle mismatch")
        rate, wave = read_wave(args.output / f"{name}_velocity.wav")
        if name.startswith("hammer"):
            record["figure_comparison"] = {"final_energy_J": float(trace["energy"][-1]), "maximum_crush_m": float(np.max(trace["elastic"])), "maximum_force_N": float(np.max(trace["force"])), "peak_displacement_m": float(np.max(trace["displacement"])), "minimum_velocity_m_s": float(np.min(trace["velocity"])), "maximum_velocity_m_s": float(np.max(trace["velocity"]))}
        record["waveform"] = metrics(rate, wave)
        record["texture"] = texture_metrics(rate, wave)
        records[name] = record
    plot_results(args.output)
    cases = []
    for interaction in ["bow", "hammer"]:
        cases.append({"title": f"Falaize–Roze {interaction}: modal and finite element projection", "group": "Falaize–Roze 2024", "reference": str((args.output / f"{interaction}_modal_velocity.wav").resolve()), "synthesis": str((args.output / f"{interaction}_fem_velocity.wav").resolve()), "reference_label": "Native 10-mode projection", "synthesis_label": "Native 20-element projection", "source_url": DOI, "notes": "Equation reconstruction using Tables 1–3 with explicit state evaluation. Both signals are native renders; no author recording or source trace is available. Bow velocity 0.1 m/s is inferred from Figure 19."})
    cases.append({"title": "Falaize–Roze hammer: literal equations and figure-derived stiffness", "reference": str((args.output / "hammer_modal_velocity.wav").resolve()), "synthesis": str((args.output / "hammer_figure_velocity.wav").resolve()), "reference_label": "Literal Table 2 and Appendix B", "synthesis_label": "Figure-inferred stiffness", "source_url": DOI, "notes": "Appendix B includes a thickness factor in elastic force that is inconsistent with Figure 10. The second native render uses stiffness 13.8/0.015=920 N/m, reproducing plotted peak force, crush, final energy and contact waveform. This inference is not author code."})
    cases.append({"title": "Falaize–Roze hammer: Figure 9 string frequencies", "reference": str((args.output / "hammer_figure55_velocity.wav").resolve()), "synthesis": str((args.output / "hammer_figure_velocity.wav").resolve()), "reference_label": "55 Hz string", "synthesis_label": "440 Hz string", "source_url": DOI, "notes": "Both use the explicitly figure-inferred hammer stiffness with the Figure 9 tension ratio 1/64. The 0.1 s duration follows the published plots."})
    cases.append({"title": "Falaize–Roze bow: state evaluation ambiguity", "reference": str((args.output / "bow_modal_velocity.wav").resolve()), "synthesis": str((args.output / "bow_midpoint_velocity.wav").resolve()), "reference_label": "Explicit state, theta=0", "synthesis_label": "Midpoint state, theta=0.5", "source_url": DOI, "notes": "Both preserve the discrete energy balance; the paper does not specify state evaluation in nonlinear dissipation. Theta=0 follows the cited framework default, not recovered paper code."})
    (args.output / "cases.json").write_text(json.dumps({"cases": cases}, indent=2) + "\n")
    benchmarks = {p.stem: json.loads(p.read_text()) for p in args.output.glob("*benchmark.json")} if args.gpu_voices else {}
    (args.output / "manifest.json").write_text(json.dumps({"source": DOI, "provenance": provenance, "records": records, "benchmarks": benchmarks, "pluck": json.loads((args.output / "pluck.json").read_text()), "tolerances": json.loads((args.output / "tolerances.json").read_text())}, indent=2, allow_nan=False) + "\n")
    for path in args.output.glob("*.csv"):
        path.unlink()
    for name in records:
        (args.output / f"{name}.json").unlink()
    print(json.dumps({name: record.get("figure19", record.get("independent_equation_oracle")) for name, record in records.items()}, indent=2))


if __name__ == "__main__":
    main()
