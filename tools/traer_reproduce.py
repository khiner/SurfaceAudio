#!/usr/bin/env python3
"""Check Traer equations against pinned later TDW author code and material data."""
import __future__
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import struct
import tempfile
from types import SimpleNamespace
import urllib.request

import numpy as np
from scipy import signal
from scipy.io import wavfile
from AnalyzeRenders import read_wave, metrics

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "references/traer"


def sources(offline):
    manifest = json.loads((ROOT / "repros/traer/sources.json").read_text())
    for item in manifest["files"]:
        path = REFERENCES / item["path"]
        if not path.exists():
            if offline:
                raise RuntimeError(f"Missing pinned source: {path}")
            with urllib.request.urlopen(item["url"], timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise RuntimeError(f"Source hash changed: {item['url']}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Source hash mismatch: {path}")
    return manifest


def author_functions():
    path = REFERENCES / "tdw/Python/tdw/physics_audio/modes.py"
    spec = importlib.util.spec_from_file_location("traer_author_modes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = REFERENCES / "tdw/Python/tdw/add_ons/py_impact.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PyImpact")
    namespace = {"np": np, "sg": signal, "Modes": module.Modes}
    for name in ["_get_object_modes", "_synth_impact_modes"]:
        function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
        function.decorator_list = []
        isolated = ast.Module(body=[function], type_ignores=[])
        exec(compile(isolated, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace


def parameters(modes, rate, resonance):
    end = np.maximum(0, np.ceil(modes.decay_times * (80 + modes.powers) / 60000 * rate)).astype(int)
    return np.column_stack([modes.frequencies, modes.powers, 60000 / (modes.decay_times * resonance), end])


def relative(a, b):
    if a.shape != b.shape or not np.all(np.isfinite(b)):
        raise RuntimeError("Nonfinite or unmatched waveform")
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))


def scrape_demo(binary, output, controls, materials):
    settings = controls["scrape_demo"]
    rate = controls["sample_rate"]
    frames, taps = int(settings["seconds"] * rate), int(settings["ir_seconds"] * rate)
    profile = np.load(REFERENCES / "tdw/Python/tdw/add_ons/py_impact/scrape_surfaces/bass_wood.npy")
    data = materials[settings["material"]]
    parameters = np.column_stack([data["cf"], data["op"], 60 / np.array(data["rt"])])
    with tempfile.TemporaryDirectory(prefix="traer-scrape-") as scratch:
        scratch = Path(scratch)
        profile.astype(np.float64).tofile(scratch / "profile.f64")
        with (scratch / "config.txt").open("w") as stream:
            stream.write(f"{taps} {rate} {len(parameters)} {frames} {settings['spacing_m']} {settings['mass_kg']} "
                         f"{settings['shear_gain']} {settings['gamma']} {settings['low_m']} {settings['high_m']} {settings['cycles']}\n")
            np.savetxt(stream, parameters, fmt="%.17g")
        subprocess.run([str(binary.resolve()), "scrape", str(scratch / "config.txt"), str(scratch / "profile.f64"), str(output)], check=True)
    position = np.fromfile(output / "position.f64")
    velocity = np.fromfile(output / "velocity.f64")
    spacing = settings["spacing_m"]
    slope = (profile[2:] - profile[:-2]) / (2 * spacing)
    curvature = (profile[2:] - 2 * profile[1:-1] + profile[:-2]) / spacing ** 2
    grid = np.arange(1, len(profile) - 1) * spacing
    slope = np.interp(position, grid, slope)
    curvature = np.interp(position, grid, curvature)
    shear = velocity * slope
    force = settings["mass_kg"] * curvature * velocity ** 2 + settings["shear_gain"] * np.sign(shear) * abs(shear) ** settings["gamma"]
    _, actual_force = read_wave(output / "force.wav")
    force_error = relative(force, actual_force[:, 0])
    modes = np.loadtxt(output / "modes.txt").reshape(3, -1, 3)
    time = np.arange(taps) / rate
    responses = np.sum(10 ** ((modes[:, :, 1, None] - modes[:, :, 2, None] * time) / 20) *
                       np.cos(2 * np.pi * modes[:, :, 0, None] * time), axis=1)
    convolved = np.array([signal.fftconvolve(force, response) for response in responses])
    locations = np.pad(position.astype(np.float32), (0, taps - 1), mode="edge")
    nodes = np.array([settings["low_m"], (settings["low_m"] + settings["high_m"]) / 2, settings["high_m"]], dtype=np.float32)
    left = np.clip(np.searchsorted(nodes, locations, side="right") - 1, 0, 1)
    mix = np.clip((locations - nodes[left]) / (nodes[left + 1] - nodes[left]), 0, 1)
    indices = np.arange(len(locations))
    expected = (1 - mix) * convolved[left, indices] + mix * convolved[left + 1, indices]
    _, varying = read_wave(output / "varying.wav")
    _, constant = read_wave(output / "constant.wav")
    waveform_error = relative(expected, varying[:, 0])
    constant_error = relative(convolved[0], constant[:, 0])
    if max(force_error, waveform_error, constant_error) > 2e-4:
        raise RuntimeError(f"Scrape full record equation comparison failed: {force_error}, {waveform_error}, {constant_error}")
    gain = float(.8 / max(np.max(abs(varying)), np.max(abs(constant)), 1e-30))
    for filename, audio in [("varying.wav", varying), ("constant.wav", constant)]:
        wavfile.write(output / filename, rate, (gain * audio[:, 0]).astype(np.float32))
    for filename in ["position.f64", "velocity.f64", "modes.txt", "force.wav"]:
        (output / filename).unlink()
    return {"settings": settings, "force_relative_error": force_error, "spatial_waveform_relative_error": waveform_error,
            "constant_waveform_relative_error": constant_error, "common_listening_gain": gain,
            "scope": "Traer Equations 8–9 demonstration on a later TDW distributed 1D basswood profile and ten-mode means. "
                     "Original 2019 surfaces, trajectories and distributions are unavailable. Force and spatial parameters are prescribed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/traerReproduce")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reproduction/traer")
    args = parser.parse_args()
    provenance = sources(args.offline)
    controls = json.loads((ROOT / "repros/traer/cases.json").read_text())
    functions = author_functions()
    material_root = REFERENCES / "tdw/Python/tdw/add_ons/py_impact/material_data"
    materials = {name: json.loads((material_root / f"{name}_1_mm.json").read_text()) for name in controls["materials"]}
    clatter_data = (REFERENCES / "clatter/Clatter/Clatter.Core/Data/ImpactMaterials/metal_1_mm.bytes").read_bytes()
    if struct.unpack("<iii", clatter_data[:12]) != (10, 10, 10):
        raise RuntimeError("Unexpected Clatter material layout")
    clatter_parameters = np.frombuffer(clatter_data[12:], dtype="<f8")
    tdw_parameters = np.concatenate([materials["metal"][key] for key in ["cf", "op", "rt"]])
    if not np.array_equal(clatter_parameters, tdw_parameters):
        raise RuntimeError("Pinned Clatter and TDW material parameters disagree")
    original = SimpleNamespace(material_data=materials, rng=np.random.RandomState(controls["seed"]))
    rate = controls["sample_rate"]
    args.output.mkdir(parents=True, exist_ok=True)
    cases, comparisons = [], []
    with tempfile.TemporaryDirectory(prefix="traer-") as scratch:
        scratch = Path(scratch)
        for material in controls["materials"]:
            first = functions["_get_object_modes"](original, material)
            second = functions["_get_object_modes"](original, controls["secondary_material"])
            mode_parameters = np.concatenate([parameters(first, rate, 1), parameters(second, rate, 1)])
            frames = int(mode_parameters[:, 3].max())
            author_ir = functions["Modes"].mode_add(first.sum_modes(), second.sum_modes())
            config = scratch / "response.txt"
            with config.open("w") as stream:
                stream.write(f"{frames} {rate} {len(mode_parameters)} 0\n")
                np.savetxt(stream, mode_parameters, fmt="%.17g")
            for mass in controls["masses_kg"]:
                name = f"tdw_{material}_{mass:g}kg"
                output = args.output / name
                force = np.sin(np.linspace(0, np.pi, int(np.ceil(min(.001 * mass, .002) * rate))))
                force_path = scratch / "force.wav"
                wavfile.write(force_path, rate, force.astype(np.float32))
                run = subprocess.run([str(args.binary.resolve()), str(config), str(force_path), str(output)], check=True,
                                     capture_output=True, text=True)
                author = functions["_synth_impact_modes"](first, second, mass, 1, 1)
                reference = output / "author.wav"
                wavfile.write(reference, rate, author.astype(np.float32))
                _, ours = read_wave(output / "synthesis.wav")
                ours = ours[:, 0]
                unnormalized = signal.fftconvolve(author_ir, force)
                gain = float(1 / abs(np.max(unnormalized)))
                ours *= gain
                wavfile.write(output / "synthesis.wav", rate, ours.astype(np.float32))
                ir_reference = np.fromfile(output / "response.f64", dtype=np.float64)
                ir_error = relative(author_ir, ir_reference)
                waveform_error = relative(author, ours)
                spectrum_error = relative(np.abs(np.fft.rfft(author)), np.abs(np.fft.rfft(ours)))
                if ir_error > 2e-4 or waveform_error > 5e-4:
                    raise RuntimeError(f"Author waveform comparison failed: {name}: {ir_error}, {waveform_error}")
                comparisons.append({"case": name, "frames": len(author), "ir_relative_error": ir_error,
                                    "waveform_relative_error": waveform_error, "spectrum_relative_error": spectrum_error,
                                    "source_normalization_gain": gain, "gpu": json.loads(run.stdout),
                                    "author_metrics": metrics(rate, author[:, None]), "ours_metrics": metrics(rate, ours[:, None])})
                cases.append({"title": f"Traer descendant TDW / {material} / {mass:g} kg", "reference": str(reference.resolve()),
                              "synthesis": str((output / "synthesis.wav").resolve()),
                              "source_url": next(x["url"] for x in provenance["files"] if x["path"].endswith("py_impact.py")),
                              "notes": "Original TDW author code and distributed material data versus C++/Metal. "
                                       "Exact source normalization, no fitted gain. Later ten-mode implementation without transients or joint covariance. "
                                       "This is not a reproduction of the 2019 listening experiment."})
                (output / "response.f64").unlink()
                (output / "response.wav").unlink()
        with np.errstate(divide="ignore", invalid="ignore"):
            tiny = functions["_synth_impact_modes"](first, second, .0007, 1, 1)
        source_small_mass_finite = bool(np.all(np.isfinite(tiny)))
    scrape = scrape_demo(args.binary, args.output / "scrape_demo", controls, materials)
    cases.append({"title": "Traer Equations 8–9 basswood back-and-forth demonstration", "reference": str((args.output / "scrape_demo/constant.wav").resolve()),
                  "synthesis": str((args.output / "scrape_demo/varying.wav").resolve()), "reference_label": "Constant IR control",
                  "notes": "Left: constant IR control. Right: three spatial IRs with fixed frequencies/decays and perturbed onset powers. "
                           "Later TDW measured basswood profile and ten-mode data, prescribed trajectory and force coefficients. "
                           "Demonstration only, neither clip is an original author stimulus.",
                  "source_url": next(x["url"] for x in provenance["files"] if x["path"].endswith("bass_wood.npy"))})
    benchmarks = [json.loads(subprocess.run([str(args.binary.resolve()), "benchmark"], check=True,
                                           capture_output=True, text=True).stdout) for _ in range(3)]
    (args.output / "cases.json").write_text(json.dumps({"cases": cases}, indent=2) + "\n")
    manifest = {"method": "Traer 2019 equations / later TDW author-code comparison", "sources": provenance,
                "controls": controls, "comparisons": comparisons, "scrape_demo": scrape, "benchmark": benchmarks[1], "fresh_process_benchmarks": benchmarks,
                "source_0.7g_impact_finite": source_small_mass_finite, "clatter_tdw_metal_parameters_exact": True,
                "scope": "Original TDW method bodies execute in isolation to avoid its Unity/controller dependencies. "
                         "No calibration is fitted. Full records include individual mode cutoffs and convolution tails.",
                "gaps": ["2019 empirical joint means/covariances, 30-band transient fits, raw recordings, quilted depth maps, "
                         "and measured OptiTrack trajectories were not located in the public paper/lab/source repositories.",
                         "The Equation 5 duration bound and Section 3.3 linear mass ratio disagree with the square-root spring frequency.",
                         "Original TDW fails for the paper's 0.7 g pellet because its one-sample half-sine is zero before normalization.",
                         "Original 2019 stimuli and listening-study results remain unreproduced."]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output / "cases.json")


if __name__ == "__main__":
    main()
