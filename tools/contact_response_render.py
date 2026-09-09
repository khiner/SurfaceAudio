#!/usr/bin/env python3
"""Drive measured and fitted object responses with synthesized sustained contact forces."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTION = (
    "First compare the authors' published scraping and rolling examples with our corresponding reconstructions. "
    "Then compare measured and fitted 2023 responses driven by the same synthesized force. "
    "The latter are new cross-paper combinations, with no matching published continuous-contact recording."
)
AUTHOR_GROUPS = {"agarwal": "Agarwal: author examples vs our reconstructions",
                 "conan": "Conan: author examples vs our reconstructions"}
RESPONSE_GROUP = "Measured vs fitted responses — both rendered by us"


def fingerprint(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_float(path):
    rate, samples = wavfile.read(path)
    if rate != 44100 or samples.ndim != 1 or samples.dtype != np.float32:
        raise ValueError(f"Expected mono float32 44100 Hz input: {path}")
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError(f"Empty or nonfinite input: {path}")
    return samples


def activity(samples, frames):
    # Exclude the gate transitions and response tail. Remove local DC so steady
    # contact load cannot masquerade as sustained audible excitation.
    block = 4410
    interior = samples[block:min(frames, len(samples)) - block].astype(np.float64)
    blocks = interior[:len(interior) // block * block].reshape(-1, block)
    rms = np.std(blocks, axis=1)
    return {"window_ms": 100, "windows": len(rms),
            "fraction_above_minus40db_peak_window": float(np.mean(rms > rms.max() * .01)),
            "minimum_to_maximum_window_ac_rms": float(rms.min() / rms.max())}


def build_report(output, response_cases):
    selections = {
        "agarwal": ["wood-cell.wav", "wood-pointwise.wav", "glass-cell.wav", "glass-pointwise.wav",
                    "scrape-temporal-whole-calibrated.wav", "roll-temporal-spatial-whole.wav",
                    "roll-glass-temporal-spatial-whole.wav"],
        "conan": ["physical-full-gaussian-ir1.wav", "physical-full-gaussian-ir2.wav"],
    }
    author_cases = []
    for method, names in selections.items():
        source = ROOT / "outputs/reproduction" / method / "cases.json"
        available = {Path(case["synthesis"]).name: case for case in json.loads(source.read_text())["cases"]}
        for name in names:
            case = available[name]
            author_cases.append({**case, "group": AUTHOR_GROUPS[method],
                                 "reference": str(ROOT / case["reference"]),
                                 "synthesis": str(ROOT / case["synthesis"]),
                                 "reference_label": "Author's published audio",
                                 "synthesis_label": case.get("synthesis_label", "Our reconstruction"),
                                 "source_manifest": fingerprint(source)})
    cases = author_cases + [{**case, "group": RESPONSE_GROUP} for case in response_cases]
    for case in cases:
        for field in ["reference", "synthesis"]:
            if not Path(case[field]).is_file():
                raise FileNotFoundError(case[field])
    manifest = output / "cases.json"
    manifest.write_text(json.dumps({"title": "Sustained surface contact", "description": DESCRIPTION, "cases": cases}, indent=2) + "\n")
    subprocess.run([sys.executable, str(ROOT / "tools/BuildListeningReport.py"), str(manifest),
                    "--output", str(output / "listening")], check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/agarwalReproduce")
    parser.add_argument("--responses", type=Path, default=ROOT / "outputs/reproduction/agarwal-response-aligned")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reproduction/contact-responses")
    parser.add_argument("--report-only", action="store_true", help="Update author comparisons and the listening page using existing renders")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        cases = json.loads((output / "cases.json").read_text())["cases"]
        build_report(output, [case for case in cases if case.get("group") == RESPONSE_GROUP])
        return
    binary = args.binary.resolve()
    artifacts = [binary, binary.parent / "SurfaceAudio.metallib"]
    artifact_pins = [fingerprint(path) for path in artifacts]
    marker = output / "run.json"
    marker.unlink(missing_ok=True)
    force_sources = [
        ("scrape", ROOT / "outputs/reproduction/agarwal/scrape-temporal-whole-calibrated-force.wav",
         "Agarwal 2021 scraping equations on a measured TDW surface profile, temporal Gaussian smoothing. "
         "Motion and smoothing assumptions use the existing whole-record calibration; the original surface/motion inputs remain unavailable."),
        ("roll", ROOT / "outputs/reproduction/conan/physical-full-gaussian-seed0.wav",
         "Conan 2014 Gaussian ARMA rolling force, new seed 2014, shared innovations and duration exponent 0.29. "
         "Parameters fitted to the full author ForceModelePhy example; MP3-derived amplitude has unknown physical units."),
    ]
    force_metadata = {
        "scrape": [ROOT / "outputs/reproduction/agarwal/scrape-temporal-whole-case" / name
                   for name in ["parameters.txt", "coordinates.json", "smoothing.json", "calibration.json"]],
        "roll": [ROOT / "outputs/reproduction/conan" / name for name in ["inputs.json", "parameters.json"]],
    }
    cases, records = [], []
    for motion, source, notes in force_sources:
        source_metadata = [fingerprint(path) for path in force_metadata[motion]]
        source_samples = read_float(source)
        frames = 4 * 44100
        if len(source_samples) < frames:
            raise ValueError(f"Need four seconds of actual excitation, without looping: {source}")
        fade_frames = 882
        gate = .5 - .5 * np.cos(np.pi * np.arange(fade_frames) / (fade_frames - 1))
        envelope = np.concatenate((gate, np.ones(frames - 2 * fade_frames), gate[::-1]))
        force = (source_samples[:frames] * envelope).astype(np.float32)
        force_path = output / f"{motion}-force.f32"
        force.tofile(force_path)
        wavfile.write(output / f"{motion}-force.wav", 44100, force)
        for material in ["Wood", "Plastic", "Metal", "Glass"]:
            directory = args.responses.resolve() / "fits" / f"{material}_1"
            provenance_path = directory / "provenance.json"
            provenance = json.loads(provenance_path.read_text())
            paths = {}
            for kind, name in [("measured", "prepared.wav"), ("fitted", "fitted-response.wav")]:
                response_path = directory / name
                response_pin = fingerprint(response_path)
                if response_pin["sha256"] != provenance["artifact_sha256"][name]:
                    raise ValueError(f"Response changed since fitting: {response_path}")
                response = read_float(response_path)
                taps_path = output / f"{material.lower()}-{kind}-response.f32"
                response.tofile(taps_path)
                destination = output / f"{motion}-{material.lower()}-{kind}.wav"
                command = [str(binary), "filter", str(force_path), str(taps_path), str(destination), "44100"]
                subprocess.run(command, check=True, cwd=ROOT)
                rendered = read_float(destination)
                # Independent full-record double-precision FFT oracle, including
                # the entire response tail, validates the production GPU FIR.
                reference = fftconvolve(force.astype(np.float64), response.astype(np.float64))
                if len(rendered) != len(reference):
                    raise RuntimeError(f"Incomplete convolution tail: {destination}")
                relative_error = float(np.linalg.norm(rendered - reference) / max(np.linalg.norm(reference), 1e-30))
                if relative_error > 1e-4:
                    raise RuntimeError(f"GPU convolution relative RMS error {relative_error}: {destination}")
                ac = rendered.astype(np.float64) - np.mean(rendered, dtype=np.float64)
                gain = min(.1 / np.sqrt(np.mean(ac * ac)), .89 / np.max(np.abs(ac)))
                audition = output / f"{motion}-{material.lower()}-{kind}-audition.wav"
                wavfile.write(audition, 44100, (ac * gain).astype(np.float32))
                records.append({"motion": motion, "material": material, "response_kind": kind,
                                "force_source": fingerprint(source), "force": fingerprint(force_path),
                                "force_source_metadata": source_metadata,
                                "response": response_pin, "response_provenance": fingerprint(provenance_path),
                                "output": fingerprint(destination), "audition": fingerprint(audition),
                                "audition_gain": float(gain), "frames": len(rendered),
                                "gpu_relative_rms_error": relative_error,
                                "force_activity": activity(force, frames), "output_activity": activity(rendered, frames),
                                "command": command})
                paths[kind] = str(destination)
                print(f"{destination.name}: {len(rendered) / 44100:.3f}s, GPU error {relative_error:.3g}", flush=True)
            cases.append({"title": f"{motion.capitalize()} — {material.lower()}",
                          "reference": paths["measured"], "synthesis": paths["fitted"],
                          "reference_label": "Same synthesized force × measured response",
                          "synthesis_label": "Same synthesized force × fitted response",
                          "notes": notes + " Four-second force excerpt, 20 ms half-cosine contact onset/release gates, complete response tail. "
                                   "Measured response is onset-aligned and median-DC-subtracted. "
                                   "Object example 1 chosen uniformly across materials; fixed response, no position-dependent contact coupling."})
    if artifact_pins != [fingerprint(path) for path in artifacts]:
        raise RuntimeError("Synthesis executable or Metal library changed during rendering")
    marker.write_text(json.dumps({"description": DESCRIPTION, "synthesis_artifacts": artifact_pins,
                                 "force_frames": 176400, "gate_frames": 882, "records": records}, indent=2, allow_nan=False) + "\n")
    build_report(output, cases)


if __name__ == "__main__":
    main()
