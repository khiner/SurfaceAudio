#!/usr/bin/env python3
"""Reproduce Poirot 2023 collision comparisons against pinned author stimuli."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import urllib.request

import numpy as np
from scipy.signal import stft, find_peaks
from AnalyzeRenders import read_wave, metrics, texture_metrics

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "references/poirot"


def fetch_sources(offline):
    provenance = json.loads((ROOT / "repros/poirot/sources.json").read_text())
    for item in provenance["files"]:
        path = REFERENCES / item["path"]
        if not path.exists() and not item["path"].endswith(".wav"):
            continue
        if not path.exists():
            if offline or item["url"].startswith("local:"):
                raise RuntimeError(f"Missing pinned source: {path}")
            url = item.get("archive_url", item["url"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise RuntimeError(f"Source hash changed: {url}")
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Source hash mismatch: {path}")
    return provenance


def comparison(reference, output):
    rate, author = read_wave(reference)
    ours_rate, ours = read_wave(output)
    if rate != ours_rate:
        raise RuntimeError("Unmatched sample rates")
    n = min(len(author), len(ours))
    author, ours = author[:n, 0], ours[:n, 0]
    pre = slice(int(.02 * rate), int(.48 * rate))
    # Estimate comparison gain only from precontact samples.
    gain = float(np.dot(author[pre], ours[pre]) / np.dot(ours[pre], ours[pre]))
    scaled = ours * gain
    result = {"preinteraction_gain": gain, "whole_record": {"author": metrics(rate, author[:, None]), "ours": metrics(rate, ours[:, None])}, "windows": {}}
    for name, begin, end in [("preinteraction", .02, .48), ("interaction", .5, .8), ("decay", .8, 2.)]:
        a, b = author[int(begin * rate):int(end * rate)], scaled[int(begin * rate):int(end * rate)]
        kwargs = dict(fs=rate, window=("kaiser", 5), nperseg=2048, noverlap=1800, boundary=None)
        ma, mb = np.abs(stft(a, **kwargs)[2]), np.abs(stft(b, **kwargs)[2])
        result["windows"][name] = {"waveform_relative_error": float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30)), "spectrogram_relative_error": float(np.linalg.norm(ma - mb) / max(np.linalg.norm(ma), 1e-30)), "author_texture": texture_metrics(rate, a[:, None]), "ours_texture": texture_metrics(rate, b[:, None])}
    # Measure first-mode sidebands separately from aggregate spectral error.
    result["first_mode_peaks_hz"] = []
    for begin in [.51, .6, .8]:
        peaks = []
        for waveform in [author, scaled]:
            z = waveform[int(begin * rate):int((begin + .08) * rate)]
            spectrum = np.abs(np.fft.rfft(z * np.hanning(len(z)), n=262144))
            indices, _ = find_peaks(spectrum, distance=50)
            indices = [i for i in indices if 240 < i * rate / 262144 < 570]
            indices = sorted(indices, key=lambda i: spectrum[i], reverse=True)[:2]
            peaks.append([float(i * rate / 262144) for i in indices])
        result["first_mode_peaks_hz"].append({"start": begin, "author": peaks[0], "ours": peaks[1]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/poirotReproduce")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reproduction/poirot")
    args = parser.parse_args()
    provenance = fetch_sources(args.offline)
    calibration = json.loads((ROOT / "repros/poirot/calibration.json").read_text())
    base_modes = np.array(calibration["modes"])
    _, base = read_wave(REFERENCES / "sounds" / calibration["source"])
    args.output.mkdir(parents=True, exist_ok=True)
    records, cases = [], []
    source_items = {Path(i["path"]).name: i for i in provenance["files"]}
    selected = [(2, 1, 1), (2, 1, 5), (2, 1, 9), (2, 2, 1), (2, 3, 1), (3, 1, 1), (3, 2, 5)]
    with tempfile.TemporaryDirectory(prefix="poirot-") as scratch:
        config = Path(scratch) / "model.txt"
        for model, x, y in selected:
            name = f"M{model}_x{x}_y{y}.wav"
            reference = REFERENCES / "sounds" / name
            rate, author = read_wave(reference)
            pre = slice(0, int(.48 * rate))
            gain = float(np.dot(author[pre, 0], base[pre, 0]) / np.dot(base[pre, 0], base[pre, 0]))
            modes = base_modes.copy()
            modes[:, 2] *= gain
            position = [.5, 1 / 3, 5 / 12][x - 1]
            height_ratio = {1: .005, 5: .58, 9: .995}[y]
            height = height_ratio * modes[0, 2] * np.exp(-modes[0, 1] * .5) * np.sin(np.pi * position)
            scale = calibration["collision_calibration"]["normalized_power_scale"] / modes[0, 2] ** 2
            threshold, slope = (340, .0006) if model == 2 else (4000, .0001)
            variants = [("source_calibrated", 1)] + ([("literal_eq16", 0)] if (model, x, y) == (2, 1, 1) else [])
            for variant, shape in variants:
                output = args.output / f"{name[:-4]}_{variant}.wav"
                with config.open("w") as stream:
                    stream.write(f".5 .00125 {height:.17g} {position:.17g} {threshold} {slope} {scale:.17g} 1 {shape} 1 {len(modes)}\n")
                    np.savetxt(stream, modes, fmt="%.17g")
                subprocess.run([str(args.binary.resolve()), "signal", str(config), str(output), f"{len(author)/rate:.17g}"], check=True)
                detail = comparison(reference, output)
                detail.update({"name": name, "variant": variant, "height": height, "power_scale": scale, "training_case": name in calibration["collision_calibration"]["training"]})
                records.append(detail)
                notes = "Author stimulus versus signal reconstruction. Initial modes fitted before contact. Recipient weighting enforces paper power conservation. Author-measured phase reset at t0 is reproduced. "
                notes += "Unnormalized modal shape realizes author sidebands; scalar power calibration uses M2_x1_y1 and M2_x1_y5 only." if shape else "Literal normalized Eq.16 suppresses the near-fundamental sideband shift heard in the author stimulus."
                cases.append({"title": f"Poirot {name[:-4]} / {variant}", "reference": str(reference), "synthesis": str(output.resolve()), "notes": notes, "source_url": source_items[name]["archive_url"]})
        for x, y in [(1, 1), (1, 5), (1, 9)]:
            name = f"M1_x{x}_y{y}.wav"
            reference = REFERENCES / "sounds" / name
            params = calibration["physical_preinteraction_fit"]
            height = {1: .0065, 5: .49, 9: .985}[y]
            config.write_text(f"{params['WaveSpeed']} {params['Stiffness']} {params['Loss0']} {params['Loss1']} .5 {height} .3\n")
            output = args.output / f"{name[:-4]}_physical_reference.wav"
            run = subprocess.run([str(args.binary.resolve()), "physical", str(config), str(output), "5"], check=True, capture_output=True, text=True)
            detail = comparison(reference, output)
            detail.update({"name": name, "variant": "physical_reference", "diagnostics": run.stdout.strip()})
            records.append(detail)
            cases.append({"title": f"Poirot {name[:-4]} / physical reference", "reference": str(reference), "synthesis": str(output.resolve()), "notes": "Independent energy-based finite-difference stiff-string reference. Dispersion/loss use physical author precontact modal estimates. Readout x/L=.3 is an assumption; original FD code and observation point are unpublished. Phase and waveform equivalence remain unestablished.", "source_url": source_items[name]["archive_url"]})
    (args.output / "cases.json").write_text(json.dumps({"cases": cases}, indent=2) + "\n")
    (args.output / "manifest.json").write_text(json.dumps({"method": "Poirot 2023", "calibration": calibration, "source_status": "Archived author stimuli, pinned by SHA256. No author implementation located.", "acceptance": "Equation/CPU-GPU tests pass separately. Source errors are diagnostics, not perceptual equivalence certification.", "comparisons": records}, indent=2) + "\n")
    print(args.output / "cases.json")


if __name__ == "__main__":
    main()
