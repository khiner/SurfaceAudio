#!/usr/bin/env python3
"""Validate Lee rolling analysis/synthesis on separately identified FoleyAutomatic author media."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

import numpy as np
from scipy import linalg, signal
from scipy.io import wavfile
from AnalyzeRenders import read_wave, metrics, texture_metrics, reconstruction_metrics

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "references/lee"


def relative(a, b):
    if a.shape != b.shape or not np.isfinite(b).all():
        raise RuntimeError("Nonfinite or mismatched reference")
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))


def sources(offline, controls):
    manifest = json.loads((ROOT / "repros/lee/sources.json").read_text())
    video = REFERENCES / "foleyautomatic.mpeg"
    for item in manifest["files"]:
        path = REFERENCES / item["path"]
        if not path.exists() and item["kind"] == "author_video":
            if offline:
                raise RuntimeError(f"Missing pinned source: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(item["url"], timeout=60) as response:
                path.write_bytes(response.read())
        if not path.exists() and item["kind"] == "decoded_excerpt":
            case = next(x for x in controls["cases"] if x["name"] == path.stem)
            decoded = REFERENCES / "decoded.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-ac", "1", "-c:a", "pcm_f32le", str(decoded)], check=True)
            rate, data = read_wave(decoded)
            wavfile.write(path, rate, data[int(case["start_seconds"] * rate):int(case["end_seconds"] * rate), 0].astype(np.float32))
            decoded.unlink()
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Pinned source changed: {path}")
    return manifest


def notch_filter(data, notches, inverse=False):
    for omega, bandwidth in notches:
        radius = np.exp(-bandwidth / 2)
        zero, pole = (.95 * radius, radius) if inverse else (radius, .95 * radius)
        data = signal.lfilter([1, -2 * zero * np.cos(omega), zero * zero], [1, -2 * pole * np.cos(omega), pole * pole], data)
    return data


def independent_checks(directory, controls):
    parameters = json.loads((directory / "parameters.json").read_text())
    filters = [np.array(x) for x in json.loads((directory / "filters.json").read_text())]
    rate, input_data = read_wave(directory / "input_mono.wav")
    _, native = read_wave(directory / "native.wav")
    data = input_data[:, 0]
    tail = int(np.ceil(rate * controls["tail_seconds"]))
    reference = np.zeros(len(native))
    maximum_lpc_error = 0
    maximum_float_impulse_error = 0
    for contact in parameters["contacts"]:
        first, frames = contact["frame"], contact["frames"]
        segment = data[first:first + frames]
        for b, (band, decimation) in enumerate(zip(contact["bands"], [8, 8, 4, 2])):
            analyzed = signal.upfirdn(filters[b], segment, down=decimation)
            whitened = notch_filter(analyzed, band["notches"], True)
            order = len(band["denominator"]) - 1
            correlation = np.array([np.dot(whitened[:len(whitened) - k], whitened[k:]) for k in range(order + 1)])
            if correlation[0] > 1e-25:
                solved = linalg.solve_toeplitz(correlation[:order], -correlation[1:])
                maximum_lpc_error = max(maximum_lpc_error, relative(np.r_[1., solved], np.array(band["denominator"])))
            length = (frames + tail + len(filters[b])) // decimation + 1
            impulse = np.zeros(length)
            impulse[0] = band["gain"]
            precise = signal.lfilter([1], band["denominator"], impulse)
            limited = signal.lfilter(np.array([1], dtype=np.float32), np.array(band["denominator"], dtype=np.float32), impulse.astype(np.float32))
            maximum_float_impulse_error = max(maximum_float_impulse_error, relative(precise, limited))
            subband = notch_filter(precise, band["notches"])
            rendered = signal.upfirdn(filters[b] * (-1 if b else 1), subband, up=decimation)
            delay = (len(filters[b]) - 1) // 2
            reference[first:first + frames + tail] += rendered[delay:delay + frames + tail]
    waveform_error = relative(reference, native[:, 0])
    if waveform_error > 2e-4 or maximum_lpc_error > 2e-5:
        raise RuntimeError(f"Independent Lee reference failed: waveform={waveform_error}, LPC={maximum_lpc_error}")
    wavfile.write(directory / "independent.wav", rate, reference.astype(np.float32))
    high = -signal.firwin(controls["highpass_taps"], controls["highpass_hz"], fs=rate, window="hamming")
    high[(len(high) - 1) // 2] += 1
    filtered = signal.convolve(data, high, mode="same")
    prefix = np.r_[0, np.cumsum(filtered * filtered)]
    n = np.arange(len(data))
    width = controls["envelope_frames"]
    envelope = (prefix[np.minimum(len(data), n + (width + 1) // 2)] - prefix[np.maximum(0, n - width // 2)]) / width
    first = len(high) // 2 + width // 2
    end = len(data) - len(high) // 2 - (width + 1) // 2 + 1
    envelope[:first] = 0
    envelope[max(first, end):] = 0
    above = envelope > controls["threshold"] * envelope.max()
    onsets = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    onsets = onsets[(onsets > first) & (onsets < end)]
    actual_onsets = np.array([x["frame"] for x in parameters["contacts"]])
    if not np.array_equal(onsets, actual_onsets):
        raise RuntimeError("Independent Eq1-5 onset times differ")
    bank_input = np.random.default_rng(2010).normal(size=4096)
    bank_output = np.zeros_like(bank_input)
    for b, d in enumerate([8, 8, 4, 2]):
        split = signal.upfirdn(filters[b], bank_input, down=d)
        merged = signal.upfirdn(filters[b] * (-1 if b else 1), split, up=d)
        bank_output += merged[len(filters[b]) - 1:len(filters[b]) - 1 + len(bank_input)]
    bank_error = relative(bank_input, bank_output)
    if bank_error > .001:
        raise RuntimeError("Four-band QMF reconstruction error exceeds declared approximation")
    return {"independent_waveform_relative_error": waveform_error, "independent_lpc_relative_error": maximum_lpc_error,
            "onsets_exact": True, "qmf_complete_record_relative_error": bank_error,
            "maximum_float32_impulse_relative_error": maximum_float_impulse_error,
            "input_metrics": metrics(rate, input_data), "native_metrics": metrics(rate, native),
            "input_texture": texture_metrics(rate, input_data), "native_texture": texture_metrics(rate, native)}


def record_comparison(rate, source, native):
    size = max(len(source), len(native))
    a, b = np.pad(source, (0, size - len(source))), np.pad(native, (0, size - len(native)))
    window = max(1, round(rate * .01))
    count = size // window
    ea = np.sqrt(np.mean(a[:count * window].reshape(count, window) ** 2, axis=1))
    eb = np.sqrt(np.mean(b[:count * window].reshape(count, window) ** 2, axis=1))
    frequencies, pa = signal.welch(a, rate, nperseg=4096)
    _, pb = signal.welch(b, rate, nperseg=4096)
    audible = (frequencies >= 20) & (frequencies < min(20000, rate / 2))
    pa, pb = pa[audible], pb[audible]
    pa, pb = pa / max(pa.sum(), 1e-30), pb / max(pb.sum(), 1e-30)
    difference = 10 * np.log10(np.maximum(pa, 1e-10)) - 10 * np.log10(np.maximum(pb, 1e-10))
    return {"rms_delta_db": float(20 * np.log10(max(np.linalg.norm(b), 1e-30) / max(np.linalg.norm(a), 1e-30))),
            "envelope_correlation_10ms": float(np.corrcoef(ea, eb)[0, 1]),
            "energy_weighted_spectral_distance_db": float(np.sqrt(np.sum(pa * difference ** 2))),
            "unweighted_spectral_distance_db": float(np.sqrt(np.mean(difference ** 2)))}


def parameter_comparison(directory):
    cpu = json.loads((directory / "parameters.json").read_text())["contacts"]
    gpu = json.loads((directory / "gpu_parameters.json").read_text())["contacts"]
    error, gain_error, notch_error, changed = 0., 0., 0., 0
    for a, b in zip(cpu, gpu):
        for x, y in zip(a["bands"], b["bands"]):
            error = max(error, relative(np.array(x["denominator"]), np.array(y["denominator"])))
            gain_error = max(gain_error, abs(x["gain"] - y["gain"]) / max(abs(x["gain"]), 1e-30))
            if len(x["notches"]) != len(y["notches"]):
                changed += 1
            elif x["notches"]:
                notch_error = max(notch_error, float(np.max(np.abs(np.array(x["notches"]) - np.array(y["notches"])))))
    return {"maximum_lpc_relative_error": error, "maximum_gain_relative_error": gain_error,
            "maximum_notch_absolute_error_radians": notch_error, "changed_notch_count_bands": changed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/leeReproduce")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reproduction/lee")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--keep-diagnostics", action="store_true", help="Retain intermediate audio and trajectories after successful checks")
    args = parser.parse_args()
    controls = json.loads((ROOT / "repros/lee/controls.json").read_text())
    provenance = sources(args.offline, controls)
    args.output.mkdir(parents=True, exist_ok=True)
    cases, comparisons, diagnostics = [], [], []
    for item in controls["cases"]:
        name = item["name"]
        directory = args.output / name
        case_controls = {**controls, **item}
        settings = [case_controls[key] for key in ["threshold", "envelope_frames", "tail_seconds", "maximum_notches", "highpass_hz", "highpass_taps"]]
        command = [str(args.binary.resolve()), str(REFERENCES / f"{name}.wav"), str(directory), *map(str, settings)]
        subprocess.run(command, check=True)
        comparison = independent_checks(directory, case_controls)
        rate, source = read_wave(directory / "input_mono.wav")
        _, native = read_wave(directory / "native.wav")
        comparison["input_native_comparison"] = record_comparison(rate, source[:, 0], native[:, 0])
        comparison["reconstruction"] = reconstruction_metrics(rate, source[:, 0], native[:, 0])
        comparison["gpu_parameter_comparison"] = parameter_comparison(directory)
        comparison.update({"case": name, "kind": item["kind"], "timings": json.loads((directory / "timings.json").read_text())})
        ablation = directory / "without_notches"
        subprocess.run([*command[:2], str(ablation), *map(str, [*settings[:3], 0, *settings[4:]])], check=True)
        _, no_notches = read_wave(ablation / "native.wav")
        comparison["without_notches_metrics"] = metrics(rate, no_notches)
        comparison["without_notches_input_comparison"] = record_comparison(rate, source[:, 0], no_notches[:, 0])
        shutil.copyfile(REFERENCES / f"{name}.wav", directory / "author_input.wav")
        comparisons.append(comparison)
        limitation = (f" Detected {comparison['timings']['contacts']} contacts; input/output envelope correlation "
                      f"{comparison['input_native_comparison']['envelope_correlation_10ms']:.3f}. "
                      f"Weak-output intervals contain {comparison['reconstruction']['reference_energy_missing_fraction']:.1%} of input energy. "
                      "These measurements do not establish perceptual equivalence.")
        cases.append({"title": f"Lee 2010 cross-paper input / {name}", "reference": str((directory / "author_input.wav").resolve()),
                      "synthesis": str((directory / "native.wav").resolve()), "source_url": provenance["files"][0]["url"],
                      "reference_label": "Shared input recording: FoleyAutomatic 2001",
                      "synthesis_label": "Lee contact-filter reconstruction", "notes": item["kind"] + " Lee method reconstruction, not original Lee audio. "
                      "The named FoleyAutomatic input is shared with Lagrange; it is not an original Lee synthesis. "
                      "Raw source scale, published LPC orders, explicit QMF/onset/notch/truncation choices. No fitted gain." + limitation})
        diagnostics.append({"title": f"Lee diagnostic: two native notch settings / {name}", "reference": str((ablation / "native.wav").resolve()),
                      "synthesis": str((directory / "native.wav").resolve()), "reference_label": "Without notch estimation",
                      "notes": "Same input/onsets/orders, independently refitted LPC with versus without notch whitening. "
                      "Neither is an author Lee synthesis." + limitation})
    manifest = {"method": "Lee, Depalle and Scavone 2010 reconstruction", "sources": provenance, "controls": controls,
                "comparisons": comparisons,
                "gaps": ["Original Lee source code and author example WAVs were not recovered, so published audio similarity is unverified.",
                         "Paper does not provide QMF taps, onset threshold/averaging length, peak settings, or impulse truncation.",
                         "64-tap near-PR QMF and explicit settings reconstruct the method; they are not recovered author parameters.",
                         "Quadratic fitting uses reciprocal-spectrum dB peaks; the paper leaves its fitting domain and neighborhood unspecified.",
                         "No time-warped or gain-fitted waveform is used as the reference, and no listening-study equivalence is claimed."]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "cases.json").write_text(json.dumps({"cases": cases}, indent=2) + "\n")
    (args.output / "diagnostics.json").write_text(json.dumps({"title": "Lee notch diagnostics", "cases": diagnostics}, indent=2) + "\n")
    if not args.keep_diagnostics:
        for item in controls["cases"]:
            directory = args.output / item["name"]
            for name in ("input_mono.wav", "gpu.wav", "gpu_pipeline.wav", "independent.wav", "half_speed.wav", "reverse_trajectory.wav"):
                (directory / name).unlink(missing_ok=True)
            shutil.rmtree(directory / "without_notches")
        (args.output / "diagnostics.json").unlink(missing_ok=True)
    print(args.output / "manifest.json")


if __name__ == "__main__":
    main()
