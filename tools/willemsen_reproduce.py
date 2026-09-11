#!/usr/bin/env python3
"""Reproduce published Willemsen figures and optionally execute the pinned MATLAB oracle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import urllib.request

COMMIT = "12898a254e69cfbd68705ab028dd8894e4f05dfe"
BASE = f"https://raw.githubusercontent.com/SilvinWillemsen/ElastoPlastic/{COMMIT}/"
ROOT = Path(__file__).resolve().parents[1]


def author_reference(output, frames, offline=False):
    import numpy as np
    source, hashes = {}, {}
    expected = json.loads((ROOT / "repros/willemsen/source.json").read_text())["author_oracle"]["sha256"]
    cache = ROOT / "references/willemsen/author" / COMMIT
    for name in ("StiffStringElastoPlastic.m", "unscaledCreateStringNR.m"):
        path = cache / name
        data = path.read_bytes() if path.exists() else b""
        if hashlib.sha256(data).hexdigest() != expected[name]:
            if offline:
                raise RuntimeError(f"Optional author oracle source is missing or changed: {path}")
            data = urllib.request.urlopen(BASE + name).read()
            if hashlib.sha256(data).hexdigest() != expected[name]:
                raise RuntimeError(f"Author source hash mismatch: {name}")
            cache.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
        source[name] = data.decode().replace("\r\n", "\n")
    script = source["StiffStringElastoPlastic.m"]
    script = script[:script.index("startPos =", script.index("    u = uNext;"))]
    start = script.index("    if drawString == true")
    end = script.index("    % Save output", start)
    script = script[:start] + script[end:]
    script = re.sub(r'bowModel\s*==\s*"([^"]+)"', r'strcmp(bowModel,"\1")', script)
    script = re.sub(r'bowModel\s*~=\s*"([^"]+)"', r'!strcmp(bowModel,"\1")', script)
    for old, new in (("Fs = 44100 * 5;", "Fs = 44100;"), ("f0 = 100.00;", "f0 = 440;"),
                     ("FnInit = 1;", "FnInit = 5;"), ("lengthSound = Fs*2;", f"lengthSound = {frames};")):
        if old not in script:
            raise RuntimeError(f"Pinned source did not contain {old}")
        script = script.replace(old, new)
    script += '\ndlmwrite("author.csv", [(0:lengthSound-1)\',out1(:),out3(:),Fsave(:),vRelSave(:),zSave(:),zDotSave(:)], "precision",17);\n'
    with tempfile.TemporaryDirectory(prefix="willemsen-author-") as temp:
        directory = Path(temp)
        (directory / "render.m").write_text(script)
        (directory / "unscaledCreateStringNR.m").write_text(source["unscaledCreateStringNR.m"])
        environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
        subprocess.run(["octave", "--no-gui", "--quiet", "render.m"], cwd=directory, env=environment, check=True)
        values = np.loadtxt(directory / "author.csv", delimiter=",")
    np.savetxt(output / "author.csv", values, delimiter=",", comments="",
               header="frame,displacement,bow_displacement,force,velocity,z,rate")
    return {"commit": COMMIT, "sha256": hashes,
            "adaptations": ["Override Fs=44100, f0=440, FnInit=5 and requested duration to match published Figure 8 controls.",
                            "Remove plotting blocks and use strcmp for Octave string comparisons.",
                            "Retain numerical update, initial state, bristle damping, point contact and boundary conventions unchanged."]}


def analyze(output, include_author=False):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.io import wavfile
    cpp = np.genfromtxt(output / "trajectory.csv", delimiter=",", names=True)
    metrics = {"source_commit": COMMIT, "comparison": "Unaligned physical samples; no gain, pitch, time or parameter fitting."}
    trajectories = [("C++ source convention", cpp)]
    if include_author:
        author = np.genfromtxt(output / "author.csv", delimiter=",", names=True)
        if len(author) != len(cpp):
            raise ValueError("Author and C++ durations differ")
        wave = author["displacement"]
        wavfile.write(output / "author_displacement_m.wav", 44100, wave.astype(np.float32))
        wavfile.write(output / "author_listen.wav", 44100, (.8*wave/max(abs(wave))).astype(np.float32))
        trajectories.append(("Pinned MATLAB in Octave", author))
        metrics["author_relative_l2"] = {}
        for name in ("displacement", "bow_displacement", "force", "velocity", "z", "rate"):
            metrics["author_relative_l2"][name] = float(np.linalg.norm(cpp[name]-author[name])/np.linalg.norm(author[name]))
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    for label, data in trajectories:
        time = (np.arange(len(data))+1)/44100
        window = (time >= .25) & (time <= .26)
        axes[0, 0].plot(time[window], data["displacement"][window], label=label)
        axes[0, 1].plot(time[window], data["bow_displacement"][window], label=label)
        window = slice(22050, 22550)
        axes[1, 0].plot(data["velocity"][window], data["force"][window], linewidth=.8, label=label)
        tail = data["displacement"][22050:]
        spectrum = abs(np.fft.rfft((tail-tail.mean()) * np.hanning(len(tail))))
        frequencies = np.fft.rfftfreq(len(tail), 1/44100)
        axes[1, 1].plot(frequencies, 20*np.log10(np.maximum(spectrum/spectrum.max(), 1e-8)), label=label)
    axes[0, 0].set(title="Figure 8: string pickup", xlabel="Time (s)", ylabel="Displacement (m)")
    axes[0, 1].set(title="Figure 8: bow pickup", xlabel="Time (s)", ylabel="Displacement (m)")
    axes[1, 0].set(title="Figure 9: hysteresis", xlabel="Relative velocity (m/s)", ylabel="Friction force (N)")
    axes[1, 1].set(title="Sustained displacement spectrum", xlabel="Frequency (Hz)", ylabel="Magnitude (dB)", xlim=(0, 5000), ylim=(-100, 0))
    pickup = np.loadtxt(ROOT / "repros/willemsen/figure8_pickup.csv", delimiter=",", skiprows=1)
    bow = np.loadtxt(ROOT / "repros/willemsen/figure8_bow.csv", delimiter=",", skiprows=1)
    loop = np.loadtxt(ROOT / "repros/willemsen/figure9.csv", delimiter=",", skiprows=1)
    for axis, data, column in ((axes[0, 0], pickup, "displacement"), (axes[0, 1], bow, "bow_displacement")):
        axis.plot(data[:, 0], data[:, 1], color="black", linestyle="--", label="Published EPS")
        raw = np.interp(data[:, 0], (np.arange(len(cpp))+1)/44100, cpp[column])
        metrics[f"figure8_{column}_relative_l2"] = float(np.linalg.norm(raw-data[:, 1])/np.linalg.norm(data[:, 1]))
    axes[1, 0].plot(loop[:, 0], loop[:, 1], color="black", linestyle="--", label="Published EPS")
    simulated = np.column_stack([cpp["velocity"][22000:23000], cpp["force"][22000:23000]])
    distance = np.min(np.sum((loop[:, None, :]-simulated[None, :, :])**2, axis=2), axis=1)
    metrics["figure9_nearest_sample_rms_mixed_units"] = float(np.sqrt(np.mean(distance)))
    metrics["figure9_published_force_range_n"] = [float(min(loop[:, 1])), float(max(loop[:, 1]))]
    metrics["figure9_cpp_force_range_n"] = [float(min(cpp["force"][22050:22550])), float(max(cpp["force"][22050:22550]))]
    for axis in axes.flat:
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=170)
    bounds = {"figure8_displacement_relative_l2": 3e-5, "figure8_bow_displacement_relative_l2": 3e-5,
              "figure9_nearest_sample_rms_mixed_units": 1e-6}
    if any(not np.isfinite(metrics[key]) or metrics[key] > bound for key, bound in bounds.items()):
        raise RuntimeError("Published figure reproduction exceeded EPS coordinate precision")
    if include_author and any(not np.isfinite(value) or value > 1e-9 for value in metrics["author_relative_l2"].values()):
        raise RuntimeError("Pinned author trajectory comparison failed")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/willemsenReproduce")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reproduction/willemsen")
    parser.add_argument("--frames", type=int, default=44100)
    parser.add_argument("--offline", action="store_true", help="Use checked-in fixtures and only cached optional author sources")
    parser.add_argument("--author-oracle", action="store_true", help="Download pinned MATLAB sources and run them in Octave")
    parser.add_argument("--keep-diagnostics", action="store_true", help="Retain intermediate audio and trajectories after successful checks")
    args = parser.parse_args()
    if args.frames < 23000:
        parser.error("Figure comparison requires at least 23000 frames")
    fixture_manifest = json.loads((ROOT / "repros/willemsen/source.json").read_text())["fixtures"]
    for name, metadata in fixture_manifest.items():
        if hashlib.sha256((ROOT / "repros/willemsen" / name).read_bytes()).hexdigest() != metadata["sha256"]:
            raise RuntimeError(f"Published fixture hash mismatch: {name}")
    args.output.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(args.binary.resolve()), str(args.output.resolve()), str(args.frames)], check=True)
    if args.author_oracle:
        provenance = author_reference(args.output, args.frames, args.offline)
        (args.output / "author_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    (args.output / "reference.json").write_text(json.dumps(analyze(args.output, args.author_oracle), indent=2)+"\n")
    cases = []
    paper_url = "https://dafx.de/paper-archive/2019/DAFx2019_paper_18.pdf"
    if args.author_oracle:
        cases.append({"title": "Willemsen Figure 8: original author source and C++ reproduction", "group": "Willemsen 2019",
                      "reference": str((args.output / "author_listen.wav").resolve()),
                      "synthesis": str((args.output / "figure_listen.wav").resolve()),
                      "reference_label": "Pinned original MATLAB executed in Octave", "synthesis_label": "C++ source-convention reproduction",
                      "source_url": paper_url,
                      "notes": "Identical 440 Hz nominal string, 5 N force, 0.1 m/s speed, 0.25 bow position and zero noise. "
                               "Commit 12898a254e69cfbd68705ab028dd8894e4f05dfe reproduces the published EPS figures. "
                               "Both files use the same peak normalization; comparisons use physical samples without alignment or fitting."})
    diagnostics = [{"title": "Willemsen diagnostic: two native numerical conventions", "group": "Willemsen 2019",
                  "reference": str((args.output / "paper_listen.wav").resolve()),
                  "synthesis": str((args.output / "figure_listen.wav").resolve()),
                  "reference_label": "Native literal equations and Table 1", "synthesis_label": "Native original figure-source conventions",
                  "source_url": paper_url,
                  "notes": "Both are native C++ renders with the printed Figure 8 controls. "
                           "The author figure code uses bristle damping 1 instead of 0.1, a shorter clamped grid, point contact, "
                           "different initial state and separately scaled contact losses. Only the source convention matches the published figure. "
                           "These are dry displacement signals without a body or radiation model."}]
    (args.output / "diagnostics.json").write_text(json.dumps({"title": "Willemsen numerical conventions", "cases": diagnostics}, indent=2)+"\n")
    (args.output / "cases.json").write_text(json.dumps({"cases": cases}, indent=2)+"\n")
    if not args.keep_diagnostics:
        for name in ("author.csv", "trajectory.csv"):
            (args.output / name).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
