#!/usr/bin/env python3
"""Reproduce the archived Matusiak bowed-string computation and compare actual author outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from AnalyzeRenders import texture_metrics

ROOT = Path(__file__).resolve().parents[1]
REVISION = "0e2d39d0d62460370f1de3945a4f1a2c984b038c"
SOURCE = "https://github.com/wasilakis/the_bowed_string"
AUDIO = "https://www.mdw.ac.at/upload/MDWeb/iwk/mp3/FastSautille.mp3"


def generate_author_reference(source, destination):
    """Run original author arithmetic and an explicitly identified diagonal correction."""
    from scipy.io import loadmat
    text = source.read_text()
    body = text[text.index("str_par ="):text.index("%% plotting")]
    interpolation = text[text.index("function [Iu,Ju]"):]
    original_term = " + 1/M * 1/O3;"
    if body.count(original_term) != 1:
        raise ValueError("Author coupling expression changed")
    with tempfile.TemporaryDirectory(prefix="matusiak-octave-") as temporary:
        directory = Path(temporary)
        results = []
        for name, corrected in [("original", False), ("diagonal", True)]:
            computation = body.replace(original_term, " + 1/M * 1/O3 * eye(M);") if corrected else body
            function = "author_" + name
            (directory / (function + ".m")).write_text("function " + function + "()\n" + computation + '\nsave("-mat7-binary", "result.mat", "F_bridge", "F_fr", "H_t", "v");\nend\n' + interpolation)
            subprocess.run(["octave-cli", "--quiet", "--no-gui", "--eval", function], cwd=directory, check=True)
            results.append(loadmat(directory / "result.mat"))
        original, diagonal = results
        data = np.column_stack([original["F_bridge"].ravel(), diagonal["F_bridge"].ravel(), original["F_fr"].mean(axis=0), diagonal["F_fr"].mean(axis=0), original["H_t"].ravel(), diagonal["H_t"].ravel(), diagonal["v"][2]])
        data.astype("<f8").tofile(destination)


def compare(reference, synthesis):
    difference = synthesis - reference
    _, _, a = stft(reference, 44100, nperseg=1024, noverlap=768)
    _, _, b = stft(synthesis, 44100, nperseg=1024, noverlap=768)
    floor = max(np.abs(a).max(), np.abs(b).max()) * 1e-5
    selected = np.maximum(np.abs(a), np.abs(b)) > floor
    db = 20 * np.log10(np.maximum(np.abs(a), floor) / np.maximum(np.abs(b), floor))
    window = np.ones(441) / 441
    ar = np.sqrt(np.convolve(reference**2, window, "valid"))
    br = np.sqrt(np.convolve(synthesis**2, window, "valid"))
    return {"relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(reference)),
            "max_absolute_force_n": float(np.abs(difference).max()),
            "stft_active_bin_rmse_db": float(np.sqrt(np.mean(db[selected]**2))),
            "rms_envelope_relative_l2": float(np.linalg.norm(ar-br) / np.linalg.norm(ar)),
            "stft": {"fft": 1024, "hop": 256, "floor_db": -100}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/matusiakReproduce")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reproduction/matusiak")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--author-reference", action="store_true", help="Regenerate both author variants with GNU Octave")
    parser.add_argument("--gpu-voices", type=int, default=128, help="Full distributed GPU comparison batch; zero skips GPU")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference_dir = ROOT / "references/matusiak"
    source_dir = reference_dir / "the_bowed_string"
    if not args.offline:
        if not source_dir.exists():
            subprocess.run(["git", "clone", "--quiet", SOURCE, str(source_dir)], check=True)
            subprocess.run(["git", "-C", str(source_dir), "checkout", "--quiet", REVISION], check=True)
        audio_path = reference_dir / "FastSautille.mp3"
        if not audio_path.exists():
            urllib.request.urlretrieve(AUDIO, audio_path)
    fixture = ROOT / "repros/matusiak/Reference.f64"
    provenance = json.loads((fixture.with_suffix(".json")).read_text())
    if hashlib.sha256(fixture.read_bytes()).hexdigest() != provenance["sha256"]:
        raise RuntimeError("Author fixture hash mismatch")
    if args.author_reference:
        if hashlib.sha256((source_dir / "bowed_string_with_comments.m").read_bytes()).hexdigest() != provenance["source_sha256"]:
            raise RuntimeError("Archived author source hash mismatch")
        regenerated = output / "author_regenerated.f64"
        generate_author_reference(source_dir / "bowed_string_with_comments.m", regenerated)
        a = np.fromfile(fixture, "<f8")
        b = np.fromfile(regenerated, "<f8")
        if not np.allclose(a, b, rtol=1e-8, atol=1e-10):
            raise RuntimeError("Regenerated actual author output differs from pinned fixture")
        regenerated.unlink()
    subprocess.run([str(args.binary.resolve()), str(output), ".5", "2.3433", str(args.gpu_voices)], check=True)
    author = np.fromfile(fixture, "<f8").reshape(22050, 7)
    native = np.fromfile(output / "trace.f64", "<f8").reshape(22050, 8)
    for name, column in [("author_original_bridge.wav", 0), ("author_diagonal_bridge.wav", 1)]:
        wavfile.write(output / name, 44100, author[:, column].astype(np.float32))
    comparisons = {"original_author_vs_native": compare(author[:, 0], native[:, 0]),
                   "diagonal_author_vs_native": compare(author[:, 1], native[:, 0])}
    cases = [
        {"title": "Matusiak 2025: archived author computation", "reference": str(output / "author_original_bridge.wav"),
         "synthesis": str(output / "bridge_force.wav"), "source_url": SOURCE,
         "notes": "Full 0.5 s raw bridge force at 44100 Hz. All author defaults and interpolation indexing retained. Ours applies Eq. 59's diagonal bow-hair compliance; the archived code adds its scalar to every coupling entry. No cello body filtering or fitted parameters."},
        {"title": "Matusiak 2025: isolated equation correction oracle", "reference": str(output / "author_diagonal_bridge.wav"),
         "reference_label": "Actual author Octave with diagonal correction", "synthesis": str(output / "bridge_force.wav"), "source_url": SOURCE,
         "notes": "Author MATLAB computation executed in GNU Octave, changing only the final IJ_mat scalar term to scalar times eye(M). Native C++ follows the published diagonal coupling. Raw force units, identical time origin and duration."}]
    if args.gpu_voices:
        _, gpu = wavfile.read(output / "gpu_bridge_force.wav")
        comparisons["gpu_fp32_vs_author_fp64"] = compare(author[:, 1], gpu.astype(np.float64))
        cases.append({"title": "Matusiak 2025: complete distributed model on Metal", "reference": str(output / "bridge_force.wav"),
                      "reference_label": "Our certified FP64 CPU", "synthesis": str(output / "gpu_bridge_force.wav"), "synthesis_label": "Our FP32 Metal",
                      "source_url": SOURCE, "notes": "Complete FDTD transverse/torsional string, five contact points, compliant bow and implicit elasto-plastic solve. GPU comparison uses float-rounded controls. Batch timing includes allocation, upload, computation and readback. FP32 accuracy and solver failures are recorded separately."})
    supplementary = {"url": AUDIO, "status": "Author fast sautillé audio is reference-only: exact drive trajectories and measured cello impulse response are absent from the code archive."}
    audio_path = reference_dir / "FastSautille.mp3"
    if audio_path.exists():
        supplementary["sha256"] = hashlib.sha256(audio_path.read_bytes()).hexdigest()
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(audio_path), "-c:a", "pcm_f32le", str(output / "author_fast_sautille.wav")], check=True)
            supplementary["file"] = str(output / "author_fast_sautille.wav")
    manifest = {"method": "Matusiak et al. 2025", "source": provenance, "comparisons": comparisons,
                "native": json.loads((output / "native.json").read_text()), "supplementary": supplementary,
                "texture": {"author_original": texture_metrics(44100, author[:, 0]), "author_diagonal": texture_metrics(44100, author[:, 1]), "native": texture_metrics(44100, native[:, 0])},
                "trace_columns": ["bridge_N", "center_velocity_m_s", "center_friction_N", "post_step_energy_J", "energy_error_J", "bristle_dissipation_W", "nonlinear_residual", "Newton_updates"]}
    if args.gpu_voices:
        manifest["gpu"] = json.loads((output / "gpu.json").read_text())
        manifest["texture"]["gpu"] = texture_metrics(44100, gpu)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "cases.json").write_text(json.dumps({"description": "Actual archived author computations and a published-equation correction compared with native CPU and Metal implementations.", "cases": cases}, indent=2) + "\n")
    for temporary in ["trace.f64", "native.json", "gpu.json"]:
        (output / temporary).unlink(missing_ok=True)
    print(f"Matusiak author comparison: corrected relative L2 {comparisons['diagonal_author_vs_native']['relative_l2']:.3g}, original {comparisons['original_author_vs_native']['relative_l2']:.3g}; {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
