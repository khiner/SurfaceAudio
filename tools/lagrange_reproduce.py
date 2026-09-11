#!/usr/bin/env python3
"""Validate Lagrange equations and analyze pinned FoleyAutomatic author audio."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request

import numpy as np
from scipy import signal
from scipy.io import wavfile
from AnalyzeRenders import metrics, read_wave, texture_metrics, reconstruction_metrics, periodicity_comparison, band_power_fraction, spectral_motion_comparison

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "references/lagrange"
MODAL_GAINS = ("fitted", "unit", "magnitude", "causal_magnitude", "incremental_phase")
TRIGGER_FITS = ("envelope", "waveform", "split_envelope", "joint_waveform")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources(offline):
    manifest = json.loads((ROOT / "repros/lagrange/sources.json").read_text())
    for item in manifest["files"]:
        path = REFERENCES / item["path"]
        if not path.exists():
            if offline:
                raise RuntimeError(f"Missing reference input: {path}")
            with urllib.request.urlopen(item["url"], timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise RuntimeError(f"Author source changed: {item['url']}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        if digest(path) != item["sha256"]:
            raise RuntimeError(f"Reference hash mismatch: {path}")
    missing = [item for item in manifest["clips"] if not (REFERENCES / f"{item['name']}.wav").exists()]
    if missing:
        if offline:
            raise RuntimeError("Missing decoded author audio")
        with tempfile.TemporaryDirectory(prefix="lagrange-decode-") as scratch:
            decoded = Path(scratch) / "decoded.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(REFERENCES / "foleyautomatic.mpeg"),
                            "-vn", "-c:a", "pcm_f32le", str(decoded)], check=True)
            rate, audio = wavfile.read(decoded)
            for item in missing:
                wavfile.write(REFERENCES / f"{item['name']}.wav", rate,
                              audio[item["source_begin_sample"]:item["source_end_sample"]])
    for item in manifest["clips"]:
        if digest(REFERENCES / f"{item['name']}.wav") != item["sha256"]:
            raise RuntimeError(f"Decoded reference hash mismatch: {item['name']}")
    return manifest


def relative(expected, actual):
    if expected.shape != actual.shape or not np.isfinite(actual).all():
        raise RuntimeError("Nonfinite or differently sized complete waveform")
    return float(np.linalg.norm(expected - actual) / max(np.linalg.norm(expected), 1e-30))


def compare(reference, synthesis):
    overlap = synthesis[:len(reference)]
    padded = np.pad(reference, (0, max(0, len(synthesis) - len(reference))))
    gain = float(np.linalg.norm(reference) / max(np.linalg.norm(overlap), 1e-30))
    a = abs(signal.stft(reference, nperseg=1024, noverlap=768)[2])
    b = abs(signal.stft(overlap, nperseg=1024, noverlap=768)[2])
    first, second = np.sqrt(np.mean(a * a, axis=0)), np.sqrt(np.mean(b * b, axis=0))
    return {"raw_complete_waveform_relative_error": relative(padded, synthesis),
            "diagnostic_rms_matching_gain": gain,
            "rms_matched_stft_magnitude_l1": float(np.sum(abs(a - gain * b)) / max(np.sum(a), 1e-30)),
            "envelope_correlation": float(np.corrcoef(first, second)[0, 1]),
            "scope": "RMS matching is a diagnostic only. Both saved WAVs retain raw amplitudes."}


def equations(directory, original, rate, case):
    modes = np.loadtxt(directory / "modes.txt").reshape(-1, 4)
    _, impact = read_wave(directory / "impact.wav")
    _, excitation = read_wave(directory / "excitation.wav")
    impact, excitation = impact[:, 0], excitation[:, 0]
    begin, end = case["impact_interval_samples"]
    expected_impact = excitation[begin:end].copy()
    if case.get("center_impact", False):
        expected_impact -= expected_impact.mean()
    impact_error = relative(expected_impact, impact)
    if impact_error > 1e-6:
        raise RuntimeError(f"Impact extraction differs from the declared preprocessing: {impact_error}")
    def modal_filter(excitation, frames):
        padded = np.pad(excitation[:frames], (0, max(0, frames - len(excitation))))
        result = np.zeros(frames)
        for frequency, damping, real, imaginary in modes:
            pole = np.exp((-damping + 2j * np.pi * frequency) / rate)
            result += signal.lfilter([real + 1j * imaginary], [1, -pole], padded).real
        return result
    size = 1 << (2 * len(original) - 1).bit_length()
    angle = 2 * np.pi * np.arange(size) / size
    inverse_z = np.exp(-1j * angle)
    transfer = np.zeros(size, np.complex128)
    for frequency, damping, real, imaginary in modes:
        pole = np.exp((-damping + 2j * np.pi * frequency) / rate)
        gain = real + 1j * imaginary
        transfer += .5 * (gain / (1 - pole * inverse_z) + gain.conjugate() / (1 - pole.conjugate() * inverse_z))
    source = np.fft.ifft(np.fft.fft(original, size) * transfer.conjugate() /
                        np.maximum(abs(transfer) ** 2, 1e-16)).real[:len(original)]
    inverse_error = relative(source, excitation)
    _, envelope = read_wave(directory / "envelope.wav")
    _, shape = read_wave(directory / "impact_envelope.wav")
    _, deconvolved = read_wave(directory / "deconvolved.wav")
    envelope, shape = envelope[:, 0], shape[:, 0]
    if case.get("trigger_fit") == "split_envelope":
        peak = int(np.argmax(shape))
        estimated = signal.lfilter([1], shape[peak:], envelope)
        estimated = signal.lfilter([1], shape[:peak][::-1], estimated[::-1])[::-1]
    else:
        envelope_size = 1 << (len(envelope) + len(shape) - 1).bit_length()
        window = np.fft.fft(shape, envelope_size)
        floor = .03 * shape.sum()
        estimated = np.fft.ifft(np.fft.fft(envelope, envelope_size) * window.conjugate() /
                                (abs(window) ** 2 + floor * floor)).real[:len(envelope)]
    envelope_inverse_error = relative(estimated, deconvolved[:, 0])
    if envelope_inverse_error > 3e-4:
        raise RuntimeError(f"Independent envelope inverse failed: {envelope_inverse_error}")
    if case.get("trigger_fit") == "split_envelope":
        events = np.loadtxt(directory / "triggers.txt").reshape(-1, 2)
        locations = events[:, 0].astype(int) + int(np.argmax(shape)) - 1
        if len(events) and not np.allclose(events[:, 1], deconvolved[locations, 0], rtol=0, atol=0):
            raise RuntimeError("Split-envelope amplitudes or impact-origin coordinates changed")
    errors = {}
    filename, trigger_file = "synthesis.wav", "triggers.txt"
    triggers = np.loadtxt(directory / trigger_file).reshape(-1, 2)
    impulses = np.zeros(len(original))
    np.add.at(impulses, triggers[:, 0].astype(int), triggers[:, 1])
    _, actual = read_wave(directory / filename)
    expected = modal_filter(signal.fftconvolve(impulses, impact), len(actual))
    errors[filename] = relative(expected, actual[:, 0])
    _, actual_replay = read_wave(directory / "source_replay.wav")
    replay = modal_filter(excitation, len(actual_replay))
    replay_error = relative(replay, actual_replay[:, 0])
    if max(*errors.values(), replay_error) > 3e-4 or inverse_error > 3e-3:
        raise RuntimeError(f"Independent complete-record equation check failed: {errors}, {inverse_error}, {replay_error}")
    return {"impact_extraction_relative_error": impact_error, "inverse_modal_relative_error": inverse_error, "synthesis_equation_relative_errors": errors,
            "envelope_inverse_relative_error": envelope_inverse_error,
            "source_replay_equation_relative_error": replay_error,
            "source_replay_input_comparison": compare(original, actual_replay[:, 0]),
            "evaluation": "Independent NumPy Eq4-5 inverse and SciPy causal modal recurrences after full impact convolution, including every rendered tail sample."}


def waveform_fit_check(directory, original, rate, case):
    from scipy.fft import rfft, irfft
    _, source = read_wave(directory / "excitation.wav")
    _, impact = read_wave(directory / "impact.wav")
    source, impact = source[:, 0], impact[:, 0]
    triggers = np.loadtxt(directory / "triggers.txt").reshape(-1, 2)
    actual = np.zeros(len(source))
    np.add.at(actual, triggers[:, 0].astype(int), triggers[:, 1])
    observations = [(source, impact, 1.)]
    iterations = 1200
    if case.get("trigger_fit") == "joint_waveform":
        time = np.arange(len(source)) / rate
        response = np.zeros(len(source))
        for frequency, damping, real, imaginary in np.loadtxt(directory / "modes.txt").reshape(-1, 4):
            response += np.exp(-damping * time) * (real * np.cos(2 * np.pi * frequency * time) -
                                                  imaginary * np.sin(2 * np.pi * frequency * time))
        observations = [(source, impact, 1 / max(float(np.sum(source * source)), 1e-30)),
                        (original, signal.fftconvolve(impact, response), case.get("audio_weight", .01) / max(float(np.sum(original * original)), 1e-30))]
        iterations = 4800
    size = 1 << (len(source) + max(len(kernel) for _, kernel, _ in observations) - 2).bit_length()
    power, rhs = np.zeros(size // 2 + 1), np.zeros(size // 2 + 1, complex)
    for target, kernel, weight in observations:
        response = rfft(kernel, size)
        power += weight * abs(response) ** 2
        rhs += weight * response.conjugate() * rfft(target, size)
    peak = irfft(rhs, size)[:len(source)].max()
    penalty = .001 * peak
    coefficients, extrapolated = np.zeros(size), np.zeros(size)
    acceleration = 1.
    for _ in range(iterations):
        gradient = irfft(power * rfft(extrapolated) - rhs, size)
        updated = np.maximum(extrapolated - (gradient + penalty) / power.max(), 0)
        updated[len(source):] = 0
        next_acceleration = .5 * (1 + np.sqrt(1 + 4 * acceleration ** 2))
        extrapolated = updated + (acceleration - 1) / next_acceleration * (updated - coefficients)
        coefficients, acceleration = updated, next_acceleration
    error = relative(coefficients[:len(source)], actual)
    gradient = irfft(power * rfft(actual, size) - rhs, size)[:len(source)] + penalty
    kkt = float(np.max(np.where(actual > 0, abs(gradient), np.maximum(-gradient, 0))) / peak)
    actual_spectrum, cpu_spectrum = rfft(actual, size), rfft(coefficients, size)
    frequency_weights = np.full(size // 2 + 1, 2.)
    frequency_weights[[0, -1]] = 1
    output_error = float(np.sqrt(np.sum(frequency_weights * power * abs(actual_spectrum - cpu_spectrum) ** 2) /
                                 max(float(np.sum(frequency_weights * power * abs(cpu_spectrum) ** 2)), 1e-30)))
    losses = []
    for spectrum, values in [(actual_spectrum, actual), (cpu_spectrum, coefficients)]:
        loss = penalty * values.sum()
        for target, kernel, weight in observations:
            residual = irfft(rfft(kernel, size) * spectrum, size)
            residual[:len(target)] -= target
            loss += .5 * weight * np.sum(residual * residual)
        losses.append(float(loss))
    objective_error = abs(losses[0] - losses[1]) / max(losses[1], 1e-30)
    coefficient_tolerance = .01 if len(observations) > 1 else .001
    if not np.isfinite([error, kkt, output_error, objective_error]).all() or error > coefficient_tolerance or kkt > .001 or output_error > .001 or objective_error > 2e-5:
        raise RuntimeError(f"Independent fit failed: coefficients {error}, KKT {kkt}, weighted output {output_error}, objective {objective_error}")
    return {"cpu_fp64_coefficient_relative_error": error, "independent_relative_kkt": kkt,
            "weighted_reconstruction_relative_error": output_error, "objective_relative_difference": objective_error,
            "gpu_objective": losses[0], "cpu_objective": losses[1],
            "iterations": iterations, "observations": len(observations),
            "scope": "Independent SciPy FP64 weighted full-convolution fit, lambda=0.001*max(sum(w*I.T*target)), nonnegative FISTA."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/lagrangeReproduce")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--case", action="append", help="Render one named case; repeat to select multiple cases")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reproduction/lagrange")
    parser.add_argument("--modal-gains", choices=MODAL_GAINS)
    parser.add_argument("--trigger-fit", choices=TRIGGER_FITS)
    parser.add_argument("--audio-weight", type=float)
    parser.add_argument("--center-impact", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--audit", action="store_true", help="Write all quality failures to the manifest and exit unsuccessfully")
    parser.add_argument("--keep-diagnostics", action="store_true", help="Retain intermediate audio and trajectories after successful checks")
    args = parser.parse_args()
    if args.audio_weight is not None and not (np.isfinite(args.audio_weight) and args.audio_weight > 0):
        parser.error("--audio-weight must be finite and positive")
    provenance = sources(args.offline)
    controls = json.loads((ROOT / "repros/lagrange/cases.json").read_text())
    if args.case:
        unknown = set(args.case) - {case["name"] for case in controls["cases"]}
        if unknown:
            parser.error(f"Unknown cases: {', '.join(sorted(unknown))}")
        controls["cases"] = [case for case in controls["cases"] if case["name"] in args.case]
    for case in controls["cases"]:
        for key, value in [("modal_gains", args.modal_gains), ("trigger_fit", args.trigger_fit),
                           ("center_impact", args.center_impact), ("audio_weight", args.audio_weight)]:
            if value is not None:
                case[key] = value
    quality_failures = []

    def quality_failure(message):
        if not args.audit:
            raise RuntimeError(message)
        quality_failures.append(message)

    args.output.mkdir(parents=True, exist_ok=True)
    comparisons, cases, diagnostics = [], [], []
    with tempfile.TemporaryDirectory(prefix="lagrange-") as scratch:
        for case in controls["cases"]:
            output = args.output / case["name"]
            output.mkdir(parents=True, exist_ok=True)
            settings = Path(scratch) / "settings.txt"
            settings.write_text(" ".join(map(str, case["modal_interval_samples"] + case["impact_interval_samples"] +
                                              [MODAL_GAINS.index(case["modal_gains"]),
                                               TRIGGER_FITS.index(case.get("trigger_fit", "envelope")), int(case.get("center_impact", False)), case.get("audio_weight", .01)])) + "\n")
            original = REFERENCES / f"{case['name']}.wav"
            run = subprocess.run([str(args.binary.resolve()), str(original), str(settings), str(output)],
                                 capture_output=True, text=True, check=True)
            timing = json.loads(run.stdout)
            shutil.copyfile(original, output / "author.wav")
            rate, reference = read_wave(original)
            validation = equations(output, reference[:, 0], rate, case)
            if case.get("trigger_fit") in ("waveform", "joint_waveform"):
                validation["source_waveform_fit"] = waveform_fit_check(output, reference[:, 0], rate, case)
            diagnostics.append({"title": f"Filtering diagnostic / {case['title']} / recording round trip",
                          "reference": str((output / "author.wav").resolve()),
                          "synthesis": str((output / "source_replay.wav").resolve()),
                          "synthesis_label": "Inverse-filter and filter round trip",
                          "reference_label": "Shared input recording: FoleyAutomatic 2001", "source_url": provenance["files"][0]["url"],
                          "notes": "Diagnostic Eq5 source replay through the estimated modal filter, preserving the full excitation. "
                                   "It retains all input information, including any speech, and checks only inverse/filter consistency. "
                                   "This is a numerical diagnostic, not contact synthesis or evidence of physically identified forces."})
            estimator = case.get("trigger_fit", "envelope")
            synthesis_label = {"envelope": "Regularized envelope fit", "waveform": "Waveform-fit extension",
                               "split_envelope": "Published two-pass inverse with disclosed peak boundaries",
                               "joint_waveform": "Joint excitation and audio fit"}[estimator]
            renders = {}
            filename = "synthesis.wav"
            _, actual = read_wave(output / filename)
            renders[filename] = {"comparison": compare(reference[:, 0], actual[:, 0]),
                                 "reconstruction": reconstruction_metrics(rate, reference[:, 0], actual[:, 0]),
                                 "periodicity": periodicity_comparison(rate, reference[:, 0], actual[:, 0]),
                                 "spectral_motion": spectral_motion_comparison(rate, reference[:, 0], actual[:, 0]),
                                 "metrics": metrics(rate, actual), "texture": texture_metrics(rate, actual)}
            if "motion_check" in case:
                for key, minimum in case["motion_check"].items():
                    value = renders[filename]["spectral_motion"][key.removeprefix("minimum_")]
                    if not np.isfinite(value) or value < minimum:
                        quality_failure(f"Missing spectral motion in {case['name']}/{filename}: {key}={value:.3f}")
            if "reconstruction_check" in case:
                for key, bound in case["reconstruction_check"].items():
                    name = key.removeprefix("maximum_").removeprefix("minimum_")
                    values = {**renders[filename]["comparison"], **renders[filename]["metrics"], **renders[filename]["reconstruction"]}
                    value = values[name]
                    if not np.isfinite(value) or (value > bound if key.startswith("maximum_") else value < bound):
                        quality_failure(f"Reconstruction regression in {case['name']}/{filename}: {name}={value}")
            if "tone_check" in case:
                check = case["tone_check"]
                fractions = [band_power_fraction(rate, x[:len(reference)], check["frequency_hz"], check["half_width_hz"])
                             for x in (reference[:, 0], actual[:, 0])]
                excess = float(10 * np.log10(max(fractions[1], 1e-30) / max(fractions[0], 1e-30)))
                renders[filename]["tone_check"] = {"input_power_fraction": fractions[0], "output_power_fraction": fractions[1],
                                                  "excess_db": excess, "scope": "Equal input-duration, total-energy-normalized band power."}
                if not np.isfinite(excess) or excess > check["maximum_excess_db"]:
                    quality_failure(f"Spurious narrowband tone in {case['name']}/{filename}: {excess:.2f} dB excess")
            cases.append({"time_frequency": case["name"] in ("trough_pass1", "trough_pass2"),
                          "title": f"Lagrange 2010 / {case['title']} / " +
                                   synthesis_label,
                          "reference": str((output / "author.wav").resolve()), "synthesis": str((output / filename).resolve()),
                          "reference_label": "Shared input recording: FoleyAutomatic 2001", "synthesis_label": synthesis_label,
                          "source_url": provenance["files"][0]["url"],
                          "notes": "Shared FoleyAutomatic input versus native resynthesis with causal modal decay over the complete render. "
                                   "The reference is not a Lagrange author synthesis. Exact CIQS settings and outputs remain unavailable. "
                                   f"Estimator: {estimator}; modal gains: {case['modal_gains']}; impact centering: {case.get('center_impact', False)}. "
                                   f"Spectral motion {renders[filename]['spectral_motion']['spectral_shape_similarity']:.3f}; "
                                   f"short cepstra {renders[filename]['spectral_motion']['short_cepstrum_similarity']:.3f}; "
                                   f"envelope correlation {renders[filename]['comparison']['envelope_correlation']:.3f}. "
                                   "Raw WAVs have no loudness fitting or normalization."})
            comparisons.append({"case": case["name"], "timing": timing, "equations": validation, "renders": renders,
                                "author_metrics": metrics(rate, reference), "author_texture": texture_metrics(rate, reference)})
    benchmarks = [json.loads(subprocess.run([str(args.binary.resolve()), "benchmark"], check=True,
                                           capture_output=True, text=True).stdout) for _ in range(3)]
    manifest = {"method": "Lagrange, Scavone and Depalle 2010", "sources": provenance, "controls": controls,
                "benchmark": benchmarks[1], "fresh_process_benchmarks": benchmarks,
                "comparisons": comparisons, "quality_passed": not quality_failures, "quality_failures": quality_failures,
                "scope": "Complete implemented analysis/synthesis chain on audio from an author demonstration corpus cited in section VI-B. "
                         "Equation checks are independent. The exact published 2010 listening examples remain unreproduced.",
                "gaps": ["CIQS author code, original analysis settings and Lagrange resynthesized WAVs were not recovered.",
                         "Subband mask, whitening order, spline boundary condition and interval choices are disclosed reconstruction conventions.",
                         "Envelope uses a regularized full-window inverse; split_envelope uses the published two-pass inverse with zero boundary states.",
                         "Envelope refits event amplitudes; split_envelope uses Eq9 with iterative peak removal between adjacent local minima.",
                         "Modal gain conventions and complete representative impacts are calibrated on each input, not held-out results.",
                         "The controls disclose gain convention, source estimator, impact centering and joint-fit weight for each case.",
                         "Joint fitting uses a recording-length modal impulse response; rendering preserves causal decay throughout the output horizon.",
                         "Joint waveform fitting extends the published envelope estimator and fits source and audio residuals with one coefficient vector.",
                         "Incremental phase adaptation follows Lagrange, Whetsell and Depalle, DAFx 2008, equations 10-13.",
                         "The optional moving comb requires measured trajectory and wave speed. These metadata are absent from the video.",
                         "Actual perceptual listening-test equivalence and paper Table I empirical damping/frequency distributions remain unavailable."]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "cases.json").write_text(json.dumps({"cases": cases}, indent=2) + "\n")
    (args.output / "diagnostics.json").write_text(json.dumps({"title": "Lagrange filtering diagnostics", "cases": diagnostics}, indent=2) + "\n")
    print(args.output / "cases.json")
    if quality_failures:
        raise SystemExit("Quality audit failed:\n" + "\n".join(quality_failures))
    if not args.keep_diagnostics:
        for case in controls["cases"]:
            for name in ("excitation.wav", "envelope.wav", "deconvolved.wav", "source_replay.wav", "impact_envelope.wav"):
                (args.output / case["name"] / name).unlink(missing_ok=True)
        (args.output / "diagnostics.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
