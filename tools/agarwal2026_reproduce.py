#!/usr/bin/env python3
"""Render the 2026 object model using disclosed fits to the available 2023 author corpus."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import integrate, signal
from scipy.io import wavfile

from agarwal_response_reproduce import ROOT, RATE, MATERIALS, digest, initialize, preprocess, restore_response, read_wave, features, verify_inputs, save_json
from Reproduce import run, verify_inputs as verify_cached_inputs

FIXTURE = ROOT / 'repros/agarwal2026/parameters.json'
REFERENCES = ROOT / 'references/agarwal/icml2023/training'
PAPER = 'https://doi.org/10.64898/2026.01.28.702236'


def provenance(*binaries):
    paths = {Path(p).resolve() for p in binaries}
    paths.update(p.parent / 'SurfaceAudio.metallib' for p in tuple(paths))
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): digest(p) for p in sorted(paths)}


def initial_parameters(samples, seed):
    selected, metadata = initialize(samples, seed)
    rng = np.random.default_rng(seed)
    rt = metadata['broadband_rt60']['seconds']
    frequencies, power = signal.periodogram(samples, RATE, detrend=False, scaling='density')
    edges = (10 ** (np.linspace(0, 21.4 * np.log10(1 + .00437 * RATE / 2), 21) / 21.4) - 1) / .00437
    band_power = np.array([np.sum(power[(frequencies >= lo) & (frequencies < hi)]) * (frequencies[1] - frequencies[0]) for lo, hi in zip(edges[:-1], edges[1:])])
    levels = 10 * np.log10(np.maximum(band_power, 1e-16))
    parameters = np.concatenate((selected[:10], rng.uniform(-30, -10, 10), np.maximum(rng.uniform(rt - .05, rt + .05, 10), 1e-5),
                                 rng.uniform(levels - 20, levels), rng.uniform(.04, .12, 20))).astype('<f4')
    return parameters, {**metadata, 'mode_rt60_initialization': 'broadband RT60 +/- 50ms, clamped at 10us',
                        'noise_amplitude_initialization': 'uniform[band power dB - 20, band power dB]; integrated periodogram power',
                        'noise_rt60_initialization': 'uniform[.04,.12] seconds'}


def check_distribution(directory, records):
    model = json.loads((directory / 'distribution.json').read_text())
    modes = records[:, :30].reshape(-1, 3, 10).transpose(0, 2, 1).reshape(-1, 3)
    observations = [modes[(modes[:, 0] >= 20) & (modes[:, 2] >= np.float32(.01))]]
    observations += [records[records[:, 50 + b] >= np.float32(.005)][:, [30 + b, 50 + b]] for b in range(20)]
    for values, fitted in zip(observations, [model['modes'], *model['noise']]):
        factor = np.asarray(fitted['factor'])
        np.testing.assert_allclose(fitted['mean'], values.astype(float).mean(axis=0), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(factor @ factor.T, np.cov(values.astype(float), rowvar=False, ddof=1), rtol=1e-8, atol=1e-10)
    samples = np.fromfile(directory / 'samples.f32', '<f4').reshape(-1, 70)
    assert np.isfinite(samples).all() and (samples[:, :10] >= 20).all() and (samples[:, :10] <= 20000).all()
    assert (samples[:, 10:20] <= 0).all() and (samples[:, 30:50] <= 0).all()
    assert (samples[:, 20:30] >= np.float32(.01)).all() and (samples[:, 20:30] <= 1).all()
    assert (samples[:, 50:] >= np.float32(.005)).all() and (samples[:, 50:] <= 1).all()
    return {'independent_sample_covariance_check': True, 'truncation_check': True, 'draws': len(samples)}


def reference_force(frames, mass, velocity, ka, kb, scale, limit, combination=0):
    stiffness = ka + kb if combination else min(ka, kb) / (1 + min(ka, kb) / max(ka, kb))
    omega = np.sqrt(stiffness / mass)
    duration = np.pi / omega
    amplitude = scale * mass * velocity
    def function(t):
        value = amplitude * np.sin(omega * t)
        return value if np.isinf(limit) else limit * np.tanh(value / limit)
    return np.array([RATE * integrate.quad(function, i / RATE, min((i + 1) / RATE, duration), epsabs=1e-14)[0]
                             if i / RATE < duration else 0 for i in range(frames)]), duration


def relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)) if a.shape == b.shape else float('inf')


def check_impact(directory, response, *controls, excitation=None, force_tolerance=1e-4):
    force = read_wave(directory / 'force.wav')
    oracle_force, duration = reference_force(len(force), *controls)
    force_error = relative(force, oracle_force)
    expected = signal.fftconvolve(oracle_force, response)
    if excitation is not None:
        expected = signal.fftconvolve(excitation.astype(np.float32).astype(float), expected)
    waveform_error = relative(read_wave(directory / 'synthesis.wav'), expected)
    if not np.isfinite(force_error + waveform_error) or force_error > force_tolerance or waveform_error > 1e-4:
        raise ValueError(f'Impact quadrature/convolution mismatch: {force_error}, {waveform_error}')
    return {'force_relative_l2': float(force_error), 'convolution_relative_l2': float(waveform_error), 'duration_seconds': float(duration)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/agarwalObjectReproduce')
    parser.add_argument('--fit-binary', type=Path, default=ROOT / 'build/agarwalResponseFit')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/reproduction/agarwal2026')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--fit', action='store_true', help='Refit the author corpus and save replay parameters')
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--paper-optimizer', action='store_true', help='Use physical coordinates and the published Adam learning rate 2e-6')
    parser.add_argument('--keep-diagnostics', action='store_true')
    args = parser.parse_args()
    verify_cached_inputs(['agarwal-response']) if args.offline else verify_inputs()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.fit:
        retained = {'paper': PAPER, 'corpus': 'Agarwal 2023 public examples; size labels unknown; not the 2026 survey', 'fits': {}}
    else:
        retained = json.loads(FIXTURE.read_text())
    build = provenance(args.binary, args.fit_binary)
    cases, metrics = [], {'build_sha256': build, 'fits': {}, 'distributions': {}, 'impacts': {}}
    for material in MATERIALS:
        records = []
        for index in range(1, 6):
            name = f'{material}_{index}'
            source = REFERENCES / f'{name}.wav'
            directory = output / 'fits' / name
            directory.mkdir(parents=True, exist_ok=True)
            prepared, alignment = preprocess(read_wave(source), True)
            if args.fit:
                seed = 2026 + MATERIALS.index(material) * 5 + index
                initial, initialization = initial_parameters(prepared, seed)
                initial.tofile(directory / 'initial.f32')
                wavfile.write(directory / 'prepared.wav', RATE, prepared)
                run(args.fit_binary, 'fit', directory / 'prepared.wav', directory / 'initial.f32', directory, args.steps, seed,
                    2e-6 if args.paper_optimizer else .001, 'physical' if args.paper_optimizer else 'log-decay', 'linear')
                values = np.fromfile(directory / 'fitted.f32', '<f4')
                record = {'source_sha256': digest(source), 'parameters': values.tolist(), 'seed': seed, 'frames': len(prepared),
                          'alignment': alignment, 'initialization': initialization,
                          'fit': json.loads((directory / 'fit.json').read_text()), 'response_sha256': digest(directory / 'fitted.wav')}
                retained['fits'][name] = record
            else:
                record = retained['fits'][name]
                if digest(source) != record['source_sha256'] or alignment != record['alignment']:
                    raise ValueError(f'Changed response input or alignment: {name}')
                values = np.asarray(record['parameters'], dtype='<f4')
                values.tofile(directory / 'fitted.f32')
                run(args.fit_binary, 'sample', directory / 'fitted.f32', directory / 'fitted.wav', record['frames'], record['seed'])
                if digest(directory / 'fitted.wav') != record['response_sha256']:
                    raise ValueError(f'Retained 2026 response changed: {name}')
            if values.shape != (70,) or not np.isfinite(values).all():
                raise ValueError('Invalid 2026 response parameters')
            response = read_wave(directory / 'fitted.wav')
            wavfile.write(directory / 'synthesis.wav', RATE, restore_response(response, alignment))
            records.append(values)
            metrics['fits'][name] = {'input': features(prepared, frames=None), 'synthesis': features(response, frames=None), 'fit': record['fit']}
            cases.append({'title': f'Agarwal 2026 response fit / {name}', 'reference': str(source), 'synthesis': str(directory / 'synthesis.wav'),
                          'reference_label': 'Author measured response (2023 public corpus)', 'synthesis_label': 'Our 2026 response model fitted to this input',
                          'source_url': PAPER, 'notes': 'Ten modes and twenty ERB noise bands, with full-spectrum and multiresolution Huber fitting. '
                          'This is a reconstruction of a 2023 author input; the 2026 survey recordings and size labels remain unavailable.'})
        records = np.asarray(records, dtype='<f4')
        directory = output / 'materials' / material
        directory.mkdir(parents=True, exist_ok=True)
        records.tofile(directory / 'records.f32')
        eligible = (records[:, 50:] >= np.float32(.005)).sum(axis=0)
        if (eligible < 2).any():
            metrics['distributions'][material] = {'status': 'insufficient eligible observations', 'noise_band_counts': eligible.tolist(),
                                                  'reason': 'The public corpus cannot estimate sample covariance for every band after the paper exclusions.'}
        else:
            run(args.binary, 'distribution', directory / 'records.f32', directory, 1024, 2026)
            metrics['distributions'][material] = check_distribution(directory, records)
            sampled = np.fromfile(directory / 'samples.f32', '<f4').reshape(-1, 70)
            cohort, fitted = [], []
            for index in range(4):
                parameters = directory / f'sample-{index}.f32'
                wave = directory / f'sample-{index}.wav'
                sampled[index].tofile(parameters)
                run(args.fit_binary, 'sample', parameters, wave, RATE, 2026 + index)
                cohort.extend((read_wave(wave), np.zeros(RATE // 4)))
                fitted.extend((read_wave(output / 'fits' / f'{material}_{index + 1}' / 'fitted.wav'), np.zeros(RATE // 4)))
            wavfile.write(directory / 'sampled.wav', RATE, np.concatenate(cohort).astype(np.float32))
            wavfile.write(directory / 'fitted.wav', RATE, np.concatenate(fitted).astype(np.float32))
            cases.append({'title': f'Agarwal 2026 statistical responses / {material}',
                          'reference': str(directory / 'fitted.wav'), 'synthesis': str(directory / 'sampled.wav'),
                          'reference_label': 'Our four fitted responses', 'synthesis_label': 'Our four independent statistical draws',
                          'source_url': PAPER, 'notes': 'Unpaired cohorts generated by the 2026 model. The distribution uses five 2023 public inputs of unknown size. '
                          'This tests the available material cohort, not the missing 2026 material/size distributions.'})
        response_path = output / 'fits' / f'{material}_1' / 'fitted.wav'
        for suffix, mass, limit in (('0.05', .05, np.inf), ('0.5', .5, np.inf), ('clipped', .5, .1)):
            impact = output / 'impacts' / f'{material}-{suffix}'
            controls = (mass, 2, 1e7, 7.03e7, 1, limit)
            run(args.binary, 'impact', response_path, impact, *controls, 0)
            metrics['impacts'][f'{material}-{suffix}'] = check_impact(impact, read_wave(response_path), *controls)
        cases.append({'title': f'Agarwal 2026 mass control / {material}',
                      'reference': str(output / 'impacts' / f'{material}-0.05' / 'synthesis.wav'),
                      'synthesis': str(output / 'impacts' / f'{material}-0.5' / 'synthesis.wav'),
                      'reference_label': 'Our model: 50 g', 'synthesis_label': 'Our model: 500 g', 'source_url': PAPER,
                      'notes': 'Controlled synthesis comparison, both ours. Same fitted response, velocity 2 m/s, stiffnesses 1e7 and 7.03e7 N/m. '
                      'Linear force limit; gain A=1 is an explicit convention. No author impact recording is available for this pair.'})
        cases.append({'title': f'Agarwal 2026 nonlinear force / {material}',
                      'reference': str(output / 'impacts' / f'{material}-0.5' / 'synthesis.wav'),
                      'synthesis': str(output / 'impacts' / f'{material}-clipped' / 'synthesis.wav'), 'reference_label': 'Our linear impact',
                      'synthesis_label': 'Our tanh-compressed impact', 'source_url': PAPER,
                      'notes': 'Same mass, velocity, stiffness and response. The force limit cm=0.1 and A=1 are explicit demonstration choices; author calibration is unavailable.'})
    if provenance(args.binary, args.fit_binary) != build:
        raise ValueError('Executable or shaders changed during reproduction')
    if args.fit:
        retained['build_sha256'] = build
        save_json(FIXTURE, retained)
    save_json(output / 'metrics.json', metrics)
    save_json(output / 'cases.json', {'cases': cases})
    run(sys.executable, ROOT / 'tools/BuildListeningReport.py', output / 'cases.json', '--output', output / 'listening')
    if not args.keep_diagnostics:
        for pattern in ('fits/*/initial.wav', 'fits/*/initial.f32', 'fits/*/fitted.f32', 'fits/*/prepared.wav', 'fits/*/last.f32', 'fits/*/loss.csv', 'impacts/*/force.wav', 'materials/*/samples.f32', 'materials/*/sample-*.f32', 'materials/*/sample-*.wav', 'materials/*/records.f32'):
            for path in output.glob(pattern): path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
