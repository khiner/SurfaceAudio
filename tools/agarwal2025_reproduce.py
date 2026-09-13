#!/usr/bin/env python3
"""Exercise thesis micro-impact filtering with complete scraping/rolling forces and fixed responses."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

from agarwal2026_reproduce import check_impact, digest, provenance, relative, run, RATE, ROOT
from agarwal_response_reproduce import preprocess, read_wave, save_json
from AnalyzeRenders import texture_metrics
from Reproduce import verify_inputs

PAPER = 'https://hdl.handle.net/1721.1/158825'


def prepare(case, controls, output):
    source = ROOT / 'references/agarwal' / (case['profile'] + '.npy')
    profile = np.load(source).astype('<f8')
    profile -= profile.mean()
    profile.tofile(output / 'profile.f64')
    frames = round(controls['duration_seconds'] * RATE)
    t = np.arange(frames) / RATE
    mass, radius = case['mass_kg'], case['radius_m']
    eccentricity = radius * controls['eccentricity_fraction']
    if case['motion'] == 'roll':
        angular_acceleration = controls['acceleration_m_s2'] / radius
        theta = controls['start_m'] / radius + controls['speed_m_s'] / radius * t + .5 * angular_acceleration * t * t
        angular_velocity = controls['speed_m_s'] / radius + angular_acceleration * t
        position = radius * theta - eccentricity * np.sin(theta)
        velocity = (radius - eccentricity * np.cos(theta)) * angular_velocity
        normal = mass * (9.81 + eccentricity * (np.cos(theta) * angular_velocity ** 2 + np.sin(theta) * angular_acceleration))
        if normal.min() <= 0:
            raise ValueError('Prescribed rolling motion loses contact')
        normal_coordinate = 1 + 5 * (normal - normal.min()) / (normal.max() - normal.min())
    else:
        position = controls['start_m'] + controls['speed_m_s'] * t + .5 * controls['acceleration_m_s2'] * t * t
        velocity = controls['speed_m_s'] + controls['acceleration_m_s2'] * t
        normal = np.full(frames, mass * 9.81)
        weight = ((.05 - controls['alpha_constant']) / .04) ** (1 / .95)
        normal_coordinate = np.full(frames, 1 + 5 * weight)
    np.column_stack((position, velocity, normal_coordinate, np.zeros(frames))).astype('<f8').tofile(output / 'motion.f64')
    values = [RATE, frames, controls['levels'], 1, controls['spacing_m'], controls['gaussian_half_width_samples'], mass,
              controls['beta1'], controls['beta2'], radius, eccentricity, controls['rolling_stiffness'], controls['rolling_dissipation'], 1, controls['alpha_scale_m']]
    (output / 'parameters.txt').write_text(' '.join(map(str, values)) + '\n')
    return {'profile_sha256': digest(source), 'height_datum': 'mean-subtracted measured profile; integrated paths match both endpoint heights',
            'normal_force_n': [float(normal.min()), float(normal.max())],
            'normal_encoding': 'Affine normal-load fraction encoded on [1,6] for the shared atlas kernel; constant-alpha inverse for scraping',
            'path_m': [float(position.min()), float(position.max())], 'speed_m_s': [float(velocity.min()), float(velocity.max())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/agarwalObjectReproduce')
    parser.add_argument('--force-binary', type=Path, default=ROOT / 'build/agarwalReproduce')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/reproduction/agarwal2025')
    parser.add_argument('--alpha-scale', type=float, help='Curvature coefficient length unit in metres; default 1e-6 is an explicit reconstruction assumption')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--keep-diagnostics', action='store_true')
    args = parser.parse_args()
    verify_inputs(['agarwal', 'agarwal-response'])
    controls = json.loads((ROOT / 'repros/agarwal2025/cases.json').read_text())
    if args.alpha_scale is not None:
        controls['alpha_scale_m'] = args.alpha_scale
    if not np.isfinite(controls['alpha_scale_m']) or controls['alpha_scale_m'] <= 0:
        raise ValueError('Curvature scale must be positive and finite')
    build = provenance(args.binary, args.force_binary)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    responses = [preprocess(read_wave(ROOT / 'references/agarwal/icml2023/training' / (controls[field] + '.wav')), True)[0]
                 for field in ('surface_response', 'object_response')]
    response = sum(np.pad(x, (0, max(map(len, responses)) - len(x))) for x in responses).astype(np.float32)
    wavfile.write(output / 'response.wav', RATE, response)
    gate = round(controls['gate_seconds'] * RATE)
    envelope = .5 - .5 * np.cos(np.linspace(0, np.pi, gate))
    metrics, cases = {}, []
    for case in controls['cases']:
        directory = output / case['name']
        directory.mkdir(parents=True, exist_ok=True)
        metadata = prepare(case, controls, directory)
        micro_stiffness = case.get('stiffness_ball_n_m', controls['stiffness_ball_n_m'])
        arguments = [controls['oversampling'], controls['gaussian_sigma_ratio'], '--bandlimit']
        run(args.force_binary, 'prepare-temporal', directory, *arguments)
        reference_directory = directory / 'reference'
        reference_directory.mkdir(exist_ok=True)
        for name in ('profile.f64', 'motion.f64', 'parameters.txt'):
            shutil.copyfile(directory / name, reference_directory / name)
        run(args.force_binary, 'prepare-temporal-reference', reference_directory, *arguments)
        components = {}
        for name in ('scrape', 'elastic', 'damping'):
            a = np.fromfile(directory / (name + '.f32'), '<f4').astype(float)
            b = np.fromfile(reference_directory / (name + '.f32'), '<f4').astype(float)
            components[name] = {'relative_l2': relative(a, b), 'ac_relative_l2': relative(a - a.mean(), b - b.mean()),
                                'rms': float(np.sqrt(np.mean(a * a))), 'ac_rms': float(np.std(a))}
            if not np.isfinite(components[name]['relative_l2']) or max(components[name]['relative_l2'], components[name]['ac_relative_l2']) > .005:
                raise ValueError(f'GPU force component differs from FP64: {case["name"]}/{name}')
        stiffness = controls['rolling_stiffness'] if case['motion'] == 'roll' else 0
        dissipation = controls['rolling_dissipation'] if case['motion'] == 'roll' else 0
        run(args.binary, 'forces', directory, directory / 'excitation.wav', stiffness, dissipation)
        excitation = read_wave(directory / 'excitation.wav')
        excitation[:gate] *= envelope
        excitation[-gate:] *= envelope[::-1]
        wavfile.write(directory / 'excitation.wav', RATE, excitation.astype(np.float32))
        impact_controls = (case['mass_kg'], controls['micro_impact_velocity_m_s'], micro_stiffness,
                           controls['stiffness_surface_n_m'], controls['micro_impact_scale'], np.inf)
        run(args.binary, 'contact', directory / 'excitation.wav', output / 'response.wav', directory, *impact_controls, 0)
        errors = check_impact(directory, response.astype(float), *impact_controls, excitation=excitation, force_tolerance=1e-5)
        actual = read_wave(directory / 'synthesis.wav')
        unfiltered = signal.fftconvolve(excitation, response).astype(np.float32)
        wavfile.write(directory / 'unfiltered.wav', RATE, unfiltered)
        metrics[case['name']] = {'inputs': metadata, 'components': components, 'convolution_relative_l2': errors['convolution_relative_l2'], 'pulse_relative_l2': errors['force_relative_l2'],
                                'texture': texture_metrics(RATE, actual[:, None]), 'render': json.loads((directory / 'render.json').read_text())}
        cases.append({'title': f'Agarwal thesis / {case["name"]} / finite micro-impact filtering',
                      'reference': str(directory / 'unfiltered.wav'), 'synthesis': str(directory / 'synthesis.wav'),
                      'reference_label': 'Our full contact excitation with instantaneous impacts',
                      'synthesis_label': 'Our thesis extension with finite micro-impacts', 'source_url': PAPER,
                      'notes': controls['scope'] + ' Both players are our controlled renders, with the same fixed sum of two measured responses. '
                      'Scraping includes horizontal and vertical forces; rolling additionally includes elastic and dissipative forces. Complete tails retained.'})
    for motion in ('scrape', 'roll'):
        cases.append({'title': f'Agarwal thesis / {motion} / mass and radius control',
                      'reference': str(output / f'{motion}-light' / 'synthesis.wav'), 'synthesis': str(output / f'{motion}-heavy' / 'synthesis.wav'),
                      'reference_label': 'Our model: 50 g, 8 mm radius', 'synthesis_label': 'Our model: 10 kg, 50 mm radius',
                      'source_url': PAPER, 'notes': 'Published Experiment 1 mass/radius pairs, with disclosed substitute profiles and responses. '
                      'These are new model renders; original thesis audio is unavailable. Raw playback retains amplitude differences.'})
    cases.append({'title': 'Agarwal thesis / scraping / stiffness control',
                  'reference': str(output / 'scrape-hard/synthesis.wav'), 'synthesis': str(output / 'scrape-soft/synthesis.wav'),
                  'reference_label': 'Our model: hard object, 5e8 N/m', 'synthesis_label': 'Our model: soft object, 5e4 N/m',
                  'source_url': PAPER, 'notes': 'Published Experiment 2 stiffnesses, 100 g mass and 8 mm radius against a 1e8 N/m surface. '
                  'Same excitation and responses isolate the finite micro-impact duration; no bouncing or restitution is simulated.'})
    if provenance(args.binary, args.force_binary) != build:
        raise ValueError('Executable or shaders changed during reproduction')
    save_json(output / 'metrics.json', {'build_sha256': build, 'controls': controls, 'cases': metrics})
    save_json(output / 'cases.json', {'cases': cases})
    run(sys.executable, ROOT / 'tools/BuildListeningReport.py', output / 'cases.json', '--output', output / 'listening')
    if not args.keep_diagnostics:
        for case in controls['cases']:
            directory = output / case['name']
            shutil.rmtree(directory / 'reference')
            for pattern in ('*.f32', '*.f64', 'force.wav', 'excitation.wav'):
                for path in directory.glob(pattern): path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
