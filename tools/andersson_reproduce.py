#!/usr/bin/env python3
"""Prepare, validate and audition the Andersson-Kropp contact model."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from andersson_model import layer_green, punch_factor, surface_springs
from reference_assets import ROOT

DEFAULTS = {'elements': [20, 20], 'element_spacing': .001, 'thickness': .01, 'sample_rate': 51200,
            'taps': 256, 'spring_knots': 257, 'spring_fraction': .9}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def listening_input(directory):
    from scipy.ndimage import gaussian_filter
    from scipy.signal import butter, sosfilt
    directory.mkdir(parents=True, exist_ok=True)
    rate, duration, scan_spacing = 51200, 4.5, 38e-6
    rng = np.random.default_rng(2008)
    side = int(np.ceil(.02 / scan_spacing))
    surface = gaussian_filter(rng.standard_normal((side, side)), .00025 / scan_spacing, mode='wrap')
    surface = (surface - surface.mean()) * (100e-6 / surface.std())
    np.save(directory / 'surface.npy', surface)
    material = np.array([[0, 25e6, 0], [rate / 2, 50e6, 25e6]])
    np.savetxt(directory / 'material.csv', material, delimiter=',', header='frequency_Hz,storage_Pa,loss_Pa', comments='')
    time = np.arange(round(rate * duration)) / rate
    indentation = np.full(len(time), -50e-6)
    for index in range(8):
        phase = (time - (.12 + .22 * index)) / .008
        active = (phase > 0) & (phase < 1)
        indentation[active] += (200e-6 + index * 25e-6) * np.sin(np.pi * phase[active])**2
    vibration = sosfilt(butter(2, [120, 2200], btype='bandpass', fs=rate, output='sos'), rng.standard_normal(len(time)))
    vibration *= 25e-6 / vibration.std()
    gate = np.minimum(np.clip((time - 2.1) / .02, 0, 1), np.clip((4.3 - time) / .02, 0, 1))
    gate = .5 - .5 * np.cos(np.pi * gate)
    indentation += gate * (250e-6 + vibration)
    np.savetxt(directory / 'schedule.csv', np.column_stack((indentation, np.ones(len(time)))), delimiter=',',
               header='indentation_m,hold_samples', comments='', fmt=['%.17g', '%d'])
    config = {'surface': 'surface.npy', 'material': 'material.csv', 'schedule': 'schedule.csv',
              'source': 'Generated rough surface, seed 2008, 100 micrometre RMS, Gaussian smoothing sigma 0.25 mm; '
                        'prescribed 8 ms presses followed by band-limited normal vibration. '
                        'Linear complex modulus between the paper endpoints; density and Poisson ratio are chosen inputs.',
              'scan_spacing': scan_spacing, 'elements': [10, 10], 'element_spacing': .002, 'thickness': .01,
              'sample_rate': rate, 'taps': 256, 'poisson': .499, 'density': 1300, 'spring_bonded': True,
              'spring_knots': 513, 'spring_fraction': .9, 'domain_elements': 64, 'oversampling': 8}
    path = directory / 'contact.json'
    write_json(path, config)
    return path


def force_audio(run, output, config, segments=None):
    from scipy.io import wavfile
    from scipy.signal import butter, resample_poly, sosfilt
    from fractions import Fraction
    rate = config['sample_rate']
    force = np.fromfile(run / 'force.f64', dtype='<f8')
    if not len(force) or not np.all(np.isfinite(force)) or np.any(force < 0):
        raise RuntimeError('Contact-force trace is empty, nonfinite or tensile')
    signal = sosfilt(butter(2, 20, btype='highpass', fs=rate, output='sos'), force)
    ratio = Fraction(44100 / rate).limit_denominator(10000)
    signal = resample_poly(signal, ratio.numerator, ratio.denominator)
    peak = float(np.max(abs(signal)))
    if peak <= 0:
        raise RuntimeError('Contact-force trace has no audible variation')
    gain = .9 / peak
    records, cases = [], []
    if segments is None:
        segments = [('contact_force', 'Contact-force audio', 0, len(force) / rate)]
    for name, title, start, end in segments:
        samples = (gain * signal[round(start * 44100):round(end * 44100)]).astype(np.float32)
        if len(samples) != round((end - start) * 44100) or not np.isfinite(samples).all() or not np.any(samples):
            raise RuntimeError('Invalid force audio segment')
        path = output / f'{name}.wav'
        wavfile.write(path, 44100, samples)
        records.append({'file': path.name, 'interval_s': [start, end], 'peak': float(np.max(abs(samples))),
                        'rms': float(np.sqrt(np.mean(samples.astype(float)**2))),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        cases.append({'title': title, 'synthesis': str(path), 'synthesis_label': 'Computed normal contact force',
                      'notes': f'Direct force sonification. Inputs: {config["source"]} '
                               '20 Hz DC removal and a common playback gain; no added resonator.'})
    curve = np.genfromtxt(run / 'curve.csv', delimiter=',', names=True)
    metadata = {'sample_rate': 44100, 'input_sample_rate': rate, 'gain_per_newton': gain,
                'processing': 'Causal second-order 20 Hz Butterworth high-pass, then polyphase resampling; one gain shared by all clips',
                'max_force_N': float(force.max()), 'max_contact_residual_m': float(np.max(curve['residual_m'])),
                'max_newton_iterations': int(np.max(curve['iterations'])), 'clips': records}
    write_json(output / 'audio.json', metadata)
    record_cases(output, cases)
    print('Audio:', ', '.join(clip['file'] for clip in records), flush=True)


def record_cases(output, updates):
    path = output / 'cases.json'
    cases = json.loads(path.read_text())['cases'] if path.exists() else []
    titles = {case['title'] for case in updates}
    write_json(path, {'cases': [case for case in cases if case['title'] not in titles] + updates})


def prepare(config_path, output):
    config_path = config_path.resolve()
    config = DEFAULTS | json.loads(config_path.read_text())
    required = ('surface', 'material', 'schedule', 'source', 'scan_spacing', 'poisson', 'density', 'spring_bonded',
                'domain_elements', 'oversampling')
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError('Missing input settings: ' + ', '.join(missing))
    if not isinstance(config['spring_bonded'], bool) or not isinstance(config['source'], str) or not config['source'].strip():
        raise ValueError('Supply a boolean spring_bonded and a source description for the measured inputs')
    paths = {key: (config_path.parent / config[key]).resolve() for key in ('surface', 'material', 'schedule')}
    heights = np.load(paths['surface'], allow_pickle=False)
    material = np.loadtxt(paths['material'], delimiter=',', skiprows=1, ndmin=2)
    schedule = np.loadtxt(paths['schedule'], delimiter=',', skiprows=1, ndmin=2)
    if (schedule.shape[1] != 2 or not len(schedule) or not np.all(np.isfinite(schedule))
            or np.any(schedule[:, 1] < 1) or np.any(schedule[:, 1] != np.floor(schedule[:, 1]))):
        raise ValueError('Schedule columns must be indentation_m and integer hold_samples')
    elements = config['elements']
    spacing, thickness = config['element_spacing'], config['thickness']
    rate, taps = config['sample_rate'], config['taps']
    knots, fraction = config['spring_knots'], config['spring_fraction']
    print('Computing elastic-layer responses...', flush=True)
    green, response = layer_green(elements, spacing, thickness, config['poisson'], config['density'], material,
                                  rate, taps, config['domain_elements'], config['oversampling'])
    print('Computing geometry-dependent interfacial springs...', flush=True)
    tops, springs = surface_springs(heights, config['scan_spacing'], elements, spacing, material[0, 1],
                                   config['spring_bonded'], fraction, knots)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'dimensions.txt').write_text(f'{elements[1]} {elements[0]} {taps} {knots} {rate}\n')
    for name, data in [('heights', tops), ('springs', springs), ('green', green), ('schedule', schedule)]:
        np.asarray(data, dtype='<f8').tofile(output / f'{name}.f64')
    # This checks the transform normalization without altering any Green-function samples.
    static_error = float(np.max(np.abs(green.sum(axis=0) / rate - response[0].real)))
    return {'config': config, 'inputs': {key: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                                         for key, path in paths.items()}, 'static_transform_error_m_per_N': static_error}


def simulate(config, binary, output, equilibrium=False):
    provenance = prepare(config, output / 'prepared')
    print('Solving coupled contact...', flush=True)
    subprocess.run([str(binary.resolve()), str(output / 'prepared'), str(output), *(['--equilibrium'] if equilibrium else [])], check=True)
    provenance['mode'] = 'equilibrium' if equilibrium else 'time-domain'
    write_json(output / 'run.json', provenance)
    return provenance


def plot_result(output):
    curve = np.genfromtxt(output / 'curve.csv', delimiter=',', names=True, ndmin=1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    indentation, force = curve['indentation_m'], curve['force_N']
    axes[0].plot(indentation * 1e3, force)
    axes[0].set(xlabel='Indentation [mm]', ylabel='Normal force [N]')
    difference = np.diff(indentation)
    valid = difference != 0
    axes[1].plot((force[:-1] + force[1:])[valid] / 2, np.diff(force)[valid] / difference[valid] / 1e6)
    axes[1].set(xlabel='Normal force [N]', ylabel='Incremental contact stiffness [MN/m]')
    fig.tight_layout()
    fig.savefig(output / 'contact.png', dpi=150)
    plt.close(fig)


def validate(binary, output):
    subprocess.run([str(binary.parent / 'PatchContactTest')], check=True)
    subprocess.run([sys.executable, '-W', 'error', str(ROOT / 'tests/AnderssonTest.py'), '--binary', str(binary), '-v'], check=True)
    data = np.genfromtxt(ROOT / 'tests/data/andersson_punch.csv', delimiter=',', skip_header=4, names=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    errors = {}
    for bonded, label in [(True, 'Bonded base'), (False, 'Frictionless base')]:
        rows = data[(data['poisson'] == .5) & (data['bonded'] == bonded)]
        prediction = np.array([punch_factor(row['ratio'], bonded=bonded) for row in rows])
        reference = rows['normalized_stiffness']
        line, = axes[0].loglog(rows['ratio'], prediction, label=label + ': port')
        axes[0].loglog(rows['ratio'], reference, 'o', color=line.get_color(), fillstyle='none', label=label + ': published BEM')
        axes[1].semilogx(rows['ratio'], 100 * (prediction / reference - 1), label=label)
        errors[label] = float(np.max(abs(prediction / reference - 1)))
    axes[0].set(xlabel='Punch radius / layer thickness', ylabel='Stiffness / half-space stiffness')
    axes[1].set(xlabel='Punch radius / layer thickness', ylabel='Difference from published BEM [%]')
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    figure.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / 'punch_validation.png', dpi=150)
    plt.close(figure)
    write_json(output / 'validation.json', {'published_punch_max_relative_error': errors,
                                            'contact_tests': 'passed', 'layer_and_pipeline_tests': 'passed'})
    record_cases(output, [{'title': 'Finite-layer punch stiffness', 'figure': str(output / 'punch_validation.png'),
                        'source_url': 'https://doi.org/10.3390/machines11040474',
                        'notes': 'The punch calculation used by Section 2.6 is checked against Forsbach’s independent published BEM data. '
                                 'This is a component validation, not the Andersson–Kropp tread-block result. '
                                 'Running that result requires the original road scan and measured material curve.'}])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--config', type=Path, help='Measured surface, material table, indentation schedule and numerical settings')
    mode.add_argument('--validate', action='store_true', help='Run numerical tests and the independent published punch comparison')
    mode.add_argument('--listening', action='store_true', help='Render contact-force WAVs from a documented generated surface and normal-motion schedule')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/reproduction/andersson')
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/anderssonReproduce')
    parser.add_argument('--equilibrium', action='store_true', help='Solve final static equilibria using the DC compliance instead of the transient schedule')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.equilibrium and args.config is None:
        parser.error('--equilibrium requires --config')
    if args.validate:
        validate(args.binary.resolve(), output)
        return
    config = listening_input(output / 'listening_input') if args.listening else args.config
    run = output / 'audio_run' if args.listening else output
    provenance = simulate(config, args.binary, run, args.equilibrium)
    if args.config:
        plot_result(run)
        record_cases(output, [{'title': 'Tread-block force and stiffness', 'figure': str(run / 'contact.png'),
                               'notes': f'Inputs: {provenance["config"]["source"]}. Mode: {provenance["mode"]}. '
                                        'Computed force and stiffness; no matched author curve is included.'}])
    if not args.equilibrium:
        segments = [('presses', 'Repeated normal presses', 0, 2), ('vibration', 'Sustained vibrating contact', 2, 4.5)] if args.listening else None
        force_audio(run, output, provenance['config'], segments)


if __name__ == '__main__':
    main()
