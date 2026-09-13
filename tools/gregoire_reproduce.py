#!/usr/bin/env python3
"""Compare the instrumented-slider model with published Grégoire force traces."""
import argparse
import hashlib
import json
from os.path import relpath
from pathlib import Path
import re
import subprocess
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

from reference_assets import ROOT, fetch


def extract(metadata, archive):
    with tarfile.open(archive) as source:
        data = source.extractfile(metadata['member']).read()
    if hashlib.sha256(data).hexdigest() != metadata['member_sha256']:
        raise RuntimeError('Published EPS hash mismatch')
    number = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    pair = rf'{number}\s+{number}'
    paths = []
    for match in re.finditer(rf'({pair}\s+m\s+(?:{pair}\s+l\s*)+)S', data.decode()):
        points = np.array(re.findall(rf'({number})\s+({number})\s+[ml]', match.group(1)), dtype=float)
        if len(points) > 100:
            paths.append(points)
    if [len(p) for p in paths] != metadata['source_vertex_counts']:
        raise RuntimeError('Published figure path structure changed')
    calibrated = []
    for index, points in enumerate(paths):
        axes = metadata['axes'][index // metadata['paths_per_panel']]
        time = (points[:, 0] - axes['x_zero']) * 2 / (axes['x_two_seconds'] - axes['x_zero'])
        force = (points[:, 1] - axes['y_zero']) * axes['positive_force_n'] / (axes['y_positive'] - axes['y_zero'])
        if np.any(np.diff(time) < 0) or not np.all(np.isfinite(force)):
            raise RuntimeError('Invalid figure trace coordinates')
        calibrated.append((time, force))
    return calibrated


def compare(reference, candidate):
    difference = candidate - reference
    return {'rmse_n': float(np.sqrt(np.mean(difference**2))),
            'relative_rms_error': float(np.linalg.norm(difference) / max(np.linalg.norm(reference), 1e-30))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/roughReproduce')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/reproduction/rough/gregoire')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--keep-diagnostics', action='store_true')
    args = parser.parse_args()
    metadata = json.loads((ROOT / 'repros/rough/Gregoire.json').read_text())
    archive = fetch(ROOT / 'references/rough/2009.07062.tar', metadata['archive_url'], metadata['archive_sha256'], args.offline)
    paths = extract(metadata, archive)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(args.binary.resolve()), 'gregoire', str(output)], check=True)
    states = json.loads((output / 'static.json').read_text())
    native = np.genfromtxt(output / 'sensor.csv', names=True, delimiter=',')
    columns = ['asperity7_loss', 'asperity9_loss', 'asperity9_return']
    names = ['Asperity 7: contact loss', 'Asperity 9: contact loss', 'Asperity 9: return to contact']
    report = {'paper': metadata['paper'], 'doi': metadata['doi'], 'reference': metadata,
              'model': states, 'comparisons': [],
              'scope': 'Static Hertz contact forces followed by the published sensor filter; measured traces are figure vectors.',
              'limits': ['Figure quantization and plotting decimation limit reference accuracy.',
                         'The normal-contact model excludes the lateral edge impacts discussed for return to contact.',
                         'The homogeneous-box inertias and selected damping ratio affect dynamic examples, not these static force predictions.',
                         'The published dashed curves fit sensor transients; they are distinct from the paper contact-force predictions.']}
    figure, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    lowpass = butter(4, 20, fs=5000, output='sos')
    for panel, (axis, title, column) in enumerate(zip(axes, names, columns)):
        model = native[column]
        for speed_index, speed in enumerate(metadata['speeds_mm_s']):
            reference_time, reference_force = paths[5 * panel + speed_index]
            use = (native['time'] >= max(0, reference_time[0])) & (native['time'] <= min(2, reference_time[-1]))
            time = native['time'][use]
            reference = np.interp(time, reference_time, reference_force)
            result = {'panel': panel + 1, 'speed_mm_s': speed, 'seconds_compared': float(time[-1] - time[0]),
                      'time_interval_seconds': [float(time[0]), float(time[-1])],
                      'full_band': compare(reference, model[use]),
                      'below_20_hz': compare(sosfiltfilt(lowpass, reference), sosfiltfilt(lowpass, model[use]))}
            report['comparisons'].append(result)
            axis.plot(reference_time, reference_force, color=str(.25 + .15 * speed_index), linewidth=.7, label=f'Measured, {speed:g} mm/s')
        fitted_time, fitted_force = paths[5 * panel + 4]
        axis.plot(fitted_time, fitted_force, 'k--', linewidth=1.5, label='Published fitted sensor response')
        axis.plot(native['time'], model, color='#1671c5', linewidth=1.5, label='Our static contact + sensor response')
        axis.set(title=title, xlabel='Time (s)', ylabel='Force (N)', xlim=(-.05, 2))
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8, ncol=2)
    figure.savefig(output / 'Comparison.png', dpi=150)
    plt.close(figure)
    for state in states['cases']:
        if abs(sum(state['forces_n']) - 2.5) > 1e-8:
            raise RuntimeError('Static contact does not balance the published load')
    expected = [{0, 6, 8}, {0, 3, 8}]
    for state, active in zip(states['cases'], expected):
        if {i for i, force in enumerate(state['forces_n']) if force > 1e-5} != active:
            raise RuntimeError('Published active-contact identity mismatch')
    report['static_identity_and_load_checks_passed'] = True
    (output / 'Metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    cases = {'cases': [{'title': 'Local force during contact loss and return',
                        'figure': relpath(output / 'Comparison.png', ROOT),
                        'notes': 'Measured traces and the published fitted sensor response remain separate from our contact-force predictions. '
                                 'The normal-contact model excludes lateral groove-edge impacts.'}]}
    (output / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    if args.keep_diagnostics:
        for i, (time, force) in enumerate(paths):
            np.savetxt(output / f'figure14_path{i}.csv', np.column_stack([time, force]), delimiter=',', header='time_s,force_n', comments='')
    else:
        (output / 'sensor.csv').unlink()
        (output / 'static.json').unlink()
        for path in output.glob('figure14_path*.csv'):
            path.unlink()
    print(f'Published force comparison: {output / "Comparison.png"}')
    for panel, name in enumerate(names):
        subset = report['comparisons'][4 * panel:4 * panel + 4]
        print(name, 'relative RMS:', ', '.join(f'{r["full_band"]["relative_rms_error"]:.3f}' for r in subset))


if __name__ == '__main__':
    main()
