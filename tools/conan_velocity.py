#!/usr/bin/env python3
"""Reproduce velocity examples using author EPS coefficients and separately estimated missing inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.request
import zipfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter

from conan_prepare import ROOT, impacts
from conan_analyze import event_statistics, compare_events, signal_statistics, spectral_distance

BASE = 'https://www.lma.cnrs-mrs.fr/~kronland/Rolling_Sounds/'
REFERENCES = ROOT / 'references/conan/original-site'
OUTPUT = ROOT / 'outputs/reproduction/conan/velocity'
ARCHIVES = {
    'content/img/VeloAnalyse.zip': '95ba720ad40b8a1a8cc634512b3db45949fb8aaa9e3854f5946285387313686b',
    'content/snd/VeloAnalyseSnd.zip': 'abc4ecd5806c465b5b327c9ac314578d2dd8f331aceb5a911729d059f34162b9',
}
# Explicit EPS axis limits and original PostScript tick coordinates, not raster estimates.
AXES = {
    'aAmpli': (874, 6084, 4490, 396, 0, 100, -.96, -.925),
    'aDt': (874, 6084, 4490, 396, 0, 100, -.96, -.925),
    'bAmpli': (874, 6084, 4490, 396, 0, 100, .04, .24),
    'bDt': (874, 6084, 4490, 396, 0, 100, -.16, .04),
    'MuAmpli': (874, 6084, 4490, 396, 0, 100, .3, .46),
    'MuDt': (950, 6616, 4801, 405, 0, 100, 150, 195),
    'CAA': (874, 6084, 4490, 396, 0, 250, -20, 120),
    'CADt': (874, 6084, 4490, 396, 0, 250, -5000, 30000),
    'CDtDt': (874, 6084, 4490, 396, 0, 250, -2000000, 10000000),
}


def acquire(offline):
    records = []
    for name, expected in ARCHIVES.items():
        path = REFERENCES / name
        if not path.exists():
            if offline:
                raise RuntimeError(f'Missing pinned archive: {path}')
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(BASE + name, timeout=30) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError(f'Changed author archive: {BASE + name}')
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Changed local archive: {path}')
        members = []
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if not info.filename.endswith(('.eps', '.mp3')):
                    continue
                member = Path(info.filename)
                if member.is_absolute() or '..' in member.parts:
                    raise RuntimeError('Unsafe archive member')
                destination = REFERENCES / 'extracted' / member
                data = archive.read(info)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                members.append({'member': info.filename, 'path': str(destination.relative_to(ROOT)), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
        records.append({'url': BASE + name, 'path': str(path.relative_to(ROOT)), 'sha256': expected, 'members': members})
    (REFERENCES / 'velocity-manifest.json').write_text(json.dumps({'primary_page': BASE + 'AddMaterialVelocity.html', 'archives': records}, indent=2) + '\n')
    return records


def vertices(text, count):
    # MATLAB's /MP {3 1 roll moveto 1 sub {rlineto} repeat}: deltas
    # are consumed from the operand stack in reverse order after the starting point.
    assert '/MP {3 1 roll moveto 1 sub {rlineto} repeat}' in text
    result = []
    for block in re.findall(r'^([ \t]*(?:-?\d+\s+)+)MP stroke', text, re.M):
        values = np.array([int(value) for value in block.split()])
        if values[-1] != count:
            continue
        assert len(values) == 2 * count + 1
        start = values[-3:-1]
        deltas = values[:-3].reshape(-1, 2)[::-1]
        result.append(np.vstack([start, start + np.cumsum(deltas, axis=0)]))
    return result


def published_curves():
    curves = {}
    for name, axes in AXES.items():
        path = REFERENCES / 'extracted/VeloAnalyse' / (name + '.eps')
        text = path.read_text()
        count = 201 if name in ('CAA', 'CADt', 'CDtDt') else 10
        paths = vertices(text, count)
        assert len(paths) == (10 if count == 201 else 1), (name, len(paths))
        left, right, bottom, top, x_min, x_max, y_min, y_max = axes
        decoded = []
        for path_vertices in paths:
            x = x_min + (path_vertices[:, 0] - left) * (x_max - x_min) / (right - left)
            y = y_min + (path_vertices[:, 1] - bottom) * (y_max - y_min) / (top - bottom)
            if count == 10:
                assert np.max(np.abs(x - np.arange(10, 101, 10))) <= .03
            decoded.append({'postscript_vertices': path_vertices.tolist(), 'x': x.tolist(), 'y': y.tolist()})
        curves[name] = {'eps': str(path.relative_to(ROOT)), 'axes': dict(zip(('left', 'right', 'bottom', 'top', 'x_min', 'x_max', 'y_min', 'y_max'), axes)),
                        'one_coordinate_unit_y': (y_max - y_min) / abs(top - bottom), 'curves': decoded}
    (OUTPUT / 'published-curves.json').write_text(json.dumps(curves, indent=2) + '\n')
    # Independently read first vertices and axis ticks from the source EPS.
    for name, expected in (('aAmpli', -.96 + (4490 - 3139) * .035 / 4094),
                           ('bAmpli', .04 + (4490 - 638) * .20 / 4094),
                           ('MuDt', 150 + (4801 - 482) * 45 / 4396)):
        assert abs(curves[name]['curves'][0]['y'][0] - expected) < 1e-10
    return curves


def variance_factor(a1, b1):
    return (1 + b1 * b1 - 2 * a1 * b1) / (1 - a1 * a1)


def correlation_diagnostics(curves):
    records = []
    for index in range(10):
        a, b, c, d = (curves[name]['curves'][0]['y'][index] for name in ('aAmpli', 'bAmpli', 'aDt', 'bDt'))
        va, vt = variance_factor(a, b), variance_factor(c, d)
        aa, tt, at = (np.array(curves[name]['curves'][index]['y']) for name in ('CAA', 'CDtDt', 'CADt'))
        record = {'velocity_cm_s_assumed_curve_order': 10 * (index + 1),
                  'amplitude_lag1_eps': aa[101] / aa[100], 'amplitude_lag1_arma': -a + b / va,
                  'interval_lag1_eps': tt[101] / tt[100], 'interval_lag1_arma': -c + d / vt,
                  'cross_lag0_eps': at[100] / np.sqrt(aa[100] * tt[100]),
                  'cross_lag0_shared_innovation_arma': (1 + (b - a) * (d - c) / (1 - a * c)) / np.sqrt(va * vt)}
        # Independent author curves discriminate the equation's MA sign and curve order.
        assert abs(record['amplitude_lag1_eps'] - record['amplitude_lag1_arma']) < .002
        assert abs(record['interval_lag1_eps'] - record['interval_lag1_arma']) < .002
        records.append(record)
    (OUTPUT / 'correlation-checks.json').write_text(json.dumps(records, indent=2) + '\n')


def whitening(series, mean, a1, b1):
    residual = lfilter([1, a1], [1, b1], series - mean)[8:]
    return {'sigma': float(np.std(residual)), 'mean': float(residual.mean()),
            'lag1': float(np.corrcoef(residual[:-1], residual[1:])[0, 1])}


def prepare(curves, whole_record, eps_variance):
    table = {name: np.array(curve['curves'][0]['y']) for name, curve in curves.items() if name not in ('CAA', 'CADt', 'CDtDt')}
    rows, metadata = [], {'source': BASE + 'AddMaterialVelocity.html', 'whole_record_calibration': whole_record,
                          'innovation_estimator': 'eps_correlation_sum' if eps_variance else 'decoded_force_whitening', 'cases': {}}
    for index, velocity in enumerate(range(10, 101, 10)):
        name = f'velocity{velocity}'
        original = REFERENCES / f'extracted/VeloAnalyseSnd/Force_V={velocity}.mp3'
        decoded = original.with_suffix('.decoded.wav')
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(original), '-ac', '1', '-ar', '44100', '-c:a', 'pcm_f32le', '-y', str(decoded)], check=True)
        rate, raw = wavfile.read(decoded)
        raw = raw.astype(float)
        active = np.flatnonzero(raw > .05 * raw.max())
        start, end = max(0, active[0] - 44), min(len(raw), active[-1] + 44)
        split = (start + end) // 2
        calibration_end = end if whole_record else split
        threshold = .05 * raw[start:calibration_end].max()
        training = impacts(raw[start:calibration_end], rate, threshold)
        holdout = impacts(raw[split:end], rate, threshold)
        mean_a, mean_t = table['MuAmpli'][index], table['MuDt'][index] / rate
        amplitude_gain = mean_a / training[:-1, 1].mean()
        training[:, 1] *= amplitude_gain
        holdout[:, 1] *= amplitude_gain
        amplitude = whitening(training[:-1, 1], mean_a, table['aAmpli'][index], table['bAmpli'][index])
        interval = whitening(np.diff(training[:, 0]), mean_t, table['aDt'][index], table['bDt'][index])
        # Absolute correlation normalization is unpublished. This optional estimate
        # assumes centered correlation sums over the associated decoded record.
        estimated_intervals = (end - start) / table['MuDt'][index]
        eps_sigma_a = np.sqrt(curves['CAA']['curves'][index]['y'][100] / estimated_intervals
                              / variance_factor(table['aAmpli'][index], table['bAmpli'][index]))
        eps_sigma_t = np.sqrt(curves['CDtDt']['curves'][index]['y'][100] / estimated_intervals
                              / variance_factor(table['aDt'][index], table['bDt'][index])) / rate
        sigma_a, sigma_t = (eps_sigma_a, eps_sigma_t) if eps_variance else (amplitude['sigma'], interval['sigma'])
        measured = training[training[:, 2] > 0]
        predictor = measured[:, 1] ** -.29
        scale = float(np.sum(predictor * measured[:, 2]) / np.sum(predictor ** 2))
        values = [mean_a, sigma_a, table['aAmpli'][index], table['bAmpli'][index], mean_t, sigma_t, table['aDt'][index], table['bDt'][index], scale]
        rows.append(name + ' ' + ' '.join(f'{value:.12g}' for value in values))
        np.savetxt(OUTPUT / f'{name}-train.csv', training, delimiter=',', fmt='%.12g')
        np.savetxt(OUTPUT / f'{name}-holdout.csv', holdout, delimiter=',', fmt='%.12g')
        wavfile.write(OUTPUT / f'{name}-author.wav', rate, (amplitude_gain * raw[start:end]).astype(np.float32))
        wavfile.write(OUTPUT / f'{name}-holdout.wav', rate, (amplitude_gain * raw[split:end]).astype(np.float32))
        full_events = impacts(raw[start:end], rate, threshold)
        full_a = full_events[:-1, 1] * mean_a / full_events[:-1, 1].mean()
        full_t = np.diff(full_events[:, 0]) * rate
        metadata['cases'][name] = {'velocity_cm_s': velocity, 'start_sample': int(start), 'split_sample': int(split), 'end_sample': int(end),
            'published_mean_amplitude': mean_a, 'published_mean_interval_samples': table['MuDt'][index], 'published_a_amplitude': table['aAmpli'][index],
            'published_b_amplitude': table['bAmpli'][index], 'published_a_interval': table['aDt'][index], 'published_b_interval': table['bDt'][index],
            'estimated_amplitude_innovation': amplitude, 'estimated_interval_innovation': interval, 'estimated_duration_scale': scale,
            'eps_variance_estimate': {'estimated_intervals': estimated_intervals, 'detected_intervals': len(full_t),
                'amplitude_sigma': eps_sigma_a, 'interval_sigma': eps_sigma_t,
                'eps_over_decoded_amplitude_correlation_sum': curves['CAA']['curves'][index]['y'][100] / np.sum((full_a - full_a.mean()) ** 2),
                'eps_over_decoded_interval_correlation_sum': curves['CDtDt']['curves'][index]['y'][100] / np.sum((full_t - full_t.mean()) ** 2)},
            'training_amplitude_gain': amplitude_gain, 'detection_threshold_scaled': threshold * amplitude_gain,
            'training_events': len(training), 'holdout_events': len(holdout),
            'unscaled_training_amplitude_mean': mean_a / amplitude_gain, 'training_interval_mean_samples': float(np.diff(training[:, 0]).mean() * rate),
            'holdout_interval_mean_samples': float(np.diff(holdout[:, 0]).mean() * rate)}
    (OUTPUT / 'parameters.txt').write_text('\n'.join(rows) + '\n')
    (OUTPUT / 'inputs.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def analyze(metadata):
    whole_record = metadata['whole_record_calibration']
    scope = ('In-sample statistical reconstruction: author EPS coefficients/means; innovation variances inferred from plotted correlation sums divided by record-duration/mean-interval count. '
             'Centered unnormalized sums, associated-record duration, and ascending velocity curve order are assumptions. Pulse scale and unknown MP3 gain fitted over the whole decoded record. No independent holdout.'
             if whole_record else
             'Author EPS coefficients/means; innovation scales, pulse scale and unknown MP3 gain estimated from the first half only. The second half is excluded only from our missing-parameter fit. '
             'Authors may have fitted the published EPS parameters on the whole associated force record; their estimation window is unknown. Shared whole-record activity crop precedes our split.')
    result, cases = {'cases': {}, 'scope': scope}, []
    parameters = json.loads((OUTPUT / 'parameters.json').read_text())
    for name, inputs in metadata['cases'].items():
        training = np.loadtxt(OUTPUT / f'{name}-train.csv', delimiter=',')
        holdout = np.loadtxt(OUTPUT / f'{name}-holdout.csv', delimiter=',')
        comparison = training if whole_record else holdout
        reference_path = OUTPUT / f'{name}-{"author" if whole_record else "holdout"}.wav'
        _, reference = wavfile.read(reference_path)
        trials = []
        for seed in range(8):
            _, samples = wavfile.read(OUTPUT / f'{name}-seed{seed}.wav')
            events = impacts(samples.astype(float), 44100, inputs['detection_threshold_scaled'])
            trials.append({'seed': 2014 + seed, 'events': event_statistics(events), 'signal': signal_statistics(samples.astype(float)),
                'event_distance': compare_events(comparison, events), 'spectrum_distance': spectral_distance(reference.astype(float), samples.astype(float))})
        seed_checks = parameters['cases'][name]['seeds']
        event_count = sum(seed['events_including_warmup'] for seed in seed_checks)
        fractions = {field + '_fraction_including_warmup': sum(seed[field] for seed in seed_checks) / event_count
                     for field in ('amplitude_clamps', 'interval_clamps', 'duration_clamps', 'pulse_overflows')}
        result['cases'][name] = {'training': event_statistics(training), 'holdout': event_statistics(holdout), 'trials': trials,
            'guard_fractions': fractions,
            'gpu_checks': {field: parameters['cases'][name][field] for field in ('cpu_gpu_relative_rms_error', 'cpu_gpu_max_error', 'seeds')}}
        cases.append({'title': f'Conan published velocity {inputs["velocity_cm_s"]} cm/s: {"EPS variance reconstruction" if whole_record else "author-coefficient diagnostic"}',
            'reference': str(reference_path.relative_to(ROOT)), 'synthesis': str((OUTPUT / f'{name}-seed0.wav').relative_to(ROOT)),
            'notes': scope + f' New seed 2014, pulse exponent .29, no modulation. Amplitude/interval guard fractions over all seeds including warmup: {fractions["amplitude_clamps_fraction_including_warmup"]:.2%}/{fractions["interval_clamps_fraction_including_warmup"]:.2%}. Figures and recordings are velocity-labelled but not proven to use identical random realizations.',
            'source_url': BASE + 'AddMaterialVelocity.html'})
    (OUTPUT / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (OUTPUT / 'cases.json').write_text(json.dumps({'cases': cases}, indent=2) + '\n')
    for name, record in result['cases'].items():
        print(name, 'published/train/holdout interval(samples)', metadata['cases'][name]['published_mean_interval_samples'],
              metadata['cases'][name]['training_interval_mean_samples'], metadata['cases'][name]['holdout_interval_mean_samples'],
              'mean KS A/T', *(np.mean([trial['event_distance'][field] for trial in record['trials']]) for field in ('amplitude_ks', 'interval_ks')))


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/conanReproduce')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--fetch-only', action='store_true')
    parser.add_argument('--eps-variance', action='store_true', help='Reconstruct using assumed EPS correlation-sum normalization and full-record pulse/gain calibration; no independent holdout.')
    args = parser.parse_args()
    acquire(args.offline)
    if args.fetch_only:
        return
    if args.eps_variance:
        OUTPUT = OUTPUT.with_name('velocity-eps-variance')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    curves = published_curves()
    correlation_diagnostics(curves)
    metadata = prepare(curves, args.eps_variance, args.eps_variance)
    subprocess.run([str(args.binary), str(OUTPUT), '--parameters', str(OUTPUT / 'parameters.txt')], check=True)
    analyze(metadata)


if __name__ == '__main__':
    main()
