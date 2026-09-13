#!/usr/bin/env python3
"""Render rough-contact models and compare their physical outputs with published figures."""
import argparse
import hashlib
import io
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
from PIL import Image
from scipy.signal import correlate, welch

from AnalyzeRenders import read_wave, texture_metrics
from reference_assets import ROOT, fetch, require_packages


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def spectrum_trace(document, source):
    data = document.extract_image(document[source['page_index']].get_images()[source['image_index']][0])['image']
    if hashlib.sha256(data).hexdigest() != source['image_sha256']:
        raise RuntimeError('Published spectrum image hash mismatch')
    red, green, blue = np.moveaxis(np.array(Image.open(io.BytesIO(data)).convert('RGB'), dtype=float), 2, 0)
    mask = (blue > 90) & (blue - red > 60) & (blue - green > 40)
    x0, x1, y0, y1 = source['axes_pixels']
    low, high = source['ordinate_db']
    f0, f1 = np.log10(source['frequency_hz'])
    points = []
    for x in range(x0 + 1, x1):
        y = np.flatnonzero(mask[y0 + 1:y1, x]) + y0 + 1
        if len(y):
            levels = high - (np.array([np.median(y), y.max(), y.min()]) - y0) * (high - low) / (y1 - y0)
            points.append([10 ** (f0 + (x - x0) * (f1 - f0) / (x1 - x0)), *levels])
    if len(points) != source['columns']:
        raise RuntimeError('Published spectrum pixel extraction changed')
    return np.array(points).T


def spectral_summary(frequency, db):
    grid = np.geomspace(100, 10000, 241)
    values = np.interp(np.log10(grid), np.log10(frequency), db)
    slope = np.polyfit(np.log10(grid[grid >= 1000]), values[grid >= 1000], 1)[0]
    return {'slope_1_to_10khz_db_per_decade': float(slope)}, float(np.mean(values[grid <= 1000]))


def assemien(output, metadata, paper, keep):
    import pymupdf
    document = pymupdf.open(paper)
    rate, audio = read_wave(output / 'vibration.wav')
    velocity = audio[:, 0] / metadata['audio_gain']
    reference = json.loads((ROOT / 'repros/rough/Assemien.json').read_text())
    if metadata['speed_m_s'] != .1 or metadata['target_rq_m'] != 17e-6 or metadata['contact_law'] != {'K': 1e16, 'chi': 3e17, 'n': 1.5, 'm': 1.5}:
        raise ValueError('The published Assemien comparison requires the Appendix 4 contact law, Rq 17 micrometres and 0.1 m/s')
    if rate != 44100 or abs(len(velocity) / rate - metadata['duration_s']) > 1 / rate:
        raise RuntimeError('Assemien WAV duration or sampling rate differs from the native metrics')
    if len(velocity) < rate:
        raise ValueError('The published spectral comparison requires at least one second')
    level = lambda samples: float(20 * np.log10(np.sqrt(np.mean(samples ** 2)) / reference['velocity_reference_m_s']))
    report = {'reference': reference, 'wav_sha256': hashlib.sha256((output / 'vibration.wav').read_bytes()).hexdigest(),
              'receiver_level_db': level(velocity), 'analysis_interval_s': [.5, len(velocity) / rate],
              'analysis_level_db': level(velocity[rate // 2:]), 'texture': texture_metrics(rate, audio), 'spectra': [],
              'limits': ['Figures contain raster spectra rather than raw measurements or author simulation samples.',
                         'Spectral overlays subtract each curve\'s mean dB value over 100–1000 Hz; absolute receiver level is compared separately.',
                         'The native analysis uses 0.5 s to the recording end; the author stationary interval and simulated PSD settings are unavailable.',
                         'The measured plate is freely suspended; the simulated plate is simply supported.',
                         'The published contact coefficients were calibrated against these levels and spectra.']}
    bending = 210e9 * .004 ** 3 / (12 * (1 - .3 ** 2))
    frequencies = np.sort([np.pi / 2 * np.sqrt(bending / (7800 * .004)) * ((p / .6) ** 2 + (q / .4) ** 2)
                           for p in range(1, 201) for q in range(1, 201)])
    report['plate_mode_frequency_check'] = {
        'equations': 'Equations 3.7 and 3.12, with the Appendix 4 dimensions and material properties.',
        'mode_indices': reference['table3_9_mode_indices'], 'printed_hz': reference['table3_9_frequencies_hz'],
        'complete_ssss_basis_hz': [float(frequencies[k - 1]) for k in reference['table3_9_mode_indices']],
        'modes_at_or_below_printed_maximum': int(np.count_nonzero(frequencies <= 13510.63))}
    figure, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    for source in reference['figures']:
        frequency, db, lower, upper = spectrum_trace(document, source)
        result, offset = spectral_summary(frequency, db)
        report['spectra'].append({'label': source['label'], **result})
        line, = axes[0].semilogx(frequency, db - offset, label=source['label'], linewidth=1)
        axes[0].fill_between(frequency, lower - offset, upper - offset, color=line.get_color(), alpha=.12)
        if keep:
            np.savetxt(output / f'figure{source["page_index"] + 1}_spectrum.csv', np.column_stack([frequency, db, lower, upper]),
                       delimiter=',', header='frequency_hz,median_db,lower_db,upper_db', comments='')
    for window, style in [(256, '--'), (8192, '-')]:
        frequency, power = welch(velocity[rate // 2:], rate, nperseg=window)
        db = 10 * np.log10(np.maximum(power[1:], 1e-40))
        result, offset = spectral_summary(frequency[1:], db)
        label = f'Our simulation, Hann {window}'
        report['spectra'].append({'label': label, **result, 'window_samples': window, 'overlap_samples': window // 2})
        axes[0].semilogx(frequency[1:], db - offset, style, linewidth=.8, label=label)
    axes[0].set(xlim=(100, 20000), ylim=(-75, 35), xlabel='Frequency (Hz)', ylabel='Relative spectral level (dB)')
    axes[0].legend(fontsize=8)
    block = rate // 10
    chunks = velocity[:len(velocity) // block * block].reshape(-1, block)
    times = (np.arange(len(chunks)) + .5) * block / rate
    levels = 20 * np.log10(np.maximum(np.sqrt(np.mean(chunks ** 2, axis=1)), 1e-30) / reference['velocity_reference_m_s'])
    axes[1].plot(times, levels, label='Our receiver, 100 ms RMS')
    axes[1].axhspan(*reference['measured_level_range_db'], alpha=.15, color='black', label='Reported measured level range')
    axes[1].axhline(reference['simulated_level_db'], color='tab:orange', linestyle='--', label='Reported simulation level, Table 3.5')
    axes[1].set(xlabel='Time (s)', ylabel='Velocity level (dB re 1 nm/s)')
    axes[1].legend(fontsize=8)
    axes[2].plot(np.arange(len(velocity)) / rate, velocity * 1000, linewidth=.3)
    axes[2].set(xlabel='Time (s)', ylabel='Receiver velocity (mm/s)')
    for axis in axes:
        axis.grid(alpha=.2)
    figure.savefig(output / 'Comparison.png', dpi=150)
    plt.close(figure)
    if not keep:
        for path in output.glob('figure*_spectrum.csv'):
            path.unlink()
    write_json(output / 'Comparison.json', report)
    return {'title': 'Rough slider on a plate · 100 mm/s', 'synthesis': relpath(output / 'vibration.wav', ROOT),
            'synthesis_label': 'Our simulated receiver velocity', 'figure': relpath(output / 'Comparison.png', ROOT),
            'notes': 'The plots compare published measured and simulated spectra and vibration levels. Author WAVs are unavailable. '
                     'Playback is peak normalized surface velocity; physical levels use the recorded inverse gain.'}


def event_summary(events):
    count = events['completed']
    histograms = events['histograms']
    if not count or sum(histograms['force_n']['counts']) != count or sum(histograms['duration_s']['counts']) != count:
        raise RuntimeError('Incomplete contact-event histograms')
    if sum(histograms['positive_work_j']['counts']) + sum(histograms['negative_work_magnitude_j']['counts']) + events['zero_work_events'] != count:
        raise RuntimeError('Signed-work histograms do not cover every complete event')
    return {'complete_events': count, 'left_censored': events['left_censored'], 'right_censored': events['right_censored'],
            'total_nodal_work_j': events['total_nodal_work_j'],
            'force_below_threshold_fraction': [value / count for value in events['force_below_threshold']],
            'duration_below_100us_fraction': events['duration_below_100us'] / count}


def histogram_edges(histogram):
    return np.logspace(histogram['log10_minimum'], histogram['log10_maximum'], histogram['interior_bins'] + 1)


def histogram_extent(histogram):
    edges = histogram_edges(histogram)
    occupied = np.flatnonzero(histogram['counts'])
    if not len(occupied):
        return edges[0], edges[0]
    return edges[max(0, occupied[0] - 1)], edges[min(len(edges) - 1, occupied[-1])]


def dang_paths(reference, archive, y_sign=-1):
    with tarfile.open(archive) as source:
        data = source.extractfile(reference['member']).read()
    if hashlib.sha256(data).hexdigest() != reference['member_sha256']:
        raise RuntimeError('Published Dang figure hash mismatch')
    paths = re.findall(rf'q 1 0 0 {y_sign} 0 [\d.]+ cm\s+(.*?)S Q', data.decode(), re.S)
    if len(paths) != reference['path_count']:
        raise RuntimeError('Published Dang figure paths changed')
    return paths


def dang_modal_audit(reference, archive):
    source = reference['modal_figure']
    begin, end = source['marker_path_range']
    paths = dang_paths(source, archive, y_sign=1)[begin:end]
    points = [np.array(re.findall(r'([-\d.]+)\s+([-\d.]+)\s+[ml]', path), dtype=float) for path in paths]
    if len(points) != len(source['mode_indices']) or any(p.shape != (2, 2) or p[0, 0] != p[1, 0] for p in points):
        raise RuntimeError('Published resonance marker geometry changed')
    x0, f0, x1, f1 = source['frequency_axis']
    frequencies = f0 + (np.array([p[0, 0] for p in points]) - x0) * (f1 - f0) / (x1 - x0)
    factor = np.pi / 2 * np.sqrt(source['young_pa'] * source['thickness_m'] ** 2 / (12 * source['density_kg_m3']))
    hypotheses = []
    for length in source['beam_lengths_m']:
        predicted = factor * np.array(source['mode_indices']) ** 2 / length ** 2
        hypotheses.append({'length_m': length, 'frequencies_hz': predicted.tolist(),
                           'marker_relative_error': (frequencies / predicted - 1).tolist(),
                           'table7_mode_periods_s': (length ** 2 / (factor * np.array(source['table7_mode_indices']) ** 2)).tolist()})
    return {'reference': source, 'marker_frequencies_hz': frequencies.tolist(), 'geometry_comparisons': hypotheses,
            'limit': 'The dashed lines identify theoretical resonances; they are not measured eigenfrequency samples.'}


def dang_levels(reference, archive):
    paths = dang_paths(reference, archive)
    x0, v0, x1, v1 = reference['speed_axis']
    y0, level0, y1, level1 = reference['level_axis']
    rows = []
    for group, label in enumerate(reference['roughness_labels_um']):
        for i, speed in enumerate(reference['speeds_m_s']):
            values = np.array(re.findall(r'[-+]?\d+(?:\.\d+)?', paths[reference['marker_start'] + 7 * group + i]), float)
            if group < 3:
                x, y = values[:[6, 8, 4][group]].reshape(-1, 2).mean(axis=0)
            elif group == 3:
                x, y = values[:2] + values[2:4] / 2
            else:
                points = values.reshape(-1, 2)
                x, y = (points.min(axis=0) + points.max(axis=0)) / 2
            decoded_speed = v0 * (v1 / v0) ** ((x - x0) / (x1 - x0))
            if abs(np.log(decoded_speed / speed)) > 2e-4:
                raise RuntimeError('Published marker speed differs from its expected axis position')
            rows.append({'ra_label_um': label, 'speed_m_s': speed, 'level_db': level0 + (y - y0) * (level1 - level0) / (y1 - y0)})
    return rows


def dang_profile_audit(reference, archive):
    source = reference['profile_figure']
    paths = dang_paths(source, archive, 1)
    first, end = source['profile_path_range']
    points = np.concatenate([np.array(re.findall(r'([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+[ml]', path), float)
                             for path in paths[first:end]])
    points = points[np.argsort(points[:, 0], kind='stable')]
    x, index, count = np.unique(points[:, 0], return_index=True, return_counts=True)
    x0, low, x1, high = source['x_axis']
    if len(points) != source['vertices'] or len(x) != source['distinct_x'] or max(abs(x[0] - x0), abs(x[-1] - x1)) > 1e-3:
        raise RuntimeError('Published Dang profile coordinates changed')
    means = np.add.reduceat(points[:, 1], index) / count
    estimates = []
    for reduction, heights in [('first', points[index, 1]), ('mean', means), ('last', points[index + count - 1, 1])]:
        for samples in [1024, 2048, 4096]:
            values = np.interp(np.linspace(x0, x1, samples), x, heights)
            values -= values.mean()
            acf = correlate(values, values, mode='full', method='fft')[samples - 1:]
            acf /= acf[0]
            crossing = np.flatnonzero(acf < np.exp(-1))
            if not len(crossing):
                raise RuntimeError('Published profile has no first 1/e correlation crossing')
            i = crossing[0]
            lag = i - 1 + (acf[i - 1] - np.exp(-1)) / (acf[i - 1] - acf[i])
            estimates.append({'repeated_x_reduction': reduction, 'resampled_points': samples,
                              'first_1e_crossing_um': float(lag * (high - low) / (samples - 1))})
    bounds = [min(row['first_1e_crossing_um'] for row in estimates), max(row['first_1e_crossing_um'] for row in estimates)]
    return {'source': source, 'printed_extent_um': high - low, 'correlation_range_under_printed_axis_um': bounds,
            'method': 'Mean-subtracted finite-record autocorrelation, normalized by zero lag; linear interpolation of the first 1/e crossing.',
            'estimates': estimates,
            'limit': 'The plotted profile and tabulated correlation length disagree under the printed axis; original units and sampling remain unresolved.'}


def dang_profile_generation(metadata):
    kind = metadata.get('profile_generation', 'unspecified')
    note = {'exact_period': 'Surfaces use the supplied generator’s exact requested periodic grid.',
            'power_of_two_period_then_crop': 'These retained results use enlarged periodic surfaces cropped to the beam grids; the current generator uses exact periods.'}.get(
                kind, 'The surface-generation convention is unrecorded; these results do not establish exact-period reproduction.')
    return kind, note


def dang_sweep(output, reference, archive, seed=None):
    published = dang_levels(reference, archive)
    data = np.atleast_1d(np.genfromtxt(output / 'Sweep.csv', delimiter=',', names=True))
    if seed is None:
        summary = json.loads((output / 'Sweep.json').read_text())
        if not summary['full_record_batch_repeatability'] or abs(summary['duration_s'] - .45) > 1e-12:
            raise ValueError('The sweep requires a completed full-duration repeatability check')
    else:
        data = data[data['seed'] == seed]
        summary = {'cases': 42, 'seeds': 1, 'duration_s': .45, 'scope': 'single_realization', 'full_record_batch_repeatability': None}
    generation, generation_note = dang_profile_generation(summary)
    seeds = sorted(set(data['seed']))
    labels, speeds, windows = [3, 5, 8, 10, 20, 30], reference['speeds_m_s'], [0., .1, .2]
    cases = len(seeds) * len(labels) * len(speeds)
    if len(seeds) < (3 if seed is None else 1) or summary['seeds'] != len(seeds) or summary['cases'] != cases or len(data) != cases * len(windows):
        raise ValueError('The selected data must contain complete roughness/speed/window grids and the required seed count')
    if any(not np.isfinite(data[name]).all() for name in data.dtype.names) or np.any(data['space_time_velocity_rms_m_s'] <= 0):
        raise ValueError('Sweep metrics contain nonfinite or nonpositive vibration levels')
    keys = ['window_start_s', 'seed', 'ra_label_um', 'speed_m_s']
    ordered = data[np.lexsort(tuple(data[key] for key in reversed(keys)))]
    expected = np.stack(np.meshgrid(windows, seeds, labels, speeds, indexing='ij'), axis=-1).reshape(-1, len(keys))
    if not np.array_equal(np.column_stack([ordered[key] for key in keys]), expected) or np.any(abs(ordered['duration_s'] - .45) > 1e-12):
        raise ValueError('Sweep has duplicate, missing or mismatched-duration cases')
    cube = (20 * np.log10(ordered['space_time_velocity_rms_m_s'] / 1e-9)).reshape(len(windows), len(seeds), len(labels), len(speeds))
    report = {'reference': reference, 'published_points': published, 'run': summary, 'seeds': seeds, 'duration_s': .45, 'windows': [],
              'configuration_status': reference['configuration_status'],
              'profile_generation': generation,
              'csv_sha256': hashlib.sha256((output / 'Sweep.csv').read_bytes()).hexdigest(),
              'limits': [*reference['configuration_limits'], generation_note,
                         'Each roughness realization is shared across sliding speeds; random innovations are paired across roughness labels.',
                         'All runs retain the published 5 micrometre grid and 0.1 microsecond time step.',
                         'The common 0.45 s duration preserves full overlap; the paper gives one second in prose and five seconds in Table 7.',
                         'Window sensitivity and variation across seeds are reported separately.' if len(seeds) > 1 else
                         'One surface realization provides no estimate of variation across seeds; full-record batch repeatability is not certified by this analysis.']}
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    figure.suptitle(f'{len(seeds)} realization' + ('s' if len(seeds) > 1 else '') + ' per roughness level' + (f' · seed {seed}' if seed is not None else ''))
    for w, begin in enumerate(windows):
        speed_slopes = np.polyfit(np.log10(speeds), cube[w].reshape(-1, len(speeds)).T, 1)[0].reshape(len(seeds), len(labels)) / 20
        roughness_slopes = np.polyfit(np.log10(labels), cube[w].transpose(0, 2, 1).reshape(-1, len(labels)).T, 1)[0].reshape(len(seeds), len(speeds)) / 20
        mean = cube[w].mean(axis=0)
        differences = [mean[labels.index(row['ra_label_um']), speeds.index(row['speed_m_s'])] - row['level_db'] for row in published]
        report['windows'].append({'start_s': begin, 'end_s': .45, 'levels_db_by_seed_roughness_speed': cube[w].tolist(),
                                  'speed_exponents_by_seed_roughness': speed_slopes.tolist(),
                                  'roughness_exponents_by_seed_speed': roughness_slopes.tolist(),
                                  'mean_level_difference_from_published_db': float(np.mean(differences)),
                                  'level_rmse_from_published_db': float(np.sqrt(np.mean(np.square(differences)))),
                                  'mean_normal_force_range_n': [float(f(data['mean_normal_force_n'][data['window_start_s'] == begin])) for f in (np.min, np.max)]})
        axes[1, 0].errorbar(labels, speed_slopes.mean(axis=0), yerr=speed_slopes.std(axis=0, ddof=1) if len(seeds) > 1 else None, marker='.', label=f'{begin:g}–0.45 s')
        axes[1, 1].errorbar(speeds, roughness_slopes.mean(axis=0), yerr=roughness_slopes.std(axis=0, ddof=1) if len(seeds) > 1 else None, marker='.', label=f'{begin:g}–0.45 s')
    for r, label in enumerate(labels):
        line, = axes[0, 0].semilogx(speeds, cube[1, :, r].mean(axis=0), marker='.', label=f'Ra{label}')
        axes[0, 0].fill_between(speeds, cube[1, :, r].min(axis=0), cube[1, :, r].max(axis=0), color=line.get_color(), alpha=.12)
        points = [p for p in published if p['ra_label_um'] == label]
        axes[0, 0].plot([p['speed_m_s'] for p in points], [p['level_db'] for p in points], '--', color=line.get_color(), linewidth=.8)
    for v, speed in enumerate(speeds):
        axes[0, 1].semilogx(labels, cube[1, :, :, v].mean(axis=0), marker='.', label=f'{speed:g} m/s')
    axes[0, 0].set(title='Solid: our 0.1–0.45 s mean/range; dashed: published' if len(seeds) > 1 else 'Solid: our 0.1–0.45 s values; dashed: published', xlabel='Sliding speed (m/s)', ylabel='Spatial velocity level (dB re 1 nm/s)')
    axes[0, 1].set(title='Our 0.1–0.45 s ensemble mean' if len(seeds) > 1 else 'Our 0.1–0.45 s single realization', xlabel='Roughness label (µm)', ylabel='Spatial velocity level (dB re 1 nm/s)')
    axes[1, 0].plot(reference['roughness_labels_um'], reference['speed_exponents'], 'kx', label='Published fitted exponents')
    axes[1, 0].set(xscale='log', xlabel='Roughness label (µm)', ylabel='Speed exponent n')
    axes[1, 1].axhspan(*reference['roughness_exponent_range'], color='black', alpha=.12, label='Published exponent range')
    axes[1, 1].set(xscale='log', xlabel='Sliding speed (m/s)', ylabel='Roughness exponent m')
    for axis in axes.flat:
        axis.grid(alpha=.2)
        axis.legend(fontsize=7, ncol=2)
    figure.savefig(output / 'Scaling.png', dpi=150)
    plt.close(figure)
    write_json(output / 'Scaling.json', report)
    return {'title': 'Roughness and sliding-speed scaling' + (' · one realization per roughness level' if len(seeds) == 1 else ''), 'figure': relpath(output / 'Scaling.png', ROOT),
            'notes': generation_note + f' Six roughness levels, seven speeds and {len(seeds)} realization' + ('s' if len(seeds) > 1 else '') + ' per roughness level. '
                     'Levels use a physical 1 nm/s reference; the lower plots show analysis-window sensitivity. '
                     + ('Error bars show standard deviations across seeds.' if len(seeds) > 1 else 'Variation across seeds remains unmeasured.')}


def dang(output, metadata, reference, archive):
    if metadata['speed_m_s'] != .7 or metadata['ra_label_um'] != 5 or metadata['duration_s'] > metadata['full_overlap_end_s']:
        raise ValueError('The local Dang comparison fixture requires Ra5 at 0.7 m/s and a recording within full overlap')
    populations = metadata['contact_events']
    generation, generation_note = dang_profile_generation(metadata)
    event_rate = metadata['event_sample_rate_hz']
    cadence = f'{event_rate / 1e6:g} MHz' if event_rate >= 1e6 else f'{event_rate / 1e3:g} kHz'
    rate, audio = read_wave(output / 'vibration.wav')
    if rate != 44100 or abs(len(audio) / rate - metadata['duration_s']) > 1 / rate:
        raise RuntimeError('Dang WAV duration or sampling rate differs from the native metrics')
    report = {'doi': '10.1007/s00466-013-0870-7', 'reference': 'Figure 12 and accompanying text; Ra5, 0.7 m/s.',
              'configuration_status': reference['configuration_status'],
              'profile_generation': generation,
              'event_sample_rate_hz': event_rate, 'thesis_event_sample_rate_hz': 100000,
              'surface_profile_audit': dang_profile_audit(reference, archive), 'modal_figure_audit': dang_modal_audit(reference, archive),
              'bodies': {body: event_summary(populations[body]) for body in ['resonator', 'slider']},
              'force_thresholds_n': [.78, 7.8, 78], 'published_force_below_threshold_approximate': [.30, .57, 1.],
              'published_duration_below_100us_lower_bound': .90, 'texture': texture_metrics(rate, audio),
              'limits': [*reference['configuration_limits'], generation_note,
                         'Resonator and slider nodes are reported separately; the paper does not clearly specify its event-node population.',
                         f'Our event recording rate is {cadence}; the thesis specifies 100 kHz. Brief contacts depend on the recording cadence.',
                         'Each nodal force includes its slave force and all interpolated master reactions.',
                         'Our slider weight is 7.6518 N from the thesis width convention; the paper cites 0.78 N.',
                         'The recording includes initial contact and ends before the slider leaves full overlap.',
                         'Exact author roughness arrays and seeds are unavailable.']}
    figure, rows = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    figure.suptitle(f'Contact statistics · {cadence} recording; thesis: 100 kHz')
    extent = {key: np.array([histogram_extent(body['histograms'][key]) for body in populations.values()])
              for key in ['force_n', 'duration_s', 'positive_work_j', 'negative_work_magnitude_j']}
    force_limits = min(extent['force_n'][:, 0].min(), .78) / 1.2, max(extent['force_n'][:, 1].max(), 78) * 1.2
    duration_limits = min(extent['duration_s'][:, 0].min(), 1e-4) / 1.2, max(extent['duration_s'][:, 1].max(), 1e-4) * 1.2
    work_limit = 1.2 * max(extent['positive_work_j'][:, 1].max(), extent['negative_work_magnitude_j'][:, 1].max(), 1e-12)
    for body, axes in zip(['resonator', 'slider'], rows):
        events = populations[body]
        count, histograms = events['completed'], events['histograms']
        for axis, key, label in zip(axes, ['force_n', 'duration_s'], ['Peak nodal force (N)', 'Contact duration (s)']):
            histogram = histograms[key]
            axis.step(histogram_edges(histogram), np.cumsum(histogram['counts'][:-1]) / count, where='post', label=body.capitalize())
            axis.set(xscale='log', xlabel=label, ylabel='Fraction of complete events')
        positive, negative = histograms['positive_work_j'], histograms['negative_work_magnitude_j']
        edges = histogram_edges(positive)
        negative_cdf = np.cumsum(negative['counts'][::-1][:-1]) / count
        zero_cdf = (sum(negative['counts']) + events['zero_work_events']) / count
        positive_cdf = zero_cdf + np.cumsum(positive['counts'][:-1]) / count
        axes[2].step(np.r_[-edges[::-1], 0, edges], np.r_[negative_cdf, zero_cdf, positive_cdf], where='post', label=body.capitalize())
        axes[2].set(xscale='symlog', xlabel='Signed nodal work (J)', ylabel='Fraction of complete events')
        axes[2].set_xscale('symlog', linthresh=1e-12)
        axes[2].xaxis.get_major_locator().set_params(numticks=7)
        axes[0].set_xlim(*force_limits)
        axes[1].set_xlim(*duration_limits)
        axes[2].set_xlim(-work_limit, work_limit)
        axes[0].plot([.78, 7.8, 78], [.30, .57, 1.], 'kx', label='Published approximate fractions')
        axes[1].plot([1e-4], [.9], 'kx', label='Published lower bound')
        for axis in axes:
            axis.set(ylim=(0, 1.02), title=body.capitalize())
            axis.legend(fontsize=7)
            axis.grid(alpha=.2)
    figure.savefig(output / 'Comparison.png', dpi=150)
    plt.close(figure)
    write_json(output / 'Comparison.json', report)
    return {'title': 'Rough beams · local Ra5 fixture · 0.7 m/s', 'synthesis': relpath(output / 'vibration.wav', ROOT),
            'synthesis_label': 'Our simulated beam velocity', 'figure': relpath(output / 'Comparison.png', ROOT),
            'notes': generation_note + ' Local fixture combining journal parameters, a separate thesis example and inferred settings. '
                     f'The plot uses {cadence} contact snapshots; the thesis specifies 100 kHz. '
                     'It compares each body’s complete nodal-event statistics with the published approximate thresholds. '
                     'The paper’s cited slider weight differs from the mass implied by the thesis input convention. Author WAVs are unavailable.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['dang', 'assemien', 'all'], default='all')
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/roughReproduce')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/reproduction/rough')
    parser.add_argument('--paper', type=Path, help='Use a local copy of the pinned Assemien thesis')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--analyze-only', action='store_true')
    parser.add_argument('--sweep', action='store_true', help='Run and compare the 126-case Dang roughness/speed ensemble')
    parser.add_argument('--scaling-seed', type=int, help='Analyze one complete seed from Sweep.csv; requires --analyze-only')
    parser.add_argument('--keep-diagnostics', action='store_true')
    args = parser.parse_args()
    if args.sweep and args.method == 'assemien':
        parser.error('--sweep requires dang or all')
    if args.scaling_seed is not None and (not args.analyze_only or args.sweep or args.method == 'assemien'):
        parser.error('--scaling-seed requires --analyze-only and dang or all, without --sweep')
    methods = ['dang', 'assemien'] if args.method == 'all' else [args.method]
    if 'assemien' in methods:
        require_packages(['PyMuPDF'])
    prepared = []
    for method in methods:
        output = args.output.resolve() / method
        reference = json.loads((ROOT / 'repros/rough' / ('Dang.json' if method == 'dang' else 'Assemien.json')).read_text())
        if method == 'assemien':
            source = fetch(args.paper or ROOT / 'references/rough/assemien2023.pdf', reference['url'], reference['sha256'], args.offline)
        else:
            source = fetch(ROOT / 'references/rough/1310.5252.tar', reference['archive_url'], reference['archive_sha256'], args.offline)
        prepared.append((method, output, reference, source))
    for method, output, reference, source in prepared:
        if not args.analyze_only:
            parameters = (['dang-sweep', str(output), '0.45', '3'] if args.sweep else ['dang', str(output), '0.45', '0.7', '5', '2013']) if method == 'dang' else ['assemien', str(output), '2', '0.1', '2023']
            subprocess.run([str(args.binary.resolve()), *parameters], check=True)
        metadata = json.loads((output / 'Metrics.json').read_text())
        if method == 'assemien':
            case = assemien(output, metadata, source, args.keep_diagnostics)
        else:
            case = dang(output, metadata, reference, source)
        cases = [case]
        if method == 'dang' and (args.sweep or args.scaling_seed is not None):
            cases.append(dang_sweep(output, reference, source, args.scaling_seed))
        write_json(output / 'cases.json', {'cases': cases})
        print(f'{method}: {output / "Comparison.json"}')


if __name__ == '__main__':
    main()
