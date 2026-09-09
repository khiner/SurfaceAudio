#!/usr/bin/env python3
"""Estimate spatial modal parameters from published ablations; render the full paper IR model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from scipy import signal, optimize
from scipy.ndimage import uniform_filter1d
from agarwal_reproduce import ROOT, wave, stats


def modal_peak_ranking(f, power, force_power=None, smoothing_bins=9, floor_ratio=1e-8):
    f, power = np.asarray(f), np.asarray(power)
    if f.ndim != 1 or f.shape != power.shape or f.size < 3 or not np.isfinite(f).all() or not np.isfinite(power).all() or np.any(np.diff(f) <= 0) or np.any(power < 0):
        raise ValueError('Modal ranking requires ordered frequencies and finite nonnegative power')
    active = (f > 80) & (f < 14000)
    if not np.any(active):
        raise ValueError('No frequencies in the modal analysis band')
    if force_power is None:
        score, prominence = power, power.max() * 1e-5
    else:
        force_power = np.asarray(force_power)
        if force_power.shape != power.shape or not np.isfinite(force_power).all() or np.any(force_power < 0) or int(smoothing_bins) != smoothing_bins or smoothing_bins < 1 or smoothing_bins % 2 != 1 or not np.isfinite(floor_ratio) or not 0 < floor_ratio <= 1:
            raise ValueError('Invalid force power, smoothing width or relative floor')
        smoothed = uniform_filter1d(force_power.astype(float), int(smoothing_bins))
        maximum = smoothed[active].max()
        if maximum <= 0:
            raise ValueError('Cannot rank response modes from silent in-band excitation')
        score = power / np.maximum(smoothed, maximum * floor_ratio)
        # Anti-alias stopband zeros outside the analysis band must not set its peak threshold.
        prominence = score[active].max() * 1e-5
    peaks, _ = signal.find_peaks(score, distance=5, prominence=prominence)
    peaks = peaks[active[peaks]]
    return peaks[np.argsort(score[peaks])[::-1]], score


def spatial_track_parameters(samples, morph, fs, frequencies, force=None, *, window=8192):
    """Historical tracker with an explicit sorted seed catalogue and auditable observations.

    Frequency regions use neighboring seed midpoints capped at +/-15 percent.
    The optional force affects only the historical amplitude-ratio estimate.
    Window defaults to8192; the hop remains1024 for explicit resolution sensitivities.
    """
    if not isinstance(window, (int, np.integer)) or window < 1024:
        raise ValueError("Track window must be an integer of at least1024 samples")
    samples, morph, frequencies = map(np.asarray, [samples, morph, frequencies])
    if samples.ndim != 1 or len(samples) < window or morph.shape != samples.shape or frequencies.ndim != 1 or not len(frequencies) or not np.isfinite(samples).all() or not np.isfinite(morph).all() or not np.isfinite(frequencies).all() or np.any(np.diff(frequencies) <= 0):
        raise ValueError('Require finite full-support samples, matching locations and sorted distinct seed frequencies')
    if not np.isfinite(fs) or fs <= 0 or np.any((frequencies <= 80) | (frequencies >= min(14000, fs / 2))):
        raise ValueError('Invalid track sample rate or seed frequency range')
    if force is not None:
        force = np.asarray(force)
        if force.shape != samples.shape or not np.isfinite(force).all():
            raise ValueError('Historical amplitude-ratio estimation requires a finite aligned force')
    endpoints = np.column_stack([frequencies, frequencies])
    amplitude_ratios = np.ones(len(frequencies))
    evidence, diagnostics = [], []
    ff, tt, z = signal.stft(samples - samples.mean(), fs, nperseg=window, noverlap=window-1024, boundary=None, padded=False)
    local_power = abs(z)**2
    force_power = None
    if force is not None:
        _, _, force_z = signal.stft(force - force.mean(), fs, nperseg=window, noverlap=window-1024, boundary=None, padded=False)
        force_power = uniform_filter1d(uniform_filter1d(abs(force_z)**2, 9, axis=0), 3, axis=1)
    locations = np.interp(tt*fs, np.arange(len(samples)), morph)
    for index, frequency in enumerate(frequencies):
        low = max(.85*frequency, (frequencies[index-1]+frequency)/2 if index else 80)
        high = min(1.15*frequency, (frequency+frequencies[index+1])/2 if index+1 < len(frequencies) else 14000)
        bins = np.flatnonzero((ff >= low) & (ff <= high))
        if len(bins) < 3:
            raise ValueError("Track region must contain at least three FFT bins")
        observed, positions, weights, log_amplitudes, columns = [], [], [], [], []
        for column, location in enumerate(locations):
            peak = bins[np.argmax(local_power[bins, column])]
            if peak == bins[0] or peak == bins[-1]:
                continue
            lp = np.log(np.maximum(local_power[peak-1:peak+2, column], 1e-30))
            curvature = lp[0]-2*lp[1]+lp[2]
            offset = .5*(lp[0]-lp[2])/curvature if curvature < 0 else 0
            columns.append(column)
            observed.append((peak+np.clip(offset, -.5, .5))*fs/window)
            positions.append(location)
            weights.append(local_power[peak, column])
            if force_power is not None:
                log_amplitudes.append(.5*np.log(local_power[peak, column]/max(force_power[peak, column], force_power.max()*1e-8)))
        if len(observed) >= 4 and np.ptp(positions) > .1:
            weight = np.sqrt(np.array(weights)/max(weights))
            design = np.column_stack([np.ones(len(positions)), positions])
            coefficients = np.linalg.lstsq(design*weight[:, None], np.log(observed)*weight, rcond=None)[0]
            endpoints[index] = np.clip(np.exp([coefficients[0], coefficients.sum()]), low, high)
            if force_power is not None:
                amplitude_fit = np.linalg.lstsq(design*weight[:, None], np.array(log_amplitudes)*weight, rcond=None)[0]
                amplitude_ratios[index] = np.exp(np.clip(amplitude_fit[1], -4, 4))
            evidence.append({'mode': index, 'frequency0': endpoints[index, 0], 'frequency1': endpoints[index, 1], 'amplitude_ratio': amplitude_ratios[index], 'windows': len(observed)})
        diagnostics.append({'mode': index, 'seed_frequency_hz': float(frequency), 'bounds_hz': [float(low), float(high)],
                            'frequency_endpoints_hz': endpoints[index].tolist(), 'accepted_windows': len(observed),
                            'fitted': len(observed) >= 4 and bool(np.ptp(positions) > .1),
                            'observations': [{'stft_column': int(column), 'center_seconds': float(tt[column]),
                                              'location': float(location), 'frequency_hz': float(observation), 'power_weight': float(power)}
                                             for column, location, observation, power in zip(columns, positions, observed, weights)]})
    return endpoints, amplitude_ratios, evidence, diagnostics


def modal_parameters(samples, morph, fs, train, spatial, force=None):
    samples = samples[:train]
    f, power = signal.welch(samples - samples.mean(), fs, nperseg=8192)
    ranked, _ = modal_peak_ranking(f, power)
    peaks = np.sort(ranked[:50])
    frequencies = f[peaks]
    widths = signal.peak_widths(power, peaks, rel_height=.5)[0]*fs/8192
    decays = np.clip(1/(np.pi*widths), .002, .08)
    endpoints = np.column_stack([frequencies, frequencies])
    amplitude_ratios = np.ones(len(frequencies))
    evidence = []
    if spatial:
        endpoints, amplitude_ratios, evidence, _ = spatial_track_parameters(samples, morph[:train], fs, frequencies, None if force is None else force[:train])
    return endpoints, decays, amplitude_ratios, evidence


def fit_amplitudes(basis, target, fs, train):
    _, _, z = signal.stft(basis[:, :train], fs, nperseg=4096, noverlap=2048, boundary=None, padded=False, scaling='psd', axis=-1)
    cross = np.einsum('mft,nft->fmn', z, z.conj(), optimize=True).real*(2/z.shape[-1])
    f, power = signal.welch(target[:train], fs, nperseg=4096, noverlap=2048, detrend=False)
    active = (f >= 80) & (f <= 14000)
    cross, power = cross[active], power[active]
    floor = power.max()*1e-8
    weights = np.sqrt(power/power.max())
    scale = np.sqrt(power.max()/np.maximum(np.max(np.trace(cross, axis1=1, axis2=2)), 1e-30))
    def residual(log_amplitudes):
        a = np.exp(log_amplitudes)
        predicted = np.einsum('i,fij,j->f', a, cross, a, optimize=True)
        return np.concatenate([.25*weights*np.log(np.maximum(predicted, floor)/np.maximum(power, floor)),
                               2*(np.sqrt(np.maximum(predicted, 0))-np.sqrt(power))/np.sqrt(power.max())])
    def jacobian(log_amplitudes):
        a = np.exp(log_amplitudes)
        qa = np.einsum('fij,j->fi', cross, a, optimize=True)
        predicted = np.sum(qa*a[None, :], axis=1)
        derivative = 2*qa*a[None, :]
        logarithmic = .25*weights[:, None]*derivative/np.maximum(predicted[:, None], floor)
        logarithmic[predicted <= floor] = 0
        amplitude = derivative/np.sqrt(np.maximum(predicted[:, None], 1e-30)*power.max())
        amplitude[predicted <= 0] = 0
        return np.vstack([logarithmic, amplitude])
    fit = optimize.least_squares(residual, np.full(len(basis), np.log(scale)), jac=jacobian,
                                 bounds=(np.log(scale)-15, np.log(scale)+15), max_nfev=300)
    return np.exp(fit.x), {'evaluations': fit.nfev, 'status': fit.status, 'optimality': fit.optimality, 'combined_objective_rms': float(np.sqrt(np.mean(fit.fun**2)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT/'build/agarwalSpatialReproduce')
    parser.add_argument('--case', choices=['roll', 'roll-glass'], default='roll-glass')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--whole-record', action='store_true')
    parser.add_argument('--object-modes', type=Path, help='Keep an independently fitted fixed ball response; frequency, decay seconds, amplitude columns')
    args = parser.parse_args()
    out = ROOT/'outputs/reproduction/agarwal'
    source = args.source or out/(args.case+'-temporal'+('-whole' if args.whole_record else '')+'-case')
    domain = json.loads((source/'smoothing.json').read_text())['domain']
    tag = source.name.removesuffix('-case').removesuffix('-whole')+'-spatial'+('-whole' if args.whole_record else '')
    directory = out/(tag+'-case')
    physical = np.loadtxt(source/'parameters.txt')
    fs, reference = wave(ROOT/f'references/agarwal/{args.case}-author.wav')
    frames, taps = int(physical[1]), int(physical[3])
    train = frames if args.whole_record else int(.75*frames)
    calibration_path = source/'calibration.json'
    source_calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else None
    force_note = source_calibration.get('force_scope', '') if source_calibration else ''
    if not args.whole_record:
        expected_reference = f'references/agarwal/{args.case}-author.wav'
        if not source_calibration or source_calibration.get('training_frames', frames) > train or source_calibration.get('total_frames') != frames or source_calibration.get('reference') != expected_reference:
            raise RuntimeError('Held-out reconstruction requires verified training-only source parameters; regenerate the source case or use --whole-record')
    coordinates_path = source/'coordinates.json'
    coordinates = json.loads(coordinates_path.read_text()) if coordinates_path.exists() else None
    shutil.copytree(source, directory, dirs_exist_ok=True)
    physical[13] = 1
    (directory/'parameters.txt').write_text(' '.join(str(int(x)) if i < 4 else str(x) for i, x in enumerate(physical))+'\n')
    morph = np.fromfile(directory/'morph.f32', dtype='<f4')
    material = 'wood' if args.case == 'roll' else 'glass'
    _, surface = wave(ROOT/f'references/agarwal/ablations/{material}-surface.wav')
    _, ball = wave(ROOT/f'references/agarwal/ablations/{material}-ball.wav')
    matrix = np.column_stack([surface, ball, np.ones(len(reference))])
    mixture = np.linalg.lstsq(matrix[:train], reference[:train], rcond=None)[0]
    force = np.fromfile(directory/'scrape.f32', dtype='<f4').astype(float)+physical[11]*np.fromfile(directory/'elastic.f32', dtype='<f4')+physical[12]*np.fromfile(directory/'damping.f32', dtype='<f4')
    frequencies, decays, amplitude_ratios, tracks = modal_parameters(surface, morph, fs, train, True, force)
    object_frequencies, object_decays, _, _ = modal_parameters(ball, morph, fs, train, False)
    object_input = None
    if args.object_modes:
        if not args.whole_record:
            raise RuntimeError('External object modes require explicit whole-record scope until their training provenance is checked')
        object_modes = np.loadtxt(args.object_modes, ndmin=2)
        if object_modes.shape[1] not in (3, 4) or not 1 <= len(object_modes) <= 50 or not np.isfinite(object_modes).all() or np.any(object_modes[:, :3] <= 0) or np.any(object_modes[:, 0] >= fs / 2) or mixture[1] <= 0:
            raise ValueError('Invalid fixed object modes or nonpositive author mixture gain')
        object_frequencies = np.column_stack([object_modes[:, 0], object_modes[:, 0]])
        object_decays = object_modes[:, 1]
        object_input = {'path': str(args.object_modes.resolve()), 'sha256': hashlib.sha256(args.object_modes.read_bytes()).hexdigest(),
                        'scope': 'Supplied fixed ball response; amplitudes scaled only by the full/surface/ball author mixture coefficient',
                        'mixture_gain': float(mixture[1])}
    lag = np.arange(taps)/fs
    minimum = np.finfo(np.float32).tiny
    envelopes = np.maximum(np.exp(-lag[None, :]/decays[:, None]), minimum)
    object_envelopes = np.maximum(np.exp(-lag[None, :]/object_decays[:, None]), minimum)
    for endpoint in [0, 1]:
        frequencies[:, endpoint].astype('<f8').tofile(directory/f'surface{endpoint}_frequencies.f64')
        np.maximum(envelopes*amplitude_ratios[:, None]**endpoint, minimum).astype('<f8').tofile(directory/f'surface{endpoint}_amplitudes.f64')
    object_frequencies[:, 0].astype('<f8').tofile(directory/'object_frequencies.f64')
    object_envelopes.astype('<f8').tofile(directory/'object_amplitudes.f64')
    prefix = out/tag
    subprocess.run([str(args.binary), str(directory), str(prefix), '--basis'], check=True)
    count = len(frequencies)+len(object_frequencies)
    basis = np.fromfile(str(prefix)+'-basis.f32', dtype='<f4').reshape(count, frames+taps-1).astype(float)
    surface_amplitudes, surface_fit = fit_amplitudes(basis[:len(frequencies)], mixture[0]*surface, fs, train)
    if object_input:
        object_amplitudes = object_modes[:, 2] * mixture[1]
        object_fit = {'fixed_external_modes': object_input}
    else:
        object_amplitudes, object_fit = fit_amplitudes(basis[len(frequencies):], mixture[1]*ball, fs, train)
    for endpoint in [0, 1]:
        np.maximum(envelopes*surface_amplitudes[:, None]*amplitude_ratios[:, None]**endpoint, minimum).astype('<f8').tofile(directory/f'surface{endpoint}_amplitudes.f64')
    np.maximum(object_envelopes*object_amplitudes[:, None], minimum).astype('<f8').tofile(directory/'object_amplitudes.f64')
    subprocess.run([str(args.binary), str(directory), str(prefix)], check=True)
    _, audio = wave(prefix.with_suffix('.wav'))
    predicted = np.sum(basis*np.concatenate([surface_amplitudes, object_amplitudes])[:, None], axis=0)
    basis_error = float(np.linalg.norm(audio-predicted)/np.linalg.norm(predicted))
    if not np.isfinite(basis_error) or basis_error > .0001:
        raise RuntimeError(f'Spatial summed render differs from individual modal basis: {basis_error}')
    result = {'scope': 'Estimated spatial mode tracks and separate object response from published ablation features. No reference samples enter force or IR. Original physical event inputs remain unknown.',
              'source_case': str(source), 'source_calibration': source_calibration, 'source_coordinates': coordinates, 'whole_record_fit': args.whole_record, 'surface_tracks': tracks,
              'force_scope': force_note, 'fixed_object_input': object_input,
              'surface_fit': surface_fit, 'object_fit': object_fit, 'summed_basis_relative_l2': basis_error,
              'training': stats(reference, audio, fs, 0, train), 'held_out': stats(reference, audio, fs, train, frames)}
    (out/(tag+'-metrics.json')).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    manifest = out/'cases.json'
    cases = json.loads(manifest.read_text())['cases'] if manifest.exists() else []
    synthesis = str(prefix.with_suffix('.wav').relative_to(ROOT))
    cases = [case for case in cases if case['synthesis'] != synthesis]
    fit_scope = 'Whole event used for calibration; no held-out accuracy claim.' if args.whole_record else 'First 75% used for calibration; final 25% excluded from parameter estimation.'
    coordinate_note = '' if not coordinates else f" Coordinate hypothesis: {coordinates['hypothesized_paper_coordinate_system']}; full measured profile: {coordinates['full_measured_profile']}. Author coordinate units remain unverified."
    cases.append({'title': f'Agarwal {source.name.removesuffix("-case")}: full spatial IR ('+('whole-record fit)' if args.whole_record else 'held-out fit)'),
                  'reference': f'references/agarwal/{args.case}-author.wav', 'synthesis': synthesis,
                  'notes': 'Author-associated texture drives the paper force equations. '+force_note+' Surface frequencies and amplitudes vary with output location; a separate fixed object response is added. Modal tracks and decays are estimated from published ablations, not recovered measured IRs. '+fit_scope+coordinate_note+(' Fixed ball modes come from a separate time-resolved contact fit and are retained here.' if object_input else ''),
                  'source_url': 'https://mcdermottlab.mit.edu/scraping_rolling.html'})
    manifest.write_text(json.dumps({'cases': cases}, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
