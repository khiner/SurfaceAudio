#!/usr/bin/env python3
"""Compare fixed-seed Conan syntheses against disjoint author force holdouts."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch, fftconvolve
from scipy.stats import ks_2samp
from conan_prepare import ROOT, OUTPUT, REFERENCES, NAMES, impacts, read

SOURCE = 'https://kronland.fr/publications/a-synthesis-model-with-intuitive-control-capabilities-for-rolling-sounds/'


def load(path):
    rate, samples = wavfile.read(path)
    assert rate == 44100 and samples.ndim == 1
    return samples.astype(np.float64)


def event_statistics(events):
    amplitude, intervals = events[:-1, 1], np.diff(events[:, 0])
    durations = events[:, 2]
    return {'events': len(events), 'amplitude_mean': float(amplitude.mean()), 'amplitude_sd': float(amplitude.std()),
            'interval_mean_seconds': float(intervals.mean()), 'interval_sd_seconds': float(intervals.std()),
            'amplitude_interval_correlation': float(np.corrcoef(amplitude, intervals)[0, 1]),
            'amplitude_lag1': float(np.corrcoef(amplitude[:-1], amplitude[1:])[0, 1]),
            'interval_lag1': float(np.corrcoef(intervals[:-1], intervals[1:])[0, 1]),
            'duration_median_seconds': float(np.median(durations[durations > 0]))}


def band_power(samples):
    frequency, power = welch(samples, 44100, nperseg=8192)
    edges = np.geomspace(30, 14000, 55)
    bands = np.array([np.sum(power[(frequency >= left) & (frequency < right)]) for left, right in zip(edges[:-1], edges[1:])])
    return bands / max(bands.sum(), 1e-30)


def signal_statistics(samples):
    ac = samples - samples.mean()
    envelope = np.sqrt(np.mean(ac[:len(ac) // 441 * 441].reshape(-1, 441) ** 2, axis=1))
    frequency, power = welch(ac, 44100, nperseg=8192)
    return {'dc': float(samples.mean()), 'rms': float(np.sqrt(np.mean(samples ** 2))),
            'ac_rms': float(np.sqrt(np.mean(ac ** 2))), 'envelope_cv_10ms': float(envelope.std() / envelope.mean()),
            'spectral_centroid_hz': float(np.sum(frequency * power) / power.sum()),
            'power_above_2khz_fraction': float(power[frequency > 2000].sum() / power.sum())}


def spectral_distance(first, second):
    a, b = band_power(first), band_power(second)
    active = a > a.max() * 1e-4
    return {'normalized_band_total_variation': float(.5 * np.abs(a - b).sum()),
            'band_db_rmse_40db_reference_range': float(np.sqrt(np.mean((10 * np.log10(np.maximum(a[active], 1e-15) / np.maximum(b[active], 1e-15))) ** 2)))}


def compare_events(reference, candidate):
    return {'amplitude_ks': float(ks_2samp(reference[:, 1], candidate[:, 1]).statistic),
            'interval_ks': float(ks_2samp(np.diff(reference[:, 0]), np.diff(candidate[:, 0])).statistic)}


def retain_case_audio(output):
    cases = json.loads((output / "cases.json").read_text())["cases"]
    retained = {(ROOT / case[field]).resolve() for case in cases for field in ("reference", "synthesis")}
    for path in output.glob("*.wav"):
        if path.resolve() not in retained:
            path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-diagnostics", action="store_true", help="Retain trial WAVs after successful analysis")
    args = parser.parse_args()
    inputs = json.loads((OUTPUT / 'inputs.json').read_text())
    parameters = json.loads((OUTPUT / 'parameters.json').read_text())
    result = {'inputs': inputs, 'force': {}, 'audio': {}}
    cases = []
    for key, data in inputs['cases'].items():
        train = np.loadtxt(OUTPUT / f'{key}-train.csv', delimiter=',')
        holdout = np.loadtxt(OUTPUT / f'{key}-holdout.csv', delimiter=',')
        reference = load(OUTPUT / f'{key}-holdout.wav')
        record = {'training': event_statistics(train), 'holdout': event_statistics(holdout),
                  'training_holdout_distance': compare_events(holdout, train), 'holdout_signal': signal_statistics(reference), 'variants': {}}
        # The pulse-shape diagnostic uses measured held-out peak times/amplitudes,
        # but only the training duration law. It is not a new stochastic synthesis.
        model = parameters['cases'][key + '-gaussian']
        reconstructed = np.zeros(len(reference))
        for time, amplitude, _ in holdout:
            duration = model['duration_scale'] * amplitude ** (-model['duration_exponent'])
            left, right = max(0, int((time - duration / 2) * 44100)), min(len(reference), int((time + duration / 2) * 44100) + 1)
            t = np.arange(left, right) / 44100 - time
            reconstructed[left:right] += amplitude * np.where(np.abs(t) <= duration / 2, .5 * (1 + np.cos(2 * np.pi * t / duration)), 0)
        record['heldout_peak_pulse_reconstruction_relative_rms'] = float(np.linalg.norm(reconstructed - reference) / np.linalg.norm(reference))
        wavfile.write(OUTPUT / f'{key}-pulse-diagnostic.wav', 44100, reconstructed.astype(np.float32))
        for kind in ('gaussian', 'empirical'):
            name = key + '-' + kind
            records = []
            for seed in range(8):
                samples = load(OUTPUT / f'{name}-seed{seed}.wav')
                events = impacts(samples, 44100, data['detection_threshold'])
                records.append({'seed': 2014 + seed, 'events': event_statistics(events), 'signal': signal_statistics(samples),
                                'event_distance': compare_events(holdout, events), 'spectrum_distance': spectral_distance(reference, samples)})
            record['variants'][kind] = records
        full_reference = load(OUTPUT / f'{key}-author.wav')
        full_events = np.loadtxt(OUTPUT / f'{key}-full.csv', delimiter=',')
        record['full_record_fit'] = {'reference': event_statistics(full_events), 'variants': {}}
        for kind in ('gaussian', 'empirical'):
            records = []
            for seed in range(8):
                samples = load(OUTPUT / f'{key}-full-{kind}-seed{seed}.wav')
                events = impacts(samples, 44100, data['detection_threshold'])
                records.append({'seed': 2014 + seed, 'events': event_statistics(events), 'signal': signal_statistics(samples),
                                'event_distance': compare_events(full_events, events), 'spectrum_distance': spectral_distance(full_reference, samples)})
            record['full_record_fit']['variants'][kind] = records
        if key != 'physical':
            _, author_synth = read('ForceSynth' + ('2' if key == 'force2' else ''))
            author_events = impacts(author_synth, 44100, data['detection_threshold'])
            record['author_synthesis'] = {'events': event_statistics(author_events), 'signal': signal_statistics(author_synth),
                                         'event_distance': compare_events(holdout, author_events), 'spectrum_distance': spectral_distance(reference, author_synth)}
        cases.append({'title': f'Conan {NAMES[key]}: full-record calibrated force',
                      'reference': str((OUTPUT / f'{key}-author.wav').relative_to(ROOT)),
                      'synthesis': str((OUTPUT / f'{key}-full-gaussian-seed0.wav').relative_to(ROOT)),
                      'notes': 'Descriptive calibration uses all source events. New Gaussian shared innovations with fixed seed 2014; no replay of measured events. Paper exponent 0.29 and fitted pulse scale. This same-record statistical reconstruction is separate from the held-out predictive comparisons.', 'source_url': SOURCE})
        if key != 'physical':
            author_name = 'ForceSynth' + ('2' if key == 'force2' else '')
            cases.append({'title': f'Conan author proposed-scheme {author_name}: independent new synthesis',
                          'reference': str((REFERENCES / f'{author_name}.decoded.wav').relative_to(ROOT)),
                          'synthesis': str((OUTPUT / f'{key}-full-gaussian-seed0.wav').relative_to(ROOT)),
                          'notes': 'Author proposed-scheme demo compared with our new seed 2014 calibrated on its paired physics-based ForceOrig recording. Exact author fit parameters, random seed, and MP3 export gain are unpublished. Level matching is available; raw files preserve their scales.', 'source_url': SOURCE})
        result['force'][key] = record
        cases.append({'title': f'Conan {NAMES[key]}: held-out force vs new Gaussian synthesis',
                      'reference': str((OUTPUT / f'{key}-holdout.wav').relative_to(ROOT)),
                      'synthesis': str((OUTPUT / f'{key}-gaussian-seed0.wav').relative_to(ROOT)),
                      'notes': 'First half of the decoded author force estimates ARMA coefficients, innovation scales and pulse scale. The reference here is its disjoint second half. New fixed seed 2014, paper exponent 0.29, no modulation. Record halves have different event statistics. MP3-derived amplitudes have unknown physical units.', 'source_url': SOURCE})
    for index in (1, 2):
        reference = load(OUTPUT / f'ir{index}-author.wav')
        for kind in ('gaussian', 'empirical'):
            name = f'physical-{kind}-ir{index}'
            candidate = load(OUTPUT / f'{name}.wav')
            result['audio'][name] = {'reference': signal_statistics(reference), 'synthesis': signal_statistics(candidate[:4 * 44100]),
                                     'spectrum_distance': spectral_distance(reference, candidate[:4 * 44100])}
            if kind == 'gaussian':
                cases.append({'title': f'Conan author response {index}: new calibrated rolling force',
                    'reference': str((OUTPUT / f'ir{index}-author.wav').relative_to(ROOT)),
                    'synthesis': str((OUTPUT / f'{name}.wav').relative_to(ROOT)),
                    'notes': 'New Gaussian ARMA force, seed 2014, calibrated on the first half of ForceModelePhy. Convolved on Metal with the author response; response delay and gain identified on the first half of the published source/filter pair. Includes the complete response tail. Different random events prevent pointwise matching.', 'source_url': SOURCE})
        for kind in ('gaussian', 'empirical'):
            name = f'physical-full-{kind}-ir{index}'
            candidate = load(OUTPUT / f'{name}.wav')
            result['audio'][name] = {'reference': signal_statistics(reference), 'synthesis': signal_statistics(candidate[:4 * 44100]),
                                     'spectrum_distance': spectral_distance(reference, candidate[:4 * 44100])}
            if kind == 'gaussian':
                cases.append({'title': f'Conan author response {index}: full-record calibrated reproduction',
                    'reference': str((OUTPUT / f'ir{index}-author.wav').relative_to(ROOT)),
                    'synthesis': str((OUTPUT / f'{name}.wav').relative_to(ROOT)),
                    'notes': 'Final descriptive fit uses all ForceModelePhy events. New seed 2014, shared Gaussian ARMA innovations, paper exponent 0.29, no modulation. Author IR delay and gain identified on the first half of the published source/filter pair. This is an in-sample statistical reconstruction, not independent predictive validation. Complete convolution tail retained.', 'source_url': SOURCE})
        default = load(OUTPUT / f'paper-default-ir{index}.wav')
        result['audio'][f'paper-default-ir{index}'] = {'spectrum_distance': spectral_distance(reference, default[:4 * 44100])}
    for name, record in result['audio'].items():
        index = name[-1]
        force_name = name.rsplit('-ir', 1)[0]
        source = load(OUTPUT / f'{force_name}-seed0.wav')[:4 * 44100]
        response = load(OUTPUT / f'ir{index}-calibrated.wav')
        oracle = fftconvolve(source, response)
        actual = load(OUTPUT / f'{name}.wav')
        record['gpu_convolution_relative_rms_error'] = float(np.linalg.norm(actual - oracle) / np.linalg.norm(oracle))
        record['gpu_convolution_max_error'] = float(np.max(np.abs(actual - oracle)))
    checks = {}
    for key, data in result['audio'].items():
        checks[f'{key}_full_tail_gpu_convolution'] = data['gpu_convolution_relative_rms_error'] < 1e-5
    for key, data in inputs['ir_diagnostics'].items():
        checks[f'author_ir{key}_source_filter_correlation'] = data['full_correlation'] > .98
    for key, data in result['force'].items():
        checks[f'{key}_heldout_raised_cosine_relative_rms'] = data['heldout_peak_pulse_reconstruction_relative_rms'] < .12
        trials = data['full_record_fit']['variants']['gaussian']
        checks[f'{key}_full_fit_amplitude_ks'] = float(np.mean([trial['event_distance']['amplitude_ks'] for trial in trials])) < .15
        checks[f'{key}_full_fit_interval_ks'] = float(np.mean([trial['event_distance']['interval_ks'] for trial in trials])) < .16
        checks[f'{key}_full_fit_spectral_tv'] = float(np.mean([trial['spectrum_distance']['normalized_band_total_variation'] for trial in trials])) < .18
    for index in (1, 2):
        fitted = result['audio'][f'physical-full-gaussian-ir{index}']['spectrum_distance']['normalized_band_total_variation']
        generic = result['audio'][f'paper-default-ir{index}']['spectrum_distance']['normalized_band_total_variation']
        checks[f'ir{index}_calibration_improves_spectral_distribution'] = fitted < generic
    result['regression_checks'] = checks
    (OUTPUT / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (OUTPUT / 'cases.json').write_text(json.dumps({'cases': cases}, indent=2) + '\n')
    if not all(checks.values()):
        raise RuntimeError('Published-record regression failed: ' + ', '.join(key for key, passed in checks.items() if not passed))
    for key, record in result['force'].items():
        print(key, 'train A/dt', record['training']['amplitude_mean'], record['training']['interval_mean_seconds'],
              'holdout A/dt', record['holdout']['amplitude_mean'], record['holdout']['interval_mean_seconds'],
              'pulse relative RMS', record['heldout_peak_pulse_reconstruction_relative_rms'])
        for kind, records in record['variants'].items():
            print(kind, 'mean KS amplitude/interval', np.mean([r['event_distance']['amplitude_ks'] for r in records]), np.mean([r['event_distance']['interval_ks'] for r in records]),
                  'mean spectral TV/dB', np.mean([r['spectrum_distance']['normalized_band_total_variation'] for r in records]), np.mean([r['spectrum_distance']['band_db_rmse_40db_reference_range'] for r in records]))
    for key, record in result['audio'].items():
        print(key, record['spectrum_distance'])
    if not args.keep_diagnostics:
        retain_case_audio(OUTPUT)


if __name__ == '__main__':
    main()
