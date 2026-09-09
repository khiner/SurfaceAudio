#!/usr/bin/env python3
"""Prepare disjoint fit/holdout data from Conan's lossy companion recordings."""
import json
from pathlib import Path
import warnings
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, correlation_lags, fftconvolve, find_peaks, peak_widths

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / 'references/conan'
OUTPUT = ROOT / 'outputs/reproduction/conan'
NAMES = {'physical': 'ForceModelePhy', 'force1': 'ForceOrig', 'force2': 'ForceOrig2'}


def read(name):
    rate, samples = wavfile.read(REFERENCES / (name + '.decoded.wav'))
    assert rate == 44100 and samples.ndim == 1
    return rate, samples.astype(np.float64)


def impacts(samples, rate, threshold):
    # Decoded MP3 ringing creates hundreds of false local maxima near zero.
    peaks, _ = find_peaks(samples, height=threshold, prominence=.4 * threshold, distance=10)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        widths = peak_widths(samples, peaks, rel_height=.5)[0]
    curvature = samples[peaks - 1] - 2 * samples[peaks] + samples[peaks + 1]
    offset = .5 * (samples[peaks - 1] - samples[peaks + 1]) / curvature
    amplitude = samples[peaks] - .25 * (samples[peaks - 1] - samples[peaks + 1]) * offset
    duration = 2 * widths / rate
    # Small, weak peaks are retained for event statistics, but not duration fitting.
    duration[(amplitude < 3 * threshold) | (duration < .0003) | (duration > .004)] = 0
    return np.column_stack(((peaks + offset) / rate, amplitude, duration))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = {'sample_rate': 44100, 'cases': {}, 'ir_diagnostics': {}}
    for key, name in NAMES.items():
        rate, samples = read(name)
        # Whole-record activity determines the crop and midpoint before the disjoint fit/holdout split.
        active = np.flatnonzero(samples > .05 * samples.max())
        start, end = max(0, active[0] - 44), min(len(samples), active[-1] + 44)
        midpoint = (start + end) // 2
        # Detector floor fixed using training data only. Never tune against holdout.
        threshold = .05 * samples[start:midpoint].max()
        train = impacts(samples[start:midpoint], rate, threshold)
        full = impacts(samples[start:end], rate, threshold)
        np.savetxt(OUTPUT / f'{key}-full.csv', full, delimiter=',', fmt='%.12g')
        holdout = impacts(samples[midpoint:end], rate, threshold)
        np.savetxt(OUTPUT / f'{key}-train.csv', train, delimiter=',', fmt='%.12g')
        np.savetxt(OUTPUT / f'{key}-holdout.csv', holdout, delimiter=',', fmt='%.12g')
        wavfile.write(OUTPUT / f'{key}-holdout.wav', rate, samples[midpoint:end].astype(np.float32))
        wavfile.write(OUTPUT / f'{key}-author.wav', rate, samples[start:end].astype(np.float32))
        metadata['cases'][key] = {'source': name, 'start_sample': int(start), 'split_sample': int(midpoint), 'end_sample': int(end),
                                  'training_events': len(train), 'holdout_events': len(holdout), 'detection_threshold': float(threshold),
                                  'training_seconds': (midpoint-start)/rate, 'holdout_seconds': (end-midpoint)/rate}
    rate, force = read('ForceModelePhy')
    for index in (1, 2):
        _, response = read(f'ir{index}')
        _, reference = read(f'ForceConv{index}')
        convolution = fftconvolve(force, response)
        split = len(reference) // 2
        correlations = correlate(convolution[:split + len(response)], reference[:split], method='fft')
        lag = int(correlation_lags(split + len(response), split)[np.argmax(np.abs(correlations))])
        assert lag >= 0
        aligned = convolution[lag:lag + len(reference)]
        split = len(reference) // 2
        # Alignment and scalar gain are channel identification, not force synthesis.
        # Fit gain on first half; report reconstruction error on the second half.
        gain = float(np.sum(aligned[:split] * reference[:split]) / np.sum(aligned[:split] ** 2))
        holdout_error = float(np.linalg.norm(gain * aligned[split:] - reference[split:]) / np.linalg.norm(reference[split:]))
        trimmed = gain * response[lag:]
        wavfile.write(OUTPUT / f'ir{index}-calibrated.wav', rate, trimmed.astype(np.float32))
        wavfile.write(OUTPUT / f'ir{index}-author.wav', rate, reference.astype(np.float32))
        wavfile.write(OUTPUT / f'ir{index}-oracle.wav', rate, (gain * aligned).astype(np.float32))
        metadata['ir_diagnostics'][str(index)] = {'removed_samples': lag, 'gain_fit_first_half': gain,
            'discarded_response_energy_fraction': float(np.sum(response[:lag] ** 2) / np.sum(response ** 2)),
            'full_correlation': float(np.corrcoef(aligned, reference)[0, 1]), 'holdout_relative_rms_error': holdout_error,
            'alignment_fit': 'Delay and scalar gain fitted on first half only.'}
    (OUTPUT / 'inputs.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
