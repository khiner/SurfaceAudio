"""Check spectral-motion diagnostics against controlled moving and stationary spectra."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from AnalyzeRenders import spectral_motion_comparison, reconstruction_metrics

rate = 44100
time = np.arange(3 * rate) / rate
frequency = 900 + 600 * np.sin(2 * np.pi * time / 3)
phase = 2 * np.pi * np.cumsum(frequency) / rate
moving = np.sin(phase) + .3 * np.sin(2 * phase)
rng = np.random.default_rng(2010)
spectrum = np.fft.rfft(moving)
stationary = np.fft.irfft(abs(spectrum) * np.exp(1j * rng.uniform(-np.pi, np.pi, len(spectrum))), n=len(moving))
identity = spectral_motion_comparison(rate, moving, moving)
scaled = spectral_motion_comparison(rate, moving, .001 * moving)
wrong = spectral_motion_comparison(rate, moving, stationary)
reversed_motion = spectral_motion_comparison(rate, moving, moving[::-1])
silent = spectral_motion_comparison(rate, np.zeros_like(moving), np.zeros_like(moving))
for key in ('spectral_shape_similarity', 'short_cepstrum_similarity'):
    assert abs(identity[key] - 1) < 1e-12
    assert abs(scaled[key] - 1) < 1e-12
    assert wrong[key] < .2, (key, wrong)
    assert reversed_motion[key] < .2, (key, reversed_motion)
    assert silent[key] == 0
reference = np.ones(rate)
ring = .1 * np.sin(2 * np.pi * 3000 * np.arange(rate) / rate)
late_ring = reconstruction_metrics(rate, reference, np.concatenate([reference, np.zeros(rate // 2), ring]))
no_tail = reconstruction_metrics(rate, reference, reference)
partial_tail = reconstruction_metrics(rate, reference, np.concatenate([reference, np.full(7, .25)]))
assert abs(late_ring['tail_peak_rms_over_reference'] - .1 / np.sqrt(2)) < 1e-12
assert late_ring['output_tail_energy_fraction'] > .004
assert no_tail['tail_peak_rms_over_reference'] == no_tail['output_tail_energy_fraction'] == 0
assert partial_tail['tail_peak_rms_over_reference'] == .25
print('Spectral-motion and delayed-tail checks passed')
