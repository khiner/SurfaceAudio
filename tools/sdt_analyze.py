#!/usr/bin/env python3
"""Report full-record audio and motion comparisons for sdtReproduce outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path, nargs='?', default=Path('outputs/reproduction/sdt'))
args = parser.parse_args()
records = {}
for name in ['impact', 'rolling', 'scraping', 'friction', 'friction_startup']:
    rate, upstream = wavfile.read(args.directory / (name + '_upstream.wav'))
    metal_rate, metal = wavfile.read(args.directory / (name + '_metal.wav'))
    assert rate == metal_rate and upstream.shape == metal.shape
    upstream, metal = upstream.astype(float), metal.astype(float)
    _, reference_power = welch(upstream - upstream.mean(), rate, nperseg=min(8192, len(upstream)))
    _, metal_power = welch(metal - metal.mean(), rate, nperseg=min(8192, len(metal)))
    trace = np.fromfile(args.directory / (name + '_upstream.f64'), dtype=np.float64).reshape(-1, 7)
    records[name] = {
        'upstream_peak': float(np.max(np.abs(upstream))),
        'upstream_ac_rms': float(np.std(upstream)),
        'upstream_late_ac_rms': float(np.std(upstream[min(rate, len(upstream)//2):])),
        'upstream_dc': float(np.mean(upstream)),
        'metal_to_upstream_ac_rms': float(np.std(metal)/np.std(upstream)) if np.std(upstream) else 1,
        'spectral_power_relative_l1': float(np.sum(np.abs(metal_power-reference_power))/np.sum(reference_power)) if np.sum(reference_power) else 0,
        'max_abs_velocity0': float(np.max(np.abs(trace[:, 4]))),
        'terminal_velocity0': float(trace[-1, 4]),
    }
path = args.directory / 'audio_metrics.json'
path.write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
print(path)
