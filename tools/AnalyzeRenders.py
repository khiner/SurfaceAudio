"""Shared WAV loading and complete-record metrics for listening reports."""
import struct

import numpy as np
from scipy import ndimage, signal

def read_wave(path):
    data = path.read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"Invalid RIFF WAV: {path}")
    offset, fmt, payload = 12, None, None
    while offset + 8 <= len(data):
        tag, size = struct.unpack_from("<4sI", data, offset)
        chunk = data[offset + 8:offset + 8 + size]
        if len(chunk) != size:
            raise ValueError(f"Truncated chunk: {path}")
        if tag == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", chunk)
            if fmt[0] == 0xfffe:
                if len(chunk) < 40 or struct.unpack_from("<H", chunk, 18)[0] != fmt[-1]:
                    raise ValueError(f"Unsupported extensible WAV: {path}")
                fmt = (struct.unpack_from("<H", chunk, 24)[0], *fmt[1:])
        elif tag == b"data":
            payload = chunk
        offset += 8 + size + size % 2
    if fmt is None or payload is None:
        raise ValueError(f"Missing WAV chunks: {path}")
    encoding, channels, rate, _, _, bits = fmt
    if (encoding, bits) == (3, 32):
        samples = np.frombuffer(payload, dtype="<f4").astype(np.float64)
    elif (encoding, bits) == (1, 16):
        samples = np.frombuffer(payload, dtype="<i2").astype(np.float64) / 32768
    elif (encoding, bits) == (1, 32):
        samples = np.frombuffer(payload, dtype="<i4").astype(np.float64) / 2147483648
    elif (encoding, bits) == (1, 24):
        triplets = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        packed = triplets[:, 0] | (triplets[:, 1] << 8) | (triplets[:, 2] << 16)
        samples = ((packed ^ 0x800000) - 0x800000).astype(np.float64) / 8388608
    else:
        raise ValueError(f"Unsupported WAV encoding: {path}")
    samples = samples.reshape(-1, channels)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError(f"Empty or nonfinite WAV: {path}")
    return rate, samples

def metrics(rate, channels):
    samples = channels[:, 0]
    mean = float(np.mean(samples))
    rms = float(np.sqrt(np.mean(samples * samples)))
    ac = samples - mean
    # Whole-record energy spectrum retains onset transients at either boundary.
    power = np.abs(np.fft.rfft(ac)).astype(np.float64) ** 2
    power[1:-1 if len(ac) % 2 == 0 else None] *= 2
    frequencies = np.fft.rfftfreq(len(ac), 1 / rate)
    total = float(power.sum())
    return {
        "sample_rate": rate, "frames": len(samples), "channels": channels.shape[1],
        "duration_seconds": len(samples) / rate, "mean": mean, "rms": rms,
        "ac_rms": float(np.sqrt(np.mean(ac * ac))),
        "absolute_mean_over_rms": abs(mean) / rms if rms else 0,
        "peak": float(np.max(np.abs(samples))),
        "ac_spectral_centroid_hz": float(np.sum(frequencies * power) / total) if total else 0,
        "power_fraction_above_15khz": float(power[frequencies > 15000].sum() / total) if total else 0,
    }


def texture_metrics(rate, channels):
    """Level-independent texture descriptors; comparisons require matching event windows."""
    samples = np.asarray(channels, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[:, None]
    if not np.isfinite(rate) or rate <= 0 or samples.ndim != 2 or not samples.shape[1] or len(samples) < 2 or not np.isfinite(samples).all():
        raise ValueError('Require finite mono/stereo audio and a positive sample rate')
    samples = samples - samples.mean(axis=0)
    rms = np.sqrt(np.mean(samples * samples))
    normalized = samples / rms if rms else samples
    frequency, power = signal.welch(normalized, rate, nperseg=min(8192, len(samples)), axis=0)
    power = power.mean(axis=1)
    total = max(float(power.sum()), 1e-30)
    audible = power[frequency >= 20]
    bands = [(20, 80), (80, 250), (250, 1000), (1000, 4000), (4000, 10000), (10000, rate / 2)]
    window = max(1, round(rate * .01))
    count = len(samples) // window
    envelope = np.sqrt(np.mean(normalized[:count * window].reshape(count, window, -1) ** 2, axis=(1, 2))) if count else np.empty(0)
    centered = envelope - envelope.mean() if count else envelope
    env_rate = rate / window
    modulation = {}
    if count >= 2:
        mf, mp = signal.welch(centered, env_rate, nperseg=min(512, count))
        for low, high in [(0.5, 2), (2, 8), (8, 32)]:
            modulation[f'{low:g}-{high:g}'] = float(mp[(mf >= low) & (mf < high)].sum() / max(float(mp.sum()), 1e-30))
    autocorrelation = {}
    for milliseconds in [10, 50, 100, 250]:
        lag = max(1, round(milliseconds * env_rate / 1000))
        norm = np.linalg.norm(centered[:-lag]) * np.linalg.norm(centered[lag:])
        autocorrelation[str(milliseconds)] = float(np.dot(centered[:-lag], centered[lag:]) / norm) if count > lag and norm else None
    trend = ndimage.gaussian_filter1d(envelope, .2 * env_rate, mode='reflect') if count else envelope
    relative = envelope / np.maximum(trend, max(float(envelope.max()) * .01, 1e-30)) - 1 if count and rms else np.zeros_like(envelope)
    concentration = []
    if len(samples) >= 2048:
        _, _, spectrum = signal.spectrogram(normalized, rate, window='hann', nperseg=2048, noverlap=1536, scaling='spectrum', axis=0)
        spectrum = spectrum.mean(axis=1)
        energy = spectrum.sum(axis=0)
        active = energy > max(float(energy.max()) * 1e-4, 1e-30)
        concentration = np.partition(spectrum[:, active], -3, axis=0)[-3:].sum(axis=0) / energy[active]
    return {
        'duration_seconds': len(samples) / rate,
        'channel_aggregation': 'Mean channel power with per-channel DC removal',
        'psd_band_fraction': {f'{low:g}-{high:g}': float(power[(frequency >= low) & (frequency < high)].sum() / total) for low, high in bands if low < high},
        'spectral_flatness': float(np.exp(np.log(np.maximum(audible, 1e-30)).mean()) / max(float(audible.mean()), 1e-30)) if len(audible) and rms else 0.,
        'spectral_peak_fraction': float(power.max() / total),
        'frame_top3_bin_fraction_median': float(np.median(concentration)) if len(concentration) else None,
        'amplitude_kurtosis': float(np.mean(normalized ** 4)),
        'absolute_amplitude_quantiles': np.quantile(np.abs(normalized), [.5, .9, .99]).tolist(),
        'envelope_cv': float(envelope.std() / max(float(envelope.mean()), 1e-30)) if count else 0.,
        'local_envelope_rms': float(np.sqrt(np.mean(relative ** 2))) if count else 0.,
        'envelope_autocorrelation_ms': autocorrelation,
        'envelope_modulation_fraction_hz': modulation,
    }
