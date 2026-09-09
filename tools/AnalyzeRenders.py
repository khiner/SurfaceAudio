"""Shared WAV loading and complete-record metrics for listening reports."""
import struct

import numpy as np

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
