#!/usr/bin/env python3
"""Reproduce Conan CMJ2014 interactions using the author video, visible controls and an isolated object hit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

import numpy as np
from scipy import signal, optimize
from scipy.io import wavfile

from AnalyzeRenders import read_wave, metrics, texture_metrics

ROOT = Path(__file__).resolve().parents[1]
RATE = 44100
URL = 'https://kronland.fr/wp-content/uploads/2015/05/ConanEtAlCMJ2014.mp4'
PAGE = 'https://kronland.fr/publications/an-intuitive-synthesizer-of-continuous-interaction-sounds-rubbing-scratching-and-rolling/'
VIDEO_HASH = '0256b0f30f88e261553f6bfa84d6e3adcaf52790d8d6d5570e8306b4798e4bcb'
CASES = [('rubbing', 59, 65, 0), ('rolling', 68, 74, 4*np.pi/3), ('scratching', 78, 84, 2*np.pi/3), ('transition', 107, 117, None)]


def pin(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def video_controls(video):
    command = ['ffmpeg', '-v', 'error', '-i', str(video), '-vf', 'fps=20,crop=252:260:142:50,scale=126:130', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    data = subprocess.run(command, capture_output=True, check=True).stdout
    frames = np.frombuffer(data, np.uint8).reshape(-1, 130, 126, 3)
    cursor, level = [], []
    previous = np.array([269., 160.])
    missing = 0
    for frame in frames:
        orange = (frame[:, :, 0] > 140) & (frame[:, :, 0] > 1.15*frame[:, :, 1]) & (frame[:, :, 1] > 65) & (frame[:, :, 2] < 100)
        orange[110:] = False
        y, x = np.where(orange)
        if len(x):
            previous = np.array([np.median(x)*2 + 142, np.median(y)*2 + 50])
        else:
            missing += 1
        delta = previous - [269, 160]
        cursor.append([float(np.arctan2(-delta[1], delta[0]) % (2*np.pi)), float(min(np.linalg.norm(delta)/88, 1))])
        meter = frame[117:121]
        colored = (meter[:, :, 1] > 140) & (meter[:, :, 0] < 220) & (meter[:, :, 2] < 100)
        level.append(float(np.count_nonzero(np.any(colored, axis=0))/110))
    return np.asarray(cursor), np.asarray(level), {'command': command, 'frames_without_cursor': missing, 'sample_rate_hz': 20,
        'cursor': 'Orange centroid, action disk center269,160 radius88 pixels; radius clipped to unit disk. Fixed labeled prototypes used in isolated action clips.',
        'level': 'Green/yellow occupancy of visible action-panel indicator divided by110 reduced-image columns. Used as an inferred relative gesture drive; the video does not identify its numeric calibration. No reference waveform envelope is fed into synthesis.'}


def controls_file(path, angles, level, cutoff, gains=1, density=100):
    gains = np.broadcast_to(gains, len(level))
    cutoffs = np.broadcast_to(cutoff, len(level))
    rows = [[2205, angle, radius, .5, float(v), .5, 0, density, 1, 1, float(cutoff), float(g)]
            for (angle, radius), v, cutoff, g in zip(angles, level, cutoffs, gains)]
    np.savetxt(path, rows, fmt='%.12g', header='frames angle radius size velocity roughness asymmetry scratch_density friction_sigma pulse_samples cutoff_hz gain')
    return rows


def render(binary, controls, response, output, seed):
    subprocess.run([str(binary), str(controls), str(response), str(output), str(seed)], check=True)
    rate, samples = read_wave(output)
    if rate != RATE or samples.shape[1] != 1:
        raise ValueError('Unexpected native output format')
    return samples[:, 0]


def spectrum(samples):
    frequency, power = signal.welch(samples - samples.mean(), RATE, nperseg=4096)
    selected = (frequency >= 80) & (frequency <= 12000)
    return power[selected] / max(float(power[selected].sum()), 1e-30)


def spectrum_error(reference, generated):
    floor = max(float(reference.max())*1e-6, 1e-30)
    return float(np.sqrt(np.mean((10*np.log10(np.maximum(reference, floor)/np.maximum(generated, floor)))**2)))


def envelope(samples):
    block = 441
    x = samples[:len(samples)//block*block].reshape(-1, block)
    return np.sqrt(np.mean(x*x, axis=1))


def comparison(reference, generated):
    a, b = envelope(reference), envelope(generated)
    common = min(len(a), len(b))
    a, b = a[:common], b[:common]
    psd_a, psd_b = spectrum(reference), spectrum(generated)
    return {'reference': metrics(RATE, reference[:, None]), 'synthesis': metrics(RATE, generated[:, None]),
        'reference_texture': texture_metrics(RATE, reference[:, None]), 'synthesis_texture': texture_metrics(RATE, generated[:, None]),
        'normalized_psd_rms_db': spectrum_error(psd_a, psd_b),
        'envelope_correlation_10ms': float(np.corrcoef(a, b)[0, 1]) if np.std(a)*np.std(b) else None,
        'envelope_relative_l2': float(np.linalg.norm(a-b)/max(np.linalg.norm(a), 1e-30)),
        'rms_ratio': float(np.sqrt(np.mean(generated*generated)/max(np.mean(reference*reference), 1e-30))),
        'scope': 'Stochastic realization comparison, not a pointwise waveform equivalence gate or perceptual listening study.'}


def infer_modes(hit, training):
    frequency, power = signal.periodogram(hit, RATE, nfft=131072)
    peaks, _ = signal.find_peaks(10*np.log10(np.maximum(power, 1e-20)), distance=35, prominence=3)
    peaks = peaks[(frequency[peaks] > 150) & (frequency[peaks] < 10000)]
    peaks = np.sort(peaks[np.argsort(power[peaks])[-40:]])
    width = signal.peak_widths(power, peaks, rel_height=.5)[0]*(frequency[1]-frequency[0])
    decay = np.clip(1/(np.pi*width), .001, .3)
    frequencies = frequency[peaks]
    f, target_power = signal.welch(training, RATE, nperseg=4096)
    selected = (f >= 150) & (f <= 10000)
    z = np.exp(-2j*np.pi*f[selected, None]/RATE)
    poles = np.exp(-1/(RATE*decay))*np.exp(2j*np.pi*frequencies/RATE)
    transfer = (z*poles.imag)/((1-z*poles)*(1-z*poles.conj()))
    normalization = np.linalg.norm(transfer, axis=0)
    transfer /= normalization
    target = np.sqrt(target_power[selected])
    target /= np.linalg.norm(target)
    initial = np.sqrt(np.interp(frequencies, f, target_power))
    initial /= np.linalg.norm(initial)
    floor = float(target.max())*1e-3
    def residual(amplitudes):
        return np.log(np.maximum(abs(np.einsum('ij,j->i', transfer, amplitudes)), floor))-np.log(np.maximum(target, floor))
    fit = optimize.least_squares(residual, initial, bounds=(0, np.inf), max_nfev=80)
    amplitudes = fit.x/normalization
    keep = amplitudes > 1e-12
    modes = np.column_stack([frequencies[keep], decay[keep], amplitudes[keep]])
    if not np.isfinite(modes).all() or not len(modes):
        raise ValueError('Invalid inferred object modes')
    time = np.arange(int(np.ceil(np.log(1000)*max(decay[keep])*RATE)))/RATE
    response = np.einsum('i,ij->j', modes[:, 2], np.exp(-time/modes[:, 1, None])*np.sin(2*np.pi*modes[:, 0, None]*time))
    return modes, response, {'candidate_modes': len(peaks), 'retained_modes': len(modes), 'optimizer_converged': bool(fit.success),
        'training_log_spectral_rms_db': float(np.sqrt(np.mean(residual(fit.x)**2))*20/np.log(10)),
        'source': 'Frequencies and inverse pi half-power-bandwidth decays from the independent isolated hit. Positive amplitudes fitted on rubbing59..61s only to normalized log magnitude of the exact discrete damped-sine transfer. Negligible amplitudes below1e-12 omitted. No heldout or transition waveform used.',
        'limitations': 'The spacebar excitation is unknown; its recording is not assumed to be a unit impulse. Frequency/decay peak estimation and steady-white-drive amplitude inference are approximate; original modal presets are unavailable.'}


def candidate_filter(force, level, cutoff):
    if cutoff == 0:
        return force.copy()
    state = np.zeros(1)
    result = np.zeros_like(force)
    for index, velocity in enumerate(level):
        section = slice(index*2205, (index+1)*2205)
        pole = np.exp(-2*np.pi*cutoff*velocity/RATE)
        result[section], state = signal.lfilter([1-pole], [1, -pole], force[section], zi=state)
        if velocity == 0:
            result[section] = 0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT/'build/continuousReproduce')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/reproduction/continuous')
    args = parser.parse_args()
    binary, output = args.binary.resolve(), args.output.resolve()
    reference = ROOT/'references/continuous'
    reference.mkdir(parents=True, exist_ok=True)
    video = reference/'ConanEtAlCMJ2014.mp4'
    if not video.exists():
        if args.offline:
            raise FileNotFoundError('Offline author video is missing: '+str(video))
        with urllib.request.urlopen(URL, timeout=60) as response:
            video.write_bytes(response.read())
    original = pin(video)
    if original['sha256'] != VIDEO_HASH:
        raise ValueError('Author video changed from the pinned original')
    output.mkdir(parents=True, exist_ok=True)
    source = {'page': PAGE, 'url': URL, 'original': original, 'license': 'Author companion media; no redistribution license was advertised. Downloaded original remains under references/.',
        'unavailable': ['Original source code', 'Numeric material/shape/modal presets', 'Gesture time series and velocity mapping', 'Exact stochastic seeds'],
        'legacy_page': 'https://www.lma.cnrs-mrs.fr/users/kronland/public_html/CMJ2014/ returned404; current kronland.fr/CMJ2014 is an image attachment.'}
    write_json(reference/'manifest.json', source)
    with tempfile.TemporaryDirectory(prefix='continuous-') as temp:
        decoded = Path(temp)/'video.wav'
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-vn', '-ac', '1', '-c:a', 'pcm_f32le', '-y', str(decoded)], check=True)
        rate, channels = read_wave(decoded)
        if rate != RATE:
            raise ValueError('Author sample rate changed')
        authored = channels[:, 0]
        onset = int(47.19*RATE) + int(np.flatnonzero(abs(authored[int(47.19*RATE):int(47.3*RATE)]) > .005)[0]) - 44
        response = authored[onset:onset+int(.24*RATE)].copy()
        response_path = output/'object-response.wav'
        wavfile.write(response_path, RATE, response.astype(np.float32))
        modes, response, mode_inference = infer_modes(response, authored[59*RATE:61*RATE])
        modes_path = output/'object-modes.txt'
        np.savetxt(modes_path, modes, fmt='%.12g')
        cursor, level, extraction = video_controls(video)
        report = {'source': source, 'binary': pin(binary), 'metallib': pin(binary.parent/'SurfaceAudio.metallib'),
            'scripts': [pin(Path(__file__)), pin(ROOT/'tools/AnalyzeRenders.py'), pin(ROOT/'tools/BuildListeningReport.py')],
            'object_response': {'source_sample': onset, 'frames': int(.24*RATE), 'file': pin(response_path),
                'scope': 'Isolated author plastic-plate hit extracted before continuous gestures; independent of training/heldout continuous examples. Measured FIR retains AAC and object response, not an inferred force or playback of a continuous author signal.'},
            'modal_object': {**mode_inference, 'file': pin(modes_path)},
            'control_extraction': extraction, 'training_seconds': [0, 2], 'heldout_seconds': [2, 6],
            'inference': 'After shared object inference from independent hit plus first2seconds rubbing, fit one cutoff from [0,1000,3000,8000,20000]Hz (0 bypasses) and one constant output gain per fixed action; scratching also selects density from100,300,1000events/s. Score is normalized PSD RMS dB +3*abs(log(envelopeCV/referenceCV)), using only its first2seconds. Remaining4seconds held out. Transition interpolates fitted gains and cutoffs (bypass represented by Nyquist for interpolation) and inherits fitted scratch density with no transition-audio fit. Numeric defaults for size, roughness, density and pulse duration are explicit in control files.',
            'cases': [], 'complete': False}
        cases, fitted = [], {}
        for index, (name, begin, end, anchor) in enumerate(CASES):
            target = authored[begin*RATE:end*RATE]
            target_path = output/(name+'-author.wav')
            wavfile.write(target_path, RATE, target.astype(np.float32))
            positions = cursor[begin*20:end*20].copy()
            velocity = level[begin*20:end*20]
            if anchor is not None:
                positions[:] = [anchor, 1]
            controls = output/(name+'-controls.txt')
            synthesis_path = output/(name+'.wav')
            if anchor is not None:
                candidates = []
                train = target[:2*RATE]
                reference_spectrum = spectrum(train)
                reference_envelope = envelope(train)
                reference_cv = float(reference_envelope.std()/reference_envelope.mean())
                for density in ([100, 300, 1000] if name == 'scratching' else [100]):
                    controls_file(controls, positions, velocity, 0, density=density)
                    raw_path = Path(temp)/(name+'-raw.wav')
                    render(binary, controls, modes_path, raw_path, 2014+index)
                    _, raw_force = read_wave(raw_path.with_name(raw_path.stem+'-force.wav'))
                    for cutoff in [0, 1000, 3000, 8000, 20000]:
                        filtered = candidate_filter(raw_force[:, 0], velocity, cutoff)
                        trial = signal.fftconvolve(filtered, response)[:len(target)]
                        generated = trial[:2*RATE]
                        gain = float(np.sqrt(np.mean(train*train)/max(np.mean(generated*generated), 1e-30)))
                        spectral = spectrum_error(reference_spectrum, spectrum(generated))
                        env = envelope(generated)
                        cv = float(env.std()/env.mean())
                        score = spectral + 3*abs(np.log(cv/reference_cv))
                        candidates.append({'cutoff_hz': cutoff, 'scratch_density': density, 'gain': gain, 'training_normalized_psd_rms_db': spectral, 'training_envelope_cv': cv, 'selection_score': score})
                choice = min(candidates, key=lambda row: row['selection_score'])
                cutoff, gain = choice['cutoff_hz'], choice['gain']
                fitted[name] = choice
                controls_file(controls, positions, velocity, cutoff, gain, choice['scratch_density'])
            else:
                candidates = []
                weights = []
                for angle, radius in positions:
                    sector = angle*3/(2*np.pi)
                    first = min(int(sector), 2)
                    w = np.full(3, (1-radius)/3)
                    w[first] += radius*(1-sector+first)
                    w[(first+1)%3] += radius*(sector-first)
                    weights.append(w)
                gains = np.asarray(weights) @ np.array([fitted[k]['gain'] for k in ['rubbing', 'scratching', 'rolling']])
                cutoffs = np.asarray(weights) @ np.array([fitted[k]['cutoff_hz'] or RATE/2 for k in ['rubbing', 'scratching', 'rolling']])
                controls_file(controls, positions, velocity, cutoffs, gains, fitted['scratching']['scratch_density'])
            actual = render(binary, controls, modes_path, synthesis_path, 2014+index)
            matched = actual[:len(target)]
            split = 2*RATE if anchor is not None else 0
            result = {'name': name, 'video_seconds': [begin, end], 'reference': pin(target_path), 'synthesis': pin(synthesis_path), 'controls': pin(controls),
                'native': json.loads(synthesis_path.with_suffix('.json').read_text()), 'candidates': candidates,
                'training': comparison(target[:split], matched[:split]) if split else None,
                'heldout': comparison(target[split:], matched[split:]), 'whole_event': comparison(target, matched)}
            report['cases'].append(result)
            notes = 'Author CMJ2014 plastic-plate video. The modal object frequencies/decays come from an isolated earlier hit, with amplitudes inferred from the first2seconds of rubbing. Gesture drive is inferred from the visible indicator. '
            notes += 'Cutoff, constant gain and scratch density selected using the first2seconds; final4seconds held out. A new stochastic realization; synthetic modal tail retained through60dB decay.' if anchor is not None else 'Continuous disk controls extracted from the video; source/filter state retained across updates. Gains and cutoff inherited from other clips; this entire transition clip is held out.'
            cases.append({'title': 'Conan CMJ2014 / '+name, 'reference': str(target_path), 'synthesis': str(synthesis_path), 'notes': notes, 'source_url': PAGE})
            print(name, result['heldout']['normalized_psd_rms_db'], result['heldout']['envelope_correlation_10ms'], flush=True)
        report['complete'] = True
        write_json(output/'metrics.json', report)
        write_json(output/'cases.json', {'title': 'Conan CMJ2014 continuous interactions', 'cases': cases})
    subprocess.run([sys.executable, str(ROOT/'tools/BuildListeningReport.py'), str(output/'cases.json'), '--output', str(output/'listening')], check=True)
    print('Wrote '+str(output/'cases.json'))


if __name__ == '__main__':
    main()
