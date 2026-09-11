#!/usr/bin/env python3
"""Reproduce Agarwal 2023 response fitting and joint material distributions."""
import argparse
import hashlib
import html
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import urllib.request
from urllib.parse import quote

import numpy as np
import scipy
from scipy import signal, stats
from scipy.io import wavfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
REFERENCES = ROOT / 'references/agarwal/icml2023'
SOURCE = 'https://mcdermottlab.mit.edu/ICML2023/sound_website.html'
MATERIALS = ('Wood', 'Plastic', 'Metal', 'Glass')
RATE = 44100
BAND_EDGES = np.geomspace(20, 20000, 33)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthesis_artifacts(binary):
    metallib = binary.parent / 'SurfaceAudio.metallib'
    return {'executable': {'path': str(binary), 'sha256': digest(binary)},
            'metallib': {'path': str(metallib), 'sha256': digest(metallib)}}


def environment_metadata(binary):
    compiler = None
    cache = binary.parent / 'CMakeCache.txt'
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.startswith('CMAKE_CXX_COMPILER:FILEPATH='):
                compiler_path = line.split('=', 1)[1]
                version = subprocess.run([compiler_path, '--version'], check=True, capture_output=True, text=True).stdout.strip()
                compiler = {'configured_path': compiler_path, 'current_version': version,
                            'cmake_cache_sha256': digest(cache),
                            'scope': 'Configured compiler observed at harness startup; executable and metallib hashes identify actual artifacts'}
                break
    return {'python': sys.version, 'python_executable': sys.executable, 'numpy': np.__version__,
            'scipy': scipy.__version__, 'platform': platform.platform(), 'machine': platform.machine(),
            'macos_version': platform.mac_ver()[0], 'compiler': compiler}


def read_wave(path):
    rate, samples = wavfile.read(path)
    if rate != RATE or samples.ndim != 1:
        raise ValueError(f'{path}: expected mono 44100 Hz')
    if np.issubdtype(samples.dtype, np.integer):
        samples = samples.astype(np.float64) / 2 ** (np.iinfo(samples.dtype).bits - 1)
    samples = samples.astype(np.float64)
    if not np.isfinite(samples).all():
        raise ValueError(f'{path}: nonfinite audio')
    return samples


def verify_inputs():
    pins_path = ROOT / 'docs/ReferenceInputs.json'
    pins = [entry for entry in json.loads(pins_path.read_text())['files'] if entry['method'] == 'agarwal-response']
    if len(pins) != 100:
        raise ValueError('Expected 100 committed agarwal-response reference pins')
    for entry in pins:
        path = ROOT / entry['path']
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(entry['source_url'], timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != entry['sha256']:
                raise ValueError(f'Download hash mismatch: {entry["path"]}')
            path.write_bytes(data)
        if digest(path) != entry['sha256']:
            raise ValueError(f'Hash mismatch: {entry["path"]}')
    return {'files': pins, 'sha256': digest(pins_path)}


def preprocess(samples, align):
    dc = float(np.median(samples)) if align else 0.
    centered = samples - dc
    energy = np.cumsum(centered ** 2)
    if energy[-1] <= 0:
        raise ValueError('Cannot prepare a response without signal energy')
    onset = int(np.searchsorted(energy, .001 * energy[-1])) if align else 0
    removed = float(energy[onset - 1] / energy[-1]) if onset else 0.
    metadata = {'align_onset': align, 'dc_offset': dc, 'dc_estimator': 'whole-record median' if align else 'none',
                'tail_50ms_median': float(np.median(samples[-2205:])), 'onset_frame': onset,
                'onset_seconds': onset / RATE, 'removed_centered_energy_fraction': removed,
                'original_frames': len(samples), 'prepared_frames': len(samples) - onset,
                'onset_rule': 'first cumulative centered-energy crossing of 0.1%; retain crossing sample' if align else 'original recording start'}
    return centered[onset:].astype(np.float32), metadata


def restore_response(response, metadata):
    if len(response) != metadata['prepared_frames']:
        raise ValueError('Response length differs from prepared input')
    restored = np.full(metadata['original_frames'], metadata['dc_offset'], dtype=np.float64)
    restored[metadata['onset_frame']:] += response
    return restored.astype(np.float32)


def validate_fit_provenance(directory, binary):
    path = directory / 'provenance.json'
    if not path.exists():
        raise ValueError(f'{directory}: missing successful-fit provenance; regenerate the fit')
    provenance = json.loads(path.read_text())
    if not provenance.get('synthesis_artifacts', {}).get('metallib', {}).get('sha256'):
        raise ValueError(f'{directory}: missing successful-fit shader provenance; regenerate the fit')
    initial_names = ('initial-response.wav', 'initial.wav')
    for name, expected in provenance['artifact_sha256'].items():
        if name in initial_names and not (directory / name).exists():
            continue
        artifact = directory / name
        if not artifact.exists() or digest(artifact) != expected:
            raise ValueError(f'{directory}: stale or altered fit artifact {name}')
    response = directory / 'initial-response.wav'
    if not response.exists():
        subprocess.run([str(binary), 'sample', str(directory / 'initial.f32'), str(response),
                        str(len(read_wave(directory / 'prepared.wav'))), str(provenance['fit_configuration']['seed'])], check=True)
    if not (directory / 'initial.wav').exists():
        wavfile.write(directory / 'initial.wav', RATE, restore_response(read_wave(response), provenance['preprocessing']))
    for name in initial_names:
        if digest(directory / name) != provenance['artifact_sha256'][name]:
            raise ValueError(f'{directory}: regenerated initialization differs from successful fit: {name}')
    return provenance


def broadband_rt60(samples):
    width = 441
    power = np.convolve(samples ** 2, np.ones(width) / width, mode='full')[:len(samples)]
    peak = int(np.argmax(power))
    db = 10 * np.log10(np.maximum(power / max(power[peak], 1e-30), 1e-30))
    crossing = np.flatnonzero(db[peak:] <= -60)
    if len(crossing):
        value = max(float(crossing[0]) / RATE, .001)
        return value, {'method': '10ms causal power first -60dB crossing after maximum', 'seconds': value}
    region = (np.arange(len(samples)) > peak) & (db <= -5) & (db >= -25)
    if np.count_nonzero(region) >= 10:
        slope = float(np.polyfit(np.arange(len(samples))[region] / RATE, db[region], 1)[0])
        if slope < 0:
            value = min(max(-60 / slope, .001), 10.)
            return value, {'method': 'no -60dB crossing; extrapolate -5 to -25dB power slope, bounded 1ms..10s', 'seconds': value}
    value = max(len(samples) / RATE, .001)
    return value, {'method': 'no crossing or negative decay fit; record duration fallback', 'seconds': value}


def initialize(samples, seed):
    rng = np.random.default_rng(seed)
    frequencies, power = signal.periodogram(samples, RATE, detrend=False)
    f2 = frequencies ** 2
    weighting = (12194. ** 2 * f2 ** 2 / np.maximum(
        (f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194. ** 2), 1e-30))
    weighted_db = 10 * np.log10(np.maximum(power * weighting ** 2, 1e-30)) + 2.
    peaks, _ = signal.find_peaks(weighted_db, prominence=2.)
    peaks = peaks[(frequencies[peaks] >= 20) & (frequencies[peaks] <= 20000)]
    selected = []
    for peak in peaks[np.argsort(-weighted_db[peaks], kind='stable')]:
        if all(abs(frequencies[peak] - frequencies[other]) >= 100 for other in selected):
            selected.append(int(peak))
            if len(selected) == 10:
                break
    if len(selected) != 10:
        raise ValueError(f'Only {len(selected)} peaks satisfy initialization constraints; no fabricated modes added')
    rt60, decay = broadband_rt60(samples)
    params = np.concatenate((frequencies[selected], rng.uniform(-30, -10, 10),
                             rt60 * rng.uniform(.5, 1.5, 10), rng.uniform(-15, -5, 10), rng.uniform(.04, .12, 10)))
    return params.astype('<f4'), {'seed': seed, 'frequencies_hz': frequencies[selected].tolist(),
                                'broadband_rt60': decay, 'prominence_units_assumption': 'dB of A-weighted periodogram power',
                                'mode_rt60_initialization_assumption': 'uniform[0.5,1.5] times broadband RT60'}


def gaussian(data):
    data = np.asarray(data, dtype=np.float64)
    mean = data.mean(axis=0)
    factor = (data - mean).T / np.sqrt(len(data))
    covariance = factor @ factor.T
    return mean, factor, {'mean': mean.tolist(), 'covariance_mle': covariance.tolist(),
                          'covariance_rank': int(np.linalg.matrix_rank(factor)), 'observations': len(data),
                          'dimensions': data.shape[1], 'estimator': 'Gaussian MLE covariance 1/n; exact centered-data factor; no jitter or diagonalization'}


def material_parameters(parameters, count, seed, heldout=None):
    if heldout is not None:
        parameters = np.delete(parameters, heldout, axis=0)
    rng = np.random.default_rng(seed)
    modes = np.stack((parameters[:, :10], parameters[:, 10:20], parameters[:, 20:30]), axis=-1).reshape(-1, 3)
    noise = parameters[:, 30:50]
    mm, mf, mode_model = gaussian(modes)
    nm, nf, noise_model = gaussian(noise)
    rejects = {'modes': 0, 'noise': 0}

    def draw(mean, factor, valid, kind):
        for _ in range(100000):
            sample = mean + factor @ rng.standard_normal(factor.shape[1])
            if valid(sample):
                return sample
            rejects[kind] += 1
        raise ValueError(f'{kind} joint Gaussian physical-domain rejection exceeded 100000 attempts')

    output = []
    for _ in range(count):
        mode = np.array([draw(mm, mf, lambda x: 20 <= x[0] <= 20000 and x[2] > 0, 'modes') for _ in range(10)])
        bands = draw(nm, nf, lambda x: np.all(x[10:] > 0), 'noise')
        output.append(np.concatenate((mode[:, 0], mode[:, 1], mode[:, 2], bands)))
    return np.asarray(output, dtype='<f4'), {'seed': seed, 'mode': mode_model, 'noise': noise_model,
        'rejections': rejects, 'sampling_domain': 'reject whole triples outside frequency[20,20000]Hz or RT60>0; reject whole noise vectors with any nonpositive RT60; amplitudes unbounded'}


def features(samples, frames=RATE):
    used = np.pad(samples[:frames], (0, max(0, frames - len(samples)))) if frames else samples
    frequencies, power = signal.periodogram(used, RATE, detrend=False)
    bands = np.array([power[(frequencies >= low) & (frequencies < high)].sum()
                      for low, high in zip(BAND_EDGES[:-1], BAND_EDGES[1:])])
    bands /= max(bands.sum(), 1e-30)
    energy = used ** 2
    cumulative = np.cumsum(energy) / max(energy.sum(), 1e-30)
    values = {'rms': float(np.sqrt(np.mean(energy))), 'peak': float(np.max(np.abs(used))),
              'centroid_hz': float(np.sum(frequencies * power) / max(power.sum(), 1e-30)),
              't50_seconds': float(np.searchsorted(cumulative, .5) / RATE),
              't90_seconds': float(np.searchsorted(cumulative, .9) / RATE),
              'original_frames': len(samples), 'analyzed_frames': len(used), 'truncated': bool(frames and len(samples) > frames)}
    return {'features': values, 'normalized_band_power': bands.tolist()}


def cohort_comparison(reference, candidate):
    a = np.array([r['normalized_band_power'] for r in reference])
    b = np.array([r['normalized_band_power'] for r in candidate])
    return {'reference_count': len(a), 'candidate_count': len(b),
            'pooled_normalized_band_total_variation': float(.5 * np.abs(a.mean(axis=0) - b.mean(axis=0)).sum()),
            'feature_ks_statistics': {key: float(stats.ks_2samp([r['features'][key] for r in reference],
                                      [r['features'][key] for r in candidate]).statistic)
                                      for key in ('rms', 'centroid_hz', 't50_seconds', 't90_seconds')},
            'interpretation': 'descriptive distribution distances; no matched sample pairing or statistical acceptance claim'}


def sample_cohort(binary, directory, parameters, seed, analyze_only):
    directory.mkdir(parents=True, exist_ok=True)
    batch_path = directory / 'parameters.f32'
    if analyze_only and (not batch_path.exists() or batch_path.read_bytes() != parameters.tobytes()):
        raise ValueError(f'{directory}: cached sampling parameters differ; regenerate audio with the same seed and fitted inputs')
    if not analyze_only:
        parameters.tofile(batch_path)
    records = []
    for index, row in enumerate(parameters):
        path = directory / f'{index + 1:02d}.wav'
        parameter_path = directory / f'{index + 1:02d}.f32'
        if analyze_only and (not parameter_path.exists() or parameter_path.read_bytes() != row.tobytes()):
            raise ValueError(f'{parameter_path}: cached parameters differ')
        if not analyze_only:
            row.tofile(parameter_path)
        if not analyze_only:
            subprocess.run([str(binary), 'sample', str(parameter_path), str(path), str(RATE), str(seed + index)], check=True,
                           stdout=subprocess.DEVNULL)
        record = features(read_wave(path))
        record.update(wav=str(path), sha256=digest(path), parameter_sha256=digest(parameter_path), noise_seed=seed + index)
        records.append(record)
    return records


def listening_report(output, result, cases, keep_diagnostics=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    report = output / 'listening'
    report.mkdir(exist_ok=True)
    assets = report / 'audio'
    assets.mkdir(exist_ok=True)

    def player(path, label, name):
        x = read_wave(path)
        ac = x - x.mean()
        rms, peak = np.sqrt(np.mean(ac ** 2)), np.max(np.abs(ac))
        gain = min(.1 / rms if rms else 1., .89 / peak if peak else 1.)
        raw_url = quote(os.path.relpath(path.resolve(), report.resolve()), safe='/')
        wavfile.write(assets / f'{name}-level.wav', RATE, (ac * gain).astype(np.float32))
        return f'<div>{html.escape(label)}<audio controls preload="none" src="audio/{name}-level.wav" data-level="audio/{name}-level.wav" data-raw="{raw_url}"></audio><a href="{raw_url}">Raw WAV</a></div>'

    sections = []
    for case in cases:
        name = case['name']
        panels = [player(pathlib.Path(case[field]), label, f'{name}-{field}') for field, label in
                  [('reference', 'Author measured IR'), *([('initial', 'Initialization')] if keep_diagnostics else []),
                   ('synthesis', 'Fitted on this recording')]]
        causal = [player(pathlib.Path(case[field]), label, f'{name}-{field}') for field, label in
                  [('prepared_reference', 'Prepared causal reference'), *([('initial_response', 'Initial causal response')] if keep_diagnostics else []),
                   ('fitted_response', 'Fitted causal response')]]
        offset = case['preprocessing']['onset_seconds']
        dc = case['preprocessing']['dc_offset']
        sections.append(f'<section><h2>{name}: calibrated reconstruction</h2><p>Full-record players restore delay {offset:.6f} seconds and DC {dc:.8f}. The raw author recording is unchanged.</p><div class="players">{"".join(panels)}</div><details><summary>Causal reference and responses</summary><div class="players">{"".join(causal)}</div></details></section>')
    for material, data in result.get('materials', {}).items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), constrained_layout=True)
        for key, label, color in [('author', 'Author generated', '#355e91'), ('generated', 'Our generated', '#b45309')]:
            cohort = data[key]
            bands = np.asarray([r['normalized_band_power'] for r in cohort])
            db = 10 * np.log10(np.maximum(bands, 1e-12))
            x = np.sqrt(BAND_EDGES[:-1] * BAND_EDGES[1:])
            axes[0].semilogx(x, np.median(db, axis=0), label=label, color=color)
            axes[0].fill_between(x, *np.quantile(db, [.1, .9], axis=0), color=color, alpha=.15)
            times = np.sort([r['features']['t90_seconds'] for r in cohort])
            axes[1].step(times, np.arange(1, len(times) + 1) / len(times), where='post', label=label, color=color)
        axes[0].set(xlabel='Frequency (Hz)', ylabel='Normalized band power (dB)')
        axes[1].set(xlabel='Time to 90% energy (seconds)', ylabel='Empirical cumulative fraction')
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend()
        fig.savefig(report / f'{material}.png', dpi=130)
        plt.close(fig)
        columns = []
        for key, label in [('author', 'Author generated cohort'), ('generated', 'Our independent generated cohort')]:
            panels = [player(pathlib.Path(row['wav']), f'{label} {i + 1}', f'{material}-{key}-{i + 1}')
                      for i, row in enumerate(data[key])]
            columns.append(f'<div><h3>{label}</h3>{"".join(panels)}</div>')
        distance = data['comparison']['pooled_normalized_band_total_variation']
        sections.append(f'<section><h2>{material}: material distribution</h2><p>Pooled spectral total variation {distance:.3f}. Curves show median and 10–90% range. Samples have independent seeds and no one-to-one correspondence.</p><img src="{material}.png" alt="Cohort spectral and decay distributions"><details><summary>Listen to all 40 cohort samples</summary><div class="cohorts">{"".join(columns)}</div></details></section>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Agarwal 2023 response reproduction</title><style>body{font:16px/1.5 system-ui;max-width:1150px;margin:30px auto;padding:0 20px;background:#fafafa;color:#202a35}section{background:white;border:1px solid #ddd;padding:20px;margin:20px 0}audio{display:block;width:100%}.players{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}.cohorts{display:grid;grid-template-columns:1fr 1fr;gap:30px}img{width:100%}@media(max-width:650px){.players,.cohorts{grid-template-columns:1fr}}</style><h1>Agarwal 2023 response reproduction</h1><p>Measured examples are fitted directly: these are calibrated reconstructions. New material samples use joint Gaussian parameter distributions. Author and new generated samples are unpaired cohorts. Leave-one-out metrics in <a href="../metrics.json">metrics.json</a> exclude the evaluated recording from distribution fitting. They are exploratory with five examples per material, not perceptual validation.</p><p><label><input id="level" type="checkbox" checked>Match playback level and remove DC</label>. Playback copies target AC RMS 0.1 with peak ceiling 0.89. Raw audio retains recorded or synthesized gain. Cohort metrics use a common one-second crop or zero padding, including truncation of long metal recordings. Optional onset alignment subtracts median DC and trims less than 0.1% centered energy before fitting. Full-record comparisons restore the removed delay and median DC. Causal players preserve the fitted response and full remaining tail. Author measured peaks are approximately 0.9 and generated peaks approximately 0.8, consistent with separate peak normalization. Original normalization code and random seeds are unpublished. Raw RMS is descriptive; level-invariant spectrum and decay comparisons are primary.</p>'''
    page += '\n'.join(sections)
    page += '''<script>document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>document.querySelectorAll('audio').forEach(b=>{if(a!==b)b.pause()})));document.querySelector('#level').addEventListener('change',e=>document.querySelectorAll('audio').forEach(a=>{a.pause();a.src=e.target.checked?a.dataset.level:a.dataset.raw}));</script></html>'''
    (report / 'index.html').write_text(page)
    for path in assets.glob('*-raw.wav'):
        path.unlink()
    if not keep_diagnostics:
        for pattern in ('*-initial-level.wav', '*-initial_response-level.wav'):
            for path in assets.glob(pattern):
                path.unlink()


def self_test():
    z = np.arange(-2., 3.)
    coefficients = np.concatenate((np.arange(1., 11.) * .1, np.arange(1., 11.) * .0001))
    means = np.concatenate((np.full(10, -10.), np.full(10, .1)))
    params = np.tile(np.concatenate((np.arange(200., 1200., 100.), np.full(10, -20.), np.full(10, .1), means)), (5, 1))
    params[:, 30:] += z[:, None] * coefficients
    draws, model = material_parameters(params, 4000, 57)
    expected = 2 * np.outer(coefficients, coefficients)
    actual = np.array(model['noise']['covariance_mle'])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    empirical = np.cov(draws[:, 30:].T, bias=True)
    assert np.linalg.norm(empirical - expected) / np.linalg.norm(expected) < .06
    assert model['noise']['covariance_rank'] == 1
    repeated, _ = material_parameters(params, 4000, 57)
    assert np.array_equal(draws, repeated)
    excluded, _ = material_parameters(params, 20, 57, heldout=2)
    changed = params.copy()
    changed[2, 30:40] += 30
    excluded_again, _ = material_parameters(changed, 20, 57, heldout=2)
    assert np.array_equal(excluded, excluded_again)
    included, _ = material_parameters(changed, 20, 57)
    assert not np.array_equal(excluded, included)
    waveform = np.r_[np.full(100, .125), [1., -.5, .25], np.full(80, .125)]
    prepared, metadata = preprocess(waveform, True)
    assert metadata['onset_frame'] == 100 and metadata['dc_offset'] == .125
    assert metadata['removed_centered_energy_fraction'] == 0
    np.testing.assert_array_equal(restore_response(prepared, metadata), waveform.astype(np.float32))
    print('Passed joint covariance, deterministic sampling, withheld-record exclusion, known-delay alignment, and DC restoration checks')


def render_retained(binary, output):
    import tempfile
    record = json.loads((ROOT / 'repros/agarwal-response/parameters.json').read_text())
    verify_inputs()
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    with tempfile.TemporaryDirectory(prefix='response-') as temporary:
        for name, fit in record['fits'].items():
            parameters = pathlib.Path(temporary) / 'parameters.f32'
            np.asarray(fit['parameters'], dtype='<f4').tofile(parameters)
            destination = output / (name + '-response.wav')
            subprocess.run([str(binary), 'sample', str(parameters), str(destination),
                            str(fit['preprocessing']['prepared_frames']), str(fit['seed'])], check=True)
            if digest(destination) != fit['waveform_sha256']:
                raise ValueError(f'{name}: response differs from retained fit')
            restored = output / (name + '.wav')
            wavfile.write(restored, RATE, restore_response(read_wave(destination), fit['preprocessing']))
            cases.append({'title': 'Agarwal 2023 ' + name, 'reference': str(REFERENCES / 'training' / (name + '.wav')),
                          'synthesis': str(restored), 'notes': record['scope'], 'source_url': SOURCE})
    save_json(output / 'cases.json', {'cases': cases})
    subprocess.run([sys.executable, str(ROOT / 'tools/BuildListeningReport.py'), str(output / 'cases.json'), '--output', str(output / 'listening')], cwd=ROOT, check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=pathlib.Path, default=ROOT / 'build/agarwalResponseFit')
    parser.add_argument('--output', type=pathlib.Path, default=ROOT / 'outputs/reproduction/agarwal-response')
    parser.add_argument('--steps', type=int, default=140000)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--case', default='all', choices=['all'] + [f'{m}_{i}' for m in MATERIALS for i in range(1, 6)])
    parser.add_argument('--retained', action='store_true', help='Render the twenty committed fitted responses without optimization')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--sample-only', action='store_true', help='Reuse existing fits, then synthesize all material and leave-one-out cohorts and report')
    parser.add_argument('--analyze-only', action='store_true', help='Reuse fitted and sample WAVs; regenerate initialization and recompute distribution metrics')
    parser.add_argument('--self-test', action='store_true', help='Verify analytic joint covariance and leave-one-out exclusion without audio or executable')
    parser.add_argument('--skip-loo', action='store_true', help='Bounded run without leave-one-out sampling; report marks omitted validation')
    parser.add_argument('--align-onset', action='store_true', help='Subtract median DC and trim before 0.1% cumulative energy; restore delay and DC for full-record comparisons')
    parser.add_argument('--loss-scale', choices=('db', 'ln', 'linear'), default='db')
    parser.add_argument('--learning-rate', type=float, default=2e-6)
    parser.add_argument('--parameter-scale-mode', choices=('physical', 'scaled', 'log-decay'), default='physical',
                        help='Physical parameters, frequency/amplitude scaling, or scaling with log RT60 optimization')
    parser.add_argument("--keep-diagnostics", action="store_true", help="Retain initialization WAVs and their listening players")
    args = parser.parse_args()
    if args.retained:
        render_retained(args.binary.resolve(), args.output.resolve())
        return
    if sum((args.prepare_only, args.sample_only, args.analyze_only)) > 1:
        parser.error('--prepare-only, --sample-only, and --analyze-only are mutually exclusive')
    if args.self_test:
        self_test()
        return
    args.binary, args.output = args.binary.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = verify_inputs()
    environment = environment_metadata(args.binary)
    result = {'source': SOURCE, 'environment': environment, 'reference_pins_sha256': manifest['sha256'], 'verified_files': len(manifest['files']),
              'seed': args.seed, 'fit_phase': 'reused existing fits' if args.sample_only or args.analyze_only else 'fit',
              'requested_optimizer': None if args.sample_only or args.analyze_only else {'steps': args.steps, 'learning_rate': args.learning_rate, 'parameter_scale_mode': args.parameter_scale_mode, 'loss_scale': args.loss_scale},
              'align_onset': args.align_onset,
              'parameter_layout': 'little-endian float32 frequency10,modeDb10,modeRt6010,noiseDb10,noiseRt6010',
              'cohort_analysis_frames': RATE, 'band_edges_hz': BAND_EDGES.tolist(), 'fits': {}, 'materials': {}}
    measured_peaks = [float(np.max(np.abs(read_wave(REFERENCES / 'training' / f'{m}_{i}.wav')))) for m in MATERIALS for i in range(1, 6)]
    generated_peaks = [float(np.max(np.abs(read_wave(REFERENCES / 'resynth' / m.lower() / f'{i}.wav')))) for m in MATERIALS for i in range(1, 21)]
    result['author_gain_evidence'] = {'measured_peaks': measured_peaks, 'generated_peaks': generated_peaks,
        'interpretation': 'All measured peaks approximately 0.9 and generated peaks approximately 0.8, consistent with separate peak normalization; original code unavailable. Learned amplitudes inherit author digital normalization. Raw RMS differences are descriptive; level-invariant spectral and decay comparisons are primary.'}
    cases = []
    for material_index, material in enumerate(MATERIALS):
        for index in range(1, 6):
            name = f'{material}_{index}'
            if args.case != 'all' and args.case != name:
                continue
            path = REFERENCES / 'training' / f'{name}.wav'
            directory = args.output / 'fits' / name
            directory.mkdir(parents=True, exist_ok=True)
            prepared, preprocessing = preprocess(read_wave(path), args.align_onset)
            preprocessing['source_sha256'] = digest(path)
            prepared_path = directory / 'prepared.wav'
            preprocessing_path = directory / 'preprocessing.json'
            reuse = args.analyze_only or args.sample_only
            provenance = validate_fit_provenance(directory, args.binary) if reuse else None
            if reuse:
                if provenance['original_source_sha256'] != digest(path):
                    raise ValueError(f'{name}: source differs from successful fit')
                if not preprocessing_path.exists() or json.loads(preprocessing_path.read_text()) != preprocessing:
                    raise ValueError(f'{name}: cached preprocessing or source differs; regenerate the fit with consistent --align-onset')
                if not np.array_equal(read_wave(prepared_path), prepared.astype(np.float64)):
                    raise ValueError(f'{name}: prepared audio differs from source preprocessing')
            else:
                wavfile.write(prepared_path, RATE, prepared)
                save_json(preprocessing_path, preprocessing)
            params, metadata = initialize(prepared.astype(np.float64), args.seed + material_index * 5 + index)
            initial_path = directory / 'initial.f32'
            if (args.analyze_only or args.sample_only) and (not initial_path.exists() or initial_path.read_bytes() != params.tobytes()):
                raise ValueError(f'{name}: cached initialization differs; use the original seed')
            if not (args.analyze_only or args.sample_only):
                params.tofile(initial_path)
                save_json(directory / 'initialization.json', metadata)
            if args.prepare_only:
                print(f'Prepared {name}: RT60 {metadata["broadband_rt60"]["seconds"]:.4f}s', flush=True)
                continue
            if not (args.analyze_only or args.sample_only):
                (directory / 'provenance.json').unlink(missing_ok=True)
                fit_artifacts = synthesis_artifacts(args.binary)
                subprocess.run([str(args.binary), 'fit', str(prepared_path), str(directory / 'initial.f32'), str(directory),
                                str(args.steps), str(metadata['seed']), str(args.learning_rate), args.parameter_scale_mode, args.loss_scale], check=True)
                if synthesis_artifacts(args.binary) != fit_artifacts:
                    raise ValueError(f'{name}: executable or Metal library changed during fitting; no successful provenance recorded')
                for kind in ('initial', 'fitted'):
                    rendered = directory / f'{kind}.wav'
                    response_path = directory / f'{kind}-response.wav'
                    shutil.copyfile(rendered, response_path)
                    wavfile.write(rendered, RATE, restore_response(read_wave(response_path), preprocessing))
            info = json.loads((directory / 'fit.json').read_text())
            fit = np.fromfile(directory / 'fitted.f32', dtype='<f4')
            if fit.shape != (50,) or not np.isfinite(fit).all():
                raise ValueError(f'{name}: invalid fitted parameters')
            if not reuse:
                artifact_names = ('prepared.wav', 'preprocessing.json', 'initialization.json', 'initial.f32', 'fitted.f32',
                                  'initial-response.wav', 'fitted-response.wav', 'initial.wav', 'fitted.wav', 'fit.json')
                provenance = {'original_source_sha256': digest(path), 'fit_binary_sha256': fit_artifacts['executable']['sha256'],
                    'synthesis_artifacts': fit_artifacts, 'environment': environment,
                    'fit_configuration': {key: info[key] for key in ('steps', 'seed', 'learning_rate', 'parameter_scale_mode')},
                    'loss_scale': args.loss_scale, 'preprocessing': preprocessing,
                    'artifact_sha256': {name: digest(directory / name) for name in artifact_names}}
                save_json(directory / 'provenance.json', provenance)
            result['fits'][name] = {'provenance': provenance, 'optimizer': info, 'initialization': metadata, 'preprocessing': preprocessing,
                'prepared_reference': features(prepared.astype(np.float64), frames=None),
                'initial_response': features(read_wave(directory / 'initial-response.wav'), frames=None),
                'fitted_response': features(read_wave(directory / 'fitted-response.wav'), frames=None),
                'reference': features(read_wave(path), frames=None), 'initial': features(read_wave(directory / 'initial.wav'), frames=None),
                'fitted': features(read_wave(directory / 'fitted.wav'), frames=None), 'parameter_sha256': digest(directory / 'fitted.f32')}
            cases.append({'name': name, 'title': f'Agarwal 2023 {name}: calibrated reconstruction', 'reference': str(path),
                          'initial': str(directory / 'initial.wav'), 'synthesis': str(directory / 'fitted.wav'),
                          'prepared_reference': str(prepared_path), 'initial_response': str(directory / 'initial-response.wav'),
                          'fitted_response': str(directory / 'fitted-response.wav'), 'preprocessing': preprocessing,
                          'notes': 'Same-record calibrated reconstruction with ten modes and ten ERB noise bands. Full-record comparison restores the recorded onset delay and median DC after optional alignment; causal response files are separate. Raw gain retained. Paper implementation details are partly unspecified.', 'source_url': SOURCE})
    result['actual_fit_configurations'] = {name: {**row['provenance']['fit_configuration'], 'loss_scale': row['provenance']['loss_scale'], 'fit_binary_sha256': row['provenance']['fit_binary_sha256'], 'metallib_sha256': row['provenance']['synthesis_artifacts']['metallib']['sha256']} for name, row in result['fits'].items()}
    save_json(args.output / 'run.json', {k: v for k, v in result.items() if k not in ('fits', 'materials')})
    if args.prepare_only:
        return
    if args.case == 'all':
        for material_index, material in enumerate(MATERIALS):
            parameters = np.array([np.fromfile(args.output / 'fits' / f'{material}_{i}' / 'fitted.f32', dtype='<f4') for i in range(1, 6)])
            seed = args.seed + 10000 + material_index * 1000
            draws, model = material_parameters(parameters, 20, seed)
            directory = args.output / 'materials' / material
            generated = sample_cohort(args.binary, directory, draws, seed, args.analyze_only)
            save_json(directory / 'distribution.json', model)
            author = []
            for i in range(1, 21):
                path = REFERENCES / 'resynth' / material.lower() / f'{i}.wav'
                author.append(dict(features(read_wave(path)), wav=str(path), sha256=digest(path)))
            data = {'distribution': model, 'author': author, 'generated': generated,
                    'comparison': cohort_comparison(author, generated), 'leave_one_out': []}
            if not args.skip_loo:
                for heldout in range(5):
                    loo_seed = seed + 100 + heldout * 100
                    loo_draws, loo_model = material_parameters(parameters, 20, loo_seed, heldout=heldout)
                    loo_dir = directory / f'leave-out-{heldout + 1}'
                    cohort = sample_cohort(args.binary, loo_dir, loo_draws, loo_seed, args.analyze_only)
                    save_json(loo_dir / 'distribution.json', loo_model)
                    heldout_prepared, heldout_preprocessing = preprocess(read_wave(REFERENCES / 'training' / f'{material}_{heldout + 1}.wav'), args.align_onset)
                    holdout = features(heldout_prepared.astype(np.float64))
                    data['leave_one_out'].append({'heldout': f'{material}_{heldout + 1}',
                        'training_records': [f'{material}_{i + 1}' for i in range(5) if i != heldout],
                        'distribution': loo_model, 'heldout_features': holdout, 'heldout_preprocessing': heldout_preprocessing, 'generated': cohort,
                        'comparison': cohort_comparison([holdout], cohort),
                        'heldout_feature_percentile': {key: float(np.mean([r['features'][key] <= holdout['features'][key] for r in cohort]))
                                                      for key in ('centroid_hz', 't50_seconds', 't90_seconds')}})
            data['leave_one_out_status'] = 'omitted by --skip-loo' if args.skip_loo else 'five folds; four calibration records per distribution; twenty new samples per fold'
            result['materials'][material] = data
            print(f'Analyzed {material}: pooled spectral TV {data["comparison"]["pooled_normalized_band_total_variation"]:.4f}', flush=True)
    save_json(args.output / 'metrics.json', result)
    save_json(args.output / 'cases.json', {'cases': cases})
    listening_report(args.output, result, cases, args.keep_diagnostics)
    if not args.keep_diagnostics:
        for case in cases:
            for field in ("initial", "initial_response"):
                pathlib.Path(case.pop(field)).unlink(missing_ok=True)
        save_json(args.output / "cases.json", {"cases": cases})
    print(f'Wrote {args.output / "listening/index.html"}')


if __name__ == '__main__':
    main()
