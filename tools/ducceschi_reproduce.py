#!/usr/bin/env python3
"""Execute author oracles and render the 2021 energy-quadratisation collision models."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from ducceschi_models import mass_barrier, string_model, snare, tromba
from reference_assets import ROOT, fetch, write_json

OUTPUT = ROOT/'outputs/reproduction/ducceschi'
REFERENCE = ROOT/'references/ducceschi'


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a), 1e-30))


def file_hash(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def author_oracle(name, duration, directory, paper=False, kind='frets', rate=44100, stiffness=None, overrides=None):
    source = (REFERENCE/f'{name}.m').read_text()
    settings = {'fs0': rate, 'T': duration, 'flagPlot': 0, 'EnCheck': 1}
    if paper:
        settings.update(r=np.sqrt(.0063/(8000*np.pi)), T0=100, B=1e12)
        if name == 'stringHammer': settings['v0'] = .5
        if name == 'stringFret': settings['hb0'] = -.0001
    if name == 'stringFret': settings['flagBarr'] = 1 if kind == 'backboard' else 2
    if stiffness is not None: settings['B'] = stiffness
    settings.update(overrides or {})
    edits = {}
    for key, value in settings.items():
        source, count = re.subn(rf'(?m)^({key}\s*=)[^;]+;', rf'\g<1> {value:.17g} ;', source, count=1)
        if count: edits[key] = value
    if name == 'massBarr':
        source = source[:source.index('% Plot Stuff')]
        columns = 'out,Et,outforce,outchi,outNR,outJSV'
    elif name == 'stringHammer': columns = 'out,Et,outFD,EtFD'
    else: columns = 'outFD,EtFD,out,Et'
    target = directory/'author.f64'
    source += f"\nf=fopen('{str(target).replace(chr(39), chr(39)*2)}','wb');fwrite(f,[{columns}].','double');fclose(f);\n"
    script_hash = hashlib.sha256(source.encode()).hexdigest()
    cache_path = directory/'oracle-cache.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    reuse = target.exists() and cache.get('script_sha256') == script_hash and cache.get('output_sha256') == file_hash(target)
    (directory/'oracle.m').write_text(source)
    write_json(directory/'oracle-settings.json', {'source': str((REFERENCE/f'{name}.m').relative_to(ROOT)), 'overrides': edits, 'output_columns': columns.split(','), 'plot_tail_removed': name == 'massBarr'})
    if not reuse:
        process = subprocess.run(['octave', '--quiet', str(directory/'oracle.m')], capture_output=True, text=True)
        (directory/'octave.log').write_text(process.stdout+process.stderr)
        if process.returncode: raise RuntimeError(f'Author script failed: {directory}/octave.log')
        write_json(cache_path, {'script_sha256': script_hash, 'output_sha256': file_hash(target)})
    result = np.fromfile(target, dtype='<f8').reshape(-1, len(columns.split(',')))
    if not np.isfinite(result[:, :2]).all(): raise RuntimeError(f'Nonfinite author N-IT output: {directory}')
    return result


def render(binary, name, model, duration, gpu=True):
    directory = OUTPUT/name; directory.mkdir(parents=True, exist_ok=True)
    metadata = model.write(directory/'model.txt')
    frames = int(duration*model.rate)
    subprocess.run([str(binary), str(directory/'model.txt'), str(frames), str(directory), *(['--gpu'] if gpu else [])], check=True)
    cpu = np.fromfile(directory/'cpu.f64', '<f8').reshape(-1, 11)
    metal = np.fromfile(directory/'gpu.f32', '<f4').reshape(-1, 11).astype(float) if gpu else None
    metadata['frames'] = frames
    metadata['model_sha256'] = file_hash(directory/'model.txt')
    scale = max(float(np.max(np.abs(cpu[:, 1]))), float(np.sum(np.abs(cpu[:, 9]))/model.rate), 1e-30)
    metadata['energy_balance_relative'] = float(abs(cpu[:, 10].sum())/scale)
    metadata['maximum_step_balance_relative'] = float(np.max(np.abs(cpu[:, 10]))/scale)
    if metadata['energy_balance_relative'] > 1e-7 or metadata['maximum_step_balance_relative'] > 1e-7:
        raise RuntimeError(f'Energy balance failed: {name}: {metadata}')
    metadata['relative_energy_variation'] = float(np.max(np.abs(cpu[:, 1]-cpu[0, 1]))/max(abs(cpu[0, 1]), 1e-30))
    if not np.isfinite(cpu).all(): raise RuntimeError(f'Nonfinite CPU result: {name}')
    conservative = model.bow_force == 0 and not np.any(model.damping) and model.loss.nnz == 0
    if conservative and metadata['relative_energy_variation'] > 1e-7:
        raise RuntimeError(f'Energy drift: {name}: {metadata}')
    if gpu:
        if not np.isfinite(metal).all(): raise RuntimeError(f'Nonfinite Metal result: {name}')
        metadata['metal_waveform_relative_l2'] = relative(cpu[:, 0], metal[:, 0])
        metadata['metal_energy_relative_l2'] = relative(cpu[:, 1], metal[:, 1])
        metal_scale = max(float(np.max(np.abs(metal[:, 1]))), float(np.sum(np.abs(metal[:, 9]))/model.rate), 1e-30)
        metadata['metal_energy_balance_relative'] = float(abs(metal[:, 10].sum())/metal_scale)
        if metadata['metal_energy_balance_relative'] > .005:
            raise RuntimeError(f'Metal energy balance failed: {name}: {metadata}')
        if conservative and metadata['metal_energy_relative_l2'] > .01:
            raise RuntimeError(f'Metal conserved energy mismatch: {name}: {metadata}')
    write_json(directory/'checks.json', metadata)
    return directory, cpu, metal, metadata


def audio_case(directory, title, reference, synthesis, rate, notes, reference_label, synthesis_label='Metal FP32'):
    signals = np.array([reference, synthesis], dtype=float)
    signals -= signals.mean(axis=1, keepdims=True)
    gain = .8/max(float(np.max(np.abs(signals))), 1e-30)
    fade = min(int(.01*rate), signals.shape[1]//2)
    envelope = np.ones(signals.shape[1]); envelope[:fade] = np.linspace(0, 1, fade); envelope[-fade:] = np.linspace(1, 0, fade)
    for name, signal in zip(('reference', 'synthesis'), signals):
        wavfile.write(directory/f'{name}.wav', rate, (signal*envelope*gain).astype(np.float32))
    return {'title': title, 'group': 'Collision synthesis', 'reference': str((directory/'reference.wav').relative_to(ROOT)),
            'synthesis': str((directory/'synthesis.wav').relative_to(ROOT)), 'reference_label': reference_label,
            'synthesis_label': synthesis_label, 'notes': notes, 'playback_gain': gain}


def source_checks(binary):
    checks = {}
    jobs = [('mass-author', mass_barrier(), 'massBarr', .02, False, 'frets', False),
            ('hammer-author', string_model(), 'stringHammer', .025, False, 'frets', False),
            ('hammer-paper', string_model(paper=True), 'stringHammer', .025, True, 'frets', False)]
    for paper in (False, True):
        for kind in ('frets', 'backboard'):
            for modal in (False, True):
                name = f'{kind}-{"paper" if paper else "author"}-{"modal" if modal else "fd"}'
                jobs.append((name, string_model(kind, paper, modal), 'stringFret', .035, paper, kind, modal))
    for name, model, script, duration, paper, kind, modal in jobs:
        print('Checking', name, flush=True)
        directory, cpu, _, metadata = render(binary, name, model, duration)
        # The author fret script produces both discretisations in one run.
        sibling = OUTPUT/name.replace('-modal', '-fd')
        author = author_oracle(script, duration, sibling if modal else directory, paper, kind)
        start = 2 if modal else 0
        errors = {'output': relative(author[:, start], cpu[:, 0]), 'energy': relative(author[:, start+1], cpu[:, 1])}
        if script == 'massBarr':
            errors.update(force=relative(author[:, 2], cpu[:, 2]), auxiliary=relative(author[:, 3], cpu[:, 3]))
        metadata['author_relative_l2'] = errors
        write_json(directory/'checks.json', metadata)
        print(name, errors, 'Metal', metadata['metal_waveform_relative_l2'], flush=True)
        if max(errors.values()) > 1e-5: raise RuntimeError(f'Author comparison failed: {name}: {errors}')
        checks[name] = metadata
        fig, ax = plt.subplots(2, 1, figsize=(9, 5), constrained_layout=True)
        time = np.arange(len(cpu))/model.rate*1000
        ax[0].plot(time, author[:, start]*1000, label='Author MATLAB'); ax[0].plot(time, cpu[:, 0]*1000, '--', label='C++ FP64')
        ax[0].set(ylabel='Displacement (mm)', title=name); ax[0].legend()
        ax[1].plot(time, cpu[:, 1]/cpu[0, 1]-1); ax[1].set(xlabel='Time (ms)', ylabel='Relative energy change')
        fig.savefig(directory/'comparison.png', dpi=140); plt.close(fig)
    return checks


def instrument_checks(binary):
    from ducceschi_oracle import mechanical_oracle
    checks = {}
    for name, model in [('snare', snare()), ('tromba-figure10', tromba(rate=352800, bow_force=1))]:
        directory, cpu, _, metadata = render(binary, name+'-independent', model, .05, gpu=False)
        output, force, q, psi, contact_energy = mechanical_oracle(model, len(cpu))
        state = np.fromfile(directory/'state.f64', '<f8')
        n, c = len(model.mass), len(model.barrier)
        metadata['independent_relative_l2'] = {'output': relative(output, cpu[:, 0]), 'first_contact_force': relative(force, cpu[:, 2]),
                                               'final_coordinates': relative(q, state[:n]), 'final_auxiliary': relative(psi, state[3*n:3*n+c])}
        errors = metadata['independent_relative_l2']
        if max(errors[key] for key in ('output', 'final_coordinates', 'final_auxiliary')) > 1e-5 or errors['first_contact_force'] > .05:
            raise RuntimeError(f'Independent mechanical solve failed: {name}: {metadata}')
        metadata['peak_contact_energy_joules'] = contact_energy
        write_json(directory/'checks.json', metadata)
        if contact_energy == 0: raise RuntimeError(f'No contact exercised: {name}')
        checks[name] = metadata
    write_json(OUTPUT/'independent-checks.json', checks)
    return checks


def paper_checks(binary):
    checks = {}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for column, stiffness in enumerate((1e6, 1e8, 1e10)):
        name = f'figure1-k{stiffness:.0e}'
        model = mass_barrier(stiffness=stiffness, exponent=1.3, frequency=100)
        directory, cpu, _, metadata = render(binary, name, model, .03, gpu=False)
        author = author_oracle('massBarr', .03, directory, stiffness=stiffness, overrides={'al': 1.3, 'f0': 100, 'tol0': 1e-14})
        metadata['author_output_relative_l2'] = relative(author[:, 0], cpu[:, 0])
        metadata['author_force_relative_l2'] = relative(author[:, 2], cpu[:, 2])
        if max(metadata['author_output_relative_l2'], metadata['author_force_relative_l2']) > 1e-6:
            raise RuntimeError(f'Figure 1 source comparison failed: {metadata}')
        checks[name] = metadata
        time = np.arange(len(cpu))/model.rate
        axes[0, column].plot(time, cpu[:, 0]*1000, label='N-IT port')
        axes[0, column].plot(time, author[:, 4]*1000, '--', label='Author IT-2')
        axes[0, column].plot(time, author[:, 5]*1000, ':', label='Author IT-1')
        axes[0, column].set(title=f'Kη={stiffness:.0e}', ylabel='Displacement (mm)')
        axes[1, column].plot(time, cpu[:, 2]); axes[1, column].set(xlabel='Time (s)', ylabel='Collision force (N)')
    axes[0, 0].legend(); fig.savefig(OUTPUT/'figure1.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    factors = np.array([1, 2, 4, 8, 16, 32])
    for ax, stiffness in zip(axes.flat, (1e2, 1e7, 1e11, 1e15)):
        errors = []
        for factor in factors:
            rate = int(44100*factor)
            model = mass_barrier(rate=rate, stiffness=stiffness)
            # Figure 2 specifies +0.5; the downloadable source instead uses -0.5.
            model.q -= 1.5/rate; model.previous -= 1.5/rate
            _, cpu, _, _ = render(binary, f'figure2-k{stiffness:.0e}-of{factor}', model, .05, gpu=False)
            exact = -(len(cpu)/rate+model.previous[0]/1.5-np.pi*np.sqrt(.01/stiffness))*1.5
            errors.append(abs(cpu[-1, 0]-exact))
        slope = float(-np.polyfit(np.log(factors[:4]), np.log(errors[:4]), 1)[0])
        if not ((1.6 < slope < 2.4) if stiffness <= 1e7 else (.7 < slope < 1.3)):
            raise RuntimeError(f'Unexpected Figure 2 convergence: {stiffness}: {errors}, slope {slope}')
        checks[f'figure2-k{stiffness:.0e}'] = {'oversampling': factors.tolist(), 'absolute_displacement_error_m': errors, 'order_first_four': slope}
        ax.loglog(factors, errors, 'o-'); ax.set(title=f'Kη={stiffness:.0e}; order {slope:.2f}', xlabel='Oversampling', ylabel='Absolute displacement error (m)'); ax.grid(True, which='both')
    fig.savefig(OUTPUT/'figure2.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(4, 1, figsize=(9, 8), constrained_layout=True)
    for ax, stiffness, factor in zip(axes, (1e7, 1e10, 1e12, 1e12), (1, 1, 1, 5)):
        name = f'figure4-k{stiffness:.0e}-of{factor}'
        model = string_model(paper=True, rate=44100*factor, stiffness=stiffness)
        directory, cpu, _, metadata = render(binary, name, model, .025, gpu=False)
        author = author_oracle('stringHammer', .025, directory, paper=True, rate=model.rate, stiffness=stiffness)
        metadata['author_output_relative_l2'] = relative(author[:, 0], cpu[:, 0])
        tolerance = 1e-5
        if factor == 5:
            steps = int(.001/.5*model.rate)
            rounding = steps*np.finfo(float).eps/(1-steps*np.finfo(float).eps)*.001
            sensitivity = []
            for sign in (-1, 1):
                diagnostic = OUTPUT/f'figure4-roundoff-{sign:+d}'; diagnostic.mkdir(exist_ok=True)
                perturbed = author_oracle('stringHammer', .025, diagnostic, paper=True, rate=model.rate, stiffness=stiffness, overrides={'x0': -.001+sign*rounding})
                sensitivity.append(relative(author[:, 0], perturbed[:, 0]))
            metadata['initial_roundoff_probe_m'] = float(rounding)
            metadata['author_roundoff_sensitivity_relative_l2'] = sensitivity
            metadata['nominal_sample_match'] = metadata['author_output_relative_l2'] <= 1e-5
            tolerance = max(tolerance, max(sensitivity))
        if metadata['author_output_relative_l2'] > tolerance: raise RuntimeError(f'Figure 4 source comparison failed: {name}')
        checks[name] = metadata
        time = np.arange(len(cpu))/model.rate*1000
        ax.plot(time, cpu[:, 0]*1000, label='N-IT port'); ax.plot(time, author[:, 0]*1000, ':', label='Author N-IT'); ax.plot(time, author[:, 2]*1000, '--', label='Author IT-2')
        ax.set(title=f'Kη={stiffness:.0e}, fs={model.rate} Hz', xlabel='Time (ms)', ylabel='Displacement (mm)')
    axes[0].legend(); fig.savefig(OUTPUT/'figure4.png', dpi=150); plt.close(fig)
    write_json(OUTPUT/'paper-checks.json', checks)
    return [{'title': 'Figure 4: hammer stiffness and sample rate', 'group': 'Published experiments', 'figure': str((OUTPUT/'figure4.png').relative_to(ROOT)), 'notes': 'The four published stiffness/sample-rate combinations, with executed author N-IT and IT-2 checks.'},
            {'title': 'Figure 1: nonlinear mass–barrier collisions', 'group': 'Published experiments', 'figure': str((OUTPUT/'figure1.png').relative_to(ROOT)), 'notes': 'Published parameters, N-IT port and executed author IT-1/IT-2 comparisons.'},
            {'title': 'Figure 2: convergence against the analytic collision', 'group': 'Published experiments', 'figure': str((OUTPUT/'figure2.png').relative_to(ROOT)), 'notes': 'The published half-step initial alignment and four barrier stiffnesses; error evaluated after collision at 50 ms.'}]


def clean_outputs(directory):
    for name in ('cpu.f64', 'gpu.f32', 'state.f64', 'model.txt', 'oracle.m', 'octave.log'):
        for path in directory.rglob(name):
            path.unlink()
    manifest = directory/'cases.json'
    cases = json.loads(manifest.read_text())['cases'] if manifest.exists() else []
    figures = {(ROOT/case['figure']).resolve() for case in cases if 'figure' in case}
    for path in directory.rglob('comparison.png'):
        if path.resolve() not in figures:
            path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT/'build/ducceschiReproduce')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--checks-only', action='store_true')
    parser.add_argument('--keep-diagnostics', action='store_true', help='Retain raw solver outputs, model inputs, scripts, logs and extra plots')
    args = parser.parse_args()
    for record in json.loads((ROOT/'docs/ReferenceInputs.json').read_text())['files']:
        if record['method'] == 'ducceschi': fetch(ROOT/record['path'], record['url'], record['sha256'], args.offline)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checks = source_checks(args.binary)
    checks.update(instrument_checks(args.binary))
    if args.checks_only:
        write_json(OUTPUT/'metrics.json', checks)
        if not args.keep_diagnostics: clean_outputs(OUTPUT)
        return
    cases = paper_checks(args.binary)
    for name, title, model, script, kind in [
            ('hammer-audio', 'Hammer striking a string', string_model(paper=True), 'stringHammer', 'frets'),
            ('frets-audio', 'String against twelve frets', string_model('frets', True), 'stringFret', 'frets'),
            ('backboard-audio', 'String against a curved backboard', string_model('backboard', True), 'stringFret', 'backboard')]:
        print('Rendering', name, flush=True)
        directory, cpu, _, metadata = render(args.binary, name, model, 1.5)
        author = author_oracle(script, 1.5, directory, True, kind)
        error = relative(author[:, 0], cpu[:, 0]); metadata['author_waveform_relative_l2'] = error
        if error > 1e-4: raise RuntimeError(f'Author audio mismatch: {name}: {error}')
        write_json(directory/'checks.json', metadata)
        cases.append(audio_case(directory, title, author[:, 0], cpu[:, 0], model.rate, 'Paper parameters; author MATLAB and port use the same pickup, duration and playback gain.', 'Executed author MATLAB', 'C++ FP64'))
        checks[name] = metadata
    for name, title, model, duration, notes in [
            ('snare', 'Wire striking a drum membrane', snare(), 1.5, 'Section V wire–membrane coupling. Figure 7 wire position corrected from 0.3 m to 0.03 m; contact exponent 1.3 and stiffness 10¹² N/m^(α+1) are supplied because the caption omits them.'),
            ('tromba', 'Bowed tromba marina with rattling bridge', tromba(), 3., 'Section VI with the predecessor’s Table 1 parameters, α=β=1.3, bow speed −0.2 m/s and force 0.1 N.')]:
        print('Rendering', name, flush=True)
        directory, cpu, metal, metadata = render(args.binary, name, model, duration)
        cases.append(audio_case(directory, title, cpu[:, 0], metal[:, 0], model.rate, notes, 'C++ FP64'))
        checks[name] = metadata
    for name in ('mass-author', 'hammer-paper', 'frets-paper-fd', 'backboard-paper-modal'):
        cases.append({'title': name, 'group': 'Author numerical comparisons', 'figure': str((OUTPUT/name/'comparison.png').relative_to(ROOT)), 'notes': 'Executed author MATLAB and C++ displacement, with the port’s discrete energy residual.'})
    write_json(OUTPUT/'metrics.json', checks)
    write_json(OUTPUT/'native-provenance.json', {'binary': str(args.binary.resolve()), 'sha256': file_hash(args.binary)})
    write_json(OUTPUT/'cases.json', {'paper': 'https://doi.org/10.1121/10.0005008', 'cases': cases})
    if not args.keep_diagnostics: clean_outputs(OUTPUT)


if __name__ == '__main__':
    main()
