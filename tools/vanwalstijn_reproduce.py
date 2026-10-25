#!/usr/bin/env python3
"""Validate both mass-barrier schemes against author code and paper figures; render audio."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pymupdf
from scipy.io import wavfile
from scipy.signal import savgol_filter

from reference_assets import ROOT, fetch, write_json

PAPER = 'https://doi.org/10.1016/j.ifacol.2024.08.283'
FIELDS = ('rate', 'mass', 'stiffness', 'damping', 'barrier', 'kappa', 'exponent', 'restitution', 'max_velocity', 'regularization', 'friction', 'cap', 'modal', 'implicit')
AUTHOR_COLUMNS = {'x': 0, 'y': 1, 'vx': 2, 'vy': 3, 'Fc': 5, 'Ff': 6, 'H': 7, 'Qc': 9}


def relative(reference, actual):
    return float(np.linalg.norm(actual - reference) / max(np.linalg.norm(reference), 1e-30))


def excitation(c, experiment):
    t = np.arange(int(np.ceil(experiment['duration'] * c['rate']))) / c['rate']
    if experiment['drive'] == 'resonance':
        omega = np.sqrt(c['stiffness']/c['mass'] - (c['damping']/(2*c['mass']))**2)
        f = experiment['amplitude'] * np.column_stack([-np.cos(omega*t), np.sin(omega*t)])
    else:
        f = np.column_stack([np.full(len(t), experiment['normal_force']), -experiment['amplitude']*np.cos(2*np.pi*experiment['frequency']*t)])
    if experiment.get('hann'):
        f *= (.5 - .5*np.cos(2*np.pi*t/experiment['duration']))[:, None]
    return f


def simulate(binary, directory, c, experiment, gpu=False, author=False):
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'controls.txt').write_text(' '.join(str(c[key]) for key in FIELDS) + '\n')
    excitation(c, experiment).astype('<f8').tofile(directory/'forces.f64')
    subprocess.run([str(binary), str(directory/'controls.txt'), str(directory/'forces.f64'), str(directory), *(['--gpu'] if gpu else [])], check=True)
    result = np.fromfile(directory/'cpu.f64', dtype='<f8').reshape(-1, 15)
    metadata = json.loads((directory/'native.json').read_text())
    scale = max(abs(metadata['work']), metadata['dissipation'], float(result[:, 7].max()), 1e-30)
    metadata['relative_energy_balance'] = abs(metadata['energy_balance_residual'])/scale
    metadata['max_step_energy_balance'] = float(np.abs(result[:, 12]).max())/scale
    if not np.isfinite(result).all() or metadata['relative_energy_balance'] > 1e-8 or metadata['minimum_contact_loss'] < -1e-12:
        raise RuntimeError(f'Energy or finiteness check failed: {directory}: {metadata}')
    if gpu:
        metal = np.fromfile(directory/'gpu.f32', dtype='<f4').reshape(-1, 15).astype(float)
        metadata['metal_velocity_relative_l2'] = relative(result[:, 2:4], metal[:, 2:4])
        metadata['metal_minimum_contact_loss'] = float(metal[:, 9].min())
        if not np.isfinite(metal).all() or metadata['metal_velocity_relative_l2'] > .01 or metal[:, 9].min() < 0 or metal[:, 11].min() < 0:
            raise RuntimeError(f'Metal check failed: {directory}: {metadata}')
    write_json(directory/'checks.json', metadata)
    reference, comparison = author_oracle(directory, c, result) if author else (None, {})
    return result, reference, {**metadata, **comparison}


def octave_quote(path):
    return "'" + str(path).replace("'", "''") + "'"


def author_oracle(directory, c, actual, require_match=True):
    columns = AUTHOR_COLUMNS if c['implicit'] else {**AUTHOR_COLUMNS, 'psi': 11}
    names = ('m', 'k', 'r', 'xb', 'kap', 'alp', 'cr', 'vim', 'chi', 'thetd')
    assignments = ';'.join(f'par.{name}={c[key]:.17g}' for name, key in zip(names, FIELDS[1:11]))
    script = f"""addpath({octave_quote(ROOT/'references/vanwalstijn')});
f=fopen({octave_quote(directory/'forces.f64')},'rb');a=fread(f,[2,Inf],'double');fclose(f);
inp.x=a(1,:);inp.y=a(2,:);{assignments};
o=sim{'IMP' if c['implicit'] else 'EXP'}func(inp,par,{c['rate']},{c['cap']},{c['modal']});
f=fopen({octave_quote(directory/'author.f64')},'wb');fwrite(f,[{';'.join('o.'+name for name in columns)}],'double');fclose(f);
"""
    (directory/'oracle.m').write_text(script)
    process = subprocess.run(['octave', '--quiet', str(directory/'oracle.m')], capture_output=True, text=True, check=True)
    (directory/'octave.log').write_text(process.stdout + process.stderr)
    if 'maxiter reached' in process.stdout:
        raise RuntimeError(f'Author Newton solver failed to converge: {directory}')
    reference = np.fromfile(directory/'author.f64', dtype='<f8').reshape(-1, len(columns))
    errors, nonfinite = {}, {}
    for i, (name, column) in enumerate(columns.items()):
        finite = np.isfinite(reference[:, i])
        nonfinite[name] = int((~finite).sum())
        errors[name] = relative(reference[finite, i], actual[finite, column])
    # The source evaluates 0/0 for the friction coefficient at zero-load rest.
    if any(nonfinite[name] for name in ('x', 'y', 'vx', 'vy', 'Fc')) or (require_match and max(errors.values()) > 1e-4):
        raise RuntimeError(f'Author comparison failed: {directory}: {errors}, {nonfinite}')
    checks = {'relative_l2': errors, 'author_nonfinite_samples': nonfinite}
    write_json(directory/'author-checks.json', checks)
    return reference, checks


def path_points(drawing):
    items = drawing['items']
    return np.array([list(items[0][1])] + [list(item[2]) for item in items])


def paper_traces(output, drawings):
    traces = {}
    # The pinned PDF stores each plotted sample as a line segment, including dashed paths.
    for name, paths, duration, zero, scale in [
        ('figure3-x', [34], .020, 88.14306640625, 73777.16),
        ('figure3-y', [35, 36, 37, 38], .020, 88.14306640625, 73777.16),
        ('figure4-x', [75], .025, 278.10980224609375, 188182.2),
        ('figure4-y', [76], .025, 278.10980224609375, 188182.2),
    ]:
        segments = []
        for path in paths:
            xy = path_points(drawings[path])
            segments.append(np.column_stack([(xy[:, 0]-342.2370910644531)/(535.6531982421875-342.2370910644531)*duration, (zero-xy[:, 1])/scale]))
        traces[name] = np.concatenate(segments)
        np.savetxt(output/(name+'-published.csv'), traces[name], delimiter=',', header='time_s,displacement_m')
    t, y = traces['figure4-y'].T
    v = savgol_filter(y, 31, 4, deriv=1, delta=1/44100)
    a = savgol_filter(y, 31, 4, deriv=2, delta=1/44100)
    slip = (np.abs(v) > .005) & (t > .001) & (t < .024)
    force = .001*a + .002*v + 1000*y + .07*(3-.025)*np.sign(v)
    design = np.column_stack([np.cos(2*np.pi*100*t), np.sin(2*np.pi*100*t)])
    coefficients = np.linalg.lstsq(design[slip], force[slip], rcond=None)[0]
    write_json(output/'figure4-drive-recovery.json', {'cosine_N': float(coefficients[0]), 'sine_N': float(coefficients[1]),
               'method': 'Invert the tangential equation on slipping segments using a 31-sample fourth-degree Savitzky-Golay derivative and approximately 2.975 N contact load. Round the recovered sinusoid to -0.25*cos(2*pi*100*t) N before simulation.'})
    return traces


def plot_trajectory(output, name, c, low, high, traces):
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    errors = {}
    for i, axis in enumerate(axes):
        time, published = traces[name+'-'+'xy'[i]].T
        actual = np.interp(time, np.arange(len(low))/c['rate'], low[:, i])
        errors['xy'[i]] = relative(published, actual)
        axis.plot(np.arange(len(high))/c['rate'], high[:, i]*1000, color='black', lw=1, label='Implicit / 1024×')
        axis.plot(np.arange(len(low))/c['rate'], low[:, i]*1000, '--', label='Explicit / 44.1 kHz')
        axis.plot(time, published*1000, '.', ms=1.5, label='Published vector trace')
        axis.set(ylabel='xy'[i]+' (mm)')
        axis.legend(fontsize=8)
    axes[-1].set(xlabel='Time (s)')
    fig.tight_layout()
    fig.savefig(output/(name+'.png'), dpi=150)
    plt.close(fig)
    return errors


def convergence(binary, output, c, experiment, oversampling, drawings):
    reference, _, checks = simulate(binary, output/'convergence-reference', {**c, 'rate': c['rate']*oversampling, 'implicit': 1, 'cap': 0}, experiment)
    results = {'reference': checks, 'curves': {}}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for implicit, axis in enumerate(axes):
        for cap in [0, 1]:
            errors, steps = [], []
            for power in range(9):
                factor = 2**power
                p = {**c, 'rate': c['rate']*factor, 'implicit': implicit, 'cap': cap}
                actual, _, _ = simulate(binary, output/f'convergence-{implicit}-{cap}-{factor}', p, experiment)
                expected = reference[::oversampling//factor, :2][:len(actual)]
                errors.append(relative(expected, actual[:, :2]))
                steps.append(1/p['rate'])
            order = float(np.polyfit(np.log(steps[-4:]), np.log(errors[-4:]), 1)[0])
            key = ('implicit' if implicit else 'explicit') + ('-capped' if cap else '-uncapped')
            points = path_points(drawings[(218 if implicit else 471) + 19*cap])[::-1]
            published = 10**(-1 + (74.08807373046875-points[:, 1])/16.60968)
            results['curves'][key] = {'dt': steps, 'relative_l2': errors, 'order_last_four': order,
                                     'published_relative_l2': published.tolist(), 'published_curve_relative_l2': relative(published, np.array(errors))}
            axis.loglog(steps, errors, 'o-', label=f'{"Capped" if cap else "Uncapped"} / order {order:.2f}')
            axis.loglog(steps, published, ':', color=axis.lines[-1].get_color(), label='Published '+('capped' if cap else 'uncapped'))
            if not (.8 < order < 1.2 if not implicit else 1.7 < order < 2.4):
                raise RuntimeError(f'Unexpected convergence order: {key}: {order}')
            print(key, 'convergence order', order, flush=True)
        axis.set(title='Implicit' if implicit else 'Explicit', xlabel='Time step (s)', ylabel='Relative displacement error')
        axis.legend()
        axis.grid(True, which='both', alpha=.2)
    fig.tight_layout()
    fig.savefig(output/'figure5.png', dpi=150)
    plt.close(fig)
    # Retain the uncapped stiff case's source disagreement alongside the matched capped case.
    for cap in [0, 1]:
        directory = output/f'convergence-1-{cap}-1'
        actual = np.fromfile(directory/'cpu.f64', dtype='<f8').reshape(-1, 15)
        _, comparison = author_oracle(directory, {**c, 'implicit': 1, 'cap': cap}, actual, require_match=bool(cap))
        results['author-implicit-'+str(cap)] = comparison
    write_json(output/'convergence.json', results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT/'build/vanwalstijnReproduce')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/reproduction/vanwalstijn')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if not shutil.which('octave'):
        parser.error('Octave is required to execute the pinned author reference')
    sources = json.loads((ROOT/'repros/vanwalstijn/Sources.json').read_text())
    sources['files'] = [record for record in json.loads((ROOT/'docs/ReferenceInputs.json').read_text())['files'] if record['method'] == 'vanwalstijn']
    for record in sources['files']:
        fetch(ROOT/record['path'], record['url'], record['sha256'], args.offline)
    cfg = json.loads((ROOT/'repros/vanwalstijn/Experiments.json').read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'experiments.json', cfg)
    write_json(output/'sources.json', sources)
    paper = pymupdf.open(ROOT/'references/vanwalstijn/paper.pdf')
    traces = paper_traces(output, paper[4].get_drawings())
    checks, cases = {}, []
    for name in ('figure3', 'figure4'):
        experiment = cfg[name]
        c = {**cfg['controls'], **experiment.get('controls', {})}
        for implicit in [0, 1]:
            directory = output/(name+('-implicit' if implicit else '-explicit'))
            actual, _, checks[directory.name] = simulate(args.binary, directory, {**c, 'implicit': implicit}, experiment, gpu=True, author=True)
            if not implicit:
                low = actual
        high, _, _ = simulate(args.binary, output/(name+'-reference'), {**c, 'rate': c['rate']*cfg['oversampling'], 'implicit': 1}, experiment)
        checks[name+'-published'] = plot_trajectory(output, name, c, low, high[::cfg['oversampling']][:len(low)], traces)
        checks[name+'-published']['stick_fraction'] = float(np.mean(low[:, 3] == 0))
        print(name, checks[name+'-published'], flush=True)
        if max(checks[name+'-published'][key] for key in ('x', 'y')) > .02:
            raise RuntimeError(f'Published trajectory mismatch: {name}')
    literal = {**cfg['controls'], **cfg['figure4']['controls'], 'barrier': 2e-8}
    literal_result, _, _ = simulate(args.binary, output/'figure4-caption-literal', literal, cfg['figure4'])
    time, published = traces['figure4-x'].T
    checks['figure4-caption-literal'] = {'x_relative_l2': relative(published, np.interp(time, np.arange(len(literal_result))/literal['rate'], literal_result[:, 0]))}
    convergence(args.binary, output, {**cfg['controls'], **cfg['figure5']['controls']}, cfg['figure5'], cfg['oversampling'], paper[5].get_drawings())
    # Exercise the original excitation-2 defaults and both optional algorithm switches.
    for implicit in [0, 1]:
        c = {**cfg['controls'], 'implicit': implicit, 'cap': 0, 'modal': 0}
        experiment = {**cfg['figure4'], 'duration': .02, 'normal_force': 1, 'amplitude': 1.1}
        directory = output/f'author-excitation2-{implicit}'
        _, _, checks[directory.name] = simulate(args.binary, directory, c, experiment, author=True)
    for name, experiment_name, implicit, overrides in [
        ('inelastic-collisions', 'figure3', 0, {}),
        ('elastic-collisions', 'figure3', 0, {'restitution': .7, 'friction': .15}),
        ('stick-slip-explicit', 'figure4', 0, {}),
        ('stick-slip-implicit', 'figure4', 1, {}),
    ]:
        experiment = {**cfg[experiment_name], 'duration': cfg['audio_seconds']}
        c = {**cfg['controls'], **experiment.get('controls', {}), **overrides, 'implicit': implicit}
        directory = output/name
        actual, reference, checks[name] = simulate(args.binary, directory, c, experiment, gpu=True, author=True)
        metal = np.fromfile(directory/'gpu.f32', dtype='<f4').reshape(-1, 15)
        signals = [reference[:, 2:4].sum(axis=1), metal[:, 2:4].sum(axis=1)]
        gain = .8/max(float(np.abs(signal).max()) for signal in signals)
        fade = np.ones(len(actual))
        count = round(.01*c['rate'])
        fade[:count] = np.linspace(0, 1, count)
        fade[-count:] = np.linspace(1, 0, count)
        wavs = []
        for label, signal in zip(('author', 'metal'), signals):
            path = directory/(label+'.wav')
            wavfile.write(path, c['rate'], (signal*gain*fade).astype(np.float32))
            rms = float(np.sqrt(np.mean((signal*gain*fade)**2)))
            if not np.isfinite(signal).all() or rms < .001:
                raise RuntimeError(f'Empty or nonfinite audio: {path}')
            wavs.append({'path': str(path.relative_to(ROOT)), 'rms': rms, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        checks[name].update(playback_gain=gain, audio=wavs)
        cases.append({'title': name.replace('-', ' ').capitalize(), 'source_url': PAPER,
                      'reference': str(directory/'author.wav'), 'synthesis': str(directory/'metal.wav'),
                      'reference_label': 'Executed author MATLAB', 'synthesis_label': 'Our Metal '+('implicit' if implicit else 'explicit')+' scheme',
                      'notes': 'Two-second drive; audio is vx + vy with a shared gain and 10 ms fades. '+
                      ('Normal force 3 N, tangential drive 0.25 N at 100 Hz, friction 0.07.' if experiment_name == 'figure4' else f"0.1 N quadrature drive at resonance; restitution {c['restitution']}, friction {c['friction']}.")})
        print(name, 'author error', max(checks[name]['relative_l2'].values()), 'Metal velocity error', checks[name]['metal_velocity_relative_l2'], flush=True)
    write_json(output/'checks.json', checks)
    cases += [{'title': 'Published '+name.replace('figure', 'Figure '), 'figure': str((output/(name+'.png')).relative_to(ROOT)),
               'source_url': PAPER, 'notes': cfg[name].get('provenance', 'Displacement compared with the published vector paths.')} for name in ('figure3', 'figure4', 'figure5')]
    write_json(output/'cases.json', {'title': 'van Walstijn et al. 2024', 'cases': cases})
    print('Wrote', output/'cases.json')


if __name__ == '__main__':
    main()
