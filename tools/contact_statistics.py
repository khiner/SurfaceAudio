#!/usr/bin/env python3
"""Reproduce Ponthus separation statistics and Zouabi correlated bouncing distributions."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

from reference_assets import ROOT, fetch, require_packages


def figure_source(method, figure, offline):
    import pymupdf
    reference = next(r for r in json.loads((ROOT / 'docs/ReferenceInputs.json').read_text())['files'] if r['method'] == method)
    source = fetch(ROOT / reference['path'], reference['url'], reference['sha256'], offline)
    executable = shutil.which('gs')
    if not executable:
        raise RuntimeError('Install Ghostscript to read the author vector figures: brew install ghostscript')
    eps, pdf = source.parent / f'{figure}.eps', source.parent / f'{figure}.pdf'
    with tarfile.open(source) as archive:
        eps.write_bytes(archive.extractfile(f'{figure}.eps').read())
    subprocess.run([executable, '-q', '-dSAFER', '-dBATCH', '-dNOPAUSE', '-sDEVICE=pdfwrite', '-dEPSCrop',
                    f'-sOutputFile={pdf}', str(eps)], check=True)
    return pymupdf.open(pdf)


def axes(drawings, minimum_width=200, minimum_height=90):
    horizontal = [d['rect'] for d in drawings if d['fill'] is None and d['rect'].width > minimum_width and d['rect'].height < .01]
    return [(r.x0, r.y0, r.x1, b.y0) for r in horizontal for b in horizontal
            if minimum_height < b.y0 - r.y0 < 260 and abs(b.x0 - r.x0) < .01 and abs(b.x1 - r.x1) < .01]


def ponthus_reference(offline):
    import numpy as np
    records = []
    for figure, maximum in [('figure7a', 6), ('figure7b', 1), ('figure7c', 1.2)]:
        with figure_source('ponthus', figure, offline) as doc:
            drawings = doc[0].get_drawings()
            left, top, right, bottom = axes(drawings)[0]
            ticks = [d['rect'] for d in drawings if d['rect'].width < .01 and 2.5 < d['rect'].height < 4
                     and abs(d['rect'].y1 - bottom) < .01]
            x0, x5 = sorted({round(r.x0, 3) for r in ticks})[:2]
            values = []
            for d in drawings:
                r = d['rect']
                if d['fill'] is None and len(d['items']) == 4 and all(item[0] == 'c' for item in d['items']) and left < r.x0 < right:
                    x, y = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
                    values.append([10 ** (5 * (x - x0) / (x5 - x0)), maximum * (bottom - y) / (bottom - top)])
            values = np.array(sorted(values))
            if len(values) != 15:
                raise RuntimeError(f'Unexpected author marker count in {figure}: {len(values)}')
            records.append(values)
    return records


def zouabi_reference(offline):
    import numpy as np
    result = []
    with figure_source('zouabi', 'FIGURE2', offline) as doc:
        drawings = doc[0].get_drawings()
        panels = sorted(axes(drawings), key=lambda r: (round(r[1] / 20), r[0]))
        if len(panels) != 4:
            raise RuntimeError('Expected four published bouncing panels')
        for left, top, right, bottom in panels:
            xticks = sorted({round(d['rect'].x0, 3) for d in drawings if d['color'] and d['color'][0] < .2 and d['rect'].width < .001
                             and 1.5 < d['rect'].height < 2.1 and abs(d['rect'].y1 - bottom) < .01 and left < d['rect'].x0 < right})
            yticks = sorted({round(d['rect'].y0, 3) for d in drawings if d['rect'].height < .001
                             and 2 < d['rect'].width < 3 and abs(d['rect'].x0 - left) < .01 and top < d['rect'].y0 <= bottom + .01}, reverse=True)
            if len(xticks) != 3 or len(yticks) != 3:
                raise RuntimeError('Unexpected author histogram tick calibration')
            x0, x5, _ = xticks
            points = []
            for d in drawings:
                if not d['color'] or not .79 < d['color'][0] < .81:
                    continue
                for item in d['items']:
                    if item[0] != 'l':
                        continue
                    a, b = item[1:]
                    x, y = (a.x + b.x) / 2, (a.y + b.y) / 2
                    if abs(a.y - b.y) < .001 and .3 < abs(a.x - b.x) < 1.2 and left < x < right and top < y < bottom:
                        points.append([5 * (x - x0) / (x5 - x0), .1 * (bottom - y) / (bottom - yticks[1]), 5 * abs(a.x - b.x) / (x5 - x0)])
            points = np.array(sorted(points))
            if len(points) < 200 or not .98 < np.sum(points[:, 1] * points[:, 2]) < 1.02:
                raise RuntimeError('Author histogram extraction lost probability mass')
            result.append(points)
    return result


def zouabi_excitation(offline, output, filters):
    import numpy as np
    import matplotlib.pyplot as plt
    figure, panels = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    errors = {}
    with figure_source('zouabi', 'FIGURE1', offline) as doc:
        drawings = doc[0].get_drawings()
        bounds = sorted(r for r in axes(drawings, 120, 80) if r[1] < 120)
        if len(bounds) != 2:
            raise RuntimeError('Expected two published excitation panels')
        for panel, correlation, (left, top, right, bottom) in zip(panels, (10, 1.11), bounds):
            ticks = sorted({d['rect'].y0 for d in drawings if d['rect'].height < .001 and 0 < d['rect'].width < 4
                            and abs(d['rect'].x0 - left) < .01 and top < d['rect'].y0 < bottom})
            if len(ticks) != 2:
                raise RuntimeError('Unexpected excitation response tick calibration')
            points = []
            for d in drawings:
                if not d['color'] or d['color'][0] > .2:
                    continue
                for item in d['items']:
                    if item[0] != 'l':
                        continue
                    a, b = item[1:]
                    if abs(a.x - b.x) > 1e-5 and abs(a.y - b.y) > 1e-5:
                        points.extend((p.x, p.y) for p in (a, b) if left < p.x < right and top < p.y < bottom)
            points = np.unique(points, axis=0)
            frequency = 10 ** (-1 + 2 * (points[:, 0] - left) / (right - left))
            author = 10 ** (2 * (ticks[0] - points[:, 1]) / (ticks[1] - ticks[0]))
            magnitude = 1 / np.sqrt(((1 - frequency**2)**2 + 4 * frequency**2 / correlation**2) * (1 + frequency**2 / 100))
            dt, transition = filters[correlation]
            matrix = np.array(transition).reshape(3, 4)
            dc = np.linalg.solve(np.eye(3) - matrix[:, :3], matrix[:, 3])[2]
            response = np.array([abs(np.linalg.solve(np.exp(1j * w * np.float32(dt)) * np.eye(3) - matrix[:, :3], matrix[:, 3])[2] / dc)
                                 for w in frequency])
            error = float(np.sqrt(np.mean(np.log10(author / response)**2)))
            if len(points) < 100 or error > .002:
                raise RuntimeError('Published excitation response differs from the specified filter')
            errors[correlation] = {'published_log10_rms': error, 'published_relative_rms': float(np.sqrt(np.mean((response / author - 1)**2))),
                                   'continuous_relative_rms': float(np.sqrt(np.mean((response / magnitude - 1)**2)))}
            panel.loglog(frequency, author, '.', color='.5', ms=3, label='Published analytical curve')
            panel.loglog(frequency, response, lw=1, label='Our RK4 filter')
            panel.set(title=f'τcorr = {correlation:g}', xlabel='Angular frequency / Ωc', ylabel='Filter magnitude / DC gain')
            panel.grid(alpha=.2)
            panel.legend(fontsize=8)
    figure.savefig(output / 'Excitation.png', dpi=160)
    plt.close(figure)
    return errors


def audition(path, samples, input_rate, gain):
    import numpy as np
    from scipy.signal import resample_poly
    from scipy.io import wavfile
    from fractions import Fraction
    ratio = Fraction(44100 / input_rate).limit_denominator(10000)
    output = resample_poly(np.asarray(samples, dtype=float), ratio.numerator, ratio.denominator) * gain
    if not np.isfinite(output).all() or not len(output):
        raise RuntimeError('Invalid sonification output')
    wavfile.write(path, 44100, output.astype(np.float32))


@contextmanager
def run(args, output, *parameters):
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(args.binary), args.method, str(output), *map(str, parameters)], check=True)
    yield json.loads((output / 'Run.json').read_text())
    if not args.keep_diagnostics:
        names = ('track.f32', 'slider.f32', 'separation.f32') if args.method == 'ponthus' else ('statistics.bin', 'trajectory.f32')
        for name in ('Run.json', *names):
            (output / name).unlink()
        if not any(output.iterdir()):
            output.rmdir()


def ponthus(args, output):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import welch
    from scipy.stats import skew
    reference = ponthus_reference(args.offline)
    rows, cases = [], []
    figure, panels = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for i, (values, label) in enumerate(zip(reference, ['Mean separation', 'Standard deviation', 'Skewness'])):
        panels[0, i].semilogx(values[:, 0], values[:, 1], 'o', ms=3, color='.5', label='Published simulations')
        panels[0, i].set(xlabel='N = 4 L² / (π λ₀²)', ylabel=label)
    for nodes in (64, 128, 256, 512, 1024):
        folder = output / str(nodes)
        with run(args, folder, nodes, 15, 4, .46) as record:
            values = np.fromfile(folder / 'separation.f32', dtype='<f4').reshape(15, -1) / np.sqrt(2)
            statistics = np.array([values.mean(axis=1), values.std(axis=1), skew(values, axis=1)]).T
            measured, spread = statistics.mean(axis=0), statistics.std(axis=0, ddof=1)
            author = np.array([np.interp(np.log(record['patches']), np.log(curve[:, 0]), curve[:, 1]) for curve in reference])
            record.update(mean=float(measured[0]), deviation=float(measured[1]), skewness=float(measured[2]),
                          ensemble_deviation=spread.tolist(), published_interpolation=author.tolist(), difference=(measured - author).tolist())
            for i in range(3):
                panels[0, i].errorbar(record['patches'], measured[i], yerr=spread[i], fmt='o', ms=4, color='#1671c5', label='Our 15-draw ensemble' if nodes == 64 else None)
            samples = values[0]
            step = record['spacing'] / record['wavelength']
            f, psd = welch(values - values.mean(axis=1, keepdims=True), fs=1 / step, nperseg=min(2048, values.shape[1]), axis=1)
            panels[1, 0].plot(np.arange(len(samples)) * step, samples, label=f'L/λ₀={record["size_over_wavelength"]:.1f}', lw=.7)
            panels[1, 1].loglog(f[1:], psd.mean(axis=0)[1:], lw=.8)
            panels[1, 2].hist(values.ravel(), bins=70, density=True, histtype='step', lw=.8)
            high = (f > record['wavelength'] * 1.2) & (f < 3 * record['wavelength'])
            record['high_frequency_psd_exponent'] = float(np.polyfit(np.log(f[high]), np.log(psd.mean(axis=0)[high]), 1)[0])
            rows.append(record)
            if nodes in (128, 1024):
                path = output / f'separation-{nodes}.wav'
                # Direct displacement sonification at a stated spatial playback rate; no acoustic response is added.
                audition(path, samples - samples.mean(), 800, .2)
                cases.append({'title': f'Fixed-orientation slider · L/λ₀ = {record["size_over_wavelength"]:.1f}',
                              'synthesis': os.path.relpath(path, ROOT), 'synthesis_label': 'Our separation sonification',
                              'notes': 'Dimensionless separation after mean removal, played at 800 spatial samples per second with gain 0.2. '
                                       'The paper reports geometric statistics and provides no reference audio.'})
    for i, field in enumerate(('mean', 'deviation', 'skewness')):
        for prefix, label, style in [('evt', 'Independent Gaussian maxima', '--'), ('preumont', 'Preumont approximation', ':')]:
            panels[0, i].semilogx([r['patches'] for r in rows], [r.get(f'{prefix}_{field}', 1.1395470994) for r in rows],
                                 style, color='.3', lw=.8, label=label)
    panels[0, 0].legend(fontsize=7)
    panels[1, 0].set(xlabel='Sliding distance / λ₀', ylabel='Separation / (√2 Rq)')
    panels[1, 0].legend(fontsize=7)
    panels[1, 1].set(xlabel='Spatial frequency × λ₀', ylabel='Centered separation PSD')
    panels[1, 2].set(xlabel='Separation / (√2 Rq)', ylabel='Probability density')
    for panel in panels.flat:
        panel.grid(alpha=.2)
    figure.savefig(output / 'Comparison.png', dpi=160)
    plt.close(figure)
    cases.insert(0, {'title': 'Published separation statistics', 'figure': os.path.relpath(output / 'Comparison.png', ROOT),
                     'notes': 'Published circles are recovered from Figure 7 for α=4 and b=0.46. Error bars show variation across fifteen independent surface pairs.'})
    return rows, cases


def zouabi(args, output):
    import numpy as np
    import matplotlib.pyplot as plt
    reference = zouabi_reference(args.offline)
    dtype = np.dtype([('steps', '<u8'), ('sticking', '<u8'), ('counts', '<u4', 4), ('sums', '<f4', 5), ('histogram', '<u4', 200)], align=True)
    rows, cases, gains = [], [], {}
    figure, panels = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    targets = json.loads((ROOT / 'repros/zouabi/Reference.json').read_text())['figure_2_caption_means']
    for i, (acceleration, correlation) in enumerate([(5, 1.11), (.3, 1.11), (5, 10), (.3, 10)]):
        panel = panels.flat[i]
        author = reference[i]
        mass = author[:, 1] * author[:, 2]
        author_mean = float(np.sum(author[:, 0] * mass) / mass.sum())
        author_negative = float(np.sum(mass * np.clip(.5 - author[:, 0] / author[:, 2], 0, 1)) / mass.sum())
        author_cdf = (np.cumsum(mass) - .5 * mass) / mass.sum()
        panel.plot(author[:, 0], author[:, 1], color='.5', label=f'Published histogram · mean {author_mean:.2f}')
        for relative in (False, True):
            folder = output / f'{i+1}-{"thesis" if relative else "letter"}'
            with run(args, folder, acceleration, correlation, 1000, 1000, 524288 if relative else 0, .005, int(relative)) as record:
                data = np.fromfile(folder / 'statistics.bin', dtype=dtype)
                counts = data['counts'].astype(np.uint64).sum(axis=0)
                steps, sticking = int(data['steps'].sum()), int(data['sticking'].sum())
                sums = data['sums'].astype(float).sum(axis=0)
                if len(data) != 1000 or counts[0] != 900000:
                    raise RuntimeError('Bouncing ensemble reached the step limit before collecting every flight')
                if counts[3] != counts[0] or counts[1] > counts[3] or sticking > steps:
                    raise RuntimeError('Invalid take-off pair or sticking counts')
                pdf = data['histogram'].astype(np.uint64).sum(axis=0) / (counts[0] * .1)
                centers = np.arange(200) * .1 - 4.95
                mean = float(sums[0] / counts[0])
                cdf_error = float(np.max(np.abs((np.cumsum(pdf) - .5 * pdf) * .1 - np.interp(centers, author[:, 0], author_cdf))))
                record.update(mean_velocity=mean,
                              published_histogram_mean=author_mean, published_caption_mean=targets[i],
                              published_negative_fraction=author_negative, published_histogram_mass=float(mass.sum()),
                              mean_difference=mean - author_mean, cdf_maximum_difference=cdf_error,
                              mean_flight_time=float(sums[1] / counts[0]), negative_velocity_fraction=float(counts[2] / counts[0]),
                              takeoff_pairs=int(counts[3]), descending_fraction=float(counts[1] / counts[3]), sticking_time_fraction=sticking / steps,
                              plate_mean_squares=(sums[2:] / steps).tolist(), histogram_mass=float(pdf.sum() * .1))
                rows.append(record)
                panel.plot(centers, pdf, label=f'{"Thesis" if relative else "Letter"} sticking · mean {mean:.2f}', lw=1)
                if relative:
                    trace = np.fromfile(folder / 'trajectory.f32', '<f4').reshape(-1, 4)[10000:]
                    deficit = float(np.maximum(trace[:, 0] - trace[:, 2], 0).max())
                    interpolation_bound = np.max(np.abs(np.diff(trace[:, 1]))) * record['dt'] / 8
                    roundoff = 32 * np.finfo(np.float32).eps * max(1., np.max(np.abs(trace[:, [0, 2]])))
                    record.update(maximum_height_deficit=deficit, height_deficit_bound=float(interpolation_bound + roundoff))
                    if not np.isfinite(trace).all() or deficit > interpolation_bound + roundoff:
                        raise RuntimeError('Bouncing trajectory exceeds interpolation and float roundoff bounds')
                    paths = [output / f'plate-{correlation:g}.wav', output / f'ball-{i+1}.wav']
                    if correlation not in gains:
                        gains[correlation] = .7 / max(1., np.max(np.abs(trace[:, [1, 3]])))
                        audition(paths[0], trace[:, 1], 100000, gains[correlation])
                    record['sonification_gain'] = float(gains[correlation])
                    audition(paths[1], trace[:, 3], 100000, gains[correlation])
                    cases.append({'title': f'Λ={acceleration:g}, τcorr={correlation:g} · thesis sticking rule',
                                  'reference': os.path.relpath(paths[0], ROOT), 'synthesis': os.path.relpath(paths[1], ROOT),
                                  'reference_label': 'Our driving-plate velocity', 'synthesis_label': 'Our ball velocity',
                                  'notes': 'Two local velocity sonifications with equal gain and Ωc=500 s⁻¹. '
                                           'Published results are the statistical curves above; these players contain no author audio.'})
        panel.set(title=f'Figure 2({"abcd"[i]}): Λ={acceleration:g}, τcorr={correlation:g}', xlabel='Take-off velocity / σw', ylabel='Probability density', xlim=(-5, 12))
        panel.grid(alpha=.2)
        panel.legend(fontsize=8)
    figure.savefig(output / 'Comparison.png', dpi=160)
    plt.close(figure)
    cases.insert(0, {'title': 'Published take-off velocity distributions', 'figure': os.path.relpath(output / 'Comparison.png', ROOT),
                     'notes': 'Gray curves are extracted from the four published Figure 2 histograms. Each local curve uses 900,000 retained flights. '
                              'Means are integrated from the plotted bars; the caption gives different means for panels (a) and (c). '})
    excitation = zouabi_excitation(args.offline, output, {r['correlation_time']: (r['dt'], r['plate_transition']) for r in rows})
    for record in rows:
        record['excitation_response'] = excitation[record['correlation_time']]
    cases.insert(0, {'title': 'Published excitation filter', 'figure': os.path.relpath(output / 'Excitation.png', ROOT),
                     'notes': 'Figure 1 analytical curves match filter magnitude, although the axes say PSD. '
                              'Our curve evaluates the actual RK4 coefficients uploaded to Metal, with unit DC gain.'})
    return rows, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['ponthus', 'zouabi'], required=True)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/contactStatistics')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--keep-diagnostics', action='store_true')
    args = parser.parse_args()
    require_packages(['numpy', 'scipy', 'matplotlib', 'PyMuPDF'])
    import matplotlib
    matplotlib.use('Agg')
    args.binary = args.binary.resolve()
    output = (args.output or ROOT / f'outputs/reproduction/{args.method}').resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, cases = globals()[args.method](args, output)
    (output / 'Metrics.json').write_text(json.dumps({'method': args.method, 'cases': rows}, indent=2) + '\n')
    (output / 'cases.json').write_text(json.dumps({'cases': cases}, indent=2) + '\n')
    print(f'Comparison: {output / "Comparison.png"}')


if __name__ == '__main__':
    main()
