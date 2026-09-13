#!/usr/bin/env python3
"""Build the offline paper catalog, or a standalone comparison from a JSON manifest."""
import argparse
import fcntl
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
from types import SimpleNamespace
from urllib.parse import quote, unquote

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, welch, stft

from AnalyzeRenders import metrics, read_wave, texture_metrics
from ListeningCatalog import ROOT, OUTPUTS, PAPERS, catalog


def url(path, output):
    return quote(os.path.relpath(Path(path).resolve(), output.resolve()), safe='/')


def audio_links(root):
    return {(page.parent / unquote(html.unescape(link))).resolve()
            for page in root.rglob('*.html')
            for link in re.findall(r'data-(?:raw|level)="([^"]+)"', page.read_text())}


def playback_url(rate, samples, output, cache):
    encoded = io.BytesIO()
    wavfile.write(encoded, rate, samples.astype(np.float32))
    payload = encoded.getvalue()
    path = cache / (hashlib.sha256(payload).hexdigest() + '.wav')
    if not path.exists():
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(payload)
        temporary.replace(path)
    return url(path, output)


def comparison_plot(signals, output, time_frequency):
    key = json.dumps([(s['sha256'], label) for label, s in signals]) + str(time_frequency)
    name = 'comparison-' + hashlib.sha256(key.encode()).hexdigest()[:20] + '.png'
    path = output / name
    if path.exists():
        return name
    fig, axes = plt.subplots(4 if time_frequency else 2, 1, figsize=(9, 9 if time_frequency else 4.5), constrained_layout=True)
    for index, (label, data) in enumerate(signals):
        rate, samples = read_wave(ROOT / data['source'])
        mono = samples.mean(axis=1)
        mono -= mono.mean()
        rms = float(np.sqrt(np.mean(mono ** 2)))
        window = max(1, rate // 100)
        count = len(mono) // window
        envelope = np.sqrt(np.mean(mono[:count * window].reshape(count, window) ** 2, axis=1))
        color = ['#355e91', '#b45309'][index]
        axes[0].plot((np.arange(count) + .5) * window / rate, envelope / max(rms, 1e-30), label=label, color=color, linewidth=.8)
        frequency, power = welch(mono, rate, nperseg=min(8192, len(mono)))
        power /= max(float(np.trapezoid(power, frequency)), 1e-30)
        axes[1].semilogx(frequency[1:], 10 * np.log10(np.maximum(power[1:], 1e-16)), label=label, color=color, linewidth=.9)
        if time_frequency:
            frequencies, times, transform = stft(mono / max(rms, 1e-30), rate, nperseg=min(4096, len(mono)))
            axes[index + 2].pcolormesh(times, frequencies, 20 * np.log10(np.maximum(abs(transform), 1e-6)),
                                      vmin=-60, vmax=0, shading='auto', cmap='magma')
            axes[index + 2].set(title=label, xlabel='Time (s)', ylabel='Frequency (Hz)', ylim=(40, min(10000, rate / 2)), yscale='log')
    axes[0].set(xlabel='Time (s)', ylabel='10 ms RMS / record RMS')
    axes[1].set(xlabel='Frequency (Hz)', ylabel='Normalized PSD (dB/Hz)', xlim=(20, 20000))
    for axis in axes[:2]:
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return name


def cohort_plot(case, output):
    key = json.dumps([case['cohort'], case['band_edges_hz']], sort_keys=True)
    name = 'cohort-' + hashlib.sha256(key.encode()).hexdigest()[:20] + '.png'
    if (output / name).exists():
        return name
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    edges = np.asarray(case['band_edges_hz'])
    for key, label, color in [('author', 'Author generated', '#355e91'), ('generated', 'Our generated', '#b45309')]:
        rows = case['cohort'][key]
        db = 10 * np.log10(np.maximum([row['normalized_band_power'] for row in rows], 1e-12))
        centers = np.sqrt(edges[:-1] * edges[1:])
        axes[0].semilogx(centers, np.median(db, axis=0), label=label, color=color)
        axes[0].fill_between(centers, *np.quantile(db, [.1, .9], axis=0), color=color, alpha=.15)
        times = np.sort([row['features']['t90_seconds'] for row in rows])
        axes[1].step(times, np.arange(1, len(times) + 1) / len(times), where='post', label=label, color=color)
    axes[0].set(xlabel='Frequency (Hz)', ylabel='Normalized band power (dB)')
    axes[1].set(xlabel='Time to 90% energy (s)', ylabel='Cumulative fraction')
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.savefig(output / name, dpi=130)
    plt.close(fig)
    return name


def build_report(args, cache, manifest):
    papers = manifest.get('papers', [{'id': 'comparisons', 'title': manifest.get('title', 'Audio comparisons'),
                                      'subtitle': '', 'description': manifest.get('description', ''),
                                      'lineage': '', 'cases': manifest.get('cases', [])}])
    args.output.mkdir(parents=True, exist_ok=True)
    signals, records, sections, figures = {}, [], [], set()
    previous = args.output / 'Metrics.json'
    previous = json.loads(previous.read_text()) if previous.exists() else {}
    previous = previous.get('signals', {}) if isinstance(previous, dict) else {}
    player_count = 0

    def player(source, label):
        nonlocal player_count
        path = (ROOT / source).resolve()
        source = os.path.relpath(path, ROOT)
        if source not in signals:
            rate, samples = read_wave(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            old = previous.get(source, {})
            data = old if old.get('sha256') == digest else {**metrics(rate, samples), 'texture': texture_metrics(rate, samples)}
            raw = path
            if args.freeze:
                raw = cache / (digest + '-raw.wav')
                if not raw.exists():
                    shutil.copyfile(path, raw)
            raw_url = url(raw, args.output)
            playback_rate, playback = rate, samples
            if rate < 8000:
                playback_rate = 48000
                divisor = math.gcd(rate, playback_rate)
                playback = resample_poly(samples, playback_rate // divisor, rate // divisor, axis=0)
            unscaled = playback_url(playback_rate, playback, args.output, cache) if rate < 8000 else raw_url
            ac = playback - playback.mean(axis=0)
            rms, peak = float(np.sqrt(np.mean(ac ** 2))), float(np.max(np.abs(ac)))
            gain = min(.1 / rms if rms else 1, .89 / peak if peak else 1)
            matched = playback_url(playback_rate, ac * gain, args.output, cache)
            signals[source] = {**data, 'source': source, 'sha256': digest, 'audition_gain': gain,
                               'playback_sample_rate': playback_rate, 'raw_url': raw_url, 'unscaled_url': unscaled, 'matched_url': matched}
        data = signals[source]
        player_count += 1
        panel = (f'<div class="player"><label class="player-label" for="audio-{player_count}">{html.escape(label)}</label>'
                 f'<audio id="audio-{player_count}" controls preload="none" data-raw="{data["unscaled_url"]}" '
                 f'data-level="{data["matched_url"]}" src="{data["matched_url"]}"></audio>'
                 f'<a class="download" download href="{data["raw_url"]}">Raw WAV · {data["duration_seconds"]:.2f} s · {data["sample_rate"]:,} Hz</a></div>')
        return panel, data

    lineage = None
    for paper in papers:
        if paper['lineage'] != lineage:
            lineage = paper['lineage']
            if lineage:
                sections.append(f'<h2>{html.escape(lineage)}</h2>')
        links = []
        source = paper.get('source_url', '')
        references = list(dict.fromkeys(case['source_url'] for case in paper['cases'] if case.get('source_url') and case['source_url'] != source))
        references = paper.get('reference_urls') or (references if len(references) <= 2 else [])
        targets = [('Source code' if 'github.com' in source else 'Paper', source)]
        targets += [('Reference material' + (f' {index + 1}' if len(references) > 1 else ''), target) for index, target in enumerate(references)]
        if not args.freeze:
            targets.append(('Method and reproduction instructions', paper.get('doc')))
        for label, target in targets:
            if target:
                href = target if target.startswith(('https://', 'http://')) else url(ROOT / target, args.output)
                links.append(f'<a href="{html.escape(href, quote=True)}">{label}</a>')
        body = [f'<div class="paper-intro"><p>{html.escape(paper["description"])}</p><div class="links">{"".join(links)}</div></div>']
        group = None
        for index, case in enumerate(paper['cases']):
            if case.get('group') != group:
                group = case.get('group')
                if group and group != 'Comparisons':
                    body.append(f'<h3>{html.escape(group)}</h3>')
            case_id = f'{paper["id"]}-{index + 1}'
            record = {'paper': paper['id'], 'id': case_id, 'title': case['title'], 'signals': []}
            body.append(f'<section class="case" id="{case_id}"><h4>{html.escape(case["title"])}</h4>')
            if case.get('notes'):
                body.append(f'<p class="case-note">{html.escape(case["notes"])}</p>')
            if 'cohort' in case:
                columns = []
                for key, label in [('author', 'Author generated'), ('generated', 'Our generated')]:
                    panels = []
                    for number, row in enumerate(case['cohort'][key], 1):
                        panel, data = player(row['wav'], f'{label} · {number}')
                        panels.append(panel)
                        record['signals'].append({'label': f'{label} · {number}', 'source': data['source']})
                    columns.append('<div><h4>' + label + '</h4>' + ''.join(panels) + '</div>')
                figure = cohort_plot(case, args.output)
                body.append('<details class="cohort-audio"><summary>Listen to both collections</summary><div class="cohorts">' + ''.join(columns) + '</div></details>')
                analysis = '<p>Spectra show the median and 10–90% range. Energy-decay distributions use a common one-second analysis window.</p>'
                switch = ''
            else:
                panels, comparison = [], []
                for field, default in [('reference', 'Reference'), ('synthesis', 'Our synthesis')]:
                    label = case.get(field + '_label', default)
                    panel, data = player(case[field], label)
                    panels.append(panel)
                    comparison.append((label, data))
                    record['signals'].append({'label': label, 'source': data['source']})
                body.append('<div class="players">' + ''.join(panels) + '</div>')
                figure = comparison_plot(comparison, args.output, case.get('time_frequency', False))
                descriptors = [('frame_top3_bin_fraction_median', 'Narrowband concentration'), ('amplitude_kurtosis', 'Amplitude kurtosis'),
                               ('envelope_cv', 'Envelope variation'), ('local_envelope_rms', 'Local envelope fluctuation')]
                rows = []
                for key, label in descriptors:
                    cells = ''.join('<td>' + (f'{data["texture"][key]:.4g}' if data['texture'][key] is not None else 'n/a') + '</td>' for _, data in comparison)
                    rows.append(f'<tr><td>{label}</td>{cells}</tr>')
                analysis = '<table><thead><tr><th>Texture descriptor</th>' + ''.join(f'<th>{html.escape(label)}</th>' for label, _ in comparison)
                analysis += '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>'
                analysis += '<p>Descriptors use the complete WAV and channel power. Envelope and spectral plots use the channel mean.</p>'
                if case.get('time_frequency'):
                    analysis += '<p>Spectrograms use AC RMS normalization and a shared −60 to 0 dB scale.</p>'
                switch = '<button class="switch" type="button">Switch A/B at current time</button>'
            figures.add(figure)
            body.append(f'<div class="actions">{switch}<details class="analysis"><summary>Signal analysis</summary>'
                        f'<img loading="lazy" src="{figure}" alt="{html.escape(case["title"], quote=True)}: spectral and temporal comparison">'
                        f'{analysis}</details></div><p class="status" role="status"></p></section>')
            records.append(record)
        count = len(paper['cases'])
        if not count:
            body.append('<p>No retained audio is available. See the reproduction instructions to generate this paper’s examples.</p>')
        noun = 'example' if count == 1 else 'examples'
        sections.append(f'<details class="paper" id="{paper["id"]}"><summary><span><span class="paper-title">{html.escape(paper["title"])}</span>'
                        f'<span class="subtitle">{html.escape(paper["subtitle"])}</span></span><span class="state"><span class="open-label">Expand +</span>'
                        f'<span class="close-label">Collapse −</span><span class="count">{count} {noun}</span></span></summary>'
                        f'<div class="paper-body">{"".join(body)}</div></details>')
        print(f'{paper["title"]}: {count} examples', flush=True)
    page = (ROOT / 'docs/Listen.html').read_text()
    replacements = {'TITLE': html.escape(manifest.get('title', 'SurfaceAudio · Listening comparisons')),
                    'COUNTS': f'{len(papers)} papers · {len(records)} examples', 'CONTENT': '\n'.join(sections)}
    page = re.sub(r'\{\{(TITLE|COUNTS|CONTENT)\}\}', lambda match: replacements[match[1]], page)
    page = re.sub(r'<title>.*?</title>', lambda _: '<title>' + replacements['TITLE'] + '</title>', page)
    if records:
        page = re.sub(r'<section id="setup">.*?</section>\n', '', page, flags=re.S)
        page = page.replace('<div id="report" hidden>', '<div id="report">')
    else:
        page = page.replace('../outputs/reproduction/listening/index.html', url(OUTPUTS / 'listening/index.html', args.output))
    (args.output / 'Metrics.json').write_text(json.dumps({'signals': signals, 'cases': records}, indent=2, allow_nan=False) + '\n')
    (args.output / 'Cases.json').write_text(json.dumps(manifest, indent=2) + '\n')
    temporary = args.output / 'index.tmp'
    temporary.write_text(page)
    temporary.replace(args.output / 'index.html')
    for path in args.output.glob('*.png'):
        if path.name not in figures:
            path.unlink()
    if args.freeze:
        files = {str(path.relative_to(args.output)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in args.output.rglob('*') if path.is_file()}
        (args.output / 'Freeze.json').write_text(json.dumps({'manifest': manifest, 'files_sha256': files}, indent=2) + '\n')
    else:
        retained = audio_links(cache.parent) | {(ROOT / source).resolve() for source in signals}
        for path in cache.glob('*.wav'):
            if path.resolve() not in retained:
                path.unlink()
    print(f'Wrote {args.output / "index.html"} ({player_count} players, {len(signals)} source WAVs)')


def write_report(output=OUTPUTS / 'listening', manifest=None, freeze=False):
    args = SimpleNamespace(output=Path(output).resolve(), freeze=freeze)
    canonical = {OUTPUTS / entry[0] / 'listening' for entry in PAPERS}
    canonical |= {OUTPUTS / name / 'listening' for name in ('agarwal-response', 'agarwal-response-aligned', 'retained-responses', 'contact-responses')}
    if not args.freeze and (args.output.resolve() in canonical or args.output.resolve() == OUTPUTS / 'listening'):
        manifest, args.output = None, OUTPUTS / 'listening'
    if (args.output / 'Freeze.json').exists() or (args.freeze and args.output.exists()):
        raise ValueError('Review output already exists or is frozen; choose a new --output directory')
    cache = args.output / 'audio' if args.freeze else (ROOT / 'outputs' if args.output.resolve().is_relative_to(ROOT / 'outputs') else args.output.resolve().parent) / 'playback'
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        build_report(args, cache, catalog() if manifest is None else manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', nargs='?', type=Path, help='Case manifest for an explicit standalone --output or --freeze')
    parser.add_argument('--output', type=Path, default=OUTPUTS / 'listening')
    parser.add_argument('--freeze', action='store_true', help='Create a self-contained snapshot in a new directory')
    args = parser.parse_args()
    write_report(args.output, json.loads(args.manifest.read_text()) if args.manifest else None, args.freeze)


if __name__ == '__main__':
    main()
