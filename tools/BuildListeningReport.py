#!/usr/bin/env python3
"""Build an offline audio comparison from an explicitly labeled JSON case manifest."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import quote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch, stft

from AnalyzeRenders import metrics, read_wave, texture_metrics


def spectrum(rate, samples):
    mono = samples.mean(axis=1)
    frequency, power = welch(mono - mono.mean(), rate, nperseg=min(8192, len(mono)))
    return frequency, power


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/reproduction/listening"))
    parser.add_argument("--freeze", action="store_true", help="Create a new review snapshot that this tool will refuse to overwrite")
    args = parser.parse_args()
    if (args.output / "Freeze.json").exists() or (args.freeze and args.output.exists()):
        parser.error("Review output already exists or is frozen; choose a new --output directory")
    manifest = json.loads(args.manifest.read_text())
    cases = manifest["cases"]
    generated_wavs = {(args.output / 'audio' / f'{index:02d}-{field}-{kind}.wav').resolve()
                      for index in range(len(cases)) for field in ('reference', 'synthesis') for kind in ('raw', 'audition')}
    if any(Path(case[field]).resolve() in generated_wavs for case in cases for field in ('reference', 'synthesis')):
        parser.error('Report destinations overlap source WAVs; choose a different --output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    assets = args.output / "audio"
    assets.mkdir(exist_ok=True)
    records, sections = [], []
    for index, case in enumerate(cases):
        group = case.get("group")
        if group and (index == 0 or group != cases[index - 1].get("group")):
            sections.append(f'<h2>{html.escape(group)}</h2>')
        record = {"title": case["title"], "notes": case["notes"], "signals": {}}
        time_frequency = case.get("time_frequency", False)
        fig, axes = plt.subplots(4 if time_frequency else 2, 1, figsize=(9, 9 if time_frequency else 4.5), constrained_layout=True)
        players = []
        for field, label, color in [("reference", "Author / upstream reference", "#355e91"), ("synthesis", "Our synthesis", "#b45309")]:
            label = case.get(field + "_label", label)
            path = Path(case[field])
            rate, samples = read_wave(path)
            data = {**metrics(rate, samples), "texture": texture_metrics(rate, samples)}
            mono = samples.mean(axis=1)
            ac = samples - samples.mean(axis=0)
            rms = float(np.sqrt(np.mean(ac ** 2)))
            peak = float(np.max(np.abs(ac)))
            gain = min(.1 / rms if rms else 1, .89 / peak if peak else 1)
            name = f"{index:02d}-{field}"
            raw_name = name + "-raw.wav"
            audition_name = name + "-audition.wav"
            if args.freeze:
                subprocess.run(['/bin/cp', '-c', str(path), str(assets / raw_name)], check=True)
                raw_url = 'audio/' + raw_name
            else:
                raw_url = quote(os.path.relpath(path.resolve(), args.output.resolve()), safe='/')
            (assets / audition_name).unlink(missing_ok=True)
            wavfile.write(assets / audition_name, rate, (ac * gain).astype(np.float32))
            data.update(source=str(path), audition_gain=gain)
            record["signals"][field] = data
            players.append(f'<div><strong>{html.escape(label)}</strong><audio controls preload="none" data-raw="{raw_url}" data-level="audio/{audition_name}" src="audio/{audition_name}"></audio><a href="{raw_url}">Raw WAV</a></div>')
            window = max(1, rate // 100)
            count = len(mono) // window
            envelope = np.sqrt(np.mean((mono[:count * window].reshape(count, window) - mono.mean()) ** 2, axis=1))
            axes[0].plot((np.arange(count) + .5) * window / rate, envelope / rms if rms else envelope, label=label, color=color, linewidth=.8)
            frequency, power = spectrum(rate, samples)
            power /= max(float(np.trapezoid(power, frequency)), 1e-30)
            axes[1].semilogx(frequency[1:], 10 * np.log10(np.maximum(power[1:], 1e-16)), label=label, color=color, linewidth=.9)
            if time_frequency:
                frequencies, times, transform = stft(mono / max(rms, 1e-30), rate, nperseg=min(4096, len(mono)))
                level = 20 * np.log10(np.maximum(abs(transform), 1e-6))
                axis = axes[2 if field == "reference" else 3]
                axis.pcolormesh(times, frequencies, level, vmin=-60, vmax=0, shading="auto", cmap="magma")
                axis.set(title=label + " — spectrum over time", xlabel="Time (s)", ylabel="Frequency (Hz)",
                         ylim=(40, min(10000, rate / 2)), yscale="log")
        axes[0].set(xlabel="Time (s)", ylabel="10 ms RMS / record RMS")
        axes[1].set(xlabel="Frequency (Hz)", ylabel="Normalized PSD (dB/Hz)", xlim=(20, 20000))
        if time_frequency:
            for axis in axes[2:]:
                axis.set_xlim(0, max(item["duration_seconds"] for item in record["signals"].values()))
        for axis in axes[:2]:
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        figure_name = f"{index:02d}-comparison.png"
        fig.savefig(args.output / figure_name, dpi=130)
        plt.close(fig)
        source = case.get("source_url", "")
        source_link = f'<a href="{html.escape(source, quote=True)}">Source</a>' if source else ""
        sections.append(f'<section id="case-{index}"><h2>{html.escape(case["title"])}</h2><p>{html.escape(case["notes"])} {source_link}</p><div class="players">{"".join(players)}</div><button class="switch">Switch A/B at current time</button><img loading="lazy" src="{figure_name}" alt="Relative amplitude envelope and normalized spectral comparisons"></section>')
        descriptors = [('frame_top3_bin_fraction_median', 'Narrowband concentration'), ('amplitude_kurtosis', 'Amplitude kurtosis'), ('envelope_cv', 'Envelope variation'), ('local_envelope_rms', 'Local envelope fluctuation')]
        rows = []
        for key, label in descriptors:
            values = [record['signals'][field]['texture'][key] for field in ['reference', 'synthesis']]
            cells = ''.join('<td>' + (f'{value:.4g}' if value is not None else 'n/a') + '</td>' for value in values)
            rows.append(f'<tr><td>{label}</td>{cells}</tr>')
        sections[-1] = sections[-1].replace('</section>', '<details><summary>Texture measures</summary><p>Computed over each complete case WAV using channel power, without stereo phase cancellation. Diagnostic differences, not perceptual equivalence scores.</p><table><tr><th>Descriptor</th><th>Reference</th><th>Ours</th></tr>' + ''.join(rows) + '</table></details></section>')
        records.append(record)
    (args.output / "Metrics.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SurfaceAudio sound comparisons</title>
<style>body{font:16px/1.5 system-ui;max-width:1050px;margin:32px auto;padding:0 20px;color:#202a35;background:#fafafa}section{background:white;border:1px solid #ddd;border-radius:8px;padding:22px;margin:24px 0}h1{font-size:28px}h2{font-size:21px}audio{display:block;width:100%;margin:8px 0}.players{display:grid;grid-template-columns:1fr 1fr;gap:24px}img{width:100%;height:auto}button{margin:12px 0;padding:8px}a{color:#355e91}@media(max-width:600px){.players{grid-template-columns:1fr}}</style>
<h1>SurfaceAudio sound comparisons</h1><p>Each case identifies its reference as an input recording, a published author example, or executed author code. Shared input recordings are evaluation material, not author-generated outputs of the methods being compared.</p>
<p><label><input id="level" type="checkbox" checked> Match playback level and remove each channel's constant DC offset</label><br>Playback copies target AC RMS 0.1 with a 0.89 peak ceiling. Only one constant gain and DC offset are applied per file. Raw WAVs preserve the case files before this playback processing; case notes identify any earlier reference extraction or gain conversion. Plots normalize level for comparison; they are not perceptual equivalence scores. Spectrograms use each record's AC RMS and a common −60 to 0 dB color scale.</p>'''
    if "title" in manifest:
        page = page.replace("SurfaceAudio sound comparisons", html.escape(manifest["title"]))
    if "description" in manifest:
        page = page.replace("Each case identifies its reference as an input recording, a published author example, or executed author code. Shared input recordings are evaluation material, not author-generated outputs of the methods being compared.", html.escape(manifest["description"]))
    page += '<details><summary>All comparisons</summary><ol>' + ''.join(f'<li><a href="#case-{index}">{html.escape(case["title"])}</a></li>' for index, case in enumerate(cases)) + '</ol></details>'
    page += "\n".join(sections)
    page += '''<script>
document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>document.querySelectorAll('audio').forEach(b=>{if(b!==a)b.pause()})));
document.querySelector('#level').addEventListener('change',e=>document.querySelectorAll('audio').forEach(a=>{a.pause();a.src=e.target.checked?a.dataset.level:a.dataset.raw}));
document.querySelectorAll('.switch').forEach(button=>button.addEventListener('click',()=>{const players=button.closest('section').querySelectorAll('audio');const from=players[0].paused?players[1]:players[0],to=from===players[0]?players[1]:players[0];const t=from.currentTime;from.pause();const start=()=>{to.currentTime=Number.isFinite(to.duration)?Math.min(t,Math.max(0,to.duration-.01)):t;to.play().catch(()=>{})};if(to.readyState>=1)start();else{to.addEventListener('loadedmetadata',start,{once:true});to.load()}}));
</script></html>'''
    (args.output / "index.html").write_text(page)
    if not args.freeze:
        sources = {Path(case[field]).resolve() for case in cases for field in ("reference", "synthesis")}
        retained = {assets / f"{index:02d}-{field}-audition.wav" for index in range(len(cases)) for field in ("reference", "synthesis")}
        for path in assets.glob("*.wav"):
            if path not in retained and path.resolve() not in sources:
                path.unlink()
    if args.freeze:
        files = {str(path.relative_to(args.output)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in args.output.rglob("*") if path.is_file()}
        (args.output / "Freeze.json").write_text(json.dumps({"manifest": manifest, "files_sha256": files}, indent=2) + "\n")
    print(f"Wrote {len(records)} listening comparisons to {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
