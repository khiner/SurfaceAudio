#!/usr/bin/env python3
"""Reconstruct Nakatsuka's cloth and copper scenes with explicit missing-input choices."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, welch
from reference_assets import ROOT, fetch


def descriptor(path):
    rate, samples = wavfile.read(path)
    if samples.ndim>1:
        samples = samples.mean(axis=1)
    if not np.isfinite(samples).all() or not np.any(samples):
        raise ValueError(f'Empty or nonfinite audio: {path}')
    divisor=math.gcd(rate,44100)
    samples=resample_poly(samples.astype(float),44100//divisor,rate//divisor)
    envelope=np.sqrt(np.mean(samples[:len(samples)//441*441].reshape(-1,441)**2,axis=1))
    envelope/=envelope.max()
    frequency, power = welch(samples,44100,nperseg=2048)
    band = (frequency>=30)&(frequency<=15000)
    normalized = power[band]/max(power[band].sum(),1e-30)
    edges=np.geomspace(80,15000,32)
    bands=np.array([power[(frequency>=lo)&(frequency<hi)].sum() for lo,hi in zip(edges[:-1],edges[1:])])
    bands/=bands.sum()
    return dict(centroid_hz=float(np.sum(frequency[band]*normalized)),
                flatness=float(np.exp(np.log(np.maximum(power[band],1e-30)).mean())/max(power[band].mean(),1e-30)),
                active_fraction=float(np.mean(envelope>.1)),peak_time_s=float(envelope.argmax()*.01)),envelope,bands


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,default=ROOT/'build/nakatsukaReproduce')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/reproduction/nakatsuka')
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--keep-diagnostics',action='store_true')
    args=parser.parse_args()
    config=json.loads((ROOT/'repros/nakatsuka/cases.json').read_text())
    reference=ROOT/'references/nakatsuka'
    for source in config['sources']:
        fetch(reference/source['file'],source['url'],source['sha256'],args.offline)
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    settings=output/'settings.txt'
    with settings.open('w') as file:
        file.write(str(len(config['cases']))+'\n')
        for case in config['cases']:
            scene = [case['scene'][key] for key in ('columns','rows','speed_m_s','sphere_radius_m','initial_height_m','shifted_adhesion','motion_start_s','motion_duration_s','settling_time_s','origin_y_m','motion_rise_s','motion_fall_s','adhesion','adhesion_distance_m')]
            material = [case['material'][key] for key in ('modulus_pa','poisson_ratio','areal_density_kg_m2','thickness_m','width_m','height_m','width_stddev_m','height_stddev_m')]
            file.write(' '.join(map(str,[*scene,*material,case['reference_speed'],case['odd_modes']]))+'\n')
    subprocess.run([str(args.binary),str(settings),str(output),str(config['seconds'])],check=True)
    cases=[]
    metrics=json.loads((output/'render.json').read_text())
    for i,case in enumerate(config['cases']):
        author=output/(case['name']+'-author.wav')
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss',str(case['author_start']),'-i',str(reference/'demo.mp4'),'-t',str(config['seconds']),'-vn','-ac','1','-c:a','pcm_f32le',str(author)],check=True)
        ours=output/f'{i}.wav'
        author_metrics,author_envelope,author_bands=descriptor(author)
        ours_metrics,ours_envelope,ours_bands=descriptor(ours)
        comparison=dict(envelope_rmse=float(np.sqrt(np.mean((author_envelope-ours_envelope)**2))),
                        spectral_rmse_db=float(np.sqrt(np.mean((10*np.log10(np.maximum(author_bands,1e-12))-10*np.log10(np.maximum(ours_bands,1e-12)))**2))))
        metrics['cases'][i].update(author=author_metrics,reconstruction=ours_metrics,comparison=comparison)
        for key,limit in config['comparison_limits'].items():
            if not np.isfinite(comparison[key]) or comparison[key]>limit:
                raise RuntimeError(f"{case['name']} {key}: {comparison[key]} exceeds {limit}")
        cases.append(dict(title=f"Nakatsuka 2017 / {case['name']} / reconstructed scene",reference=str(author),synthesis=str(ours),time_frequency=True,
                          reference_label='Author 2016 predecessor demo excerpt',synthesis_label='Our 2017 equation-based scene reconstruction',
                          notes='Surface contact points resist sliding and release during motion. Motion windows approximate the author excerpt timing; geometry, thickness and adhesion settings are inferred. This is a scene reconstruction, not an exact author-result reproduction.'))
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    manifest=output/'cases.json'
    manifest.write_text(json.dumps(dict(cases=cases),indent=2)+'\n')
    subprocess.run([sys.executable,str(ROOT/'tools/BuildListeningReport.py'),str(manifest),'--output',str(output/'listening')],check=True)
    if not args.keep_diagnostics:
        settings.unlink()


if __name__=='__main__':
    main()
