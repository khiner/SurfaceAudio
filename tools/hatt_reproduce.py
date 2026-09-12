#!/usr/bin/env python3
"""Render the published HaTT models and execute the original interpolation code."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.io import wavfile
from reference_assets import ROOT, extract, fetch


def read_model(path):
    root = ET.parse(path).getroot()
    scalar = lambda key: float(root.findtext(key))
    array = lambda node, key: np.array([float(v.text) for v in node.findall(key + '/value')])
    models = sorted(root.findall('model'), key=lambda m: int(m.findtext('modNum')))
    return dict(name=root.findtext('material'), rate=scalar('SampleRate'), mu=scalar('mu'),
                speed=scalar('maxSpeed'), force=scalar('maxForce'),
                triangles=np.array([[int(v.text)-1 for v in t] for t in root.findall('tri')]),
                models=[dict(speed=float(m.findtext('speedMod')), force=float(m.findtext('forceMod')),
                             variance=float(m.findtext('var')), gain=float(m.findtext('gain') or 1),
                             ar=array(m,'ARlsf'), ma=array(m,'MAlsf')) for m in models])


def prepare(models, frames, path, seed):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((len(models), frames)).astype(np.float32)
    with path.open('w') as out:
        def row(values):
            out.write(' '.join(map(str, values)) + '\n')
        row([len(models), frames])
        for voice, texture in enumerate(models):
            m = texture['models']
            row([texture['rate'], texture['mu'], texture['speed'], texture['force'], len(m), len(texture['triangles']), len(m[0]['ar']), len(m[0]['ma'])])
            for model in m:
                row([model['speed'],model['force'],model['variance'],model['gain'],*model['ar'],*model['ma']])
            for triangle in texture['triangles']:
                row(triangle)
            t = np.arange(frames) / texture['rate']
            speed = texture['speed'] * (.5 + .3*np.sin(2*np.pi*.7*t))
            force = texture['force'] * (.5 + .2*np.sin(2*np.pi*.43*t))
            for sample in zip(speed, force, noise[voice]):
                row(sample)
    return noise


def reference_render(coefficients, noise):
    voices, frames = noise.shape
    output = np.zeros((voices, frames))
    history = np.zeros((voices, 25))
    excitation_history = np.zeros_like(history)
    for frame in range(frames):
        c = coefficients[:, frame]
        excitation = noise[:, frame] * np.sqrt(c[:, 52])
        value = c[:, 26]*excitation - np.sum(c[:, 1:26]*history, axis=1) + np.sum(c[:, 27:52]*excitation_history, axis=1)
        output[:, frame] = value
        history[:, 1:] = history[:, :-1]
        excitation_history[:, 1:] = excitation_history[:, :-1]
        history[:, 0] = value
        excitation_history[:, 0] = excitation
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT/'build/hattReproduce')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--keep-diagnostics', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT/'repros/hatt/cases.json').read_text())
    reference = ROOT/'references/hatt'
    for source in config['sources']:
        archive = fetch(reference/source['file'],source['url'],source['sha256'],args.offline)
        if archive.suffix == '.zip':
            extract(archive, reference/archive.stem, {'.xml','.cpp','.h','.m','.pdf'})
    oracle = reference/'oracle'
    subprocess.run(['/opt/homebrew/opt/llvm/bin/clang++','-std=c++23','-O2','-I'+str(ROOT/'src'),'-I'+str(reference/'rendering/include'),str(ROOT/'tools/hatt_oracle.cxx'),str(reference/'rendering/src/AccSynthHashMatrix.cpp'),'-o',str(oracle)],check=True)
    output = ROOT/'outputs/reproduction/hatt'
    output.mkdir(parents=True,exist_ok=True)
    cases, metrics = [], []
    for rate in (1000,10000):
        models = sorted((read_model(p) for p in (reference/f'models/Models/Models{rate}Hz/XML').glob('*.xml')), key=lambda m:m['name'])
        assert len(models)==100
        folder = output/str(rate)
        folder.mkdir(exist_ok=True)
        frames = int(config['seconds']*rate)
        noise = prepare(models,frames,folder/'input.txt',config['seed'])
        subprocess.run([str(args.binary),str(folder/'input.txt'),str(folder)],check=True)
        coefficients = np.fromfile(folder/'filters.f64',dtype=np.float64).reshape(100,frames,53)
        reference_audio = reference_render(coefficients,noise)
        gpu = np.stack([wavfile.read(folder/f'{voice}.wav')[1] for voice in range(100)])
        relative = np.linalg.norm(gpu-reference_audio,axis=1)/np.maximum(np.linalg.norm(reference_audio,axis=1),1e-30)
        assert np.all(np.isfinite(gpu)) and relative.max()<1e-4,relative.max()
        record = dict(sample_rate=rate,materials=[m['name'] for m in models],maximum_numpy_gpu_relative_l2=float(relative.max()),render=json.loads((folder/'render.json').read_text()))
        executed = subprocess.run([str(oracle),str(folder/'input.txt'),str(folder/'author.f64')],check=True,timeout=120,capture_output=True,text=True)
        record['author_interpolation_messages'] = executed.stdout.splitlines()
        author = np.fromfile(folder/'author.f64',dtype=np.float64).reshape(100,frames,53)
        author_audio = reference_render(author,noise)
        assert np.isfinite(author_audio).all()
        record['author_coefficient_relative_l2'] = float(np.linalg.norm(author-coefficients)/np.linalg.norm(coefficients))
        record['author_audio_relative_l2'] = (np.linalg.norm(gpu-author_audio,axis=1)/np.maximum(np.linalg.norm(author_audio,axis=1),1e-30)).tolist()
        assert record['author_coefficient_relative_l2'] < .001
        limits = config['author_audio_relative_l2_limits'].get(str(rate),{})
        for texture, error in zip(models,record['author_audio_relative_l2']):
            assert error < limits.get(texture['name'],.02), (texture['name'],rate,error)
        if rate==10000:
            for voice,texture in enumerate(models):
                if texture['name'] not in config['listen']:
                    continue
                ours, theirs = folder/f'{voice}.wav', folder/f'{voice}-author.wav'
                wavfile.write(theirs,rate,author_audio[voice].astype(np.float32))
                cases.append(dict(title=f"HaTT 2014 / {texture['name']} / original 10 kHz model",reference=str(theirs),synthesis=str(ours),
                                  reference_label='Executed author interpolation + reference AR filter',synthesis_label='Our C++ / Metal texture vibration',
                                  notes='Identical published model, force/speed trajectory and Gaussian innovations; acceleration in m/s² played as audio. This compares texture vibration, not airborne sound or a recorded author demonstration.'))
        metrics.append(record)
        (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    manifest = output/'cases.json'
    manifest.write_text(json.dumps(dict(cases=cases),indent=2)+'\n')
    subprocess.run([sys.executable,str(ROOT/'tools/BuildListeningReport.py'),str(manifest),'--output',str(output/'listening')],check=True)
    if not args.keep_diagnostics:
        retained={Path(c[field]).resolve() for c in cases for field in ('reference','synthesis')}
        for rate in (1000,10000):
            for path in (output/str(rate)).iterdir():
                if path.suffix in ('.wav','.f64','.txt') and path.resolve() not in retained:
                    path.unlink()


if __name__=='__main__':
    main()
