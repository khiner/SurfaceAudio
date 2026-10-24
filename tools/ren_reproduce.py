#!/usr/bin/env python3
"""Render Ren 2010 textures, compare independent equations, and publish listening cases."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter

from ren_model import fractal_table, mesh_modes, normal_map
from reference_assets import ROOT, write_json

PAPER = 'https://doi.org/10.1109/VR.2010.5444799'
CONTACT = np.dtype([(name, '<f4') for name in ('u', 'v', 'vx', 'vy', 'vn', 'load', 'macro')] + [('overlap', '<u4')])
BINDING = np.dtype([('nodes', '<u4', (3,)), ('weights', '<f4', (3,))])


def relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))


def block_mesh(size, bulge):
    vertices = np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
                         [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]], dtype=float) * np.array(size)/2
    faces = [(0, 1, 2, 3), (7, 6, 5, 4), (4, 5, 1, 0), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    normals = [(0, 0, 1), (0, 0, -1), (0, -1, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0)]
    centers = [vertices[list(face)].mean(axis=0) + bulge*np.array(normal) for face, normal in zip(faces, normals)]
    triangles = [[a, face[(i+1) % 4], 8+f] for f, face in enumerate(faces) for i, a in enumerate(face)]
    return np.vstack([vertices, centers]), np.array(triangles)


def bind_path(vertices, triangles, x, y):
    # The four top faces cover the complete rectangular XY footprint.
    bindings = np.zeros(len(x), dtype=BINDING)
    assigned = np.zeros(len(x), dtype=bool)
    for triangle in triangles[:4]:
        corners = vertices[triangle, :2]
        matrix = np.vstack([corners.T, np.ones(3)])
        weights = np.linalg.solve(matrix, np.vstack([x, y, np.ones(len(x))])).T
        use = (weights.min(axis=1) >= -1e-10) & ~assigned
        weights = np.maximum(weights[use], 0)
        bindings['nodes'][use] = triangle
        bindings['weights'][use] = weights / weights.sum(axis=1, keepdims=True)
        assigned[use] = True
    if not assigned.all():
        raise ValueError('Contact path leaves the object')
    return bindings


def texture(name, count):
    u, v = np.meshgrid(np.arange(count)/count, np.arange(count)/count)
    if name == 'wood':
        return .00008*np.sin(2*np.pi*(14*u + .2*np.sin(2*np.pi*v))) + .000025*np.sin(2*np.pi*37*u)
    if name == 'tile':
        return -.0004*(np.exp(-np.sin(2*np.pi*u)**2/.012) + np.exp(-np.sin(2*np.pi*v)**2/.012))
    return .0001*np.sin(2*np.pi*9*u)*np.sin(2*np.pi*7*v) - .0003*np.exp(-np.sin(2*np.pi*2*v)**2/.015)


def gesture(name, cfg):
    rate = cfg['sample_rate']
    time = np.arange(4*rate)/rate
    c = np.zeros(len(time), dtype=CONTACT)
    active = (time >= .25) & (time < 3.25)
    t = np.clip(time - .25, 0, 3)
    x = -.09*np.cos(2*np.pi*t/1.5)
    y = .013*np.sin(2*np.pi*t/2.1)
    if name in ('impacts', 'rolling'):
        if name == 'impacts':
            events = np.array([.25, .85, 1.45, 2.05, 2.65])
            impulse = cfg['slider_mass_kg'] * .1
        else:
            # An externally driven regular 24-sided roller exchanges normal momentum at each facet transition.
            radius, speed, facets = .012, .12, 24
            events = np.arange(.25, 3.25, 2*np.pi*radius/(facets*speed))
            impulse = 2*cfg['slider_mass_kg']*speed*np.sin(np.pi/facets)
            path_radius = speed*3/(2*np.pi)
            x, y = path_radius*np.cos(2*np.pi*t/3), path_radius*np.sin(2*np.pi*t/3)
        samples = np.rint(events*rate).astype(int)
        c['macro'][samples] = impulse
        c['overlap'][samples] = 1
        c['vn'][samples] = -impulse/(2*cfg['slider_mass_kg'])
    else:
        vx = .09*2*np.pi/1.5*np.sin(2*np.pi*t/1.5)
        vy = .013*2*np.pi/2.1*np.cos(2*np.pi*t/2.1)
        if name == 'along_grain':
            x, y, vx, vy = y, x*.7, vy, vx*.7
        c['vx'], c['vy'] = vx*active, vy*active
        c['load'] = cfg['normal_force_N'] * active
        c['overlap'] = active
        c['vn'] = -c['load']/cfg['slider_mass_kg']/100
        c['macro'][round(.25*rate)] = cfg['slider_mass_kg'] * .04
    c['u'], c['v'] = x/cfg['texture_extent_m'][0] + .5, y/cfg['texture_extent_m'][1] + .5
    return c, x, y


def oracle_impulses(c, normals, table, cfg):
    rate, mass = cfg['sample_rate'], cfg['slider_mass_kg']
    speed = np.hypot(c['vx'].astype(float), c['vy'].astype(float))
    contact = (c['overlap'] != 0) & (c['vn'] < 0)
    lasting = contact & (speed > 0)
    # Match the source's FP32 texel addressing; the momentum calculation itself is FP64.
    x = np.floor((c['u'] - np.floor(c['u']))*normals.shape[1]).astype(int) % normals.shape[1]
    y = np.floor((c['v'] - np.floor(c['v']))*normals.shape[0]).astype(int) % normals.shape[0]
    n = normals[y, x].astype(float)
    momentum = -mass*(c['vx']*n[:, 0] + c['vy']*n[:, 1])/n[:, 2]
    meso = np.r_[0, np.diff(-momentum)] * (lasting & np.r_[False, lasting[:-1]])
    increments = np.where(lasting, speed/(rate*cfg['roughness_spacing_m']), 0)
    phase = np.r_[0, np.cumsum(increments[:-1])] % len(table)
    first = phase.astype(int)
    value = table[first] + (phase - first)*(table[(first+1) % len(table)] - table[first])
    micro = cfg['roughness_gain'] * c['load'] * value/rate * lasting
    return np.column_stack([c['macro']*contact, meso, micro])


def oracle_audio(impulses, bindings, modes, shapes, rate):
    drive = np.sum(shapes[bindings['nodes']] * bindings['weights'][:, :, None], axis=1) * impulses[:, None]
    audio = np.zeros(len(drive))
    for i, (frequency, decay, amplitude) in enumerate(modes.astype(float)):
        radius, angle = np.exp(-1/(rate*decay)), 2*np.pi*frequency/rate
        audio += lfilter([0, amplitude*radius*np.sin(angle)], [1, -2*radius*np.cos(angle), radius**2], drive[:, i])
    return audio


def run_case(binary, directory, cfg, mesh, response, normal, table, c, bindings):
    inputs = directory / 'input'
    inputs.mkdir(parents=True, exist_ok=True)
    modes, shapes, modal_checks = response
    last_contact = np.flatnonzero(c['overlap'])[-1] + 1
    frames = max(len(c), last_contact + int(np.ceil(np.log(1000)*modes[:, 1].max()*cfg['sample_rate'])))
    if frames > len(c):
        bindings = np.concatenate([bindings, np.repeat(bindings[-1:], frames-len(c))])
        c = np.pad(c, (0, frames-len(c)))
    normal = normal.astype('<f4')
    table = table.astype('<f4')
    dtype = np.dtype([('values', '<f4', (4,)), ('dimensions', '<u4', (3,))])
    config = np.zeros(1, dtype=dtype)
    config['values'] = [cfg['sample_rate'], cfg['slider_mass_kg'], cfg['roughness_gain'], cfg['roughness_spacing_m']]
    config['dimensions'] = [normal.shape[1], normal.shape[0], len(table)]
    for filename, data in [('parameters.bin', config), ('normals.f32', normal), ('roughness.f32', table), ('contacts.bin', c), ('bindings.bin', bindings), ('modes.f32', modes), ('shapes.f32', shapes)]:
        data.tofile(inputs / filename)
    write_json(inputs/'mesh.json', mesh)
    subprocess.run([str(binary), str(inputs), str(directory)], check=True)
    actual = np.fromfile(directory/'impulses.f32', dtype='<f4').reshape(-1, 3).astype(float)
    expected = oracle_impulses(c, normal, table, cfg)
    rate, audio = wavfile.read(directory/'audio.wav')
    reference_audio = oracle_audio(expected.sum(axis=1), bindings, modes, shapes, rate)
    checks = {**modal_checks, **json.loads((directory/'native.json').read_text()),
              'equation_relative_l2': {key: relative(expected[:, i], actual[:, i]) for i, key in enumerate(('macro', 'meso', 'micro'))},
              'audio_oracle_relative_l2': relative(reference_audio, audio),
              'peak': float(np.max(np.abs(audio))), 'rms': float(np.sqrt(np.mean(audio.astype(float)**2))),
              'tail_seconds': (len(c)-last_contact)/rate,
              'last_20ms_peak_fraction': float(np.max(np.abs(audio[-rate//50:])) / np.max(np.abs(audio)))}
    checks['wav_sha256'] = hashlib.sha256((directory/'audio.wav').read_bytes()).hexdigest()
    write_json(directory/'checks.json', checks)
    if (not np.isfinite(audio).all() or checks['rms'] <= 0 or max(checks['equation_relative_l2'].values()) > .002
            or checks['audio_oracle_relative_l2'] > .003 or modal_checks['eigen_relative_residual'] > 1e-9
            or checks['last_20ms_peak_fraction'] > .003):
        raise RuntimeError(f'Ren validation failed: {directory.name}: {checks}')
    print(directory.name, checks['audio_oracle_relative_l2'], flush=True)
    return directory/'audio.wav', checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT/'build/renReproduce')
    parser.add_argument('--config', type=Path, default=ROOT/'repros/ren/Scenes.json')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/reproduction/ren')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'scenes.json', cfg)
    vertices, triangles = block_mesh(cfg['mesh_size_m'], cfg['face_bulge_m'])
    cases, checks, materials = [], {}, {}
    pickup = np.zeros(3*len(vertices))
    pickup[3*8+2] = 1
    for material, parameters in cfg['materials'].items():
        modes, shapes, modal_checks = mesh_modes(vertices, triangles, parameters['young_Pa'], parameters['density_kg_m3'], cfg['shell_thickness_m'], parameters['gamma'], parameters['eta'], pickup, cfg['sample_rate'])
        response = modes.astype('<f4'), shapes[2::3].astype('<f4'), modal_checks
        mesh = {'vertices': vertices.tolist(), 'triangles': triangles.tolist(), 'material': parameters}
        height = texture(material, cfg['texture_resolution'])
        normal = normal_map(height, cfg['texture_extent_m'])
        table = fractal_table(cfg['roughness_samples'], parameters['noise_exponent'], cfg['seed'])
        materials[material] = mesh, response, normal, table
        fig, ax = plt.subplots(figsize=(5, 4))
        plot = ax.imshow(height*1e3, origin='lower', extent=[0, cfg['texture_extent_m'][0], 0, cfg['texture_extent_m'][1]], cmap='copper')
        fig.colorbar(plot, ax=ax, label='Height (mm)')
        ax.set(xlabel='U distance (m)', ylabel='V distance (m)', title=f'{material.capitalize()} height map (mm)')
        fig.tight_layout()
        fig.savefig(output/f'{material}-texture.png', dpi=130)
        plt.close(fig)
    scenes = [(material, action) for material in materials for action in (('scrape', 'along_grain') if material == 'wood' else ('scrape',))]
    for material, gesture_name in scenes + [('tile', 'impacts'), ('tile', 'rolling')]:
        mesh, response, normal, table = materials[material]
        c, x, y = gesture(gesture_name, cfg)
        bindings = bind_path(vertices, triangles, x, y)
        scraping = gesture_name not in ('impacts', 'rolling')
        name = f'{material}-{gesture_name}' if scraping else gesture_name
        case = {'title': f'{material.capitalize()} / '+gesture_name.replace('_', ' ') if scraping else gesture_name.capitalize(), 'source_url': PAPER}
        if scraping:
            flat = np.zeros_like(normal)
            flat[:, :, 2] = 1
            baseline, checks[name+'-micro'] = run_case(args.binary, output/(name+'-micro'), cfg, mesh, response, flat, table, c, bindings)
            case.update(reference=str(baseline), reference_label='Collision + microscopic roughness', synthesis_label='Full three-level model',
                        notes='Same mesh, motion, load, roughness and gain; the full model adds normal-map momentum impulses.')
        else:
            normal, table = np.array([[[0., 0., 1.]]]), np.zeros(2)
            case.update(synthesis_label='Mesh modal response', notes='Five equal impulses at different surface positions.' if gesture_name == 'impacts' else
                        'Macro collision impulses from a driven 24-sided roller, radius 12 mm and speed 0.12 m/s; zero contact slip gives no microscopic friction excitation.')
        synthesis, checks[name] = run_case(args.binary, output/name, cfg, mesh, response, normal, table, c, bindings)
        cases.append({**case, 'synthesis': str(synthesis)})
    write_json(output/'checks.json', checks)
    write_json(output/'cases.json', {'title': 'Ren–Yeh–Lin 2010', 'cases': cases})
    print('Wrote', output/'cases.json')


if __name__ == '__main__':
    main()
