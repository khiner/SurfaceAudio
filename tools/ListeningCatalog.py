"""Curate the retained paper comparisons for the public listening page."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / 'outputs/reproduction'

# Each entry names the reference, the local result, and the limit of the comparison.
PAPERS = [
    ('traer', 'Surface contact and object responses', 'Traer et al. · 2019', 'Statistical impacts and spatially varying responses', 'Traer',
     'Comparisons execute later TDW author code, which descends from this model. The scrape compares two local response models.'),
    ('agarwal', 'Surface contact and object responses', 'Agarwal et al. · 2021', 'Scraping and rolling from surface geometry', 'Agarwal',
     'Published audio is compared with fits using inferred motion, synthetic TDW profiles, and local resonators. '
     'Rolling texture remains substantially different. The four retained rolling fits use the vertical excitation term.'),
    ('agarwal2023', 'Surface contact and object responses', 'Agarwal, Traer & McDermott · 2023', 'Object-response fitting and material distributions', 'AgarwalResponseReproduction',
     'Twenty measured responses are fitted directly. Independently generated material samples compare distributions, with five training recordings per material.'),
    ('agarwal2025', 'Surface contact and object responses', 'Agarwal · 2022 poster / 2025 thesis', 'Finite micro-impacts in continuous contact', 'Agarwal2025',
     'All pairs compare local model settings. The thesis excitation uses responses fitted to the public 2023 corpus; exact thesis stimuli are unavailable.'),
    ('agarwal2026', 'Surface contact and object responses', 'Agarwal et al. · 2026', 'Impact-force coupling and statistical object acoustics', 'Agarwal2026',
     'Response fits use the twenty public 2023 recordings. The 1,502-response survey and fitted category distributions are unavailable. '
     'Impact and material controls compare local outputs.'),
    ('conan', 'Stochastic contact and texture', 'Conan et al. · 2014 · TASLP', 'Rolling force and velocity control', 'Conan',
     'Author forces and audio are compared with independent stochastic realizations. Full-record fits and held-out examples are identified separately. '
     'The velocity variants use different assumptions about unpublished excitation variance.'),
    ('continuous', 'Stochastic contact and texture', 'Conan et al. · 2014 · CMJ', 'Rubbing, scratching, rolling, and transitions', 'Continuous',
     'References are excerpts from the author demonstration. Gestures and resonators are inferred; transition timing remains approximate.'),
    ('hatt', 'Stochastic contact and texture', 'Culbertson et al. · 2014 · HaTT', 'Haptic texture vibration', 'Hatt',
     'Executed author interpolation and a reference AR filter share Gaussian innovations with our renderer. '
     'These are tool-acceleration signals, with a 5 kHz bandwidth, rather than an airborne-sound model.'),
    ('sdt', 'Friction and bowed strings', 'Baldan et al. · 2017 · SDT', 'Impact, rolling, scraping, and friction', 'Sdt',
     'Executed upstream code is compared with our source-convention and Metal implementations. '
     'The refined energy bound changes force near sticking; source-convention results preserve the upstream comparison.'),
    ('willemsen', 'Friction and bowed strings', 'Willemsen, Bilbao & Serafin · 2019', 'Elasto-plastic friction on a stiff string', 'Willemsen',
     'The original MATLAB figure-generating code runs in Octave. Both players contain dry string displacement, without instrument-body filtering.'),
    ('matusiak2024', 'Friction and bowed strings', 'Matusiak & Chatziioannou · 2024', 'Finite bow width, hair compliance, and torsion', 'Matusiak2024',
     'Author simulations and measured robot forces are recovered from PDF figure vectors. '
     'Numerical conventions are identified beside each output. The players audition bridge force.'),
    ('falaize', 'Friction and bowed strings', 'Falaize & Roze · 2024 / 2025', 'Energy-based bow and hammer interactions', 'Falaize',
     'All players are local equation-based variants; author audio is unavailable. '
     'Stiffness and state-evaluation ambiguities are exposed as separate comparisons. See the method notes for energy-balance limits.'),
    ('matusiak', 'Friction and bowed strings', 'Matusiak et al. · 2025', 'Passive distributed bow–string friction', 'Matusiak',
     'Executed author code is compared with our distributed model. '
     'Our Equation 59 diagonal correction is isolated in a second author-code comparison; CPU/Metal precision is shown separately.'),
    ('poirot', 'Collisions and deformation', 'Poirot et al. · 2023', 'Rattle, buzz, and obstacle collisions', 'Poirot',
     'Author stimuli are compared with the collision signal model. Calibrated cases use a corrected recipient weight and fitted splitting parameters. '
     'The literal Equation 16 and physical-reference cases remain separate.'),
    ('nakatsuka', 'Collisions and deformation', 'Nakatsuka et al. · 2017', 'Friction sound with surface deformation', 'Nakatsuka',
     'Reference excerpts come from the authors’ 2016 predecessor demonstration. '
     'Our 2017 scene reconstruction uses inferred geometry, motion, and adhesion settings.'),
    ('lagrange', 'Contact analysis and resynthesis', 'Lagrange, Scavone & Depalle · 2010', 'Excitation and resonance estimation', 'Lagrange',
     'FoleyAutomatic recordings are shared evaluation inputs. The first two reconstructions jointly fit excitation and audio, a local extension. '
     'Statistical event compression is excluded.'),
    ('lee', 'Contact analysis and resynthesis', 'Lee, Depalle & Scavone · 2010', 'Source–filter rolling analysis', 'Lee',
     'These use FoleyAutomatic inputs from another paper. Author synthesis examples and matched settings remain unavailable. '
     'The reconstructions differ substantially and demonstrate an unresolved reproduction gap.'),
]


def paper_links(doc):
    text = (ROOT / f'docs/{doc}.md').read_text()
    paper = re.search(r'^\[paper\]: (\S+)', text, re.M)
    if not paper:
        paper = re.search(r'\]\((https://[^)]+)\)', text)
    if not paper:
        paper = re.search(r'^\[source\]: (\S+)', text, re.M)
    return paper[1] if paper else ''


def curate(case, paper, variant=''):
    title = case['title']
    labels = [case.get('reference_label', 'Published author example'), case.get('synthesis_label', 'Our synthesis')]
    group, notes = 'Comparisons', ''
    if paper == 'agarwal':
        group = 'Rolling reconstructions' if 'retained' in title else 'Response and excitation comparisons'
        title = title.removeprefix('Agarwal ').replace('author-profile/material baseline', 'TDW-profile material baseline')
        title = title.replace(' (whole-record fit)', '').replace(' (temporal Gaussian)', '')
        labels[1] = 'Executed PyImpact output' if 'provided PyImpact' in title else 'Our whole-record fit'
        if 'retained' in title:
            notes = 'Vertical excitation with cell or pointwise surface sampling.'
            title = title.removeprefix('retained ').replace('-', ' · ') + ' surface sampling'
    elif paper == 'conan':
        title = title.removeprefix('Conan ').replace('author proposed-scheme ', '').replace('published velocity ', '')
        group = ('Velocity: reconstructed variance' if variant.endswith('eps-variance') else 'Velocity: published coefficients') if variant else 'Force and response comparisons'
        labels = ['Author rolling audio' if variant or 'response' in title else 'Author force', 'Our stochastic synthesis']
        if 'ForceSynth' in title:
            labels[0] = 'Author synthesized force'
    elif paper == 'sdt':
        title = title.removeprefix('SDT Pd ').replace('friction_startup', 'Friction startup')
        group = 'Additional reference' if title.startswith('Historical') else 'Executed upstream comparisons'
        labels = ['Historical SDT recording' if group == 'Additional reference' else 'Executed upstream SDT',
                  'Our Metal implementation' if title.endswith('/ metal') else 'Our source-convention implementation']
        if group == 'Additional reference':
            notes = 'The recording uses unidentified older settings; the current preset is an unmatched reference.'
    elif paper == 'matusiak2024':
        group = ('CPU / Metal precision' if 'precision' in title else
                 'Measured robot force' if 'versus measured' in title else 'Published simulation')
        title = re.sub(r' versus (author_s|author_epsilon|measured)$', '', title.removeprefix('Matusiak 2024 '))
        title = title.removesuffix(' s') + ' · S-model' if title.endswith(' s') else title.replace(' epsilon', ' · epsilon-model')
        title = title.replace('full-record CPU/Metal precision diagnostic', 'Full-record bridge force')
    elif paper == 'matusiak':
        title = title.removeprefix('Matusiak 2025: ')
        labels[0] = case.get('reference_label', 'Executed original author code')
        labels[1] = case.get('synthesis_label', 'Our model with Equation 59 correction')
        group = 'CPU / Metal precision' if 'Metal' in title else 'Executed author code'
        title = title.replace('isolated equation correction oracle', 'Author code with Equation 59 correction')
    elif paper == 'traer':
        title = title.removeprefix('Traer descendant TDW / ').removeprefix('Traer ')
        labels = ['Executed TDW author code', 'Our impact model']
        if 'Equations' in title:
            group, labels = 'Local spatial-response control', ['Our constant response', 'Our spatially varying response']
    elif paper == 'poirot':
        title = title.removeprefix('Poirot ').replace('source_calibrated', 'Calibrated signal model').replace('literal_eq16', 'Literal Equation 16')
        labels[0] = 'Author physical model' if 'physical reference' in title else 'Author signal model'
        if 'physical reference' in title:
            labels[1] = 'Our independent finite-difference model'
            notes = 'The observation point is inferred; the authors’ finite-difference code is unavailable.'
    elif paper == 'agarwal2023':
        title = case.get('name', title).replace('_', ' ')
        group, labels = 'Measured response fits', ['Author measured response', 'Our fitted response']
    elif paper == 'agarwal2026':
        if 'response fit' in title:
            group, title = 'Measured response fits · 2023 inputs', title.split(' / ')[-1].replace('_', ' ')
        else:
            group = 'Local material and impact controls'
            title = title.removeprefix('Agarwal 2026 / ').removeprefix('Agarwal 2026 ')
    elif paper == 'agarwal2025':
        group, title = 'Local excitation and object controls', title.removeprefix('Agarwal thesis / ')
    elif paper == 'hatt':
        title = title.split(' / ')[1]
        labels = ['Author interpolation + reference AR filter', 'Our C++ / Metal vibration']
    elif paper in ('lagrange', 'lee'):
        title = re.sub(r'^(Lagrange 2010 / FoleyAutomatic |Lee 2010 cross-paper input / )', '', title)
    elif paper == 'continuous':
        title = title.removeprefix('Conan CMJ2014 / ')
        labels[0] = 'Author demonstration excerpt'
    elif paper == 'nakatsuka':
        title = title.removeprefix('Nakatsuka 2017 / ').removesuffix(' / reconstructed scene')
    elif paper == 'falaize':
        title = title.removeprefix('Falaize–Roze ')
    return {**case, 'title': title[0].upper() + title[1:], 'notes': notes, 'group': group,
            'reference_label': labels[0], 'synthesis_label': labels[1]}


def causal_case(case):
    return {**case, 'group': 'Causal response fits',
            'reference': case['prepared_reference'], 'synthesis': case['fitted_response'],
            'reference_label': 'Author response after onset alignment and DC removal',
            'synthesis_label': 'Our fitted causal response'}


def cohort_cases(data):
    return [{'title': material, 'group': 'Independent material samples',
             'notes': 'Two independent collections of twenty samples; there is no one-to-one correspondence.',
             'cohort': cohort, 'band_edges_hz': data['band_edges_hz']}
            for material, cohort in data.get('materials', {}).items()]


def catalog():
    papers = []
    for key, lineage, title, subtitle, doc, description in PAPERS:
        directories = [key]
        if key == 'conan':
            directories += ['conan/velocity', 'conan/velocity-eps-variance']
        if key == 'agarwal2023':
            directories = ['agarwal-response-aligned', 'agarwal-response', 'retained-responses']
        cases, manifests, seen = [], [], set()
        for directory in directories:
            path = OUTPUTS / directory / 'cases.json'
            if not path.exists():
                continue
            manifests.append(str(path.relative_to(ROOT)))
            for case in json.loads(path.read_text())['cases']:
                if key == 'agarwal2023':
                    pair = tuple(hashlib.sha256((ROOT / case[field]).read_bytes()).hexdigest() for field in ('reference', 'synthesis'))
                    if pair in seen:
                        continue
                    seen.add(pair)
                cases.append(curate(case, key, directory if directory != key else ''))
                if key == 'agarwal2023' and 'prepared_reference' in case:
                    cases.append(causal_case(cases[-1]))
        if key == 'agarwal2023':
            for directory in directories:
                path = OUTPUTS / directory / 'metrics.json'
                if not path.exists():
                    continue
                data = json.loads(path.read_text())
                cases.extend(cohort_cases(data))
                if data.get('materials'):
                    break
            contact = OUTPUTS / 'contact-responses/cases.json'
            if contact.exists():
                manifests.append(str(contact.relative_to(ROOT)))
                for case in json.loads(contact.read_text())['cases']:
                    if 'source_manifest' not in case:
                        cases.append({**case, 'group': 'Local continuous-contact response controls',
                                      'notes': 'The same synthesized force drives measured and fitted responses. No matching author recording is available.'})
        groups = list(dict.fromkeys(case['group'] for case in cases))
        if key == 'agarwal':
            groups.sort(key=lambda group: group != 'Rolling reconstructions')
        elif key == 'matusiak2024':
            groups.sort(key=lambda group: group == 'CPU / Metal precision')
        cases.sort(key=lambda case: groups.index(case['group']))
        papers.append({'id': key, 'lineage': lineage, 'title': title, 'subtitle': subtitle, 'description': description,
                       'doc': f'docs/{doc}.md', 'source_url': paper_links(doc),
                       'reference_urls': re.findall(r'^\[(?:author|release)\]: (\S+)', (ROOT / f'docs/{doc}.md').read_text(), re.M),
                       'manifests': manifests, 'cases': cases})
    return {'title': 'SurfaceAudio · Listening comparisons', 'papers': papers}
