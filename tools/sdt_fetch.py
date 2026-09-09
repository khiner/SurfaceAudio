#!/usr/bin/env python3
"""Build an unmodified, pinned SDT C oracle and preserve author patch/audio provenance."""
import argparse
import tarfile
import tempfile
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'references/sdt'
REVISION = '0509de418e7bebc8b37866b3b4458e0acc8cf1f4'
DEST.mkdir(parents=True, exist_ok=True)
PACKAGES = [('sdt', 'SkAT-VG/SDT', '0509de418e7bebc8b37866b3b4458e0acc8cf1f4', '4fba31525e98e01b78116c2d8e84da1f8866b99a5e0855b329fc5000d93b901b'), ('json-builder', 'ChromaticIsobar/json-builder', '84eb75a53ef9b093cc8d501a7719a45409c56c6f', 'a22bda778b0286e4e57280fc9a97c3df4d267f9aab2e7e347b362002e8b3087c'), ('json-parser', 'json-parser/json-parser', '94f66d8f83c1d84ccccd7540ec2f51cf8325d272', '2ae5151213b51bf4b1baf8363c498c60f183fc8eb1a86ccc53f669b80d442f4e')]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--offline', action='store_true', help='Build only from verified cached source packages and reference inputs')
args = parser.parse_args()
source = DEST / 'source'
source.mkdir(exist_ok=True)
for name, repository, revision, digest in PACKAGES:
    archive = DEST / 'packages' / (name + '.tar.gz')
    archive.parent.mkdir(exist_ok=True)
    if not archive.exists():
        if args.offline:
            raise FileNotFoundError(archive)
        with urllib.request.urlopen(f'https://codeload.github.com/{repository}/tar.gz/{revision}', timeout=60) as response:
            archive.write_bytes(response.read())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
        raise ValueError(f'Changed source package: {archive}')
    destination = source / name
    with tempfile.TemporaryDirectory() as temporary:
        with tarfile.open(archive) as package:
            package.extractall(temporary, filter='data')
        extracted, = Path(temporary).iterdir()
        if name == 'sdt':
            shutil.copytree(extracted / 'src/SDT', destination / 'src/SDT', dirs_exist_ok=True)
            for mode in ['rolling', 'scraping', 'friction', 'impact']:
                for folder, name in [('Pd', mode + '~-help.pd'), ('MaxPackage/help', 'sdt.' + mode + '~.maxhelp')]:
                    shutil.copyfile(extracted / folder / name, DEST / name)
            for file in ['AUTHORS.txt', 'COPYING.txt', 'LICENSE.txt']:
                shutil.copyfile(extracted / file, destination / file)
        else:
            destination.mkdir(exist_ok=True)
            for file in extracted.iterdir():
                if file.is_file() and (file.suffix in ('.c', '.h') or file.name.upper().startswith(('LICENSE', 'COPYING', 'AUTHORS'))):
                    shutil.copyfile(file, destination / file.name)
repo = source / 'sdt'
builder, parser_source = source / 'json-builder', source / 'json-parser'
compiler = '/opt/homebrew/opt/llvm/bin/clang'
command = [compiler, '-O2', '-fPIC', '-dynamiclib', '-I'+str(parser_source), '-I'+str(builder)]
command += [str(p) for p in sorted((repo/'src/SDT').glob('*.c'))]
command += [str(parser_source/'json.c'), str(builder/'json-builder.c'), '-o', str(DEST/'libSDT.dylib')]
subprocess.run(command, check=True)
urls = {'RollSDT.mp3': 'https://kronland.fr/wp-content/uploads/2015/02/RollSDT.mp3'}
for file in ['d_osc.c', 'd_filter.c']:
    urls['pd_'+file] = 'https://raw.githubusercontent.com/pure-data/pure-data/0.55-2/src/'+file
urls['pd_LICENSE.txt'] = 'https://raw.githubusercontent.com/pure-data/pure-data/0.55-2/LICENSE.txt'
for name, url in urls.items():
    if not (DEST/name).exists():
        if args.offline:
            raise FileNotFoundError(DEST/name)
        with urllib.request.urlopen(url) as response:
            (DEST/name).write_bytes(response.read())
files = {str(p.relative_to(DEST)): hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.iterdir() if p.is_file() and p.name != 'manifest.json'}
manifest = {'repository': 'https://github.com/SkAT-VG/SDT', 'commit': REVISION, 'source_packages': PACKAGES, 'compiler': subprocess.check_output([compiler, '--version'], text=True), 'command': command, 'urls': urls, 'sha256': files, 'historical_audio_context': 'https://kronland.fr/publications/a-synthesis-model-with-intuitive-control-capabilities-for-rolling-sounds/'}
(DEST/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(DEST/'manifest.json')
subprocess.run(['afconvert', '-f', 'WAVE', '-d', 'LEF32', str(DEST/'RollSDT.mp3'), str(DEST/'RollSDT.wav')], check=True)
manifest['sha256']['RollSDT.wav'] = hashlib.sha256((DEST/'RollSDT.wav').read_bytes()).hexdigest()
(DEST/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
