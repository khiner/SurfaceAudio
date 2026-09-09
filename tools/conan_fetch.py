#!/usr/bin/env python3
"""Fetch author-owned Conan companion assets and preserve provenance outside git."""
import concurrent.futures
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'references/conan'
PAGES = [
    'https://kronland.fr/publications/a-synthesis-model-with-intuitive-control-capabilities-for-rolling-sounds/',
    'https://kronland.fr/subjective-evaluation/',
    'https://kronland.fr/perceptually-relevant-cue-for-the-rolling-evocation/',
]

def decode(path):
    decoded = path.with_suffix('.decoded.wav')
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-ac', '1', '-ar', '44100', '-c:a', 'pcm_f32le', '-y', str(decoded)], check=True)
    return decoded

def download(url):
    name = url.rstrip('/').split('/')[-1] or 'index'
    if not re.search(r'\.(mp3|wav|mp4|zip)$', name):
        name += '.html'
    path = TARGET / name
    try:
        if not path.exists():
            with urllib.request.urlopen(url, timeout=45) as response:
                path.write_bytes(response.read())
        record = {'url': url, 'file': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if path.suffix in ('.mp3', '.wav'):
            decoded = decode(path)
            record['decoded'] = str(decoded.relative_to(ROOT))
        return record
    except Exception as error:
        return {'url': url, 'error': str(error)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decode-only', action='store_true', help='Verify pinned local originals and regenerate decoded WAVs without network access')
    args = parser.parse_args()
    if args.decode_only:
        records = [record for record in json.loads((ROOT/'docs/ReferenceInputs.json').read_text())['files'] if record['method'] == 'conan']
        for record in records:
            path = ROOT/record['path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                raise RuntimeError(f'Changed Conan reference: {path}')
        audio = [ROOT/record['path'] for record in records if Path(record['path']).suffix in ('.mp3', '.wav')]
        for path in audio:
            decode(path)
        print(f'Regenerated {len(audio)} decoded references from verified originals')
        return
    TARGET.mkdir(parents=True, exist_ok=True)
    records = []
    urls = set()
    for url in PAGES:
        record = download(url)
        records.append(record)
        if 'file' in record:
            text = (ROOT / record['file']).read_text()
            urls.update(re.findall(r'https?://[^\s\'\"<>]+\.(?:mp3|wav|mp4|zip)', text))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        records.extend(pool.map(download, sorted(url.replace('http:', 'https:') for url in urls)))
    (TARGET / 'manifest.json').write_text(json.dumps({'source': PAGES, 'assets': records}, indent=2) + '\n')
    print(json.dumps({'downloaded': sum('file' in record for record in records), 'failed': [record for record in records if 'error' in record]}, indent=2))

if __name__ == '__main__':
    main()
