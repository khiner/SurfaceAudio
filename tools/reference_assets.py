"""Fetch pinned paper resources and extract selected toolkit files."""
import hashlib
import importlib.metadata
import importlib.util
import shlex
import sys
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def require_packages(names):
    modules = {'Pillow': 'PIL', 'PyMuPDF': 'pymupdf'}
    missing = [name for name in names if importlib.util.find_spec(modules.get(name, name)) is None]
    if missing:
        command = ' '.join(map(shlex.quote, [sys.executable, '-m', 'pip', 'install', *missing]))
        raise RuntimeError(f'Missing Python packages: {", ".join(missing)}. Install with: {command}')
    return {name: importlib.metadata.version(name) for name in names}


def fetch(path, url, digest, offline=False):
    path = Path(path)
    cached = path.exists()
    candidate = path if cached else path.with_suffix('.download')
    if not cached:
        if offline:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, candidate)
    with candidate.open('rb') as source:
        if hashlib.file_digest(source, 'sha256').hexdigest() != digest:
            raise RuntimeError(f'Reference hash mismatch: {candidate}')
    if not cached:
        candidate.replace(path)
    return path


def extract(archive, destination, suffixes):
    with zipfile.ZipFile(archive) as files:
        for name in files.namelist():
            if '__MACOSX' not in name and Path(name).suffix in suffixes:
                target = (destination / name).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError('Archive member outside destination')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(files.read(name))
