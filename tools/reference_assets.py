"""Fetch pinned paper resources and extract selected toolkit files."""
import hashlib
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


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
