#!/usr/bin/env python3
"""Build and run published-result reproductions, then generate the A/B listening page."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

from reference_assets import ROOT, require_packages


def run(*command):
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True)


def verify_inputs(methods):
    manifest = ROOT / "docs/ReferenceInputs.json"
    if not manifest.exists():
        raise RuntimeError("Missing pinned reference-input manifest")
    for record in json.loads(manifest.read_text())["files"]:
        if record["method"] not in methods:
            continue
        path = ROOT / record["path"]
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError(f"Reference input is missing or changed: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    available = ["agarwal", "sdt", "conan", "matusiak", "poirot", "continuous", "matusiak2024", "falaize", "traer", "willemsen", "lagrange", "lee", "agarwal2026", "agarwal2025", "hatt", "nakatsuka", "rough"]
    parser.add_argument("--method", nargs="+", choices=["all", *available], default=["all"])
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    parser.add_argument("--offline", action="store_true", help="Require cached reference inputs")
    parser.add_argument("--author-oracle", action="store_true", help="Execute the pinned Willemsen MATLAB reference in Octave")
    parser.add_argument("--report-only", action="store_true", help="Rebuild the listening page from existing case manifests without synthesizing")
    parser.add_argument("--keep-diagnostics", action="store_true", help="Retain regenerable intermediates after successful checks")
    args = parser.parse_args()
    methods = available if "all" in args.method else list(dict.fromkeys(args.method))
    if args.author_oracle and "willemsen" not in methods:
        parser.error("--author-oracle requires the willemsen method")
    if args.report_only:
        build_report()
        return
    packages = require_packages(["numpy", "scipy", "matplotlib", *(["Pillow", "PyMuPDF"] if "rough" in methods else [])])
    build = args.build.resolve()
    if args.offline:
        verify_inputs(methods)
    run("cmake", "-S", ROOT, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release")
    run("cmake", "--build", build, "-j8")
    output = ROOT / "outputs/reproduction"
    output.mkdir(parents=True, exist_ok=True)
    environment = {"platform": platform.platform(), "python": sys.version,
                   "packages": packages,
                   "compiler": subprocess.check_output(["/opt/homebrew/opt/llvm/bin/clang++", "--version"], text=True),
                   "build": str(build), "methods": methods}
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    diagnostics = ["--keep-diagnostics"] if args.keep_diagnostics else []
    for method in methods:
        if method == "rough":
            for name in ("gregoire", "rough"):
                run(sys.executable, ROOT / f"tools/{name}_reproduce.py", "--binary", build / "roughReproduce",
                    *(["--offline"] if args.offline else []), *diagnostics)
            verify_inputs([method])
        elif method in ("agarwal2026", "agarwal2025"):
            option, executable = ("--fit-binary", "agarwalResponseFit") if method == "agarwal2026" else ("--force-binary", "agarwalReproduce")
            run(sys.executable, ROOT / f"tools/{method}_reproduce.py", "--binary", build / "agarwalObjectReproduce",
                option, build / executable, *(["--offline"] if args.offline else []), *diagnostics)
        elif method in ("matusiak", "poirot", "continuous", "matusiak2024", "falaize", "traer", "willemsen", "lagrange", "lee", "hatt", "nakatsuka"):
            run(sys.executable, ROOT / f"tools/{method}_reproduce.py", "--binary", build / f"{method}Reproduce",
                *(["--offline"] if args.offline else []),
                *(["--author-oracle"] if method == "willemsen" and args.author_oracle else []),
                *(diagnostics if method in ("lagrange", "lee", "willemsen", "hatt", "nakatsuka") else []))
            verify_inputs([method])
        elif method == "conan":
            if not args.offline:
                run(sys.executable, ROOT / "tools/conan_fetch.py")
                run(sys.executable, ROOT / "tools/conan_velocity.py", "--fetch-only")
            verify_inputs([method])
            run(sys.executable, ROOT / "tools/conan_fetch.py", "--decode-only")
            run(sys.executable, ROOT / "tools/conan_prepare.py")
            run(build / "conanReproduce", ROOT / "outputs/reproduction/conan")
            run(sys.executable, ROOT / "tools/conan_analyze.py", *diagnostics)
            run(sys.executable, ROOT / "tools/conan_velocity.py", "--binary", build / "conanReproduce", "--offline", *diagnostics)
            run(sys.executable, ROOT / "tools/conan_velocity.py", "--binary", build / "conanReproduce", "--offline", "--eps-variance", *diagnostics)
        elif method == "sdt":
            run(sys.executable, ROOT / "tools/sdt_fetch.py", *(["--offline"] if args.offline else []))
            verify_inputs([method])
            run(build / "sdtReproduce", ROOT / "references/sdt/libSDT.dylib", ROOT / "outputs/reproduction/sdt", 6, 44100)
            run(sys.executable, ROOT / "tools/sdt_analyze.py", ROOT / "outputs/reproduction/sdt")
        else:
            run(sys.executable, ROOT / "tools/agarwal_reproduce.py", "--binary", build / "agarwalReproduce", "--temporal", "--whole-record", "--upstream")
            for case in ["roll", "roll-glass"]:
                run(sys.executable, ROOT / "tools/agarwal_spatial_fit.py", "--binary", build / "agarwalSpatialReproduce", "--case", case, "--whole-record")
            run(sys.executable, ROOT / "tools/agarwal_reproduce.py", "--binary", build / "agarwalReproduce", "--retained")
    build_report()
    for method in ([] if args.keep_diagnostics else methods):
        for extension in ("*.f32", "*.f64"):
            for path in (output / method).rglob(extension):
                path.unlink()
    if "conan" in methods:
        required = {Path(case["reference"]).resolve() for manifest in (output / "conan").rglob("cases.json")
                    for case in json.loads(manifest.read_text())["cases"]}
        for path in (ROOT / "references/conan").rglob("*.decoded.wav"):
            if path.resolve() not in required:
                path.unlink()


def build_report():
    run(sys.executable, ROOT / "tools/BuildListeningReport.py")


if __name__ == "__main__":
    main()
