"""Fetch the exact reviewed official checkouts without weights or datasets.

Usage: python scripts/prepare_research_sources.py /absolute/path --models dag kite
Existing destinations are verified, never overwritten. This script only fetches
source code; model construction is the separate, explicit execution step.
"""

import argparse
import json
from pathlib import Path
import runpy
import subprocess
import tempfile

REGISTRY = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                             "neuralforecast/models/_official_source.py"))["SOURCE_REVISIONS"]
SPARSE = {
    "dag": ["ts_benchmark"], "kite": ["ts_benchmark"],
    "apt": ["baselines/Normalization"], "glaff": ["backbone", "plugin"],
    "tgtsf": ["models", "layers"], "spectf": ["models", "layers", "utils"],
    "chronosx": ["src"], "uni2ts": ["src"],
}


def git(directory, *args):
    result = subprocess.run(["git", "-C", str(directory), *args], check=True,
                            capture_output=True, text=True, timeout=300)
    return result.stdout.strip()


def prepare(destination, models):
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for key in models:
        repo, revision = REGISTRY[key]
        target = destination / key
        if target.exists():
            if not (target / ".git").is_dir() or git(target, "rev-parse", "HEAD") != revision:
                raise ValueError(f"Refusing to overwrite an unverified source directory: {target}")
            if git(target, "status", "--porcelain", "--untracked-files=normal"):
                raise ValueError(f"Official checkout has local changes: {target}")
        else:
            with tempfile.TemporaryDirectory(prefix=key + "-", dir=destination) as tmp:
                git(tmp, "init", "-q")
                git(tmp, "remote", "add", "origin", "https://github.com/" + repo + ".git")
                git(tmp, "config", "remote.origin.promisor", "true")
                git(tmp, "config", "remote.origin.partialclonefilter", "blob:none")
                git(tmp, "fetch", "--filter=blob:none", "--depth=1", "origin", revision)
                git(tmp, "sparse-checkout", "set", "--cone", *SPARSE[key])
                git(tmp, "checkout", "--detach", revision)
                if git(tmp, "rev-parse", "HEAD") != revision:
                    raise ValueError(f"Source revision mismatch for {repo}")
                Path(tmp).rename(target)
        manifest[key] = {"repository": repo, "revision": revision, "path": str(target)}
        print(f"{key}: {revision}")
    # Write atomically so an interrupted preparation cannot leave malformed JSON.
    manifest_path = destination / "manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    existing.update(manifest)
    temp_path = destination / "manifest.json.tmp"
    temp_path.write_text(json.dumps(existing, indent=2) + "\n")
    temp_path.replace(manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--models", choices=sorted(REGISTRY), nargs="+", default=sorted(REGISTRY))
    args = parser.parse_args()
    prepare(args.destination, args.models)
