"""Download pinned, manifest-listed public iEEG files for validation.

Raw data are stored under build/validation/data and are never committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "build" / "validation" / "manifests"
DATA = ROOT / "build" / "validation" / "data"
DATASETS = {"spike": "nm000360", "hfo": "on003498"}
MANIFEST_URLS = {
    "nm000360": "https://data.nemar.org/nm000360/v1.0.2/manifest.json",
    "on003498": "https://data.nemar.org/on003498/v1.0.0/manifest.json",
}


def select(entries: list[dict], dataset: str, subject: str, run: str | None) -> list[dict]:
    prefix = f"sub-{subject}/"
    selected = [item for item in entries if item["path"].startswith(prefix)]
    if dataset == "hfo":
        if not run:
            raise ValueError("HFO data require --run")
        selected = [item for item in selected if f"run-{run}_" in item["path"]]
    else:
        interpretation = f"derivatives/sub-{subject}_task-sleep_events_interpretation.tsv"
        selected.extend(item for item in entries if item["path"] == interpretation)
    return selected


def download(item: dict, root: Path) -> Path:
    destination = root / item["path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = int(item["size"])
    if destination.is_file() and destination.stat().st_size == size and _checksum_ok(destination, item):
        return destination
    partial = destination.with_name(destination.name + ".part")
    if root.name == "on003498" and size > 16 * 1024 * 1024:
        # The NEMAR mirror currently returns HTTP 500 for this large object.
        # Ranged requests to the original OpenNeuro S3 object are reliable.
        url = "https://s3.amazonaws.com/openneuro.org/ds003498/" + item["path"]
        position = partial.stat().st_size if partial.is_file() else 0
        if position > size:
            partial.unlink()
            position = 0
        with partial.open("ab") as stream:
            while position < size:
                end = min(size, position + 4 * 1024 * 1024) - 1
                request = urllib.request.Request(url, headers={"Range": f"bytes={position}-{end}"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    block = response.read()
                if len(block) != end - position + 1:
                    raise IOError(f"Incomplete byte range for {item['path']}")
                stream.write(block)
                stream.flush()
                position += len(block)
                print(f"{item['path']}: {position}/{size}", flush=True)
    else:
        try:
            response = urllib.request.urlopen(item["bytes_url"], timeout=60)
        except urllib.error.HTTPError as exc:
            if root.name != "on003498" or exc.code != 500:
                raise
            original = "https://s3.amazonaws.com/openneuro.org/ds003498/" + item["path"]
            response = urllib.request.urlopen(original, timeout=60)
        with response, partial.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
    if partial.stat().st_size != size or not _checksum_ok(partial, item):
        raise IOError(f"Size or checksum mismatch for {item['path']}")
    partial.replace(destination)
    return destination


def _checksum_ok(path: Path, item: dict) -> bool:
    algorithm = item["checksum_algorithm"]
    if algorithm == "md5":
        digest = hashlib.md5()
    elif algorithm == "git":
        digest = hashlib.sha1()
        digest.update(f"blob {path.stat().st_size}\0".encode())
    else:
        raise ValueError(f"Unknown checksum algorithm: {algorithm}")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == item["checksum"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--run")
    parser.add_argument("--signal", action="store_true", help="Include EDF or BrainVision EEG bytes")
    args = parser.parse_args()
    dataset = DATASETS[args.dataset]
    manifest = MANIFESTS / f"{dataset}.json"
    if not manifest.is_file():
        MANIFESTS.mkdir(parents=True, exist_ok=True)
        manifest.write_bytes(urllib.request.urlopen(MANIFEST_URLS[dataset], timeout=30).read())
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    selected = select(entries, args.dataset, args.subject, args.run)
    if not args.signal:
        selected = [item for item in selected if not item["path"].endswith((".edf", ".eeg"))]
    for item in selected:
        path = download(item, DATA / dataset)
        print(item["size"], path)


if __name__ == "__main__":
    main()

