#!/usr/bin/env python3
"""Build the FLORES-200 devtest JSON used by the evaluation scripts.

Downloads the official FLORES-200 archive, checks its SHA-256, and writes
``{"sentence_<lang>": [1012 sentences], ...}`` for all 204 languages. Each
line is stripped of surrounding whitespace, and the resulting text is checked
against the digest of the copy used in the paper.

FLORES-200 is distributed under CC BY-SA 4.0 and is not redistributed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "flores200_devtest.json"

ARCHIVE_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
ARCHIVE_SHA256 = "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"
# Digest of the 204 x 1012 sentences evaluated in the paper; see text_sha256().
TEXT_SHA256 = "1785b70cfae560d30715468f32233b8592d6279a564f1eee75c6483e19fbdee1"
CONFIGURATIONS = 204
SENTENCES = 1012


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(dataset: dict[str, list[str]]) -> str:
    """Digest of every sentence, in sorted language order."""
    digest = hashlib.sha256()
    for column in sorted(dataset):
        digest.update(column.encode("utf-8"))
        digest.update(b"\0")
        for sentence in dataset[column]:
            digest.update(sentence.encode("utf-8"))
            digest.update(b"\0")
        digest.update(b"\xff")
    return digest.hexdigest()


def read_devtest(archive: Path) -> dict[str, list[str]]:
    dataset: dict[str, list[str]] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            path = Path(member.name)
            if not member.isfile() or path.parent.name != "devtest":
                continue
            if path.suffix != ".devtest":
                continue
            handle = tar.extractfile(member)
            assert handle is not None
            lines = handle.read().decode("utf-8").split("\n")
            if lines and lines[-1] == "":
                lines.pop()
            dataset[f"sentence_{path.stem}"] = [line.strip() for line in lines]
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use an already downloaded flores200_dataset.tar.gz",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as temporary:
        archive = args.archive
        if archive is None:
            archive = Path(temporary) / "flores200_dataset.tar.gz"
            print(f"Downloading {ARCHIVE_URL}", flush=True)
            urllib.request.urlretrieve(ARCHIVE_URL, archive)
        actual = file_sha256(archive)
        if actual != ARCHIVE_SHA256:
            raise SystemExit(
                f"archive SHA-256 mismatch: expected {ARCHIVE_SHA256}, got {actual}"
            )
        dataset = read_devtest(archive)

    if len(dataset) != CONFIGURATIONS or any(
        len(sentences) != SENTENCES for sentences in dataset.values()
    ):
        raise SystemExit("unexpected FLORES-200 devtest layout")
    digest = text_sha256(dataset)
    if digest != TEXT_SHA256:
        raise SystemExit(f"text digest mismatch: expected {TEXT_SHA256}, got {digest}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dataset, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}: {CONFIGURATIONS} languages x {SENTENCES} sentences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
