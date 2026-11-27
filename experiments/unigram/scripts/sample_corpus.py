#!/usr/bin/env python3
# Copyright 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

"""Sample tokenizer-training text with the byte shares of the paper's 101-language mixture.

Language ``l`` receives the first ``round(T * share_l)`` UTF-8 bytes of its
documents, where ``T`` is ``--total-bytes`` and ``share_l`` is the language's
value in ``mixture_shares.json`` divided by ``total_bytes``. A language's input
files are read in sorted path order and their documents in line order. Empty
documents are skipped, documents are taken whole, and the document that would
cross the quota is cut at the last character boundary at or below it. The
sample involves no randomness, so the sample of a smaller ``T`` is a
per-language prefix of the sample of a larger one.

Input: JSON Lines files, plain or gzip-compressed (``.gz``), with one document
per line in a ``text`` field, as in the mC4 distribution. ``--input`` is a glob
pattern in which ``{lang}`` stands for the language code.

Output: ``<lang>.jsonl`` for each language, in the same format, and
``manifest.json`` with each language's quota, text bytes, documents, input
files and SHA-256. The manifest is written last.

Usage::

    python scripts/sample_corpus.py --input 'mc4/{lang}/*.json.gz' \\
        --total-bytes 1073741824 --output-dir corpus/1gib
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import re
import zlib
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHARES = ROOT / "mixture_shares.json"
MANIFEST = "manifest.json"
PLACEHOLDER = "{lang}"


def load_shares_with_sha256(path: Path) -> tuple[dict[str, int], str]:
    """Sorted language byte counts and the SHA-256 of the bytes they came from."""
    try:
        raw = path.read_bytes()
        data = json.loads(raw)
    except (OSError, ValueError) as error:
        raise SystemExit(f"{path}: cannot read byte shares ({error})") from None
    languages = data.get("languages") if isinstance(data, dict) else None
    if not isinstance(languages, dict) or not languages:
        raise SystemExit(f"{path}: no languages")
    for lang, value in languages.items():
        if not re.fullmatch(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*", lang):
            raise SystemExit(f"{path}: invalid language code {lang!r}")
        if type(value) is not int or value <= 0:
            raise SystemExit(f"{path}: {lang} needs a positive integer byte count")
    total = data.get("total_bytes")
    if type(total) is not int or total != sum(languages.values()):
        raise SystemExit(f"{path}: total_bytes is not the sum of the language values")
    return dict(sorted(languages.items())), hashlib.sha256(raw).hexdigest()


def load_shares(path: Path) -> dict[str, int]:
    """Language code to byte count, in sorted language order."""
    return load_shares_with_sha256(path)[0]


def quotas(shares: dict[str, int], total_bytes: int) -> dict[str, int]:
    """Each language's quota: round(T * share), with share = value / total."""
    total = sum(shares.values())
    return {lang: round(total_bytes * (value / total)) for lang, value in shares.items()}


def input_files(pattern: str, lang: str) -> list[Path]:
    expanded = pattern.replace(PLACEHOLDER, lang)
    files = [Path(match) for match in sorted(glob.glob(expanded)) if Path(match).is_file()]
    if not files:
        raise SystemExit(f"no input files for {lang}: {expanded}")
    return files


def documents(path: Path):
    """Yield (line number, text) for the documents of a JSON Lines file, in order."""
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rb") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError as error:
                    raise SystemExit(f"{path}:{number}: not a JSON line ({error})") from None
                if not isinstance(record, dict) or "text" not in record:
                    raise SystemExit(f"{path}:{number}: no text field")
                text = record["text"]
                if text is not None and not isinstance(text, str):
                    raise SystemExit(f"{path}:{number}: the text field is not a string")
                yield number, text
    except (EOFError, gzip.BadGzipFile, zlib.error) as error:
        raise SystemExit(f"{path}: invalid gzip data ({error})") from None


def take_prefix(lang: str, files: list[Path], quota: int, sink) -> dict:
    """Write the first ``quota`` UTF-8 bytes of a language's documents to ``sink``."""
    digest = hashlib.sha256()
    text_bytes = count = 0
    cut = False
    inputs = []
    done = False

    def write(text: str) -> None:
        line = (json.dumps({"text": text}, ensure_ascii=False) + "\n").encode("utf-8")
        sink.write(line)
        digest.update(line)

    for path in files:
        if done:
            break
        taken = 0
        for number, text in documents(path):
            if not text:
                continue
            try:
                raw = text.encode("utf-8")
            except UnicodeEncodeError:
                raise SystemExit(f"{path}:{number}: the text is not valid Unicode") from None
            remaining = quota - text_bytes
            if len(raw) > remaining:
                # Cut at the last character boundary at or below the quota.
                piece = raw[:remaining].decode("utf-8", errors="ignore")
                if piece:
                    write(piece)
                    text_bytes += len(piece.encode("utf-8"))
                    count += 1
                    taken += 1
                    cut = True
                done = True
                break
            write(text)
            text_bytes += len(raw)
            count += 1
            taken += 1
            if text_bytes == quota:
                done = True
                break
        inputs.append({"file": str(path), "documents": taken})
    if not done:
        raise SystemExit(f"{lang}: the input ends after {text_bytes} of the {quota} bytes needed")
    return {
        "quota_bytes": quota,
        "text_bytes": text_bytes,
        "documents": count,
        "last_document_cut": cut,
        "inputs": inputs,
        "sha256": digest.hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        required=True,
        help="glob pattern of each language's JSON Lines files, with {lang} for the language code",
    )
    parser.add_argument(
        "--total-bytes",
        type=int,
        required=True,
        help="UTF-8 bytes requested for the whole sample, such as 1073741824 for 1 GiB",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shares", type=Path, default=DEFAULT_SHARES, help="byte shares (default: %(default)s)")
    args = parser.parse_args()
    if PLACEHOLDER not in args.input:
        parser.error("--input must contain {lang}")
    if args.total_bytes <= 0:
        parser.error("--total-bytes must be positive")
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error(f"{args.output_dir} exists and is not an empty directory")

    shares, shares_sha256 = load_shares_with_sha256(args.shares)
    total_share_bytes = sum(shares.values())
    targets = quotas(shares, args.total_bytes)
    files = {lang: input_files(args.input, lang) for lang in shares}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    languages = {}
    with ExitStack() as cleanup:
        for lang in shares:
            output = args.output_dir / f"{lang}.jsonl"
            with output.open("xb") as sink:
                cleanup.callback(output.unlink, missing_ok=True)
                info = take_prefix(lang, files[lang], targets[lang], sink)
            languages[lang] = {"share": shares[lang] / total_share_bytes, **info, "file": output.name}
            print(f"{lang}: {info['text_bytes']} bytes, {info['documents']} documents", flush=True)
        manifest = {
            "requested_bytes": args.total_bytes,
            "shares_sha256": shares_sha256,
            "share_total_bytes": total_share_bytes,
            "text_bytes": sum(info["text_bytes"] for info in languages.values()),
            "documents": sum(info["documents"] for info in languages.values()),
            "language_count": len(languages),
            "languages": languages,
        }
        manifest_path = args.output_dir / MANIFEST
        with manifest_path.open("x", encoding="utf-8") as sink:
            cleanup.callback(manifest_path.unlink, missing_ok=True)
            sink.write(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
        cleanup.pop_all()
    print(
        f"Wrote {args.output_dir}: {manifest['language_count']} languages, "
        f"{manifest['text_bytes']} bytes, {manifest['documents']} documents"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OSError as error:
        raise SystemExit(str(error)) from None
