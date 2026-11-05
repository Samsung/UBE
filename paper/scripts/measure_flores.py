#!/usr/bin/env python3
"""Measure FLORES-200 token counts and cross-language disparity for tokenizers.

For each ``tokenizer.json``, every devtest sentence of the 204 languages is
encoded and the per-language total and mean token counts are recorded.
The summary contains:

* ``en_avg_tokens``: mean tokens per English (``eng_Latn``) sentence;
* the token premium of a language: its mean tokens per sentence divided by
  ``en_avg_tokens``, computed for the 203 non-English languages;
* ``mean_premium``, ``median_premium``, ``max_premium`` and ``gini``, the Gini
  coefficient of those premiums (0 = every language costs the same).

Padding and truncation are disabled during measurement; special tokens added
by the tokenizer's post-processor are included.

Usage::

    python scripts/measure_flores.py path/to/tokenizer.json [more.json ...] \
        --output-dir results/flores200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLORES = ROOT / "data" / "flores200_devtest.json"
ENGLISH = "eng_Latn"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gini(values) -> float:
    """Gini coefficient (0 = perfectly equal, 1 = maximally unequal)."""
    values = np.array(sorted(values), dtype=float)
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values)))


def load_flores(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        raise SystemExit(f"{path} not found: run scripts/prepare_flores.py first")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        column.removeprefix("sentence_"): [s for s in sentences if s and s.strip()]
        for column, sentences in sorted(data.items())
        if column.startswith("sentence_")
    }


def summarize(results: dict[str, dict]) -> dict[str, float]:
    en_avg = results[ENGLISH]["avg"]
    premiums = [r["avg"] / en_avg for lang, r in results.items() if lang != ENGLISH]
    return {
        "en_avg_tokens": en_avg,
        "mean_premium": float(np.mean(premiums)),
        "median_premium": float(np.median(premiums)),
        "max_premium": float(np.max(premiums)),
        "gini": gini(premiums),
    }


def tokenizer_name(path: Path) -> str:
    """The directory name for ``.../<name>/tokenizer.json``, otherwise the file stem."""
    path = path.resolve()
    return path.parent.name if path.name == "tokenizer.json" else path.stem


def measure(tokenizer_path: Path, flores: dict[str, list[str]]) -> dict:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    results = {}
    for lang, sentences in flores.items():
        total = sum(len(encoding.ids) for encoding in tokenizer.encode_batch(sentences))
        results[lang] = {"total": total, "avg": total / len(sentences), "n": len(sentences)}
    return {
        "tokenizer": tokenizer_name(tokenizer_path),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "vocab_size": tokenizer.get_vocab_size(),
        "languages": len(results),
        "sentences_per_lang": results[ENGLISH]["n"],
        "summary": summarize(results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tokenizers", type=Path, nargs="+", help="tokenizer.json files")
    parser.add_argument("--flores", type=Path, default=DEFAULT_FLORES)
    parser.add_argument("--output-dir", type=Path, help="write one <name>.json per tokenizer")
    args = parser.parse_args()

    names = [tokenizer_name(path) for path in args.tokenizers]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        parser.error(f"tokenizers must have distinct names: {', '.join(duplicates)}")
    if args.output_dir is not None:
        inputs = {path.resolve() for path in args.tokenizers} | {args.flores.resolve()}
        clashes = []
        for name in names:
            out = (args.output_dir / f"{name}.json").resolve()
            if out in inputs or (
                out.exists() and any(source.exists() and out.samefile(source) for source in inputs)
            ):
                clashes.append(str(out))
        if clashes:
            parser.error(f"--output-dir would overwrite input files: {', '.join(clashes)}")
    flores = load_flores(args.flores)
    flores_sha256 = file_sha256(args.flores)
    print(f"FLORES-200 devtest: {len(flores)} languages", flush=True)
    print(f"{'Tokenizer':<40} {'Vocab':>7} {'EN avg':>8} {'Mean prem':>10} {'Gini':>8}")
    for path, name in zip(args.tokenizers, names):
        started = time.time()
        report = measure(path, flores) | {"flores_sha256": flores_sha256}
        summary = report["summary"]
        print(
            f"{name:<40} {report['vocab_size']:>7} {summary['en_avg_tokens']:>8.2f} "
            f"{summary['mean_premium']:>9.4f}x {summary['gini']:>8.4f}  ({time.time() - started:.0f}s)",
            flush=True,
        )
        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            out = args.output_dir / f"{name}.json"
            out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
