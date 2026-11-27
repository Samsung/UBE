#!/usr/bin/env python3
# Copyright 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

"""Compute UBE's change from UTF-8 from FLORES-200 measurements.

Each ``--setting LABEL DIRECTORY`` names one training setting and the output
directory of ``paper/scripts/measure_flores.py`` for its four tokenizers:
``ube_unigram.json``, ``bytelevel_unigram.json``, ``ube_bpe.json`` and
``bytelevel_bpe.json``. For each model, Unigram and BPE, the script computes
UBE's change relative to UTF-8 bytes (``ByteLevel``), ``100 * (UBE / UTF-8 - 1)``, in

* the mean tokens per sentence of the 3-byte group: the languages in which at
  least half of the FLORES-200 characters (Unicode scalar values, spaces
  included) take 3 or more UTF-8 bytes, 39 of the 204, as the unweighted mean
  of their per-language means;
* the mean tokens per English (``eng_Latn``) sentence;
* the Gini coefficient of the 203 non-English token premiums, as defined in
  ``measure_flores.py``;
* the total tokens of all languages.

It prints the table of the repository README, one row per setting, and with
``--output`` writes every value to a JSON file. The measurements must come
from the FLORES-200 file given with ``--flores``.

Usage::

    python scripts/compare.py --setting "1 GiB text, 16K" results/1gib_16000 \\
        --setting "1 GiB text, 32K" results/1gib_32000 --output results/comparison.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / "paper" / "scripts"))

import measure_flores  # noqa: E402

MODELS = {"unigram": ("ube_unigram", "bytelevel_unigram"), "bpe": ("ube_bpe", "bytelevel_bpe")}
GROUP_RULE = "at least half of the characters (Unicode scalar values) take 3 or more UTF-8 bytes"
MINUS = "\u2212"


def same_file(first: Path, second: Path) -> bool:
    """Whether two paths name the same file, through symbolic or hard links."""
    if first.resolve() == second.resolve():
        return True
    try:
        return first.exists() and second.exists() and first.samefile(second)
    except OSError:
        return False


def three_byte_group(flores: dict[str, list[str]]) -> list[str]:
    """Languages in which at least half of the characters take 3 or more UTF-8 bytes."""
    group = []
    for lang, sentences in sorted(flores.items()):
        chars = long = 0
        for sentence in sentences:
            chars += len(sentence)
            long += sum(1 for ch in sentence if ord(ch) >= 0x800)
        if chars and 2 * long >= chars:
            group.append(lang)
    return group


def change(ube: float, bytelevel: float) -> float | None:
    return 100 * (ube / bytelevel - 1) if bytelevel else None


def load_flores_with_sha256(path: Path) -> tuple[dict[str, list[str]], str]:
    """Check that the FLORES file has the same digest before and after the measurement loader reads it."""
    if not path.is_file():
        raise SystemExit(f"{path} not found: run paper/scripts/prepare_flores.py from the repository root")
    digest = measure_flores.file_sha256(path)
    try:
        flores = measure_flores.load_flores(path)
    except (ValueError, TypeError, AttributeError) as error:
        raise SystemExit(f"{path}: invalid FLORES-200 data ({error})") from None
    if measure_flores.file_sha256(path) != digest:
        raise SystemExit(f"{path}: the FLORES-200 file changed while it was being loaded")
    return flores, digest


def load_report(path: Path, flores: dict[str, list[str]], flores_sha256: str) -> dict:
    if not path.is_file():
        raise SystemExit(f"{path} not found")
    try:
        report = json.loads(path.read_bytes())
    except ValueError as error:
        raise SystemExit(f"{path}: invalid measurement JSON ({error})") from None
    if not isinstance(report, dict) or report.get("tokenizer") != path.stem:
        raise SystemExit(f"{path}: expected a measurement of {path.stem}")
    if report.get("flores_sha256") != flores_sha256:
        raise SystemExit(f"{path}: measured on another FLORES-200 file than --flores")
    digest = report.get("tokenizer_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise SystemExit(f"{path}: invalid tokenizer_sha256")
    if type(report.get("vocab_size")) is not int or report["vocab_size"] <= 0:
        raise SystemExit(f"{path}: invalid vocab_size")
    results = report.get("results")
    if not isinstance(results, dict) or set(results) != set(flores):
        raise SystemExit(f"{path}: the languages differ from the FLORES-200 file")
    for lang, sentences in flores.items():
        row = results[lang]
        n = row.get("n") if isinstance(row, dict) else None
        if type(n) is not int or n <= 0 or n != len(sentences):
            raise SystemExit(f"{path}: invalid sentence count for {lang} (expected {len(sentences)}, got {n!r})")
        total, avg = row.get("total"), row.get("avg")
        if type(total) is not int or total < 0 or type(avg) not in (int, float) or avg != total / n:
            raise SystemExit(f"{path}: inconsistent token total or mean for {lang}")
    if not results[measure_flores.ENGLISH]["total"]:
        raise SystemExit(f"{path}: English has zero tokens, so token premiums are undefined")
    return report


def model_values(ube: dict, bytelevel: dict, group: list[str]) -> dict:
    """UBE, UTF-8 bytes (ByteLevel) and the change for one model."""

    def quantities(report: dict) -> dict[str, float]:
        results = report["results"]
        summary = measure_flores.summarize(results)
        return {
            "group_mean_tokens": math.fsum(results[lang]["avg"] for lang in group) / len(group),
            "english_mean_tokens": summary["en_avg_tokens"],
            "gini": summary["gini"],
            "total_tokens": sum(result["total"] for result in results.values()),
        }

    u, b = quantities(ube), quantities(bytelevel)
    return {key: {"ube": u[key], "bytelevel": b[key], "change_pct": change(u[key], b[key])} for key in u}


def compare_setting(label: str, directory: Path, flores, flores_sha256: str, group: list[str]) -> dict:
    reports = {
        kind: load_report(directory / f"{kind}.json", flores, flores_sha256)
        for kinds in MODELS.values()
        for kind in kinds
    }
    sizes = {report["vocab_size"] for report in reports.values()}
    if len(sizes) != 1:
        raise SystemExit(f"{directory}: the four tokenizers have different vocabulary sizes {sorted(sizes)}")
    if len({report["tokenizer_sha256"] for report in reports.values()}) != len(reports):
        raise SystemExit(f"{directory}: multiple tokenizer kinds have the same tokenizer SHA-256")
    setting = {"label": label, "directory": str(directory), "vocab_size": sizes.pop()}
    for model, (ube_kind, bytelevel_kind) in MODELS.items():
        setting[model] = model_values(reports[ube_kind], reports[bytelevel_kind], group)
        setting[model]["tokenizer_sha256"] = {
            ube_kind: reports[ube_kind]["tokenizer_sha256"],
            bytelevel_kind: reports[bytelevel_kind]["tokenizer_sha256"],
        }
    return setting


def percent(value: float | None) -> str:
    """A change with one decimal, as in the README: -15.1% is written with a minus sign."""
    return "N/A" if value is None else f"{value:+.1f}%".replace("-", MINUS)


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    """The first column left-aligned and the others right-aligned, padded to equal widths."""
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows)]

    def line(cells: list[str]) -> str:
        padded = [cells[0].ljust(widths[0])] + [cell.rjust(width) for cell, width in zip(cells[1:], widths[1:])]
        return "| " + " | ".join(padded) + " |"

    rule = [":" + "-" * (widths[0] - 1)] + ["-" * (width - 1) + ":" for width in widths[1:]]
    return "\n".join([line(header), "| " + " | ".join(rule) + " |", *(line(row) for row in rows)])


def readme_table(settings: list[dict], group_size: int) -> str:
    header = [
        "UBE vs UTF-8 (FLORES-200 devtest)",
        f"Unigram: {group_size} languages",
        f"BPE: {group_size} languages",
        "Unigram: English",
        "Unigram: Gini",
    ]
    rows = [
        [
            " ".join(setting["label"].splitlines()).replace("\\", "\\\\").replace("|", r"\|"),
            percent(setting["unigram"]["group_mean_tokens"]["change_pct"]),
            percent(setting["bpe"]["group_mean_tokens"]["change_pct"]),
            percent(setting["unigram"]["english_mean_tokens"]["change_pct"]),
            percent(setting["unigram"]["gini"]["change_pct"]),
        ]
        for setting in settings
    ]
    return markdown_table(header, rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--setting",
        nargs=2,
        action="append",
        required=True,
        metavar=("LABEL", "DIRECTORY"),
        help="a table row label and the measure_flores.py output directory of its four tokenizers",
    )
    parser.add_argument("--flores", type=Path, default=measure_flores.DEFAULT_FLORES)
    parser.add_argument("--output", type=Path, help="write every value to this JSON file")
    args = parser.parse_args()
    if args.output is not None:
        kinds = [kind for pair in MODELS.values() for kind in pair]
        inputs = [args.flores] + [Path(directory) / f"{kind}.json" for _, directory in args.setting for kind in kinds]
        clashes = sorted({str(path) for path in inputs if same_file(args.output, path)})
        if clashes:
            parser.error(f"--output would overwrite input files: {', '.join(clashes)}")

    flores, flores_sha256 = load_flores_with_sha256(args.flores)
    if not flores.get(measure_flores.ENGLISH):
        raise SystemExit(f"{args.flores}: no {measure_flores.ENGLISH} sentences")
    if len(flores) == 1:
        raise SystemExit(f"{args.flores}: token premiums require a non-English language")
    group = three_byte_group(flores)
    if not group:
        raise SystemExit(f"{args.flores}: no language meets the 3-byte group rule")
    settings = [
        compare_setting(label, Path(directory), flores, flores_sha256, group) for label, directory in args.setting
    ]
    if args.output is not None:
        comparison = {
            "flores_sha256": flores_sha256,
            "languages": len(flores),
            "three_byte_group": {"rule": GROUP_RULE, "languages": group},
            "change": "100 * (UBE / UTF-8 - 1)",
            "settings": settings,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.output.parent) as directory:
            temporary = Path(directory) / "comparison.json"
            temporary.write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(args.output)
    print(readme_table(settings, len(group)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OSError as error:
        raise SystemExit(str(error)) from None
