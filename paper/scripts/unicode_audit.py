#!/usr/bin/env python3
"""Audit UBE, SCRIPT-BPE, and MYTE round-trip behavior on Unicode 17.

The audit separates three questions that a small hand-written probe cannot:

1. Raw identity over every Unicode scalar value.
2. Raw identity versus each method's configured normalization.
3. Raw identity on official grapheme-break and emoji sequence test data.

Unicode inputs are downloaded from versioned official URLs and checked against
hard-coded SHA-256 digests. The audit directly invokes the pinned official
MYTE and SCRIPT-BPE implementations for those methods. The normative Unicode
17 NFC target is additionally checked on NormalizationTest.txt.

Round-trip exactness is a property of each method's base representation and
normalization, not of its learned merges. Unless tokenizer files are given,
the audit therefore trains small UBE and SCRIPT-BPE tokenizers (4,096 merges
each by default) on the FLORES-200 devtest text and audits those.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

UNICODE_VERSION = "17.0.0"
MAX_CODE_POINT = 0x10FFFF
SURROGATE_START = 0xD800
SURROGATE_END = 0xDFFF
SCALAR_COUNT = (MAX_CODE_POINT + 1) - (
    SURROGATE_END - SURROGATE_START + 1
)

DEFAULT_CACHE = ROOT / "data" / "unicode_17_0_0"
DEFAULT_TRAIN_TEXT = ROOT / "data" / "flores200_devtest.json"
DEFAULT_OUTPUT = ROOT / "results" / "unicode_audit.json"
DEFAULT_REPORT = ROOT / "results" / "unicode_audit.md"
DEFAULT_MERGES = 4096
SCRIPT_TOK_COMMIT = "54a105829f302fe3a3fc5e963f150ba7d839fb53"
SCRIPT_PRETOKENIZER = "scriptenc_cb"
MYTE_REWRITER_REL = Path("src/rewrite_bytes.py")
MYTE_REWRITER_SHA256 = (
    "24454728cc5d0ca5207823d2a3d43507de8546df691ebaf5ec97a06c99e53ddd"
)

DATA_SOURCES = {
    "unicode_data": {
        "filename": "UnicodeData.txt",
        "url": (
            "https://www.unicode.org/Public/17.0.0/ucd/UnicodeData.txt"
        ),
        "sha256": (
            "2e1efc1dcb59c575eedf5ccae60f95229f706ee6d031835247d843c11d96470c"
        ),
    },
    "derived_age": {
        "filename": "DerivedAge.txt",
        "url": (
            "https://www.unicode.org/Public/17.0.0/ucd/DerivedAge.txt"
        ),
        "sha256": (
            "f8ecdf768bdc210f201abd271d9bc587825618a86a7046a8146cc816393f1998"
        ),
    },
    "normalization": {
        "filename": "NormalizationTest.txt",
        "url": (
            "https://www.unicode.org/Public/17.0.0/ucd/"
            "NormalizationTest.txt"
        ),
        "sha256": (
            "5019ffd530751a741900c849c0e010332f142a3612234639bd200b82138a87db"
        ),
    },
    "grapheme_break": {
        "filename": "GraphemeBreakTest.txt",
        "url": (
            "https://www.unicode.org/Public/17.0.0/ucd/auxiliary/"
            "GraphemeBreakTest.txt"
        ),
        "sha256": (
            "e2d134d2c52919bace503ebb6a551c1855fe1a1faec18478c78fff254a1793ec"
        ),
    },
    "emoji": {
        "filename": "emoji-test.txt",
        "url": (
            "https://www.unicode.org/Public/17.0.0/emoji/emoji-test.txt"
        ),
        "sha256": (
            "1d8a944f88d7952f7ef7c5167fef3c67995bcae24543949710231b03a201acda"
        ),
    },
}

METHODOLOGY_REFERENCES = (
    {
        "title": (
            "SentencePiece: A simple and language independent subword "
            "tokenizer and detokenizer for Neural Text Processing"
        ),
        "venue": "EMNLP System Demonstrations 2018",
        "url": "https://aclanthology.org/D18-2012/",
        "use": (
            "Separates reversible tokenization after configured normalization "
            "from recovery of the original raw string."
        ),
    },
    {
        "title": "Neural Machine Translation with Byte-Level Subwords",
        "venue": "AAAI 2020",
        "url": "https://doi.org/10.1609/aaai.v34i05.6451",
        "use": (
            "Motivates exhaustive coverage because byte-level vocabularies "
            "are intended to avoid out-of-vocabulary inputs."
        ),
    },
    {
        "title": (
            "ByT5: Towards a Token-Free Future with Pre-trained "
            "Byte-to-Byte Models"
        ),
        "venue": "TACL 2022",
        "url": "https://aclanthology.org/2022.tacl-1.17/",
        "use": (
            "Motivates testing raw UTF-8 coverage without language-specific "
            "preprocessing."
        ),
    },
    {
        "title": "Unicode Standard Annex #15: Unicode Normalization Forms",
        "venue": "Unicode 17.0.0",
        "url": "https://www.unicode.org/reports/tr15/",
        "use": (
            "Supplies the normative normalization conformance cases and "
            "expected NFC outputs."
        ),
    },
    {
        "title": "Unicode Standard Annex #29: Unicode Text Segmentation",
        "venue": "Unicode 17.0.0",
        "url": "https://www.unicode.org/reports/tr29/",
        "use": (
            "Supplies official multi-scalar grapheme-break test sequences."
        ),
    },
    {
        "title": "Unicode Standard Annex #44: Unicode Character Database",
        "venue": "Unicode 17.0.0",
        "url": "https://www.unicode.org/reports/tr44/",
        "use": (
            "Defines the category and age property files used to construct "
            "the exhaustive scalar strata."
        ),
    },
    {
        "title": "Unicode Technical Standard #51: Unicode Emoji",
        "venue": "Unicode Emoji 17.0",
        "url": "https://www.unicode.org/reports/tr51/",
        "use": "Supplies the versioned emoji sequence test set.",
    },
)

EXAMPLE_LIMIT = 12
PROGRESS_INTERVAL = 100_000


@dataclass(frozen=True)
class Case:
    suite: str
    groups: tuple[str, ...]
    identifier: str
    text: str
    expected_nfc: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return f"<external>/{resolved.name}"


def _codepoints(text: str) -> list[str]:
    return [f"U+{ord(char):04X}" for char in text]


def _parse_range(field: str) -> tuple[int, int]:
    if ".." in field:
        start, end = field.split("..", maxsplit=1)
        return int(start, 16), int(end, 16)
    value = int(field, 16)
    return value, value


def _parse_sequence(field: str) -> str:
    stripped = field.strip()
    if not stripped:
        return ""
    return "".join(chr(int(value, 16)) for value in stripped.split())


def _download_verified_sources(cache_dir: Path) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for label, specification in DATA_SOURCES.items():
        destination = cache_dir / specification["filename"]
        expected = specification["sha256"]
        if destination.is_file() and _sha256(destination) == expected:
            paths[label] = destination
            continue
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        with urllib.request.urlopen(specification["url"]) as response:
            with temporary.open("wb") as handle:
                while block := response.read(1024 * 1024):
                    handle.write(block)
        actual = _sha256(temporary)
        if actual != expected:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
            )
        os.replace(temporary, destination)
        paths[label] = destination
    return paths


def _verify_sources(cache_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for label, specification in DATA_SOURCES.items():
        path = cache_dir / specification["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != specification["sha256"]:
            raise RuntimeError(
                f"{label} SHA-256 mismatch: "
                f"expected {specification['sha256']}, got {actual}"
            )
        paths[label] = path
    return paths


def _unicode_scalar_metadata(
    unicode_data_path: Path,
    derived_age_path: Path,
) -> tuple[bytearray, bytearray]:
    # Category classes: 0=Cn, 1=assigned public, 2=Co, 3=Cs.
    category_class = bytearray(MAX_CODE_POINT + 1)
    pending: tuple[int, str] | None = None
    for raw_line in unicode_data_path.read_text(encoding="utf-8").splitlines():
        if not raw_line:
            continue
        fields = raw_line.split(";")
        code_point = int(fields[0], 16)
        name = fields[1]
        category = fields[2]
        if name.endswith(", First>"):
            pending = (code_point, category)
            continue
        if name.endswith(", Last>"):
            if pending is None or pending[1] != category:
                raise ValueError("malformed UnicodeData First/Last range")
            start = pending[0]
            pending = None
            end = code_point
        else:
            start = code_point
            end = code_point
        value = 2 if category == "Co" else 3 if category == "Cs" else 1
        category_class[start : end + 1] = bytes([value]) * (end - start + 1)
    if pending is not None:
        raise ValueError("unterminated UnicodeData range")

    # Age classes: 0=unassigned, 1=assigned through Unicode 16, 2=Unicode 17.
    age_class = bytearray(MAX_CODE_POINT + 1)
    for raw_line in derived_age_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line:
            continue
        range_field, age = [part.strip() for part in line.split(";", maxsplit=1)]
        start, end = _parse_range(range_field)
        value = 2 if age == "17.0" else 1
        age_class[start : end + 1] = bytes([value]) * (end - start + 1)
    return category_class, age_class


def _coverage_groups(
    text: str,
    category_class: bytearray,
    age_class: bytearray,
) -> tuple[str, ...]:
    """Label sequence coverage without treating overlapping strata as disjoint."""
    groups: list[str] = []
    categories = {category_class[ord(character)] for character in text}
    ages = {age_class[ord(character)] for character in text}
    if 0 in categories:
        groups.append("contains_unassigned_or_reserved")
    if 2 in categories:
        groups.append("contains_private_use")
    if 2 in ages:
        groups.append("contains_unicode_17")
    if not groups:
        groups.append("supported_through_unicode_16")
    return tuple(groups)


def _normalization_cases(
    path: Path,
    category_class: bytearray,
    age_class: bytearray,
) -> tuple[list[Case], dict[int, str]]:
    unique: dict[str, tuple[str, str]] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line or line.startswith("@"):
            continue
        fields = [part.strip() for part in line.split(";")]
        if len(fields) < 5:
            raise ValueError(
                f"malformed NormalizationTest line {line_number}"
            )
        c1, c2, c3, c4, c5 = (_parse_sequence(field) for field in fields[:5])
        for column, text, expected_nfc in (
            ("c1", c1, c2),
            ("c2", c2, c2),
            ("c3", c3, c2),
            ("c4", c4, c4),
            ("c5", c5, c4),
        ):
            previous = unique.get(text)
            if previous is not None:
                if previous[0] != expected_nfc:
                    raise ValueError(
                        "conflicting NFC expectations for "
                        f"{_codepoints(text)}"
                    )
                continue
            unique[text] = (expected_nfc, f"line-{line_number}-{column}")

    cases = [
        Case(
            suite="normalization",
            groups=(
                "nfc_stable" if text == expected_nfc else "nfc_changes",
                *_coverage_groups(text, category_class, age_class),
            ),
            identifier=identifier,
            text=text,
            expected_nfc=expected_nfc,
        )
        for text, (expected_nfc, identifier) in unique.items()
    ]
    single_nfc = {
        ord(text): expected_nfc
        for text, (expected_nfc, _) in unique.items()
        if len(text) == 1 and text != expected_nfc
    }
    return cases, single_nfc


def _grapheme_cases(
    path: Path,
    category_class: bytearray,
    age_class: bytearray,
) -> list[Case]:
    cases: list[Case] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line:
            continue
        code_points = [
            token
            for token in line.split()
            if token not in {"÷", "×"}
        ]
        text = _parse_sequence(" ".join(code_points))
        if text in seen:
            continue
        seen.add(text)
        cases.append(
            Case(
                suite="grapheme_break",
                groups=(
                    "official_sequences",
                    *_coverage_groups(text, category_class, age_class),
                ),
                identifier=f"line-{line_number}",
                text=text,
            )
        )
    return cases


def _emoji_cases(
    path: Path,
    category_class: bytearray,
    age_class: bytearray,
) -> list[Case]:
    cases: list[Case] = []
    seen: set[tuple[str, str]] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line:
            continue
        sequence_field, status = [
            part.strip() for part in line.split(";", maxsplit=1)
        ]
        text = _parse_sequence(sequence_field)
        key = (status, text)
        if key in seen:
            continue
        seen.add(key)
        cases.append(
            Case(
                suite="emoji",
                groups=(
                    status.replace("-", "_"),
                    *_coverage_groups(text, category_class, age_class),
                ),
                identifier=f"line-{line_number}",
                text=text,
            )
        )
    return cases


def _scalar_cases(
    category_class: bytearray,
    age_class: bytearray,
    single_nfc: dict[int, str],
) -> Iterator[Case]:
    for code_point in range(MAX_CODE_POINT + 1):
        if SURROGATE_START <= code_point <= SURROGATE_END:
            continue
        category = category_class[code_point]
        if category == 2:
            group = "private_use"
        elif category == 0:
            group = "unassigned_or_reserved"
        elif age_class[code_point] == 2:
            group = "assigned_in_unicode_17"
        else:
            group = "assigned_through_unicode_16"
        text = chr(code_point)
        yield Case(
            suite="unicode_scalars",
            groups=(group,),
            identifier=f"U+{code_point:04X}",
            text=text,
            expected_nfc=single_nfc.get(code_point, text),
        )


def _new_metrics() -> dict[str, Any]:
    return {
        "cases": 0,
        "raw_exact": 0,
        "declared_normalization_exact": 0,
        "nfc_expected_cases": 0,
        "nfc_exact": 0,
        "nonempty_output": 0,
        "errors": 0,
        "input_scalars": 0,
        "output_scalars": 0,
        "failure_examples": [],
    }


class Accumulator:
    def __init__(self) -> None:
        self.metrics: dict[str, dict[str, dict[str, Any]]] = defaultdict(
            lambda: defaultdict(_new_metrics)
        )

    def add(
        self,
        case: Case,
        decoded: str | None,
        declared_expected: str | None,
        error: str | None,
    ) -> None:
        for group in ("all", *case.groups):
            self._add_one(
                self.metrics[case.suite][group],
                case,
                decoded,
                declared_expected,
                error,
            )

    @staticmethod
    def _add_one(
        metrics: dict[str, Any],
        case: Case,
        decoded: str | None,
        declared_expected: str | None,
        error: str | None,
    ) -> None:
        metrics["cases"] += 1
        metrics["input_scalars"] += len(case.text)
        if case.expected_nfc is not None:
            metrics["nfc_expected_cases"] += 1
        if error is not None:
            metrics["errors"] += 1
            if len(metrics["failure_examples"]) < EXAMPLE_LIMIT:
                metrics["failure_examples"].append(
                    {
                        "identifier": case.identifier,
                        "input": _codepoints(case.text),
                        "error": error,
                    }
                )
            return
        assert decoded is not None
        assert declared_expected is not None
        metrics["output_scalars"] += len(decoded)
        if decoded:
            metrics["nonempty_output"] += 1
        raw_exact = decoded == case.text
        nfc_exact = (
            case.expected_nfc is not None
            and decoded == case.expected_nfc
        )
        if raw_exact:
            metrics["raw_exact"] += 1
        if decoded == declared_expected:
            metrics["declared_normalization_exact"] += 1
        if nfc_exact:
            metrics["nfc_exact"] += 1
        if not raw_exact and len(metrics["failure_examples"]) < EXAMPLE_LIMIT:
            example = {
                "identifier": case.identifier,
                "input": _codepoints(case.text),
                "decoded": _codepoints(decoded),
                "declared_normalization": _codepoints(declared_expected),
            }
            if case.expected_nfc is not None:
                example["expected_nfc"] = _codepoints(case.expected_nfc)
            metrics["failure_examples"].append(example)

    def serializable(self) -> dict[str, Any]:
        return {
            suite: dict(groups)
            for suite, groups in self.metrics.items()
        }


def _import_scripts() -> None:
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))


def _ube_runtime() -> dict[str, Any]:
    import tokenizers
    from tokenizers import pre_tokenizers

    if not hasattr(pre_tokenizers, "UniversalByteLevel"):
        raise RuntimeError(
            "the installed tokenizers package has no UniversalByteLevel "
            "pre-tokenizer; install the UBE tokenizers build (see README.md)"
        )
    return {
        "tokenizers_version": tokenizers.__version__,
        "python_version": sys.version.split()[0],
    }


def _load_official_myte_byte_rewriter(myte_dir: Path) -> tuple[type[Any], Path]:
    """Load ByteRewriter directly from the pinned official MYTE checkout."""
    source_path = myte_dir.resolve() / MYTE_REWRITER_REL
    if not source_path.is_file():
        raise FileNotFoundError(f"MYTE implementation missing: {source_path}")
    actual_sha256 = _sha256(source_path)
    if actual_sha256 != MYTE_REWRITER_SHA256:
        raise RuntimeError(
            "MYTE ByteRewriter hash mismatch: "
            f"expected {MYTE_REWRITER_SHA256}, got {actual_sha256}"
        )

    module_name = "_pinned_official_myte_rewrite_bytes"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load MYTE implementation: {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    byte_rewriter = getattr(module, "ByteRewriter", None)
    if byte_rewriter is None:
        raise RuntimeError(f"ByteRewriter not found in {source_path}")
    return byte_rewriter, source_path


def _load_method(
    method: str,
    tokenizer_path: Path,
) -> tuple[
    Callable[[str], str],
    Callable[[str], str],
    dict[str, Any],
]:
    if method == "ube":
        runtime = _ube_runtime()
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        # Audit the encoding itself: no padding, truncation or added special
        # tokens from a post-processor.
        tokenizer.no_padding()
        tokenizer.no_truncation()

        def round_trip(text: str) -> str:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            return tokenizer.decode(
                encoded.ids,
                skip_special_tokens=False,
            )

        normalizer = tokenizer.normalizer
        if normalizer is None:

            def declared_normalize(text: str) -> str:
                return text

        else:

            def declared_normalize(text: str) -> str:
                return normalizer.normalize_str(text)

        artifact = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        return (
            round_trip,
            declared_normalize,
            {
                "label": "UBE (ours)",
                "normalization": artifact["normalizer"],
                "pretokenizer": (artifact.get("pre_tokenizer") or {}).get("type"),
                "runtime": runtime,
            },
        )

    if method == "script":
        import unicodedata

        from script_bpe.tokenizers.bpe import BPETokenizer  # type: ignore

        tokenizer = BPETokenizer.load(str(tokenizer_path))
        pretokenizer = tokenizer.pretokenizer

        def round_trip(text: str) -> str:
            return tokenizer.decode(tokenizer.encode(text))

        normalization = pretokenizer.config.normalization
        if normalization is None:

            def declared_normalize(text: str) -> str:
                return text

        else:

            def declared_normalize(text: str) -> str:
                return unicodedata.normalize(normalization, text)

        return (
            round_trip,
            declared_normalize,
            {
                "label": "SCRIPT-BPE official implementation",
                "normalization": normalization,
                "normalization_unicode_version": unicodedata.unidata_version,
                "python_version": sys.version.split()[0],
                "remove_unassigned": pretokenizer.config.remove_unassigned,
                "pretokenizer": type(pretokenizer).__name__,
                "valid_character_count": len(pretokenizer.valid_chars),
            },
        )

    if method == "myte":
        _import_scripts()
        from myte_codec import DECOMPOSE_REL, MERGE_REL, verify_myte_source

        provenance = verify_myte_source(tokenizer_path, require_git_pin=True)
        byte_rewriter, implementation_path = (
            _load_official_myte_byte_rewriter(tokenizer_path)
        )
        decompose_rewriter = byte_rewriter(
            str(tokenizer_path / DECOMPOSE_REL)
        )
        merge_rewriter = byte_rewriter(str(tokenizer_path / MERGE_REL))

        def round_trip(text: str) -> str:
            byte_tokens = [f"{value:02x}" for value in text.encode("utf-8")]
            encoded = decompose_rewriter.rewrite_bytes(
                byte_tokens, reverse=False
            )
            encoded = merge_rewriter.rewrite_bytes(encoded, reverse=False)
            decoded = merge_rewriter.rewrite_bytes(encoded, reverse=True)
            decoded = decompose_rewriter.rewrite_bytes(decoded, reverse=True)
            decoded_bytes = bytes.fromhex(" ".join(decoded))
            return decoded_bytes.decode("utf-8", errors="ignore")

        def declared_normalize(text: str) -> str:
            return text

        return (
            round_trip,
            declared_normalize,
            {
                "label": "MYTE official implementation",
                "normalization": None,
                "decoder_utf8_errors": "ignore",
                "implementation": "src/rewrite_bytes.py::ByteRewriter",
                "implementation_sha256": _sha256(implementation_path),
                "python_version": sys.version.split()[0],
                **provenance,
            },
        )

    raise ValueError(f"unknown method: {method}")


def _run_worker(
    method: str,
    tokenizer_path: Path,
    cache_dir: Path,
    output_path: Path,
) -> None:
    sources = _verify_sources(cache_dir)
    category_class, age_class = _unicode_scalar_metadata(
        sources["unicode_data"],
        sources["derived_age"],
    )
    normalization, single_nfc = _normalization_cases(
        sources["normalization"],
        category_class,
        age_class,
    )
    grapheme = _grapheme_cases(
        sources["grapheme_break"],
        category_class,
        age_class,
    )
    emoji = _emoji_cases(
        sources["emoji"],
        category_class,
        age_class,
    )
    round_trip, declared_normalize, metadata = _load_method(
        method,
        tokenizer_path,
    )

    accumulator = Accumulator()
    started = time.monotonic()
    processed = 0

    suites: Iterable[Iterable[Case]] = (
        _scalar_cases(category_class, age_class, single_nfc),
        normalization,
        grapheme,
        emoji,
    )
    for cases in suites:
        for case in cases:
            try:
                declared_expected = declared_normalize(case.text)
                decoded = round_trip(case.text)
                error = None
            except Exception as exc:  # pragma: no cover - artifact audit
                decoded = None
                declared_expected = None
                error = f"{type(exc).__name__}: {exc}"
            accumulator.add(case, decoded, declared_expected, error)
            processed += 1
            if processed % PROGRESS_INTERVAL == 0:
                elapsed = time.monotonic() - started
                print(
                    f"[{method}] {processed:,} cases in {elapsed:.1f}s",
                    flush=True,
                )

    tokenizer_record = {
        "path": _portable_path(tokenizer_path),
        **metadata,
    }
    if tokenizer_path.is_file():
        tokenizer_record["sha256"] = _sha256(tokenizer_path)
    elif not (method == "myte" and tokenizer_path.is_dir()):
        raise FileNotFoundError(tokenizer_path)

    payload = {
        "method": method,
        "tokenizer": tokenizer_record,
        "unicode_version": UNICODE_VERSION,
        "source_sha256": {
            label: _sha256(path) for label, path in sources.items()
        },
        "suite_counts": accumulator.serializable(),
        "processed_cases": processed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[{method}] complete: {processed:,} cases -> {output_path}",
        flush=True,
    )


def _training_texts(path: Path) -> list[str]:
    """Sentences of a FLORES-style ``{"sentence_<lang>": [...]}`` JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        sentence
        for key in sorted(data)
        if key.startswith("sentence_")
        for sentence in data[key]
    ]


def _train_ube_tokenizer(text_path: Path, merges: int, output_path: Path) -> None:
    _ube_runtime()
    from tokenizers.implementations import UniversalByteLevelBPETokenizer

    tokenizer = UniversalByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        _training_texts(text_path),
        vocab_size=512 + merges,
        min_frequency=2,
        show_progress=False,
    )
    tokenizer.save(str(output_path))


def _train_script_tokenizer(
    text_path: Path,
    merges: int,
    output_path: Path,
) -> None:
    """Train SCRIPT-BPE with its official trainer (inside its environment)."""
    from script_bpe import (  # type: ignore
        BPETrainer,
        BPETrainerConfig,
        get_pretokenizer,
    )
    from script_bpe.corpus import PretokenizedCorpus  # type: ignore

    pretokenizer = get_pretokenizer(SCRIPT_PRETOKENIZER)
    with tempfile.TemporaryDirectory(prefix="script-bpe-corpus-") as corpus_dir:
        corpus = PretokenizedCorpus.from_texts(
            name="training_text",
            texts=_training_texts(text_path),
            pretokenizer=pretokenizer,
            base_path=corpus_dir,
            num_workers=4,
        )
        config = BPETrainerConfig(
            additional_vocab_size=merges,
            num_workers=4,
            verbose=False,
        )
        tokenizer = BPETrainer(pretokenizer, corpus, config).train()
    tokenizer.save(str(output_path))


def _check_script_commit(script_tok_dir: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(script_tok_dir), "rev-parse", "HEAD"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"cannot inspect SCRIPT-BPE checkout: {completed.stderr.strip()}"
        )
    actual = completed.stdout.strip()
    if actual != SCRIPT_TOK_COMMIT:
        raise RuntimeError(
            f"SCRIPT-BPE commit mismatch: expected {SCRIPT_TOK_COMMIT}, "
            f"got {actual}"
        )
    return actual


def _check_myte_source(myte_dir: Path) -> dict[str, Any]:
    _import_scripts()
    from myte_codec import verify_myte_source

    return verify_myte_source(myte_dir, require_git_pin=True)


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    # Workers are started by interpreter path; uv uses the SCRIPT-BPE project's
    # own environment.
    environment.pop("VIRTUAL_ENV", None)
    return environment


def _metric(
    payload: dict[str, Any],
    suite: str,
    group: str,
    key: str,
) -> int:
    return int(payload["suite_counts"][suite][group][key])


def _format_count(numerator: int, denominator: int) -> str:
    return f"{numerator:,}/{denominator:,}"


def _tokenizer_note(tokenizers: dict[str, dict[str, Any]]) -> str:
    def describe(method: str, label: str) -> str:
        record = tokenizers[method]
        if record["source"] == "trained":
            return (
                f"{label} was trained with {record['merges']:,} merges on "
                f"`{record['training_text']}`"
            )
        return f"{label} was loaded from a given tokenizer file"

    return (
        f"- Tokenizers: {describe('ube', 'UBE')}; "
        f"{describe('script', 'SCRIPT-BPE')}. Round-trip exactness depends "
        "on the base representation and normalization, not on learned merges."
    )


def _render_report(payload: dict[str, Any]) -> str:
    ube = payload["methods"]["ube"]
    myte = payload["methods"]["myte"]
    script = payload["methods"]["script"]
    scalar_groups = (
        (
            "Assigned through Unicode 16 (excluding Private-Use)",
            "assigned_through_unicode_16",
        ),
        ("Assigned in Unicode 17", "assigned_in_unicode_17"),
        ("Private-Use (Co)", "private_use"),
        ("Unassigned/reserved (Cn)", "unassigned_or_reserved"),
        ("All Unicode scalar values", "all"),
    )
    lines = [
        "# Unicode 17 round-trip audit",
        "",
        (
            "A raw-exact case satisfies `decode(encode(x)) == x` at the "
            "Unicode-scalar-sequence level. Exactness after each method's "
            "configured normalizer is reported separately: identity for UBE "
            "and MYTE, and NFC for SCRIPT-BPE. Unicode 17 NFC-exactness is "
            "also reported where the standard supplies an expected output. "
            "The audit is deterministic and contains no sampled accuracy "
            "estimate."
        ),
        "",
        "## Exhaustive scalar coverage",
        "",
        (
            "| Unicode 17 stratum | # scalars | UBE raw exact | "
            "MYTE raw exact | "
            "SCRIPT-BPE raw exact | "
            "SCRIPT-BPE exact after configured NFC |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, group in scalar_groups:
        cases = _metric(ube, "unicode_scalars", group, "cases")
        if not all(
            cases == _metric(method, "unicode_scalars", group, "cases")
            for method in (myte, script)
        ):
            raise AssertionError("scalar case counts differ between methods")
        lines.append(
            f"| {label} | {cases:,} "
            f"| {_format_count(_metric(ube, 'unicode_scalars', group, 'raw_exact'), cases)} "
            f"| {_format_count(_metric(myte, 'unicode_scalars', group, 'raw_exact'), cases)} "
            f"| {_format_count(_metric(script, 'unicode_scalars', group, 'raw_exact'), cases)} "
            f"| {_format_count(_metric(script, 'unicode_scalars', group, 'declared_normalization_exact'), cases)} |"
        )

    lines.extend(
        [
            "",
            "## Official Unicode sequence suites",
            "",
            (
                "| Suite | # unique inputs | UBE raw exact | "
                "MYTE raw exact | "
                "SCRIPT-BPE raw exact | "
                "SCRIPT-BPE exact after configured NFC | "
                "SCRIPT-BPE exact against Unicode 17 NFC |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, suite in (
        ("NormalizationTest.txt", "normalization"),
        ("GraphemeBreakTest.txt", "grapheme_break"),
        ("emoji-test.txt", "emoji"),
    ):
        cases = _metric(ube, suite, "all", "cases")
        if not all(
            cases == _metric(method, suite, "all", "cases")
            for method in (myte, script)
        ):
            raise AssertionError(f"{suite} case counts differ between methods")
        official_nfc_cases = _metric(
            script,
            suite,
            "all",
            "nfc_expected_cases",
        )
        official_nfc = (
            _format_count(
                _metric(script, suite, "all", "nfc_exact"),
                official_nfc_cases,
            )
            if official_nfc_cases
            else "n/a"
        )
        lines.append(
            f"| {label} | {cases:,} "
            f"| {_format_count(_metric(ube, suite, 'all', 'raw_exact'), cases)} "
            f"| {_format_count(_metric(myte, suite, 'all', 'raw_exact'), cases)} "
            f"| {_format_count(_metric(script, suite, 'all', 'raw_exact'), cases)} "
            f"| {_format_count(_metric(script, suite, 'all', 'declared_normalization_exact'), cases)} "
            f"| {official_nfc} |"
        )

    lines.extend(
        [
            "",
            "## Methodology",
            "",
            (
                "- Lossless-tokenization criterion: raw identity is reported "
                "separately from equality after each method's configured "
                "normalization."
            ),
            (
                "- Scalar universe: every value from U+0000 through U+10FFFF "
                "except the surrogate range U+D800-U+DFFF."
            ),
            (
                "- Sequence suites: every unique sequence in the Unicode 17 "
                "normalization, grapheme-break, and emoji test files."
            ),
            (
                "- NormalizationTest.txt supplies normative NFC targets. The "
                "grapheme-break and emoji files are used only as official "
                "multi-scalar stress inputs, not as tokenizer conformance "
                "oracles."
            ),
            (
                "- Inputs and tokenizer artifacts are fixed by SHA-256 in the "
                "machine-readable JSON."
            ),
            _tokenizer_note(payload["tokenizers"]),
            (
                "- MYTE directly uses the official "
                "`src/rewrite_bytes.py::ByteRewriter` for both rewrite "
                "passes and directions at the pinned repository commit; "
                "the implementation and both map files are fixed by SHA-256 "
                "in the JSON."
            ),
            "",
            "## References used to define the audit",
            "",
        ]
    )
    for index, reference in enumerate(METHODOLOGY_REFERENCES, start=1):
        lines.append(
            f"{index}. [{reference['title']}]({reference['url']}). "
            f"{reference['venue']}. {reference['use']}"
        )
    return "\n".join(lines) + "\n"


def _run_all(
    cache_dir: Path,
    output_path: Path,
    report_path: Path,
    script_tok_dir: Path,
    myte_dir: Path,
    ube_tokenizer: Path | None,
    script_tokenizer: Path | None,
    train_text: Path,
    merges: int,
) -> None:
    if shutil.which("uv") is None:
        raise SystemExit(
            "uv is required to run SCRIPT-BPE: https://docs.astral.sh/uv/"
        )
    sources = _download_verified_sources(cache_dir)
    script_commit = _check_script_commit(script_tok_dir)
    myte_provenance = _check_myte_source(myte_dir)
    for path in (ube_tokenizer, script_tokenizer):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    if None in (ube_tokenizer, script_tokenizer) and not train_text.is_file():
        raise FileNotFoundError(
            f"{train_text}: run scripts/prepare_flores.py first, or pass "
            "--ube-tokenizer and --script-tokenizer"
        )

    script_python = [
        "uv",
        "run",
        "--frozen",
        "--no-dev",
        "--project",
        str(script_tok_dir.resolve()),
        "python",
        str(Path(__file__).resolve()),
    ]
    tokenizers: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(
        prefix="ube-unicode-audit-"
    ) as temporary:
        temporary_dir = Path(temporary)
        training = {
            "merges": merges,
            "training_text": _portable_path(train_text),
            "training_text_sha256": (
                _sha256(train_text) if train_text.is_file() else None
            ),
        }
        if ube_tokenizer is None:
            ube_tokenizer = temporary_dir / "ube_tokenizer.json"
            print(f"[ube] training with {merges:,} merges", flush=True)
            _train_ube_tokenizer(train_text, merges, ube_tokenizer)
            tokenizers["ube"] = {"source": "trained", **training}
        else:
            tokenizers["ube"] = {"source": "given"}
        if script_tokenizer is None:
            script_tokenizer = temporary_dir / "script_tokenizer.json"
            print(f"[script] training with {merges:,} merges", flush=True)
            subprocess.run(
                [
                    *script_python,
                    "--train-script-tokenizer",
                    str(script_tokenizer),
                    "--train-text",
                    str(train_text.resolve()),
                    "--merges",
                    str(merges),
                ],
                check=True,
                cwd=ROOT,
                env=_subprocess_environment(),
                # The SCRIPT-BPE trainer logs every merge to stdout.
                stdout=subprocess.DEVNULL,
            )
            tokenizers["script"] = {"source": "trained", **training}
        else:
            tokenizers["script"] = {"source": "given"}
        tokenizers["myte"] = {"source": "released byte maps"}

        ube_result = temporary_dir / "ube.json"
        myte_result = temporary_dir / "myte.json"
        script_result = temporary_dir / "script.json"
        common = [
            "--worker",
            "--cache-dir",
            str(cache_dir.resolve()),
        ]
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                *common,
                "--method",
                "myte",
                "--tokenizer",
                str(myte_dir.resolve()),
                "--worker-output",
                str(myte_result),
            ],
            check=True,
            cwd=ROOT,
            env=_subprocess_environment(),
        )
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                *common,
                "--method",
                "ube",
                "--tokenizer",
                str(ube_tokenizer.resolve()),
                "--worker-output",
                str(ube_result),
            ],
            check=True,
            cwd=ROOT,
            env=_subprocess_environment(),
        )
        subprocess.run(
            [
                *script_python,
                *common,
                "--method",
                "script",
                "--tokenizer",
                str(script_tokenizer.resolve()),
                "--worker-output",
                str(script_result),
            ],
            check=True,
            cwd=ROOT,
            env=_subprocess_environment(),
        )
        methods = {
            "ube": json.loads(ube_result.read_text(encoding="utf-8")),
            "myte": json.loads(myte_result.read_text(encoding="utf-8")),
            "script": json.loads(script_result.read_text(encoding="utf-8")),
        }

    for method_payload in methods.values():
        if (
            _metric(
                method_payload,
                "unicode_scalars",
                "all",
                "cases",
            )
            != SCALAR_COUNT
        ):
            raise AssertionError("exhaustive scalar count is incomplete")

    payload = {
        "version": 1,
        "design": {
            "unicode_version": UNICODE_VERSION,
            "raw_exact_definition": (
                "decoded Unicode scalar sequence exactly equals input sequence"
            ),
            "normalization_reporting": (
                "raw exactness, equality after each method's configured "
                "normalizer, and equality to the Unicode 17 expected NFC "
                "output are reported separately"
            ),
            "scalar_universe": (
                "U+0000..U+10FFFF excluding U+D800..U+DFFF"
            ),
            "scalar_count": SCALAR_COUNT,
            "sampling": "none",
            "methodology_references": list(METHODOLOGY_REFERENCES),
        },
        "unicode_sources": {
            label: {
                **DATA_SOURCES[label],
                "observed_sha256": _sha256(path),
            }
            for label, path in sources.items()
        },
        "tokenizers": tokenizers,
        "script_bpe_commit": script_commit,
        "myte_commit": myte_provenance["commit"],
        "audit_script_sha256": _sha256(Path(__file__)),
        "methods": methods,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(payload), encoding="utf-8")
    print(f"Wrote {output_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--myte-dir",
        type=Path,
        help="checkout of https://github.com/tomlimi/MYTE",
    )
    parser.add_argument(
        "--script-tok-dir",
        type=Path,
        help="checkout of https://github.com/sanderland/script_tok",
    )
    parser.add_argument(
        "--ube-tokenizer",
        type=Path,
        help="UBE tokenizer.json to audit (default: train one)",
    )
    parser.add_argument(
        "--script-tokenizer",
        type=Path,
        help="SCRIPT-BPE tokenizer .json or .json.gz to audit (default: train one)",
    )
    parser.add_argument("--train-text", type=Path, default=DEFAULT_TRAIN_TEXT)
    parser.add_argument("--merges", type=_positive_int, default=DEFAULT_MERGES)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    # Internal modes used by the orchestrating process.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--method",
        choices=("ube", "script", "myte"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--tokenizer", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--train-script-tokenizer",
        type=Path,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.train_script_tokenizer is not None:
        _train_script_tokenizer(
            args.train_text,
            args.merges,
            args.train_script_tokenizer,
        )
        return 0

    if args.worker:
        if (
            args.method is None
            or args.tokenizer is None
            or args.worker_output is None
        ):
            parser.error(
                "--worker requires --method, --tokenizer, and --worker-output"
            )
        _run_worker(
            args.method,
            args.tokenizer,
            args.cache_dir,
            args.worker_output,
        )
        return 0

    if args.myte_dir is None or args.script_tok_dir is None:
        parser.error("--myte-dir and --script-tok-dir are required")
    _run_all(
        args.cache_dir,
        args.output,
        args.report,
        args.script_tok_dir,
        args.myte_dir,
        args.ube_tokenizer,
        args.script_tokenizer,
        args.train_text,
        args.merges,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
