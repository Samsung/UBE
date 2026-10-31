#!/usr/bin/env python3
"""MYTE-UBE: MYTE's released byte maps followed by UBE routing of residual characters.

MYTE's morphology-merge targets are not valid UTF-8, so UBE's per-character
routing is undefined on MYTE output, and routing first would remove the UTF-8
byte patterns that MYTE's maps match. MYTE-UBE therefore applies the released
maps first and routes only the characters that no merge rule rewrote:

1. MYTE decomposition pass. Every target is valid UTF-8, so the stream is
   still text.
2. MYTE morphology-merge pass, tracking which output spans a rule produced.
   Merge sources are valid UTF-8 and their targets are not, so every output
   byte is either a MYTE code byte or a byte of an untouched character.
3. UBE routing of the untouched characters: 1-2-byte UTF-8 characters keep
   their UTF-8 bytes; 3-4-byte characters are written as their UTF-16-LE
   code-unit bytes (2 for BMP characters, 4 for supplementary characters).

The representation has 512 symbols in two 256-symbol regions. Region A holds
bytes that are copied verbatim into the post-MYTE byte stream (the UTF-8 bytes
of residual characters and MYTE code bytes); region B holds routed UTF-16-LE
bytes. Symbols are written as private-use characters, U+E300..U+E3FF for
region A and U+E400..U+E4FF for region B. Decoding rebuilds the post-MYTE byte
stream exactly and then applies MYTE's released reverse maps.

Running this file measures the representation on FLORES-200 devtest and checks
every sentence: the symbols rebuild the post-MYTE byte stream exactly, and the
decoded text equals the output of MYTE's released implementation
(``src/rewrite_bytes.py::ByteRewriter``) applied forward and in reverse. Counts
are base symbols before any BPE; one symbol is one model-input position.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

from myte_codec import DECOMPOSE_REL, MERGE_REL, GreedyByteRewriter, MyteCodec, _sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLORES = ROOT / "data" / "flores200_devtest.json"
DEFAULT_OUTPUT = ROOT / "results" / "myte_ube_flores.json"

REGION_A_BASE = 0xE300  # bytes copied verbatim into the post-MYTE byte stream
REGION_B_BASE = 0xE400  # routed UTF-16-LE code-unit bytes

MYTE_REWRITER_REL = Path("src/rewrite_bytes.py")
MYTE_REWRITER_SHA256 = "24454728cc5d0ca5207823d2a3d43507de8546df691ebaf5ec97a06c99e53ddd"

ENGLISH = "eng_Latn"
# Language groups reported in the paper.
CJK_SCRIPTS = ("Hans", "Hant", "Jpan", "Hang")
OTHER_BMP_SCRIPTS = ("Tibt", "Olck", "Tfng", "Mymr")
EXPECTED_CONFIGURATIONS = 204
EXPECTED_SENTENCES = 206_448

_LEAF = -1


class SpanTrackingRewriter(GreedyByteRewriter):
    """Greedy rewriter that also reports which output spans came from rules.

    The spans concatenate to exactly the output of ``rewrite``. A span counts
    as a rule output only when its bytes differ from the consumed input, so
    explicit identity rules are treated as untouched text.
    """

    def rewrite_with_spans(self, data: bytes) -> list[tuple[bool, bytes]]:
        spans: list[tuple[bool, bytes]] = []
        root = self._root
        start = 0
        data_len = len(data)
        while start < data_len:
            node = root
            best: bytes | None = None
            best_end = start
            cursor = start
            while cursor < data_len:
                node = node.get(data[cursor])
                if node is None:
                    break
                cursor += 1
                leaf = node.get(_LEAF)
                if leaf is not None:
                    best = leaf
                    best_end = cursor
            if best is None:
                raise AssertionError(f"MYTE trie has no identity byte at {start}")
            spans.append((best != data[start:best_end], best))
            start = best_end
        return spans


class MyteUbeCodec:
    """MYTE-first encoding with UBE routing of residual characters (512 symbols)."""

    def __init__(self, myte: MyteCodec, span_merge: SpanTrackingRewriter):
        self.myte = myte
        self._span_merge = span_merge
        self.provenance = myte.provenance

    @classmethod
    def from_dir(cls, myte_dir: Path) -> "MyteUbeCodec":
        myte = MyteCodec.from_dir(myte_dir, with_reverse=True)
        decompose_rules = json.loads((myte_dir / DECOMPOSE_REL).read_text(encoding="utf-8"))
        merge_rules = json.loads((myte_dir / MERGE_REL).read_text(encoding="utf-8"))
        # The span classification relies on these properties of the released maps.
        for label, rules, valid_targets in (
            ("decompose", decompose_rules, True),
            ("merge", merge_rules, False),
        ):
            for source, target in rules.items():
                try:
                    bytes.fromhex(source).decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise AssertionError(f"{label} source is not valid UTF-8: {source}") from exc
                try:
                    bytes.fromhex(target).decode("utf-8")
                    target_valid = True
                except UnicodeDecodeError:
                    target_valid = False
                if target_valid != valid_targets and source != target:
                    raise AssertionError(f"unexpected {label} rule: {source} -> {target}")
        return cls(myte, SpanTrackingRewriter(merge_rules))

    def spans(self, text: str) -> list[tuple[bool, bytes]]:
        """Maximal post-MYTE spans as ``(is_code, bytes)``.

        Untouched bytes arrive one at a time; only their maximal runs are whole
        characters, so adjacent spans of the same kind are coalesced.
        """
        decomposed = self.myte.decompose.rewrite(text.encode("utf-8"))
        decomposed.decode("utf-8")  # decomposition targets are valid UTF-8
        coalesced: list[tuple[bool, bytearray]] = []
        for is_code, chunk in self._span_merge.rewrite_with_spans(decomposed):
            if coalesced and coalesced[-1][0] == is_code:
                coalesced[-1][1].extend(chunk)
            else:
                coalesced.append((is_code, bytearray(chunk)))
        return [(is_code, bytes(run)) for is_code, run in coalesced]

    def encode(self, text: str) -> str:
        symbols: list[str] = []
        for is_code, chunk in self.spans(text):
            if is_code:
                symbols.extend(chr(REGION_A_BASE + value) for value in chunk)
                continue
            for character in chunk.decode("utf-8"):
                utf8 = character.encode("utf-8")
                if len(utf8) <= 2:
                    symbols.extend(chr(REGION_A_BASE + value) for value in utf8)
                else:
                    symbols.extend(chr(REGION_B_BASE + value) for value in character.encode("utf-16-le"))
        return "".join(symbols)

    @staticmethod
    def post_myte_bytes(symbols: str) -> bytes:
        """Invert the routing layer back to the post-MYTE byte stream."""
        output = bytearray()
        utf16 = bytearray()
        for symbol in symbols:
            value = ord(symbol)
            if REGION_B_BASE <= value < REGION_B_BASE + 256:
                utf16.append(value - REGION_B_BASE)
                continue
            if utf16:
                output.extend(utf16.decode("utf-16-le").encode("utf-8"))
                utf16.clear()
            if REGION_A_BASE <= value < REGION_A_BASE + 256:
                output.append(value - REGION_A_BASE)
            else:
                raise ValueError(f"not a MYTE-UBE symbol: {symbol!r}")
        if utf16:
            output.extend(utf16.decode("utf-16-le").encode("utf-8"))
        return bytes(output)

    def decode(self, symbols: str) -> str:
        return self.myte.decode_bytes(self.post_myte_bytes(symbols)).decode("utf-8")


def ube_length(text: str) -> int:
    """Number of UBE base symbols (512-symbol alphabet) for ``text``."""
    total = 0
    for character in text:
        utf8_length = len(character.encode("utf-8"))
        total += utf8_length if utf8_length <= 2 else len(character.encode("utf-16-le"))
    return total


def load_official_rewriter(myte_dir: Path):
    """Load ByteRewriter from the MYTE checkout."""
    source = myte_dir.resolve() / MYTE_REWRITER_REL
    if not source.is_file():
        raise FileNotFoundError(f"MYTE implementation missing: {source}")
    actual = _sha256(source)
    if actual != MYTE_REWRITER_SHA256:
        raise RuntimeError(f"MYTE ByteRewriter hash mismatch: expected {MYTE_REWRITER_SHA256}, got {actual}")
    spec = importlib.util.spec_from_file_location("_myte_rewrite_bytes", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ByteRewriter


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summarize(per_language: dict[str, dict[str, float]]) -> dict[str, float | int]:
    averages = {language: values["avg"] for language, values in per_language.items()}
    cjk = [avg for language, avg in averages.items() if language.endswith(CJK_SCRIPTS)]
    other_bmp = [avg for language, avg in averages.items() if language.endswith(OTHER_BMP_SCRIPTS)]
    english = averages[ENGLISH]
    return {
        "english": english,
        "cjk": mean(cjk),
        "cjk_configurations": len(cjk),
        "other_bmp": mean(other_bmp),
        "other_bmp_configurations": len(other_bmp),
        "overall": mean(list(averages.values())),
        "total_positions": sum(int(values["total"]) for values in per_language.values()),
    }


def compare(reference: dict[str, dict], candidate: dict[str, dict]) -> dict[str, int]:
    changes = [int(candidate[lang]["total"]) - int(reference[lang]["total"]) for lang in reference]
    return {
        "configurations_lower": sum(change < 0 for change in changes),
        "configurations_tied": sum(change == 0 for change in changes),
        "configurations_higher": sum(change > 0 for change in changes),
    }


def load_flores(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        raise SystemExit(f"{path} not found: run scripts/prepare_flores.py first")
    data = json.loads(path.read_text(encoding="utf-8"))
    sentences = {key: value for key, value in data.items() if key.startswith("sentence_")}
    if len(sentences) != EXPECTED_CONFIGURATIONS:
        raise ValueError(f"expected {EXPECTED_CONFIGURATIONS} FLORES-200 languages, found {len(sentences)}")
    return sentences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--myte-dir", type=Path, required=True, help="checkout of github.com/tomlimi/MYTE")
    parser.add_argument("--flores", type=Path, default=DEFAULT_FLORES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    started = time.monotonic()
    flores = load_flores(args.flores)
    codec = MyteUbeCodec.from_dir(args.myte_dir)
    byte_rewriter = load_official_rewriter(args.myte_dir)
    official_decompose = byte_rewriter(str(args.myte_dir / DECOMPOSE_REL))
    official_merge = byte_rewriter(str(args.myte_dir / MERGE_REL))

    methods = ("utf8_bytes", "ube", "myte", "myte_ube")
    totals: dict[str, dict[str, int]] = {method: {} for method in methods}
    symbol_digest = hashlib.sha256()
    distinct_symbols: set[str] = set()
    sentences = 0

    for key in sorted(flores):
        language = key.removeprefix("sentence_")
        counts = dict.fromkeys(methods, 0)
        for text in flores[key]:
            raw = text.encode("utf-8")
            post_myte = codec.myte.encode_bytes(raw)
            if b"".join(chunk for _, chunk in codec.spans(text)) != post_myte:
                raise AssertionError(f"span concatenation differs from the MYTE rewriter ({language})")
            symbols = codec.encode(text)
            if codec.post_myte_bytes(symbols) != post_myte:
                raise AssertionError(f"post-MYTE byte stream not rebuilt exactly ({language})")

            hex_bytes = [f"{value:02x}" for value in raw]
            official_post = official_merge.rewrite_bytes(official_decompose.rewrite_bytes(hex_bytes))
            if bytes.fromhex("".join(official_post)) != post_myte:
                raise AssertionError(f"MYTE encoding differs from the released implementation ({language})")
            official_decoded = official_decompose.rewrite_bytes(
                official_merge.rewrite_bytes(official_post, reverse=True), reverse=True
            )
            decoded = codec.myte.decode_bytes(codec.post_myte_bytes(symbols))
            if decoded != bytes.fromhex("".join(official_decoded)):
                raise AssertionError(f"decoded output differs from MYTE's released decoder ({language})")
            if codec.decode(symbols) != decoded.decode("utf-8"):
                raise AssertionError(f"MyteUbeCodec.decode is inconsistent ({language})")

            sentences += 1
            counts["utf8_bytes"] += len(raw)
            counts["ube"] += ube_length(text)
            counts["myte"] += len(post_myte)
            counts["myte_ube"] += len(symbols)
            distinct_symbols.update(symbols)
            symbol_digest.update(key.encode("utf-8") + b"\0")
            symbol_digest.update(symbols.encode("utf-8") + b"\0")
        for method in methods:
            totals[method][language] = counts[method]
        print(f"{language}: {len(flores[key])} sentences verified", flush=True)

    if sentences != EXPECTED_SENTENCES:
        raise ValueError(f"expected {EXPECTED_SENTENCES} sentences, found {sentences}")

    per_language = {
        method: {
            language: {"total": total, "avg": total / len(flores[f"sentence_{language}"])}
            for language, total in totals[method].items()
        }
        for method in methods
    }
    summaries = {method: summarize(per_language[method]) for method in methods}
    myte_summary, myte_ube_summary = summaries["myte"], summaries["myte_ube"]
    report = {
        "dataset": {
            "configurations": len(flores),
            "sentences": sentences,
            "sha256": _sha256(args.flores),
        },
        "myte": codec.provenance | {"implementation_sha256": MYTE_REWRITER_SHA256},
        "alphabet": {"utf8_bytes": 256, "ube": 512, "myte": 256, "myte_ube": 512},
        "verification": {
            "sentences_checked": sentences,
            "post_myte_stream_exact": sentences,
            "decoded_equals_released_myte_decoder": sentences,
            "myte_ube_distinct_symbols_observed": len(distinct_symbols),
            "myte_ube_symbol_stream_sha256": symbol_digest.hexdigest(),
        },
        "summaries": summaries,
        "myte_ube_vs_myte": {
            "cjk_change": myte_ube_summary["cjk"] / myte_summary["cjk"] - 1,
            "other_bmp_change": myte_ube_summary["other_bmp"] / myte_summary["other_bmp"] - 1,
            "total_positions_change": myte_ube_summary["total_positions"] / myte_summary["total_positions"] - 1,
            **compare(per_language["myte"], per_language["myte_ube"]),
        },
        "per_language": {
            language: {method: totals[method][language] for method in methods} for language in sorted(totals["myte"])
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    labels = {"ube": "UBE", "utf8_bytes": "UTF-8 bytes", "myte_ube": "MYTE-UBE", "myte": "MYTE"}
    print(f"\n{'Representation':<14} {'Alphabet':>8} {'English':>9} {'CJK (5)':>9} {'Other BMP (7)':>14} "
          f"{'Overall':>9} {'Total positions':>16}")
    for method in ("ube", "utf8_bytes", "myte_ube", "myte"):
        s = summaries[method]
        print(f"{labels[method]:<14} {report['alphabet'][method]:>8} {s['english']:>9.3f} {s['cjk']:>9.3f} "
              f"{s['other_bmp']:>14.3f} {s['overall']:>9.3f} {s['total_positions']:>16,}")
    change = report["myte_ube_vs_myte"]
    print(f"\nMYTE-UBE vs MYTE: CJK {change['cjk_change']:+.2%}, Other BMP {change['other_bmp_change']:+.2%}, "
          f"total positions {change['total_positions_change']:+.2%}; languages lower/tied/higher "
          f"{change['configurations_lower']}/{change['configurations_tied']}/{change['configurations_higher']}")
    print(f"Verified {sentences:,} sentences: post-MYTE byte stream rebuilt exactly and decoded text equal to "
          f"MYTE's released decoder output ({time.monotonic() - started:.0f}s)")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
