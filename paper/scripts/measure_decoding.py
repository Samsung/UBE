#!/usr/bin/env python3
# Copyright 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

"""Measure partial decoding of FLORES-200 token sequences.

Every devtest sentence of the 204 languages is encoded without special tokens, and the script records:

* ``round_trip``: sentences whose token ids decode back to the sentence;
* ``streams``: each sentence is split at the token position nearest the midpoint that is also a character boundary
  (earlier on ties); the first part is the prompt and the rest is decoded incrementally as if generated, by
  - ``decode_stream``: ``decoders.DecodeStream`` started with the prompt ids;
  - ``window_back5`` / ``window_aligned``: a detokenizer that decodes a window of tokens and emits the text added
    by the newest tokens once the window text no longer ends with U+FFFD, starting five tokens before the end of
    the prompt (or at the first token), or at ``UniversalByteLevelDecoding.aligned_offset`` of that position;
  - ``final_back1`` / ``final_aligned``: one decode of the whole continuation from one token before the end of
    the prompt, or from the aligned offset of that position, keeping the text after the prompt part;
  a stream is exact when its text equals the decoded continuation;
* ``slices``: the token ids from every interior token position to the end of the sentence, decoded with
  ``Tokenizer.decode`` (``plain``) and with ``UniversalByteLevelDecoding.decode_slice``; a slice is exact when
  its text equals the following characters, preceded by one or more U+FFFD for a character that the cut divides.

Languages are processed in separate workers, with tokenizers' internal parallelism disabled in each worker.
Window-width summaries are null when there are no eligible streams.

Usage::

    python scripts/measure_decoding.py path/to/tokenizer.json [more.json ...] \
        --output-dir results/decoding
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLORES = ROOT / "data" / "flores200_devtest.json"
REPLACEMENT = "�"
COUNTS = (
    "sentences",
    "round_trip",
    "streams",
    "decode_stream",
    "window_back5",
    "window_aligned",
    "final_back1",
    "final_aligned",
    "slices",
    "slices_inside_character",
    "slices_plain",
    "slices_decode_slice",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_flores(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        raise SystemExit(f"{path} not found: run scripts/prepare_flores.py first")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        column.removeprefix("sentence_"): [s for s in sentences if s and s.strip()]
        for column, sentences in sorted(data.items())
        if column.startswith("sentence_")
    }


def tokenizer_name(path: Path) -> str:
    path = path.resolve()
    return path.parent.name if path.name == "tokenizer.json" else path.stem


def symbol_starts(text: str, universal: bool) -> list[int]:
    """Base-symbol offset of each character, and the total: UBE writes characters of three or four UTF-8
    bytes with their UTF-16LE bytes."""
    starts, position = [], 0
    for char in text:
        starts.append(position)
        width = len(char.encode("utf-8"))
        if universal and width > 2:
            width = len(char.encode("utf-16-le"))
        position += width
    return starts + [position]


def window_stream(tokenizer, ids: list[int], prompt_length: int, prefix_offset: int) -> str:
    read_offset, pieces = prompt_length, []
    for end in range(prompt_length + 1, len(ids) + 1):
        prefix_text = tokenizer.decode(ids[prefix_offset:read_offset])
        new_text = tokenizer.decode(ids[prefix_offset:end])
        if len(new_text) > len(prefix_text) and not new_text.endswith(REPLACEMENT):
            pieces.append(new_text[len(prefix_text) :])
            prefix_offset, read_offset = read_offset, end
    return "".join(pieces)


def final_text(tokenizer, ids: list[int], prompt_length: int, prefix_offset: int) -> str:
    prefix_text = tokenizer.decode(ids[prefix_offset:prompt_length])
    new_text = tokenizer.decode(ids[prefix_offset:])
    return new_text[len(prefix_text) :]


def slice_is_exact(output: str, text: str, starts: list[int], position: int) -> bool:
    index = max(i for i, start in enumerate(starts[:-1]) if start <= position)
    if starts[index] == position:
        return output == text[index:]
    rest = text[index + 1 :]
    head = output[: len(output) - len(rest)]
    return output.endswith(rest) and bool(head) and set(head) == {REPLACEMENT}


def measure_language(job: tuple[str, str, list[str]]) -> tuple[str, dict]:
    tokenizer_path, lang, sentences = job
    from tokenizers import Tokenizer, UniversalByteLevelDecoding, decoders

    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.no_padding()
    tokenizer.no_truncation()
    universal = type(tokenizer.decoder).__name__ == "UniversalByteLevel"
    decoding = UniversalByteLevelDecoding(tokenizer)
    counts = dict.fromkeys(COUNTS, 0)
    widths = []
    for text, encoding in zip(sentences, tokenizer.encode_batch(sentences, add_special_tokens=False)):
        ids = encoding.ids
        counts["sentences"] += 1
        if tokenizer.decode(ids) != text:
            continue
        counts["round_trip"] += 1
        flags = decoding.boundaries(ids)
        interior = [k for k in range(1, len(ids)) if flags[k]]
        if interior:
            k = min(interior, key=lambda position: (abs(2 * position - len(ids)), position))
            prompt, expected = ids[:k], tokenizer.decode(ids[k:])
            stream = decoders.DecodeStream(ids=prompt, skip_special_tokens=True)
            pieces = [stream.step(tokenizer, token_id) for token_id in ids[k:]]
            back5 = max(k - 5, 0)
            aligned5 = decoding.aligned_offset(prompt, k - 5)
            counts["streams"] += 1
            counts["decode_stream"] += "".join(piece or "" for piece in pieces) == expected
            counts["window_back5"] += window_stream(tokenizer, ids, k, back5) == expected
            counts["window_aligned"] += window_stream(tokenizer, ids, k, aligned5) == expected
            counts["final_back1"] += final_text(tokenizer, ids, k, k - 1) == expected
            counts["final_aligned"] += final_text(tokenizer, ids, k, decoding.aligned_offset(prompt, k - 1)) == expected
            widths.append(k - aligned5)
        starts = symbol_starts(text, universal)
        offsets = [0]
        for token_id in ids:
            offsets.append(offsets[-1] + len(tokenizer.id_to_token(token_id)))
        if offsets[-1] != starts[-1]:
            raise ValueError(f"{lang}: base symbols of the tokens do not match the text")
        for j in range(1, len(ids)):
            counts["slices"] += 1
            counts["slices_inside_character"] += not flags[j]
            counts["slices_plain"] += slice_is_exact(tokenizer.decode(ids[j:]), text, starts, offsets[j])
            counts["slices_decode_slice"] += slice_is_exact(decoding.decode_slice(ids[j:]), text, starts, offsets[j])
    return lang, {"counts": counts, "aligned_window_tokens": widths}


def _init_worker() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def measure(tokenizer_path: Path, flores: dict[str, list[str]], workers: int) -> dict:
    jobs = [(str(tokenizer_path), lang, sentences) for lang, sentences in flores.items()]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=_init_worker) as pool:
        per_language = dict(pool.map(measure_language, jobs))
    totals = {key: sum(result["counts"][key] for result in per_language.values()) for key in COUNTS}
    widths = sorted(width for result in per_language.values() for width in result["aligned_window_tokens"])
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return {
        "tokenizer": tokenizer_name(tokenizer_path),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "decoder": type(tokenizer.decoder).__name__,
        "languages": len(per_language),
        "totals": totals,
        "aligned_window_tokens": {
            "median": statistics.median(widths) if widths else None,
            "p99": widths[int(0.99 * (len(widths) - 1))] if widths else None,
            "max": widths[-1] if widths else None,
        },
        "results": {lang: result["counts"] for lang, result in per_language.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tokenizers", type=Path, nargs="+", help="tokenizer.json files")
    parser.add_argument("--flores", type=Path, default=DEFAULT_FLORES)
    parser.add_argument("--output-dir", type=Path, help="write one <name>.json per tokenizer")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    names = [tokenizer_name(path) for path in args.tokenizers]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        parser.error(f"tokenizers must have distinct names: {', '.join(duplicates)}")
    if args.output_dir is not None:
        inputs = {path.resolve() for path in args.tokenizers} | {args.flores.resolve()}
        clashes = [
            str((args.output_dir / f"{name}.json").resolve())
            for name in names
            if any(
                (args.output_dir / f"{name}.json").resolve() == path
                or (
                    (args.output_dir / f"{name}.json").exists()
                    and path.exists()
                    and (args.output_dir / f"{name}.json").samefile(path)
                )
                for path in inputs
            )
        ]
        if clashes:
            parser.error(f"--output-dir would overwrite input files: {', '.join(clashes)}")
    flores = load_flores(args.flores)
    flores_sha256 = file_sha256(args.flores)
    print(f"FLORES-200 devtest: {len(flores)} languages", flush=True)
    for path, name in zip(args.tokenizers, names):
        started = time.time()
        report = measure(path, flores, args.workers) | {"flores_sha256": flores_sha256}
        totals = report["totals"]
        print(
            f"{name}: round trip {totals['round_trip']}/{totals['sentences']}; streams exact: DecodeStream "
            f"{totals['decode_stream']}, window back 5 {totals['window_back5']}, aligned {totals['window_aligned']}, "
            f"final back 1 {totals['final_back1']}, aligned {totals['final_aligned']} of {totals['streams']}; "
            f"slices exact: plain {totals['slices_plain']}, decode_slice {totals['slices_decode_slice']} of "
            f"{totals['slices']} ({totals['slices_inside_character']} inside a character)  "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )
        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            out = args.output_dir / f"{name}.json"
            out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
