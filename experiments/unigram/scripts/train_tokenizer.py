#!/usr/bin/env python3
# Copyright 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

"""Train one tokenizer of the Unigram experiment on a corpus from sample_corpus.py.

``--kind`` names the base alphabet and the model:

* ``ube_unigram`` and ``ube_bpe``: the UBE pre-tokenizer, post-processor and
  decoder, with the 512 symbols of ``UniversalByteLevel.alphabet()`` as the
  initial alphabet;
* ``bytelevel_unigram`` and ``bytelevel_bpe``: the ByteLevel pre-tokenizer,
  post-processor and decoder, with the 256 symbols of ``ByteLevel.alphabet()``.

Both pre-tokenizers split text with the GPT-2 pattern and add no prefix space,
and no normalizer is used. The special tokens ``<s>``, ``<pad>``, ``</s>``,
``<unk>`` and ``<mask>`` take ids 0 to 4 and count toward ``--vocab-size``.
Unigram uses ``unk_token="<unk>"`` and ``max_piece_length=16`` (the library
default, counted in base symbols). BPE uses ``min_frequency=2``. Both trainers
set ``show_progress=False`` and keep the library defaults for all other arguments.

The documents are streamed to ``Tokenizer.train_from_iterator``, one sequence
per document, with the languages in code order and each language's documents
in corpus order. Before saving, the script checks the vocabulary size, the ids
of the special tokens and that every base symbol is in the vocabulary. It
writes the tokenizer and, next to it, ``<name>.meta.json`` with the settings,
the corpus manifest digest and the library version.

Usage::

    python scripts/train_tokenizer.py --corpus corpus/1gib --kind ube_unigram \\
        --vocab-size 32000 --output tokenizers/1gib_32000/ube_unigram/tokenizer.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from contextlib import ExitStack
from pathlib import Path

KINDS = ("ube_unigram", "bytelevel_unigram", "ube_bpe", "bytelevel_bpe")
SPECIAL_TOKENS = ("<s>", "<pad>", "</s>", "<unk>", "<mask>")
UNK_TOKEN = "<unk>"
MAX_PIECE_LENGTH = 16
MIN_FREQUENCY = 2
BATCH = 1000


def kind_parts(kind: str) -> tuple[str, str]:
    """(base alphabet, model) of a kind, such as ("ube", "unigram")."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; choose one of {', '.join(KINDS)}")
    base, model = kind.split("_")
    return base, model


def base_alphabet(base: str) -> list[str]:
    from tokenizers import pre_tokenizers

    if base == "ube":
        return sorted(pre_tokenizers.UniversalByteLevel.alphabet())
    return sorted(pre_tokenizers.ByteLevel.alphabet())


def build_tokenizer(kind: str):
    """An untrained tokenizer with the pipeline of the kind."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors

    base, model = kind_parts(kind)
    tokenizer = Tokenizer(models.Unigram() if model == "unigram" else models.BPE())
    tokenizer.normalizer = None
    if base == "ube":
        tokenizer.pre_tokenizer = pre_tokenizers.UniversalByteLevel(
            add_prefix_space=False, use_regex=True, trim_offsets=False
        )
        tokenizer.post_processor = processors.UniversalByteLevel(trim_offsets=False)
        tokenizer.decoder = decoders.UniversalByteLevel()
    else:
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def trainer_arguments(kind: str, vocab_size: int) -> dict:
    """The trainer arguments besides the initial alphabet."""
    _, model = kind_parts(kind)
    arguments = {"vocab_size": vocab_size, "special_tokens": list(SPECIAL_TOKENS), "show_progress": False}
    if model == "unigram":
        arguments.update(unk_token=UNK_TOKEN, max_piece_length=MAX_PIECE_LENGTH)
    else:
        arguments.update(min_frequency=MIN_FREQUENCY)
    return arguments


def build_trainer(kind: str, vocab_size: int):
    from tokenizers import trainers

    base, model = kind_parts(kind)
    trainer = trainers.UnigramTrainer if model == "unigram" else trainers.BpeTrainer
    return trainer(initial_alphabet=base_alphabet(base), **trainer_arguments(kind, vocab_size))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(corpus: Path) -> tuple[dict, str]:
    """The corpus manifest, after checking every language file, and the SHA-256 of its bytes."""
    path = corpus / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"{path} not found: create the corpus with sample_corpus.py")
    raw = path.read_bytes()
    try:
        manifest = json.loads(raw)
    except ValueError as error:
        raise SystemExit(f"{path}: invalid manifest ({error})") from None
    languages = manifest.get("languages") if isinstance(manifest, dict) else None
    if not isinstance(languages, dict) or not languages:
        raise SystemExit(f"{path}: no languages")
    requested = manifest.get("requested_bytes")
    if type(requested) is not int or requested <= 0:
        raise SystemExit(f"{path}: requested_bytes must be a positive integer")
    for lang, info in languages.items():
        filename = f"{lang}.jsonl"
        if not isinstance(info, dict) or info.get("file") != filename or Path(filename).name != filename:
            raise SystemExit(f"{path}: invalid corpus filename for {lang!r}")
        for key in ("documents", "text_bytes", "quota_bytes"):
            if type(info.get(key)) is not int or info[key] < 0:
                raise SystemExit(f"{path}: invalid {key} for {lang}")
        share = info.get("share")
        if type(share) not in (int, float) or not 0 < share <= 1 or round(requested * share) != info["quota_bytes"]:
            raise SystemExit(f"{path}: {lang} quota does not match its share of requested_bytes")
        if not 0 <= info["quota_bytes"] - info["text_bytes"] <= 3:
            raise SystemExit(f"{path}: {lang} text bytes do not match its UTF-8 quota")
    if not math.isclose(math.fsum(info["share"] for info in languages.values()), 1.0):
        raise SystemExit(f"{path}: language shares do not sum to one")
    totals = {
        "language_count": len(languages),
        "documents": sum(info["documents"] for info in languages.values()),
        "text_bytes": sum(info["text_bytes"] for info in languages.values()),
    }
    for key, total in totals.items():
        if type(manifest.get(key)) is not int or manifest[key] != total:
            raise SystemExit(f"{path}: {key} differs from the language totals")
    if not totals["documents"]:
        raise SystemExit(f"{path}: the corpus has no training documents")
    for info in languages.values():
        source = corpus / info["file"]
        if not source.is_file():
            raise SystemExit(f"{source}: missing corpus file")
        if file_sha256(source) != info.get("sha256"):
            raise SystemExit(f"{source}: SHA-256 differs from the manifest")
    return manifest, hashlib.sha256(raw).hexdigest()


def corpus_batches(corpus: Path, manifest: dict, batch: int = BATCH):
    """Batches of documents: languages in code order, documents in corpus order."""
    if batch < 1:
        raise ValueError("batch size must be positive")
    for lang in sorted(manifest["languages"]):
        info = manifest["languages"][lang]
        path = corpus / info["file"]
        digest = hashlib.sha256()
        count = text_bytes = 0
        texts = []
        with path.open("rb") as source:
            for count, line in enumerate(source, start=1):
                digest.update(line)
                try:
                    record = json.loads(line)
                    text = record.get("text") if isinstance(record, dict) else None
                    if not isinstance(text, str) or not text:
                        raise ValueError("expected a nonempty text field")
                    text_bytes += len(text.encode("utf-8"))
                except ValueError as error:
                    raise SystemExit(f"{path}:{count}: {error}") from None
                texts.append(text)
                if len(texts) == batch:
                    yield texts
                    texts = []
        if digest.hexdigest() != info["sha256"]:
            raise SystemExit(f"{path}: SHA-256 differs from the manifest")
        if count != info["documents"] or text_bytes != info["text_bytes"]:
            raise SystemExit(f"{path}: documents or text bytes differ from the manifest")
        if texts:
            yield texts


def check(tokenizer, kind: str, vocab_size: int) -> list[str]:
    """Problems with a trained tokenizer; empty when it has the expected layout."""
    base, model = kind_parts(kind)
    problems = []
    size = tokenizer.get_vocab_size(with_added_tokens=True)
    if size != vocab_size:
        problems.append(f"vocabulary size {size} instead of {vocab_size}")
    special = [tokenizer.id_to_token(i) for i in range(len(SPECIAL_TOKENS))]
    if special != list(SPECIAL_TOKENS):
        problems.append(f"ids 0 to {len(SPECIAL_TOKENS) - 1} hold {special}")
    vocab = tokenizer.get_vocab(with_added_tokens=True)
    missing = [symbol for symbol in base_alphabet(base) if symbol not in vocab]
    if missing:
        problems.append(f"{len(missing)} base symbols are not in the vocabulary")
    if model == "unigram":
        saved = json.loads(tokenizer.to_str())["model"]
        if saved.get("unk_id") != SPECIAL_TOKENS.index(UNK_TOKEN):
            problems.append(f"unk_id {saved.get('unk_id')}")
        long = [piece for piece, _ in saved["vocab"] if piece not in SPECIAL_TOKENS and len(piece) > MAX_PIECE_LENGTH]
        if long:
            problems.append(f"{len(long)} pieces are longer than {MAX_PIECE_LENGTH} base symbols")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True, help="output directory of sample_corpus.py")
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--vocab-size", type=int, required=True, help="vocabulary size, special tokens included")
    parser.add_argument("--output", type=Path, required=True, help="tokenizer.json to write")
    args = parser.parse_args()
    if not args.output.name:
        parser.error("--output must name a file")
    meta_path = args.output.with_suffix(".meta.json")
    for path in (args.output, meta_path):
        if path.exists() or path.is_symlink():
            parser.error(f"{path} exists")

    import tokenizers

    base, model = kind_parts(args.kind)
    alphabet_size = len(base_alphabet(base))
    minimum = alphabet_size + len(SPECIAL_TOKENS)
    if args.vocab_size < minimum:
        parser.error(f"--vocab-size must be at least {minimum} for the {base} alphabet and special tokens")
    manifest, manifest_sha256 = load_manifest(args.corpus)
    tokenizer = build_tokenizer(args.kind)
    trainer = build_trainer(args.kind, args.vocab_size)
    print(
        f"Training {args.kind} at {args.vocab_size} on {args.corpus}: "
        f"{manifest['text_bytes']} bytes, {manifest['documents']} documents",
        flush=True,
    )
    started = time.monotonic()
    tokenizer.train_from_iterator(
        corpus_batches(args.corpus, manifest), trainer=trainer, length=manifest["documents"]
    )
    seconds = time.monotonic() - started
    problems = check(tokenizer, args.kind, args.vocab_size)
    if problems:
        raise SystemExit(f"{args.kind}: {'; '.join(problems)}")

    serialized = tokenizer.to_str(pretty=True).encode("utf-8")
    meta = {
        "kind": args.kind,
        "model": model,
        "base_alphabet": "UniversalByteLevel" if base == "ube" else "ByteLevel",
        "base_alphabet_size": alphabet_size,
        "trainer": trainer_arguments(args.kind, args.vocab_size),
        "corpus_manifest_sha256": manifest_sha256,
        "corpus_requested_bytes": manifest["requested_bytes"],
        "corpus_text_bytes": manifest["text_bytes"],
        "corpus_documents": manifest["documents"],
        "tokenizers_version": tokenizers.__version__,
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
        "train_seconds": round(seconds, 1),
        "tokenizer_sha256": hashlib.sha256(serialized).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as cleanup:
        with args.output.open("xb") as sink:
            cleanup.callback(args.output.unlink, missing_ok=True)
            sink.write(serialized)
        with meta_path.open("x", encoding="utf-8") as sink:
            cleanup.callback(meta_path.unlink, missing_ok=True)
            sink.write(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
        cleanup.pop_all()
    print(f"Wrote {args.output} in {seconds:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OSError as error:
        raise SystemExit(str(error)) from None
