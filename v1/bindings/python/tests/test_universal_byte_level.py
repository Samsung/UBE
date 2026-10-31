# Copyright 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

"""Universal byte-level (UBE) tokenizers.

A character of one or two UTF-8 bytes is spelled with those bytes, in the GPT-2 byte-level alphabet,
and a longer one with the bytes of its UTF-16LE code units, at U+E100 + byte. A `tokenizer.json`
written by `tokenizers` 0.x with the `UniversalByteLevel` pre-tokenizer loads here as it is.
"""

import json

import pytest

from tokenizers import Tokenizer


def byte_level_alphabet():
    """GPT-2's map from bytes to characters: printable bytes stand for themselves, the rest move up."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    shifted = iter(range(256, 512))
    return [chr(b) if b in printable else chr(next(shifted)) for b in range(256)]


UTF8_ATOMS = byte_level_alphabet()
UTF16_ATOMS = [chr(0xE100 + b) for b in range(256)]


def spell(text):
    atoms = []
    for c in text:
        utf8 = c.encode("utf-8")
        if len(utf8) > 2:
            atoms.extend(UTF16_ATOMS[b] for b in c.encode("utf-16-le"))
        else:
            atoms.extend(UTF8_ATOMS[b] for b in utf8)
    return "".join(atoms)


@pytest.fixture(scope="module")
def ube(tmp_path_factory):
    """A 0.x UBE file: `<s>`, the 512 atoms, and the merges that spell 中, 文 and 中文."""
    vocab = {"<s>": 0}
    for atom in UTF8_ATOMS + UTF16_ATOMS:
        vocab[atom] = len(vocab)
    zhong, wen = spell("中"), spell("文")
    merges = [[zhong[0], zhong[1]], [wen[0], wen[1]], [zhong, wen]]
    for left, right in merges:
        vocab[left + right] = len(vocab)
    byte_map = {"add_prefix_space": False, "trim_offsets": False, "use_regex": True}
    config = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 0,
                "content": "<s>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "UniversalByteLevel", **byte_map},
        "post_processor": {"type": "UniversalByteLevel", **byte_map},
        "decoder": {"type": "UniversalByteLevel"},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": merges,
        },
    }
    path = tmp_path_factory.mktemp("ube") / "tokenizer.json"
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return Tokenizer.from_file(path)


def test_the_alphabet_has_512_atoms():
    assert len(set(UTF8_ATOMS + UTF16_ATOMS)) == 512


@pytest.mark.parametrize(
    "text, atoms",
    [("A", 1), ("é", 2), ("한", 2), ("न", 2), ("😀", 4), ("中", 1), ("中文", 1)],
)
def test_a_character_costs_its_routed_bytes_until_merged(ube, text, atoms):
    assert len(ube.encode(text, add_special_tokens=False).ids) == atoms


def test_tokens_are_spelled_in_the_alphabet(ube):
    assert ube.tokenize("中文 é", add_special_tokens=False) == [spell("中文"), spell(" "), *spell("é")]


@pytest.mark.parametrize(
    "text",
    ["Hello, world!\n", "中文 and 한국어 😀", "नमस्ते", "a literal Ġ and ", "﻿BOM", ""],
)
def test_text_round_trips(ube, text):
    assert ube.decode(ube.encode(text, add_special_tokens=False).ids) == text


def test_special_tokens(ube):
    ids = ube.encode("<s>中", add_special_tokens=False).ids
    assert ids == [0, ube.encode("中", add_special_tokens=False).ids[0]]
    assert ube.decode(ids, skip_special_tokens=False) == "<s>中"
    assert ube.decode(ids, skip_special_tokens=True) == "中"
