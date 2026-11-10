# Copyright 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

import random

from tokenizers import (
    AddedToken,
    ByteLevelBPETokenizer,
    Tokenizer,
    UniversalByteLevelBPETokenizer,
    UniversalByteLevelDecoding,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
)
from tokenizers.implementations.universal_byte_level_decoding import _second_byte_by_code_units

import pytest


CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Le renard brun rapide saute par-dessus le chien paresseux.",
    "Быстрая коричневая лиса прыгает через ленивую собаку.",
    "敏捷的棕色狐狸跳过了懒狗。",
    "素早い茶色の狐がのろまな犬を飛び越える。",
    "빠른 갈색 여우가 게으른 개를 뛰어넘는다.",
    "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है।",
    "สุนัขจิ้งจอกสีน้ำตาลกระโดดข้ามสุนัขขี้เกียจ",
]
TEXTS = CORPUS + [
    "Mixed: 中文 text, 😀 emoji, 𠀀 and 🧑‍💻, ཀུན་མཁྱེན། and ᱥᱟᱱᱛᱟᱲᱤ.",
    "前文的内容很长继续生成中文🙂𠜎",
    "é, ß and Ω stay on UTF-8; ‘quotes’ — and ॐ do not.",
]
REPLACEMENT = "�"


def plain(tokenizer):
    return Tokenizer.from_str(tokenizer.to_str())


def token_sequence(tokens, decoder):
    vocab = {token: i for i, token in enumerate(dict.fromkeys(tokens))}
    tokenizer = Tokenizer(models.BPE(vocab, []))
    tokenizer.decoder = decoder
    return tokenizer, [vocab[token] for token in tokens]


@pytest.fixture(scope="module")
def ube():
    tokenizer = UniversalByteLevelBPETokenizer()
    tokenizer.train_from_iterator(CORPUS * 20, vocab_size=700, show_progress=False)
    return plain(tokenizer)


@pytest.fixture(scope="module")
def byte_level():
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(CORPUS * 20, vocab_size=700, show_progress=False)
    return plain(tokenizer)


def symbol_width(char, ube=True):
    """Base symbols of one character: UTF-8 bytes up to two, else UTF-16LE bytes (UBE)."""
    utf8 = len(char.encode("utf-8"))
    if not ube or utf8 <= 2:
        return utf8
    return len(char.encode("utf-16-le"))


def character_starts(text, ube=True):
    starts, position = [], 0
    for char in text:
        starts.append(position)
        position += symbol_width(char, ube)
    return starts + [position]


def token_starts(tokenizer, ids):
    starts, position = [], 0
    for token_id in ids:
        starts.append(position)
        position += len(tokenizer.id_to_token(token_id))
    return starts + [position]


def window_stream(tokenizer, ids, prompt_length, prefix_offset):
    """An incremental detokenizer that decodes a window of tokens and emits the new text when it ends cleanly."""
    read_offset, pieces = prompt_length, []
    for end in range(prompt_length + 1, len(ids) + 1):
        prefix_text = tokenizer.decode(ids[prefix_offset:read_offset])
        new_text = tokenizer.decode(ids[prefix_offset:end])
        if len(new_text) > len(prefix_text) and not new_text.endswith(REPLACEMENT):
            pieces.append(new_text[len(prefix_text) :])
            prefix_offset, read_offset = read_offset, end
    return "".join(pieces)


def split_cut(output, text, starts, position):
    """Check that a decoded slice from symbol `position` is the replacement of the cut character followed by
    the following characters; return the number of replacement characters."""
    index = max(i for i, start in enumerate(starts[:-1]) if start <= position)
    if starts[index] == position:
        assert output == text[index:]
        return 0
    rest = text[index + 1 :]
    assert output.endswith(rest)
    head = output[: len(output) - len(rest)]
    assert head and set(head) == {REPLACEMENT}
    return len(head)


class TestUniversalByteLevelDecoding:
    def test_boundaries_are_the_character_starts(self, ube):
        decoding = UniversalByteLevelDecoding(ube)
        split = False
        for text in TEXTS:
            ids = ube.encode(text).ids
            assert ube.decode(ids) == text
            flags = decoding.boundaries(ids)
            chars = set(character_starts(text))
            assert flags == [start in chars for start in token_starts(ube, ids)]
            for i, flag in enumerate(flags):
                if flag:
                    assert ube.decode(ids[:i]) + ube.decode(ids[i:]) == text
            split |= not all(flags)
        assert split

    def test_aligned_offset_is_the_latest_boundary(self, ube):
        decoding = UniversalByteLevelDecoding(ube)
        for text in TEXTS:
            ids = ube.encode(text).ids
            flags = decoding.boundaries(ids)
            for offset in range(-2, len(ids) + 3):
                clamped = max(0, min(offset, len(ids)))
                aligned = decoding.aligned_offset(ids, offset)
                assert aligned <= clamped and flags[aligned]
                assert not any(flags[aligned + 1 : clamped + 1])

    def test_windows_from_aligned_offsets_decode_exactly(self, ube):
        decoding = UniversalByteLevelDecoding(ube)
        for text in TEXTS:
            ids = ube.encode(text).ids
            flags = decoding.boundaries(ids)
            for prompt_length in range(1, len(ids)):
                if not flags[prompt_length]:
                    continue
                expected = ube.decode(ids[prompt_length:])
                for back in (1, 5):
                    start = decoding.aligned_offset(ids[:prompt_length], prompt_length - back)
                    assert window_stream(ube, ids, prompt_length, start) == expected
                    prefix_text = ube.decode(ids[start:prompt_length])
                    assert ube.decode(ids[start:])[len(prefix_text) :] == expected

    def test_decode_slice_keeps_the_characters_after_the_cut(self, ube):
        decoding = UniversalByteLevelDecoding(ube)
        replaced = set()
        for text in TEXTS:
            ids = ube.encode(text).ids
            starts = character_starts(text)
            for j, position in enumerate(token_starts(ube, ids)[:-1]):
                replaced.add(split_cut(decoding.decode_slice(ids[j:]), text, starts, position))
        assert {0, 1} <= replaced <= {0, 1, 2}

    def test_decode_slice_of_every_symbol_position(self):
        tokenizer = UniversalByteLevelBPETokenizer()
        tokenizer.train_from_iterator(["x"], vocab_size=512, show_progress=False)
        tokenizer = plain(tokenizer)
        decoding = UniversalByteLevelDecoding(tokenizer)
        replaced = set()
        for text in TEXTS:
            ids = tokenizer.encode(text).ids
            starts = character_starts(text)
            assert len(ids) == starts[-1]
            for j in range(len(ids)):
                replaced.add(split_cut(decoding.decode_slice(ids[j:]), text, starts, j))
        # a slice that starts at the second byte of a high surrogate also cuts off the low surrogate
        assert replaced == {0, 1, 2}

    def test_decode_slice_with_an_open_end(self):
        tokenizer = UniversalByteLevelBPETokenizer()
        tokenizer.train_from_iterator(["x"], vocab_size=512, show_progress=False)
        tokenizer = plain(tokenizer)
        decoding = UniversalByteLevelDecoding(tokenizer)
        # Thai consonants U+0E01-U+0E07 have a low byte below 0x08, which no other reading allows.
        text = "กขฃคฅฆง" * 3
        ids = tokenizer.encode(text).ids
        for j in range(len(ids)):
            for m in range(j + 2, len(ids) + 1):
                output = decoding.decode_slice(ids[j:m], ends_at_boundary=False)
                first, last = (j + 1) // 2, m // 2
                expected = REPLACEMENT * (j % 2) + text[first:last] + REPLACEMENT * (m % 2)
                assert output == expected

    def test_token_texts(self, ube):
        decoding = UniversalByteLevelDecoding(ube)
        for text in TEXTS:
            ids = ube.encode(text).ids
            texts = decoding.token_texts(ids)
            flags = decoding.boundaries(ids)
            assert "".join(texts) == text
            assert len(texts) == len(ids)
            for i, piece in enumerate(texts):
                assert REPLACEMENT not in piece
                if flags[i + 1]:
                    assert "".join(texts[: i + 1]) == ube.decode(ids[: i + 1])
                else:
                    assert piece == ""

    def test_token_texts_match_decode_stream(self, ube):
        decoding = UniversalByteLevelDecoding(ube)
        for text in TEXTS:
            ids = ube.encode(text).ids
            stream = decoders.DecodeStream(skip_special_tokens=True)
            pieces = [stream.step(ube, token_id) or "" for token_id in ids]
            assert "".join(pieces) == "".join(decoding.token_texts(ids)) == text

    @pytest.mark.parametrize("skip_special_tokens", [True, False])
    def test_special_tokens(self, ube, skip_special_tokens):
        tokenizer = Tokenizer.from_str(ube.to_str())
        tokenizer.add_special_tokens([AddedToken("<sep>", special=True)])
        tokenizer.add_tokens([AddedToken("<plain>", special=False)])
        decoding = UniversalByteLevelDecoding(tokenizer, skip_special_tokens=skip_special_tokens)
        text = "中文<sep>😀<plain>ภาษา<sep>"
        ids = tokenizer.encode(text).ids
        full = tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        flags = decoding.boundaries(ids)
        for i, flag in enumerate(flags):
            if flag:
                head = tokenizer.decode(ids[:i], skip_special_tokens=skip_special_tokens)
                tail = tokenizer.decode(ids[i:], skip_special_tokens=skip_special_tokens)
                assert head + tail == full
                assert decoding.decode_slice(ids[i:]) == tail
        assert "".join(decoding.token_texts(ids)) == full

    def test_byte_level_tokenizers(self, byte_level):
        decoding = UniversalByteLevelDecoding(byte_level)
        replaced = set()
        for text in TEXTS:
            ids = byte_level.encode(text).ids
            starts = character_starts(text, ube=False)
            flags = decoding.boundaries(ids)
            assert flags == [start in set(starts) for start in token_starts(byte_level, ids)]
            for j, position in enumerate(token_starts(byte_level, ids)[:-1]):
                replaced.add(split_cut(decoding.decode_slice(ids[j:]), text, starts, position))
            assert "".join(decoding.token_texts(ids)) == text
        assert 0 in replaced and replaced <= {0, 1, 2, 3}

    @pytest.mark.parametrize("decoder", [decoders.UniversalByteLevel, decoders.ByteLevel])
    @pytest.mark.parametrize("skip_special_tokens", [True, False])
    def test_helpers_follow_the_decoder_on_any_ids(self, decoder, skip_special_tokens):
        tokens = sorted(pre_tokenizers.UniversalByteLevel.alphabet()) + ["", "☃Ã", "☃", "<sep>"]
        tokenizer, _ = token_sequence(tokens, decoder())
        tokenizer.add_special_tokens([AddedToken("<sep>", special=True)])
        decoding = UniversalByteLevelDecoding(tokenizer, skip_special_tokens=skip_special_tokens)
        rng = random.Random(0)
        choices = list(range(tokenizer.get_vocab_size())) + [999999]
        for _ in range(600):
            ids = [rng.choice(choices) for _ in range(rng.randrange(16))]
            full = tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
            for i, flag in enumerate(decoding.boundaries(ids)):
                if flag:
                    head = tokenizer.decode(ids[:i], skip_special_tokens=skip_special_tokens)
                    tail = tokenizer.decode(ids[i:], skip_special_tokens=skip_special_tokens)
                    assert head + tail == full
            assert "".join(decoding.token_texts(ids)) == full
            if decoder is decoders.ByteLevel:
                assert decoding.decode_slice(ids) == full

    @pytest.mark.parametrize("decoder", [decoders.UniversalByteLevel, decoders.ByteLevel])
    @pytest.mark.parametrize("skip_special_tokens", [True, False])
    def test_special_tokens_follow_native_content_selection(self, decoder, skip_special_tokens):
        tokenizer, _ = token_sequence(["x"], decoder())
        tokenizer.normalizer = normalizers.Lowercase()
        tokenizer.add_tokens([AddedToken("ABC", normalized=True)])
        tokenizer.add_special_tokens([AddedToken("abc", special=True)])
        tokenizer.add_special_tokens([AddedToken("DEF", special=True, normalized=True)])
        tokenizer.add_special_tokens([AddedToken("<sep>", special=True)])
        tokenizer.add_tokens([AddedToken("<sep>", special=False)])
        ids = [tokenizer.token_to_id(token) for token in ("ABC", "abc", "DEF", "<sep>", "x")]
        assert tokenizer.decode(ids, skip_special_tokens=True) == "defx"
        assert tokenizer.decode(ids, skip_special_tokens=False) == "abcabcdef<sep>x"
        decoding = UniversalByteLevelDecoding(tokenizer, skip_special_tokens=skip_special_tokens)
        expected = tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        assert decoding.decode_slice(ids) == expected
        assert "".join(decoding.token_texts(ids)) == expected

    def test_special_content_shared_with_model_ids(self):
        tokenizer = Tokenizer(models.Unigram([("x", 0.0), ("<sep>", 0.0), ("<sep>", 0.0)], unk_id=0))
        tokenizer.decoder = decoders.UniversalByteLevel()
        tokenizer.add_special_tokens(["<sep>"])
        decoding = UniversalByteLevelDecoding(tokenizer)
        assert tokenizer.id_to_token(1) == tokenizer.id_to_token(2) == "<sep>"
        assert decoding.decode_slice([1, 2, 0]) == tokenizer.decode([1, 2, 0]) == "x"
        assert decoding.token_texts([1, 2, 0]) == ["", "", "x"]

    @pytest.mark.parametrize("skip_special_tokens", [True, False])
    def test_slice_reconstruction_with_special_empty_and_unknown_tokens(self, skip_special_tokens):
        symbols = pre_tokenizers.UniversalByteLevel(add_prefix_space=False).pre_tokenize_str("中文")[0][0]
        tokens = ["", symbols[1], "<sep>", symbols[2:], "a"]
        tokenizer, ids = token_sequence(tokens, decoders.UniversalByteLevel())
        tokenizer.add_special_tokens([AddedToken("<sep>", special=True)])
        ids = [999999] + ids
        decoding = UniversalByteLevelDecoding(tokenizer, skip_special_tokens=skip_special_tokens)
        expected = REPLACEMENT + ("" if skip_special_tokens else "<sep>") + "文a"
        assert decoding.decode_slice(ids) == expected
        assert decoding.decode_slice(ids, ends_at_boundary=False) == expected

    def test_byte_level_uses_raw_utf8_for_whole_tokens_outside_its_alphabet(self):
        tokenizer, ids = token_sequence(["\ue100", "☃Ã", "©", "a"], decoders.ByteLevel())
        decoding = UniversalByteLevelDecoding(tokenizer)
        assert decoding.boundaries(ids) == [True] * (len(ids) + 1)
        assert decoding.decode_slice(ids) == tokenizer.decode(ids) == "\ue100☃Ã�a"
        assert decoding.token_texts(ids) == ["\ue100", "☃Ã", REPLACEMENT, "a"]

    @pytest.mark.parametrize("decoder", [None, decoders.WordPiece(), decoders.Sequence([decoders.ByteLevel()])])
    def test_decoder_type(self, decoder):
        tokenizer = Tokenizer(models.BPE())
        if decoder is not None:
            tokenizer.decoder = decoder
        with pytest.raises(ValueError, match="UniversalByteLevel or ByteLevel"):
            UniversalByteLevelDecoding(tokenizer)

    @pytest.mark.parametrize(
        "units, expected",
        [
            ([0xD800, 0xDC00], "𐀀"),
            ([0xD800, 0xD801, 0xDC00], "�𐐀"),
            ([0xD800, 0x4E2D], "�中"),
            ([0xDC00, 0xD800], "��"),
            ([0xDC00, 0xDC01], "��"),
        ],
    )
    def test_utf16_surrogates_and_odd_bytes(self, units, expected):
        symbols = [chr(0xE100 + byte) for unit in units for byte in unit.to_bytes(2, "little")]
        tokens = symbols + [chr(0xE101), "a"]
        tokenizer, ids = token_sequence(tokens, decoders.UniversalByteLevel())
        decoding = UniversalByteLevelDecoding(tokenizer)
        assert tokenizer.decode(ids) == expected + "�a"
        assert "".join(decoding.token_texts(ids)) == expected + "�a"
        for i, flag in enumerate(decoding.boundaries(ids)):
            if flag:
                assert tokenizer.decode(ids[:i]) + tokenizer.decode(ids[i:]) == tokenizer.decode(ids)

    @pytest.mark.parametrize("text", ["é", "€", "😀"])
    def test_multibyte_utf8_on_the_utf8_half(self, text):
        symbols = pre_tokenizers.ByteLevel(add_prefix_space=False).pre_tokenize_str(text)[0][0]
        tokenizer, ids = token_sequence(list(symbols), decoders.UniversalByteLevel())
        decoding = UniversalByteLevelDecoding(tokenizer)
        assert decoding.boundaries(ids) == [True] + [False] * (len(ids) - 1) + [True]
        assert decoding.token_texts(ids) == [""] * (len(ids) - 1) + [text]
        for i in range(len(ids)):
            assert decoding.decode_slice(ids[i:]) == tokenizer.decode(ids[i:])

    def test_pending_prefix_followed_by_a_flush(self):
        tokenizer, ids = token_sequence(["Ã", "a"], decoders.UniversalByteLevel())
        decoding = UniversalByteLevelDecoding(tokenizer)
        assert decoding.boundaries(ids) == [True, False, True]
        assert tokenizer.decode(ids[:1]) + tokenizer.decode(ids[1:]) == tokenizer.decode(ids)
        assert decoding.token_texts(ids) == ["", "�a"]

    def test_token_texts_assign_groups_and_final_text(self):
        tokenizer, ids = token_sequence(["aÃ", "©", "xÃ"], decoders.UniversalByteLevel())
        decoding = UniversalByteLevelDecoding(tokenizer)
        assert decoding.token_texts(ids) == ["", "aé", "x�"]
        assert decoding.token_texts(ids + [999999]) == ["", "aé", "", "x�"]

    @pytest.mark.parametrize(
        "run, second",
        [
            ([0x01, 0x0E, 0x02], False),
            ([0x0E, 0x02, 0x0E], True),
            ([0x2D, 0x4E, 0x87, 0x65], False),
            ([0x00, 0xDC, 0x00, 0xD8], False),
            ([0x00, 0xD8, 0x00, 0xDC], False),
            ([0x00, 0xD8, 0x08, 0xD8], True),
            ([0x00, 0xD8, 0x00, 0xD8], False),
            ([0x00, 0x00, 0x00], True),
            ([0x08, 0x08, 0x00, 0xDC], False),
        ],
    )
    def test_open_run_code_unit_selection(self, run, second):
        assert _second_byte_by_code_units(run) is second

    @pytest.mark.parametrize("universal", [True, False])
    def test_random_encoded_slices_and_token_texts(self, universal):
        pre = (
            pre_tokenizers.UniversalByteLevel(add_prefix_space=False, use_regex=False)
            if universal
            else pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        )
        decoder = decoders.UniversalByteLevel() if universal else decoders.ByteLevel()
        rng = random.Random(17)
        alphabet = "azéΩ中😀กขཀᱥ\ue100\ufffd" + "".join(
            chr(rng.choice((rng.randrange(0xD800), rng.randrange(0xE000, 0x110000)))) for _ in range(100)
        )
        for _ in range(100):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 20)))
            symbols = "".join(token for token, _ in pre.pre_tokenize_str(text))
            tokens = []
            while symbols:
                width = rng.randrange(1, 8)
                tokens.append(symbols[:width])
                symbols = symbols[width:]
            tokenizer, ids = token_sequence(tokens, decoder)
            decoding = UniversalByteLevelDecoding(tokenizer)
            assert tokenizer.decode(ids) == text
            assert "".join(decoding.token_texts(ids)) == text
            starts = character_starts(text, ube=universal)
            for i, position in enumerate(token_starts(tokenizer, ids)[:-1]):
                split_cut(decoding.decode_slice(ids[i:]), text, starts, position)

    def test_open_end_after_a_mixed_text_run(self):
        pre = pre_tokenizers.UniversalByteLevel(add_prefix_space=False, use_regex=False)
        text = "กข aé กข"
        symbols = pre.pre_tokenize_str(text)[0][0]
        tokenizer, ids = token_sequence(list(symbols), decoders.UniversalByteLevel())
        decoding = UniversalByteLevelDecoding(tokenizer)
        for start in (0, 1):
            for end in range(5, len(ids) + 1):
                expected = tokenizer.decode(ids[start + start % 2 : end])
                assert decoding.decode_slice(ids[start:end], ends_at_boundary=False) == REPLACEMENT * start + expected
