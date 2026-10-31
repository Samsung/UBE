# Copyright 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

from tokenizers import Tokenizer, UniversalByteLevelBPETokenizer

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


def trained(**kwargs):
    tokenizer = UniversalByteLevelBPETokenizer(**kwargs)
    tokenizer.train_from_iterator(CORPUS * 20, vocab_size=700, show_progress=False)
    return tokenizer


class TestUniversalByteLevelBPE:
    def test_train_encodes_and_decodes(self):
        tokenizer = trained()
        # all 512 base symbols are kept, so every character is covered
        assert tokenizer.get_vocab_size() == 700
        for text in CORPUS + ["Unseen: ʕ•ᴥ•ʔ 🧑‍💻 𠀀 ́"]:
            output = tokenizer.encode(text)
            assert tokenizer.decode(output.ids) == text

    def test_untrained_symbols_follow_the_routing(self):
        tokenizer = UniversalByteLevelBPETokenizer()
        tokenizer.train_from_iterator(["x"], vocab_size=512, show_progress=False)
        # no merges: one symbol per UTF-16LE byte of a three-byte character
        assert len(tokenizer.encode("中").ids) == 2
        assert len(tokenizer.encode("é").ids) == 2
        assert len(tokenizer.encode("😀").ids) == 4

    def test_offsets_follow_the_original_characters(self):
        tokenizer = UniversalByteLevelBPETokenizer()
        tokenizer.train_from_iterator(["x"], vocab_size=512, show_progress=False)
        output = tokenizer.encode("A한B")
        assert output.offsets == [(0, 1), (1, 2), (1, 2), (2, 3)]

    @pytest.mark.parametrize("add_prefix_space", [False, True])
    def test_trim_offsets_uses_the_configured_prefix_space(self, add_prefix_space):
        tokenizer = UniversalByteLevelBPETokenizer(add_prefix_space=add_prefix_space, trim_offsets=True)
        tokenizer.train_from_iterator([" Hi 빙고 🩷💚"] * 10, vocab_size=550, show_progress=False)
        output = tokenizer.encode(" Hi 빙고 🩷💚")
        assert output.offsets == [(0 if add_prefix_space else 1, 3), (4, 6), (7, 9)]
        assert tokenizer.encode("Hi 빙고 🩷💚").offsets[0][0] == 0

    @pytest.mark.parametrize("from_file", [False, True])
    def test_dropout_applies_after_training(self, from_file, tmp_path):
        tokenizer = UniversalByteLevelBPETokenizer(dropout=1.0)
        if from_file:
            path = tmp_path / "corpus.txt"
            path.write_text("Hi 빙고 🩷💚\n" * 10, encoding="utf-8")
            tokenizer.train(str(path), vocab_size=550, show_progress=False)
        else:
            tokenizer.train_from_iterator(["Hi 빙고 🩷💚"] * 10, vocab_size=550, show_progress=False)
        output = tokenizer.encode("Hi 빙고 🩷💚")
        assert output.tokens == list("HiĠ\ue159\ue1be\ue1e0\ue1acĠ\ue13e\ue1d8\ue177\ue1de\ue13d\ue1d8\ue19a\ue1dc")
        assert tokenizer.decode(output.ids) == "Hi 빙고 🩷💚"
        reloaded = Tokenizer.from_str(tokenizer.to_str())
        assert reloaded.encode("Hi 빙고 🩷💚").tokens == output.tokens

    def test_save_and_reload(self, tmp_path):
        tokenizer = trained()
        path = str(tmp_path / "tokenizer.json")
        tokenizer.save(path)
        reloaded = Tokenizer.from_file(path)
        for text in CORPUS:
            assert reloaded.encode(text).ids == tokenizer.encode(text).ids
            assert reloaded.decode(reloaded.encode(text).ids) == text
