# Copyright 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

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
SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
ROUND_TRIP_TEXTS = CORPUS + ["Unseen: ʕ•ᴥ•ʔ 🧑‍💻 𠀀 \u0301", "Hi 빙고 🩷💚", "ሰላም e\u0301"]


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.pre_tokenizer = pre_tokenizers.UniversalByteLevel(add_prefix_space=False, trim_offsets=False)
    tokenizer.post_processor = processors.UniversalByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.UniversalByteLevel()
    trainer = trainers.UnigramTrainer(
        vocab_size=700,
        special_tokens=SPECIAL_TOKENS,
        unk_token="<unk>",
        initial_alphabet=pre_tokenizers.UniversalByteLevel.alphabet(),
        show_progress=False,
    )
    corpus = tmp_path_factory.mktemp("unigram") / "corpus.txt"
    corpus.write_text("\n".join(CORPUS) + "\n", encoding="utf-8")
    tokenizer.train([str(corpus)], trainer)
    return tokenizer


class TestUniversalByteLevelUnigram:
    def test_every_base_symbol_is_in_the_vocabulary(self, tokenizer):
        alphabet = set(pre_tokenizers.UniversalByteLevel.alphabet())
        assert len(alphabet) == 512
        assert alphabet.issubset(tokenizer.get_vocab(with_added_tokens=False))

    def test_train_encodes_and_decodes(self, tokenizer):
        unk = tokenizer.token_to_id("<unk>")
        assert unk is not None
        for text in ROUND_TRIP_TEXTS:
            ids = tokenizer.encode(text).ids
            assert unk not in ids
            assert tokenizer.decode(ids) == text

    def test_save_and_reload(self, tokenizer, tmp_path):
        path = str(tmp_path / "tokenizer.json")
        tokenizer.save(path)
        reloaded = Tokenizer.from_file(path)
        for text in ROUND_TRIP_TEXTS:
            ids = tokenizer.encode(text).ids
            assert reloaded.encode(text).ids == ids
            assert reloaded.decode(ids) == text
