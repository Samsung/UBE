# Copyright 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

"""Tests for the sampling, training and comparison scripts, on small synthetic data."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import random
import shutil
import sys
import unittest
import unittest.mock
from pathlib import Path
from tempfile import TemporaryDirectory

from tokenizers import Tokenizer

EXPERIMENT = Path(__file__).resolve().parents[1]
REPOSITORY = EXPERIMENT.parents[1]
sys.path.insert(0, str(EXPERIMENT / "scripts"))
sys.path.insert(0, str(REPOSITORY / "paper" / "scripts"))

import compare  # noqa: E402
import measure_flores  # noqa: E402
import sample_corpus  # noqa: E402
import train_tokenizer  # noqa: E402

SYLLABLES = {
    "am": ["ሰ", "ላ", "ም", "አ", "ማ", "ር", "ኛ"],
    "en": ["ka", "to", "ri", "men", "the", "and", "ing", "str"],
    "hi": ["क", "ना", "मि", "ते", "स्", "हि", "न्द", "ी"],
    "ko": ["가", "나", "다", "빙", "고", "한", "국", "어"],
    "ru": ["ка", "то", "ри", "мен", "про", "ост"],
    "zh": ["的", "一", "是", "不", "了", "人", "我", "在"],
}
SHARES = {"am": 1, "en": 4, "hi": 1, "ko": 2, "ru": 1, "zh": 1}
ROUND_TRIP_TEXTS = ["Hi 빙고 🩷💚", "Unseen: ʕ•ᴥ•ʔ 🧑‍💻 𠀀 e\u0301", "ሰላም, мир! 你好\n\tनमस्ते"]


def synthetic_documents(lang: str, count: int) -> list[str]:
    rng = random.Random(f"{lang}:documents")
    documents = []
    for _ in range(count):
        words = [
            "".join(rng.choice(SYLLABLES[lang]) for _ in range(rng.randint(1, 4)))
            for _ in range(rng.randint(5, 30))
        ]
        text = " ".join(words)
        if rng.random() < 0.2:
            text += "\n" + text[: len(text) // 2]
        if rng.random() < 0.05:
            text += " 🩷💚"
        documents.append(text)
    return documents


def write_inputs(root: Path, count: int = 1000) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Two input files per language, the second gzip-compressed, with empty and null documents.

    Returns each language's documents and the text bytes of its first file.
    """
    documents, first_file_bytes = {}, {}
    for lang in SYLLABLES:
        texts = synthetic_documents(lang, count)
        documents[lang] = texts
        records = [{"text": text, "url": f"https://example.com/{i}"} for i, text in enumerate(texts)]
        records.insert(3, {"text": ""})
        records.insert(7, {"text": None})
        half = len(records) // 2
        first_file_bytes[lang] = sum(len((record["text"] or "").encode("utf-8")) for record in records[:half])
        directory = root / lang
        directory.mkdir(parents=True)
        with (directory / "part-00000.jsonl").open("w", encoding="utf-8") as handle:
            for record in records[:half]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.write("\n")
        with gzip.open(directory / "part-00001.jsonl.gz", "wt", encoding="utf-8") as handle:
            for record in records[half:]:
                handle.write(json.dumps(record) + "\n")
    return documents, first_file_bytes


def write_shares(path: Path, shares: dict[str, int]) -> Path:
    path.write_text(json.dumps({"total_bytes": sum(shares.values()), "languages": shares}), encoding="utf-8")
    return path


def run_main(module, argv: list[str]) -> int:
    with unittest.mock.patch.object(sys, "argv", [f"{module.__name__}.py", *argv]):
        with unittest.mock.patch("sys.stdout", io.StringIO()):
            return module.main()


def expected_prefix(documents: list[str], quota: int) -> list[str]:
    """The documents of a quota, computed independently: whole documents, then a character-wise cut."""
    taken, used = [], 0
    for text in documents:
        size = len(text.encode("utf-8"))
        if used + size <= quota:
            taken.append(text)
            used += size
            if used == quota:
                break
            continue
        piece = ""
        for ch in text:
            if used + len((piece + ch).encode("utf-8")) > quota:
                break
            piece += ch
        if piece:
            taken.append(piece)
        break
    return taken


def read_sample(directory: Path) -> tuple[dict, dict[str, list[str]]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    texts = {}
    for lang, info in manifest["languages"].items():
        lines = (directory / info["file"]).read_bytes().split(b"\n")
        assert lines.pop() == b""
        texts[lang] = [json.loads(line)["text"] for line in lines]
    return manifest, texts


class SampleCorpusTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.documents, self.first_file_bytes = write_inputs(self.root / "input")
        self.shares = write_shares(self.root / "shares.json", SHARES)
        self.pattern = str(self.root / "input" / "{lang}" / "*.jsonl*")

    def sample(self, total_bytes: int, name: str) -> Path:
        output = self.root / name
        argv = ["--input", self.pattern, "--total-bytes", str(total_bytes), "--output-dir", str(output)]
        self.assertEqual(run_main(sample_corpus, [*argv, "--shares", str(self.shares)]), 0)
        return output

    def test_quotas_round_the_byte_shares(self) -> None:
        self.assertEqual(sample_corpus.quotas({"a": 1, "b": 2}, 10), {"a": 3, "b": 7})
        self.assertEqual(sample_corpus.quotas({"a": 1, "b": 1}, 5), {"a": 2, "b": 2})

    def test_shipped_shares_give_the_quotas_of_the_readme_samples(self) -> None:
        shares = sample_corpus.load_shares(sample_corpus.DEFAULT_SHARES)
        self.assertEqual(len(shares), 101)
        self.assertEqual(sum(shares.values()), 371_845_316_578)
        one = sample_corpus.quotas(shares, 1 << 30)
        four = sample_corpus.quotas(shares, 4 << 30)
        self.assertEqual(
            (one["en"], one["ko"], one["haw"], sum(one.values())), (313_915_894, 93_541_326, 335_254, 1_073_741_821)
        )
        self.assertEqual((four["en"], four["ko"], sum(four.values())), (1_255_663_575, 374_165_303, 4_294_967_294))
        self.assertAlmostEqual(100 * shares["en"] / sum(shares.values()), 29.2357, places=4)

    def test_sample_takes_the_first_bytes_of_each_language(self) -> None:
        total = 100_000
        manifest, texts = read_sample(self.sample(total, "sample"))
        targets = sample_corpus.quotas(SHARES, total)
        self.assertEqual(manifest["requested_bytes"], total)
        self.assertEqual(manifest["language_count"], len(SHARES))
        for lang, info in manifest["languages"].items():
            with self.subTest(lang=lang):
                self.assertEqual(texts[lang], expected_prefix(self.documents[lang], targets[lang]))
                self.assertEqual(info["quota_bytes"], targets[lang])
                self.assertEqual(info["text_bytes"], sum(len(text.encode("utf-8")) for text in texts[lang]))
                self.assertLessEqual(targets[lang] - info["text_bytes"], 3)
                self.assertEqual(info["documents"], len(texts[lang]))
                self.assertNotIn("", texts[lang])
        self.assertEqual(manifest["text_bytes"], sum(info["text_bytes"] for info in manifest["languages"].values()))

    def test_cut_falls_on_a_character_boundary(self) -> None:
        # Korean text has 3-byte characters, so most quotas fall inside a character.
        boundaries = set()
        for total in range(30_000, 30_045, 5):
            manifest, texts = read_sample(self.sample(total, f"sample_{total}"))
            info = manifest["languages"]["ko"]
            self.assertEqual(texts["ko"], expected_prefix(self.documents["ko"], info["quota_bytes"]))
            boundaries.add(info["quota_bytes"] - info["text_bytes"])
        self.assertTrue(boundaries & {1, 2})

    def test_sample_is_deterministic_and_nested(self) -> None:
        first = self.sample(250_000, "first")
        second = self.sample(250_000, "second")
        for name in sorted(path.name for path in first.iterdir()):
            with self.subTest(file=name):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
        _, larger = read_sample(first)
        _, smaller = read_sample(self.sample(30_000, "smaller"))
        for lang in SHARES:
            with self.subTest(lang=lang):
                small, large = smaller[lang], larger[lang]
                self.assertEqual(small[:-1], large[: len(small) - 1])
                self.assertTrue(large[len(small) - 1].startswith(small[-1]))

    def test_reads_files_in_name_order_and_records_them(self) -> None:
        manifest, texts = read_sample(self.sample(250_000, "sample"))
        counts = set()
        for lang, info in manifest["languages"].items():
            with self.subTest(lang=lang):
                self.assertEqual(texts[lang], expected_prefix(self.documents[lang], info["quota_bytes"]))
                files = ["part-00000.jsonl", "part-00001.jsonl.gz"]
                if info["quota_bytes"] <= self.first_file_bytes[lang]:
                    files = files[:1]
                self.assertEqual([Path(item["file"]).name for item in info["inputs"]], files)
                self.assertEqual(sum(item["documents"] for item in info["inputs"]), info["documents"])
                counts.add(len(files))
        self.assertEqual(counts, {1, 2})

    def test_rejects_input_shorter_than_the_quota(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            self.sample(10**8, "sample")
        self.assertIn("the input ends after", str(raised.exception.code))
        self.assertEqual(list((self.root / "sample").iterdir()), [])
        self.sample(10_000, "sample")

    def test_truncated_gzip_names_the_shard(self) -> None:
        write_shares(self.shares, {"en": 1})
        self.pattern = str(self.root / "{lang}.jsonl.gz")
        path = self.root / "en.jsonl.gz"
        path.write_bytes(gzip.compress(b'{"text":"a"}\n')[:-8])
        with self.assertRaises(SystemExit) as raised:
            self.sample(10, "sample")
        self.assertIn(str(path), str(raised.exception))
        self.assertEqual(list((self.root / "sample").iterdir()), [])

    def test_rejects_a_non_empty_output_directory(self) -> None:
        output = self.root / "sample"
        output.mkdir()
        (output / "en.jsonl").write_text("", encoding="utf-8")
        with unittest.mock.patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                self.sample(60_000, "sample")
        self.assertEqual(raised.exception.code, 2)

    def test_rejects_missing_inputs_and_bad_lines(self) -> None:
        write_shares(self.shares, {**SHARES, "xx": 1})
        with self.assertRaises(SystemExit) as raised:
            self.sample(60_000, "missing")
        self.assertIn("no input files for xx", str(raised.exception.code))
        (self.root / "input" / "xx").mkdir()
        (self.root / "input" / "xx" / "part-00000.jsonl").write_text('{"body": "no text"}\n', encoding="utf-8")
        with self.assertRaises(SystemExit) as raised:
            self.sample(60_000, "bad")
        self.assertIn("no text field", str(raised.exception.code))
        self.assertEqual(list((self.root / "bad").iterdir()), [])

    def test_shares_require_language_codes(self) -> None:
        write_shares(self.shares, {"en/train": 1})
        with self.assertRaisesRegex(SystemExit, "invalid language code"):
            self.sample(10, "bad")
        self.assertFalse((self.root / "bad").exists())

    def test_manifest_hashes_the_shares_used_for_the_quotas(self) -> None:
        original = self.shares.read_bytes()
        input_files = sample_corpus.input_files

        def resolve_files(pattern, lang):
            if lang == "en":
                self.shares.write_bytes(original + b"\n")
            return input_files(pattern, lang)

        with unittest.mock.patch.object(sample_corpus, "input_files", resolve_files):
            manifest, _ = read_sample(self.sample(10_000, "sample"))
        self.assertEqual(manifest["shares_sha256"], hashlib.sha256(original).hexdigest())

    def test_cut_before_the_first_character_does_not_add_a_document(self) -> None:
        path = self.root / "one.jsonl"
        path.write_text('{"text":"🩷"}\n{"text":"a"}\n', encoding="utf-8")
        for quota in (0, 1, 3):
            with self.subTest(quota=quota):
                sink = io.BytesIO()
                info = sample_corpus.take_prefix("en", [path], quota, sink)
                self.assertEqual((info["text_bytes"], info["documents"], info["last_document_cut"]), (0, 0, False))
                self.assertEqual(info["inputs"], [{"file": str(path), "documents": 0}])
                self.assertEqual(sink.getvalue(), b"")


class TrainTokenizerTest(unittest.TestCase):
    VOCAB_SIZE = 800

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        write_inputs(cls.root / "input")
        shares = write_shares(cls.root / "shares.json", SHARES)
        cls.corpus = cls.root / "corpus"
        argv = ["--input", str(cls.root / "input" / "{lang}" / "*"), "--total-bytes", "300000"]
        run_main(sample_corpus, [*argv, "--output-dir", str(cls.corpus), "--shares", str(shares)])
        cls.paths = {}
        for kind in train_tokenizer.KINDS:
            path = cls.root / "tokenizers" / kind / "tokenizer.json"
            argv = ["--corpus", str(cls.corpus), "--kind", kind, "--vocab-size", str(cls.VOCAB_SIZE)]
            run_main(train_tokenizer, [*argv, "--output", str(path)])
            cls.paths[kind] = path

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def test_pipelines_match_the_readme_settings(self) -> None:
        pre_tokenizers = {
            "ube": {"type": "UniversalByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": True},
            "bytelevel": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        }
        post_processors = {
            "ube": {"type": "UniversalByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": True},
            "bytelevel": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": True},
        }
        decoders = {
            "ube": {"type": "UniversalByteLevel"},
            "bytelevel": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        }
        for kind, path in self.paths.items():
            with self.subTest(kind=kind):
                base, model = train_tokenizer.kind_parts(kind)
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.assertIsNone(saved["normalizer"])
                self.assertEqual(saved["pre_tokenizer"], pre_tokenizers[base])
                self.assertEqual(saved["post_processor"], post_processors[base])
                self.assertEqual(saved["decoder"], decoders[base])
                self.assertEqual(saved["model"]["type"], {"unigram": "Unigram", "bpe": "BPE"}[model])

    def test_vocabulary_holds_the_special_tokens_and_every_base_symbol(self) -> None:
        for kind, path in self.paths.items():
            with self.subTest(kind=kind):
                tokenizer = Tokenizer.from_file(str(path))
                self.assertEqual(tokenizer.get_vocab_size(with_added_tokens=True), self.VOCAB_SIZE)
                self.assertEqual(train_tokenizer.check(tokenizer, kind, self.VOCAB_SIZE), [])
                special = [tokenizer.id_to_token(i) for i in range(5)]
                self.assertEqual(special, ["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
                base, _ = train_tokenizer.kind_parts(kind)
                alphabet = train_tokenizer.base_alphabet(base)
                self.assertEqual(len(alphabet), {"ube": 512, "bytelevel": 256}[base])
                self.assertTrue(set(alphabet) <= set(tokenizer.get_vocab(with_added_tokens=False)))

    def test_unigram_pieces_and_unknown_token(self) -> None:
        for kind in ("ube_unigram", "bytelevel_unigram"):
            with self.subTest(kind=kind):
                model = json.loads(self.paths[kind].read_text(encoding="utf-8"))["model"]
                self.assertEqual(model["unk_id"], 3)
                self.assertLessEqual(max(len(piece) for piece, _ in model["vocab"][5:]), 16)

    def test_texts_round_trip(self) -> None:
        for kind, path in self.paths.items():
            with self.subTest(kind=kind):
                tokenizer = Tokenizer.from_file(str(path))
                for text in ROUND_TRIP_TEXTS:
                    ids = tokenizer.encode(text).ids
                    self.assertNotIn(3, ids)
                    self.assertEqual(tokenizer.decode(ids), text)

    def test_metadata_records_the_settings(self) -> None:
        manifest_sha256 = train_tokenizer.file_sha256(self.corpus / "manifest.json")
        for kind, path in self.paths.items():
            with self.subTest(kind=kind):
                meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
                self.assertEqual(meta["kind"], kind)
                self.assertEqual(meta["trainer"], train_tokenizer.trainer_arguments(kind, self.VOCAB_SIZE))
                self.assertEqual(meta["rayon_num_threads"], os.environ.get("RAYON_NUM_THREADS"))
                self.assertEqual(meta["tokenizers_parallelism"], os.environ.get("TOKENIZERS_PARALLELISM"))
                self.assertEqual(meta["corpus_manifest_sha256"], manifest_sha256)
                self.assertEqual(meta["tokenizer_sha256"], train_tokenizer.file_sha256(path))
        self.assertEqual(train_tokenizer.trainer_arguments("ube_bpe", 100)["min_frequency"], 2)
        unigram = train_tokenizer.trainer_arguments("bytelevel_unigram", 100)
        self.assertEqual((unigram["unk_token"], unigram["max_piece_length"]), ("<unk>", 16))

    def test_corpus_stream_keeps_language_and_document_order(self) -> None:
        manifest, _ = train_tokenizer.load_manifest(self.corpus)
        streamed = [text for batch in train_tokenizer.corpus_batches(self.corpus, manifest, batch=7) for text in batch]
        _, texts = read_sample(self.corpus)
        self.assertEqual(streamed, [text for lang in sorted(texts) for text in texts[lang]])

    def test_refuses_to_overwrite_and_unreachable_sizes(self) -> None:
        argv = ["--corpus", str(self.corpus), "--kind", "ube_bpe", "--vocab-size", str(self.VOCAB_SIZE)]
        with unittest.mock.patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                run_main(train_tokenizer, [*argv, "--output", str(self.paths["ube_bpe"])])
        self.assertEqual(raised.exception.code, 2)
        output = self.root / "too_large" / "tokenizer.json"
        argv = ["--corpus", str(self.corpus), "--kind", "ube_bpe", "--vocab-size", "1000000"]
        with self.assertRaises(SystemExit) as raised:
            run_main(train_tokenizer, [*argv, "--output", str(output)])
        self.assertIn("vocabulary size", str(raised.exception.code))
        self.assertFalse(output.exists())

    def test_rejects_a_changed_corpus_file(self) -> None:
        with TemporaryDirectory() as directory:
            corpus = Path(directory)
            for path in self.corpus.iterdir():
                (corpus / path.name).write_bytes(path.read_bytes())
            with (corpus / "ko.jsonl").open("ab") as handle:
                handle.write(b'{"text": "extra"}\n')
            with self.assertRaises(SystemExit) as raised:
                train_tokenizer.load_manifest(corpus)
        self.assertIn("SHA-256 differs", str(raised.exception.code))

    def test_manifest_totals_quotas_and_language_files_are_checked(self) -> None:
        with TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus"
            shutil.copytree(self.corpus, corpus)
            path = corpus / "manifest.json"
            original = path.read_bytes()
            for key in ("language_count", "text_bytes", "documents", "requested_bytes"):
                with self.subTest(key=key):
                    manifest = json.loads(original)
                    manifest[key] *= 2
                    path.write_text(json.dumps(manifest), encoding="utf-8")
                    with self.assertRaisesRegex(SystemExit, key):
                        train_tokenizer.load_manifest(corpus)
            manifest = json.loads(original)
            manifest["languages"]["ko"] = manifest["languages"]["en"]
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "filename"):
                train_tokenizer.load_manifest(corpus)

    def test_training_checks_the_counts_in_the_document_stream(self) -> None:
        with TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus"
            shutil.copytree(self.corpus, corpus)
            path = corpus / "manifest.json"
            manifest = json.loads(path.read_bytes())
            manifest["languages"]["zh"]["documents"] += 1
            manifest["documents"] += 1
            path.write_text(json.dumps(manifest), encoding="utf-8")
            output = Path(directory) / "tokenizer.json"
            argv = ["--corpus", str(corpus), "--kind", "ube_bpe", "--vocab-size", "800", "--output", str(output)]
            with self.assertRaisesRegex(SystemExit, "zh.jsonl: documents or text bytes"):
                run_main(train_tokenizer, argv)
            self.assertFalse(output.exists())

    def test_corpus_batches_yields_before_reading_the_next_batch(self) -> None:
        with TemporaryDirectory() as directory:
            corpus = Path(directory)
            path = corpus / "en.jsonl"
            path.write_bytes(b'{"text":"first"}\n{}\n')
            manifest = {"languages": {"en": {"file": path.name}}}
            stream = train_tokenizer.corpus_batches(corpus, manifest, batch=1)
            self.assertEqual(next(stream), ["first"])
            with self.assertRaisesRegex(SystemExit, "en.jsonl:2:"):
                next(stream)

    def test_empty_corpus_and_too_small_vocabularies_stop_before_training(self) -> None:
        with TemporaryDirectory() as directory:
            corpus = Path(directory)
            info = {"file": "en.jsonl", "documents": 0, "text_bytes": 0, "share": 1, "quota_bytes": 1}
            manifest = {
                "requested_bytes": 1,
                "language_count": 1,
                "documents": 0,
                "text_bytes": 0,
                "languages": {"en": info},
            }
            (corpus / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "no training documents"):
                train_tokenizer.load_manifest(corpus)
            for kind in train_tokenizer.KINDS:
                minimum = 517 if kind.startswith("ube") else 261
                argv = ["--corpus", str(corpus), "--kind", kind, "--vocab-size", str(minimum - 1)]
                argv += ["--output", str(corpus / "tokenizer.json")]
                with self.subTest(kind=kind), unittest.mock.patch("sys.stderr", io.StringIO()) as stderr:
                    with self.assertRaises(SystemExit):
                        run_main(train_tokenizer, argv)
                    self.assertIn(f"at least {minimum}", stderr.getvalue())

    def test_metadata_collision_keeps_the_existing_file_and_removes_the_incomplete_pair(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            sidecar = path.with_suffix(".meta.json")

            def create_sidecar(*args):
                sidecar.write_bytes(b"existing metadata")
                return []

            argv = ["--corpus", str(self.corpus), "--kind", "ube_bpe", "--vocab-size", "800", "--output", str(path)]
            with unittest.mock.patch.object(train_tokenizer, "check", create_sidecar):
                with self.assertRaises(FileExistsError):
                    run_main(train_tokenizer, argv)
            self.assertFalse(path.exists())
            self.assertEqual(sidecar.read_bytes(), b"existing metadata")

    def test_measurements_give_a_comparison_row(self) -> None:
        flores = {
            "sentence_eng_Latn": ["The quick brown fox.", "Hi there"],
            "sentence_kor_Hang": ["빠른 갈색 여우", "빙고 한국어"],
            "sentence_zho_Hans": ["敏捷的棕色狐狸", "你好"],
            "sentence_rus_Cyrl": ["Быстрая лиса", "Привет"],
        }
        flores_path = self.root / "flores.json"
        flores_path.write_text(json.dumps(flores, ensure_ascii=False), encoding="utf-8")
        results = self.root / "results"
        tokenizers = [str(path) for path in self.paths.values()]
        run_main(measure_flores, [*tokenizers, "--flores", str(flores_path), "--output-dir", str(results)])
        output = self.root / "comparison.json"
        argv = ["--setting", "tiny", str(results), "--flores", str(flores_path), "--output", str(output)]
        with unittest.mock.patch.object(sys, "argv", ["compare.py", *argv]):
            with unittest.mock.patch("sys.stdout", io.StringIO()) as stdout:
                self.assertEqual(compare.main(), 0)
        table = stdout.getvalue().splitlines()
        self.assertEqual(len(table), 3)
        self.assertTrue(table[0].startswith("| UBE vs UTF-8 (FLORES-200 devtest) | Unigram: 2 languages |"))
        self.assertTrue(table[2].startswith("| tiny "))
        comparison = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(comparison["three_byte_group"]["languages"], ["kor_Hang", "zho_Hans"])
        for model in ("unigram", "bpe"):
            for quantity in ("group_mean_tokens", "english_mean_tokens", "gini", "total_tokens"):
                values = comparison["settings"][0][model][quantity]
                if values["bytelevel"] == 0:
                    self.assertIsNone(values["change_pct"])
                else:
                    self.assertAlmostEqual(values["change_pct"], 100 * (values["ube"] / values["bytelevel"] - 1))


class CompareTest(unittest.TestCase):
    FLORES = {
        "eng_Latn": ["Hello world", "Fine."],
        "kor_Hang": ["안녕 하세요", "가a"],
        "zho_Hans": ["你好", "中文"],
        "rus_Cyrl": ["Привет", "мир"],
        "deu_Latn": ["Grüße", "gut 가"],
    }
    # Mean tokens per sentence of each language under each tokenizer.
    AVERAGES = {
        "ube_unigram": {"eng_Latn": 4.0, "kor_Hang": 6.0, "zho_Hans": 3.0, "rus_Cyrl": 5.0, "deu_Latn": 4.5},
        "bytelevel_unigram": {"eng_Latn": 5.0, "kor_Hang": 9.0, "zho_Hans": 4.0, "rus_Cyrl": 6.0, "deu_Latn": 5.0},
        "ube_bpe": {"eng_Latn": 3.0, "kor_Hang": 5.0, "zho_Hans": 2.5, "rus_Cyrl": 4.0, "deu_Latn": 3.5},
        "bytelevel_bpe": {"eng_Latn": 3.0, "kor_Hang": 5.5, "zho_Hans": 3.0, "rus_Cyrl": 4.0, "deu_Latn": 3.5},
    }

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.flores_path = self.root / "flores.json"
        self.flores_path.write_text(
            json.dumps({f"sentence_{lang}": texts for lang, texts in self.FLORES.items()}, ensure_ascii=False),
            encoding="utf-8",
        )
        self.flores_sha256 = hashlib.sha256(self.flores_path.read_bytes()).hexdigest()
        self.results = self.root / "results"
        self.results.mkdir()
        for kind, averages in self.AVERAGES.items():
            self.write_report(self.results, kind, averages)

    def write_report(self, directory: Path, kind: str, averages: dict, vocab_size: int = 100, **extra) -> None:
        results = {lang: {"total": int(2 * avg), "avg": avg, "n": 2} for lang, avg in averages.items()}
        report = {
            "tokenizer": kind,
            "tokenizer_sha256": hashlib.sha256(kind.encode()).hexdigest(),
            "vocab_size": vocab_size,
            "summary": measure_flores.summarize(results),
            "results": results,
            "flores_sha256": self.flores_sha256,
            **extra,
        }
        (directory / f"{kind}.json").write_text(json.dumps(report), encoding="utf-8")

    def run_compare(self, *settings: tuple[str, Path]) -> tuple[str, dict]:
        output = self.root / "comparison.json"
        argv = ["compare.py", "--flores", str(self.flores_path), "--output", str(output)]
        for label, directory in settings:
            argv += ["--setting", label, str(directory)]
        with unittest.mock.patch.object(sys, "argv", argv):
            with unittest.mock.patch("sys.stdout", io.StringIO()) as stdout:
                self.assertEqual(compare.main(), 0)
        return stdout.getvalue(), json.loads(output.read_text(encoding="utf-8"))

    def test_three_byte_group_counts_characters(self) -> None:
        # Korean is 5 of 6 characters at 3 bytes in one sentence and 1 of 2 in the other: 6 of 8.
        # "gut 가" puts German at 1 of 10 characters, Russian is all 2-byte.
        self.assertEqual(compare.three_byte_group(self.FLORES), ["kor_Hang", "zho_Hans"])
        self.assertEqual(compare.three_byte_group({"a": ["가a"], "b": ["가ab"], "c": ["🩷"]}), ["a", "c"])

    def test_changes_follow_the_definitions(self) -> None:
        _, comparison = self.run_compare(("setting", self.results))
        setting = comparison["settings"][0]
        self.assertEqual(comparison["three_byte_group"]["languages"], ["kor_Hang", "zho_Hans"])
        unigram = setting["unigram"]
        self.assertEqual(unigram["group_mean_tokens"]["ube"], 4.5)
        self.assertEqual(unigram["group_mean_tokens"]["bytelevel"], 6.5)
        self.assertAlmostEqual(unigram["group_mean_tokens"]["change_pct"], 100 * (4.5 / 6.5 - 1))
        self.assertAlmostEqual(unigram["english_mean_tokens"]["change_pct"], -20.0)
        gini_ube = measure_flores.gini([6 / 4, 3 / 4, 5 / 4, 4.5 / 4])
        gini_bytelevel = measure_flores.gini([9 / 5, 4 / 5, 6 / 5, 5 / 5])
        self.assertAlmostEqual(unigram["gini"]["ube"], gini_ube)
        self.assertAlmostEqual(unigram["gini"]["change_pct"], 100 * (gini_ube / gini_bytelevel - 1))
        self.assertEqual(unigram["total_tokens"]["ube"], 2 * (4 + 6 + 3 + 5 + 4.5))
        bpe = setting["bpe"]
        self.assertAlmostEqual(bpe["group_mean_tokens"]["change_pct"], 100 * (3.75 / 4.25 - 1))
        self.assertEqual(bpe["english_mean_tokens"]["change_pct"], 0.0)
        self.assertEqual(setting["vocab_size"], 100)

    def test_table_has_the_readme_layout(self) -> None:
        other = self.root / "other"
        other.mkdir()
        for kind, averages in self.AVERAGES.items():
            self.write_report(other, kind, {**averages, "kor_Hang": averages["kor_Hang"] + 1})
        table, _ = self.run_compare(("1 GiB text, 16K", self.results), ("4 GiB text, 32K", other))
        lines = table.splitlines()
        self.assertEqual(
            lines[:2],
            [
                "| UBE vs UTF-8 (FLORES-200 devtest) | Unigram: 2 languages | BPE: 2 languages "
                "| Unigram: English | Unigram: Gini |",
                "| :-------------------------------- | -------------------: | ---------------: "
                "| ---------------: | ------------: |",
            ],
        )
        cells = lines[2].split(" | ")[1:4]
        self.assertEqual(cells, [" " * 14 + "\u221230.8%", " " * 10 + "\u221211.8%", " " * 10 + "\u221220.0%"])
        self.assertTrue(lines[3].startswith("| 4 GiB text, 32K "))
        self.assertEqual(len({len(line) for line in lines}), 1)

    def test_percent_rounds_to_one_decimal_with_a_minus_sign(self) -> None:
        self.assertEqual(compare.percent(-15.098258276944287), "\u221215.1%")
        self.assertEqual(compare.percent(-9.648303000491886), "\u22129.6%")
        self.assertEqual(compare.percent(2.25), "+2.2%")

    def test_setting_label_stays_in_one_markdown_cell(self) -> None:
        label = "1 GiB | 16K\nrepeat"
        table, comparison = self.run_compare((label, self.results))
        self.assertEqual(len(table.splitlines()), 3)
        self.assertTrue(table.splitlines()[2].startswith(r"| 1 GiB \| 16K repeat "))
        self.assertEqual(comparison["settings"][0]["label"], label)

    def test_rejects_measurements_of_another_flores_file(self) -> None:
        self.write_report(self.results, "ube_bpe", self.AVERAGES["ube_bpe"], flores_sha256="0" * 64)
        with self.assertRaises(SystemExit) as raised:
            self.run_compare(("setting", self.results))
        self.assertIn("another FLORES-200 file", str(raised.exception.code))

    def test_rejects_flores_replaced_during_loading(self) -> None:
        measured = self.flores_path.read_bytes()
        other = json.loads(measured)
        other["sentence_kor_Hang"] = ["one", "two"]
        self.flores_path.write_text(json.dumps(other), encoding="utf-8")
        load_flores = measure_flores.load_flores

        def load_then_replace(path):
            sentences = load_flores(path)
            path.write_bytes(measured)
            return sentences

        with unittest.mock.patch.object(measure_flores, "load_flores", load_then_replace):
            with self.assertRaisesRegex(SystemExit, "changed"):
                self.run_compare(("setting", self.results))
        self.assertFalse((self.root / "comparison.json").exists())

    def test_unreadable_flores_data_names_the_file(self) -> None:
        for raw in (b"{", b"[]", b'{"sentence_eng_Latn": 5}'):
            with self.subTest(raw=raw):
                self.flores_path.write_bytes(raw)
                with self.assertRaises(SystemExit) as raised:
                    self.run_compare(("setting", self.results))
                self.assertIn(str(self.flores_path), str(raised.exception))

    def test_output_never_overwrites_measurements_or_flores(self) -> None:
        link = self.root / "link.json"
        link.symlink_to(self.results / "bytelevel_bpe.json")
        hardlink = self.root / "hardlink.json"
        hardlink.hardlink_to(self.flores_path)
        argv = ["--setting", "setting", str(self.results), "--flores", str(self.flores_path)]
        for path in (self.results / "ube_unigram.json", self.flores_path, link, hardlink):
            with self.subTest(path=path.name):
                before = path.read_bytes()
                with unittest.mock.patch("sys.stderr", io.StringIO()) as stderr:
                    with self.assertRaises(SystemExit) as raised:
                        run_main(compare, [*argv, "--output", str(path)])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("--output would overwrite input files", stderr.getvalue())
                self.assertEqual(path.read_bytes(), before)

    def test_output_can_be_regenerated(self) -> None:
        output = self.root / "comparison.json"
        output.write_text("earlier output\n", encoding="utf-8")
        _, first = self.run_compare(("setting", self.results))
        _, second = self.run_compare(("setting", self.results))
        self.assertEqual(first, second)
        self.assertEqual(sorted(path.name for path in self.root.iterdir() if path.name.endswith(".tmp")), [])

    def test_output_preserves_a_sibling_input_with_a_temporary_filename(self) -> None:
        self.flores_path = self.flores_path.rename(self.root / ".comparison.json.tmp")
        measured = self.flores_path.read_bytes()
        self.run_compare(("setting", self.results))
        self.assertEqual(self.flores_path.read_bytes(), measured)

    def test_failed_output_replacement_leaves_no_temporary_files(self) -> None:
        output = self.root / "comparison.json"
        output.mkdir()
        before = set(self.root.iterdir())
        argv = ["--flores", str(self.flores_path), "--setting", "setting", str(self.results), "--output", str(output)]
        with self.assertRaises(OSError):
            run_main(compare, argv)
        self.assertEqual(set(self.root.iterdir()), before)

    def test_rejects_a_renamed_measurement_or_a_repeated_tokenizer(self) -> None:
        kind = "ube_bpe"
        other = json.loads((self.results / "bytelevel_bpe.json").read_bytes())
        self.write_report(self.results, kind, self.AVERAGES[kind], tokenizer=other["tokenizer"])
        with self.assertRaisesRegex(SystemExit, "expected a measurement of ube_bpe"):
            self.run_compare(("setting", self.results))
        self.write_report(self.results, kind, self.AVERAGES[kind], tokenizer_sha256=other["tokenizer_sha256"])
        with self.assertRaisesRegex(SystemExit, "same tokenizer SHA-256"):
            self.run_compare(("setting", self.results))

    def test_rejects_inconsistent_per_language_token_counts(self) -> None:
        path = self.results / "ube_unigram.json"
        original = path.read_bytes()
        for key, value in (("n", 1), ("total", -2), ("avg", 19)):
            with self.subTest(key=key):
                report = json.loads(original)
                report["results"]["kor_Hang"][key] = value
                path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaisesRegex(SystemExit, "kor_Hang"):
                    self.run_compare(("setting", self.results))

    def test_zero_baseline_keeps_the_absolute_values(self) -> None:
        empty_group = {**self.AVERAGES["bytelevel_bpe"], "kor_Hang": 0.0, "zho_Hans": 0.0}
        self.write_report(self.results, "bytelevel_bpe", empty_group)
        table, comparison = self.run_compare(("setting", self.results))
        group = comparison["settings"][0]["bpe"]["group_mean_tokens"]
        self.assertEqual((group["ube"], group["bytelevel"]), (3.75, 0.0))
        self.assertIsNone(group["change_pct"])
        self.assertEqual(table.splitlines()[2].split(" | ")[2].strip(), "N/A")

    def test_rejects_mixed_vocabulary_sizes_and_missing_tokenizers(self) -> None:
        self.write_report(self.results, "ube_bpe", self.AVERAGES["ube_bpe"], vocab_size=200)
        with self.assertRaises(SystemExit) as raised:
            self.run_compare(("setting", self.results))
        self.assertIn("different vocabulary sizes", str(raised.exception.code))
        (self.results / "ube_bpe.json").unlink()
        with self.assertRaises(SystemExit) as raised:
            self.run_compare(("setting", self.results))
        self.assertIn("ube_bpe.json not found", str(raised.exception.code))

    def test_values_are_finite(self) -> None:
        _, comparison = self.run_compare(("setting", self.results))
        for model in ("unigram", "bpe"):
            for values in comparison["settings"][0][model].values():
                if "change_pct" in values:
                    self.assertTrue(math.isfinite(values["change_pct"]))


if __name__ == "__main__":
    unittest.main()
