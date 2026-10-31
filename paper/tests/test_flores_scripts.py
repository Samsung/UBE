"""Tests for the FLORES-200 preparation and measurement scripts."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import unittest
import unittest.mock
from pathlib import Path
from tempfile import TemporaryDirectory

from tokenizers import Tokenizer, models, pre_tokenizers, processors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import measure_flores  # noqa: E402
import prepare_flores  # noqa: E402


class MeasureFloresTest(unittest.TestCase):
    def test_gini(self) -> None:
        self.assertEqual(measure_flores.gini([2.0, 2.0, 2.0]), 0.0)
        self.assertAlmostEqual(measure_flores.gini([0.0, 0.0, 3.0]), 2 / 3)
        self.assertAlmostEqual(measure_flores.gini([3.0, 1.0, 2.0]), measure_flores.gini([1.0, 2.0, 3.0]))

    def test_summary_uses_english_as_the_reference(self) -> None:
        results = {
            "eng_Latn": {"avg": 10.0},
            "fra_Latn": {"avg": 12.0},
            "zho_Hans": {"avg": 20.0},
            "bod_Tibt": {"avg": 40.0},
        }
        summary = measure_flores.summarize(results)
        self.assertEqual(summary["en_avg_tokens"], 10.0)
        self.assertAlmostEqual(summary["mean_premium"], (1.2 + 2.0 + 4.0) / 3)
        self.assertEqual(summary["median_premium"], 2.0)
        self.assertEqual(summary["max_premium"], 4.0)
        self.assertAlmostEqual(summary["gini"], measure_flores.gini([1.2, 2.0, 4.0]))


class MeasureTokenCountsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = Tokenizer(
            models.WordLevel(
                {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3, "a": 4, "b": 5, "c": 6, "d": 7},
                unk_token="[UNK]",
            )
        )
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        # Synthetic inputs with known token counts; no language data is needed.
        self.flores = {
            "eng_Latn": ["a", "a b"],
            "kor_Hang": ["a b c", "a b c d"],
            "zho_Hans": ["a b", "a b c"],
        }

    def assert_measurement(self, extra_tokens_per_sentence: int = 0) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            self.tokenizer.save(str(path))
            original = path.read_bytes()
            report = measure_flores.measure(path, self.flores)
            self.assertEqual(path.read_bytes(), original)
            reloaded = Tokenizer.from_file(str(path))
            self.assertEqual(reloaded.padding, self.tokenizer.padding)
            self.assertEqual(reloaded.truncation, self.tokenizer.truncation)

        expected = {
            lang: {
                "total": total + 2 * extra_tokens_per_sentence,
                "avg": total / 2 + extra_tokens_per_sentence,
                "n": 2,
            }
            for lang, total in (("eng_Latn", 3), ("kor_Hang", 7), ("zho_Hans", 5))
        }
        self.assertEqual(report["results"], expected)
        self.assertEqual(report["summary"], measure_flores.summarize(expected))
        self.assertEqual(report["tokenizer_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(report["vocab_size"], 8)
        self.assertEqual(report["languages"], 3)
        self.assertEqual(report["sentences_per_lang"], 2)

    def test_counts_without_padding_or_truncation(self) -> None:
        self.assert_measurement()

    def test_counts_ignore_saved_padding(self) -> None:
        for options in ({}, {"length": 16}, {"pad_to_multiple_of": 8}, {"direction": "left", "length": 16}):
            with self.subTest(options=options):
                self.tokenizer.enable_padding(pad_id=1, pad_token="[PAD]", **options)
                self.assert_measurement()

    def test_counts_ignore_saved_truncation(self) -> None:
        for direction in ("left", "right"):
            with self.subTest(direction=direction):
                self.tokenizer.enable_truncation(max_length=2, direction=direction)
                self.assert_measurement()

    def test_counts_ignore_saved_padding_and_truncation(self) -> None:
        self.tokenizer.enable_padding(length=16, pad_id=1, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=2)
        self.assert_measurement()

    def test_counts_keep_post_processor_tokens(self) -> None:
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[("[BOS]", 2), ("[EOS]", 3)],
        )
        self.tokenizer.enable_padding(length=16, pad_id=1, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=4)
        self.assert_measurement(extra_tokens_per_sentence=2)


class MeasureFloresCliTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.tokenizer_path = self.root / "sample.json"
        self.flores_path = self.root / "flores.json"
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1, "b": 2}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.save(str(self.tokenizer_path))
        self.flores_path.write_text(
            json.dumps({"sentence_eng_Latn": ["a", "a b"], "sentence_kor_Hang": ["a b", "a b a"]}),
            encoding="utf-8",
        )

    def argv(self, tokenizers: list[Path], output_dir: Path) -> list[str]:
        return [
            "measure_flores.py", *map(str, tokenizers),
            "--flores", str(self.flores_path), "--output-dir", str(output_dir),
        ]

    def assert_collision_rejected(self, tokenizers: list[Path], output_dir: Path) -> None:
        originals = {path: path.read_bytes() for path in [*tokenizers, self.flores_path]}
        stderr = io.StringIO()
        with unittest.mock.patch.object(sys, "argv", self.argv(tokenizers, output_dir)):
            with unittest.mock.patch("sys.stderr", stderr), unittest.mock.patch("sys.stdout", io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    measure_flores.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--output-dir would overwrite input files:", stderr.getvalue())
        for path, original in originals.items():
            self.assertEqual(path.read_bytes(), original)

    def test_output_dir_never_overwrites_inputs(self) -> None:
        self.assert_collision_rejected([self.tokenizer_path], self.root)

    def test_output_path_cannot_equal_flores_input(self) -> None:
        tokenizers_dir = self.root / "tokenizers"
        tokenizers_dir.mkdir()
        tokenizer_path = tokenizers_dir / "flores.json"
        tokenizer_path.write_bytes(self.tokenizer_path.read_bytes())
        self.assert_collision_rejected([tokenizer_path], self.root)

    def assert_link_collision_rejected(self, source: Path, *, hardlink: bool) -> None:
        output_dir = self.root / "results"
        output_dir.mkdir()
        output = output_dir / "sample.json"
        if hardlink:
            output.hardlink_to(source)
        else:
            output.symlink_to(source)
        self.assert_collision_rejected([self.tokenizer_path], output_dir)

    def test_output_symlink_to_tokenizer_is_rejected(self) -> None:
        self.assert_link_collision_rejected(self.tokenizer_path, hardlink=False)

    def test_output_symlink_to_flores_is_rejected(self) -> None:
        self.assert_link_collision_rejected(self.flores_path, hardlink=False)

    def test_output_hardlink_to_tokenizer_is_rejected(self) -> None:
        self.assert_link_collision_rejected(self.tokenizer_path, hardlink=True)

    def test_output_hardlink_to_flores_is_rejected(self) -> None:
        self.assert_link_collision_rejected(self.flores_path, hardlink=True)

    def test_all_outputs_are_checked_before_any_result_is_written(self) -> None:
        second = self.root / "second.json"
        second.write_bytes(self.tokenizer_path.read_bytes())
        output_dir = self.root / "results"
        output_dir.mkdir()
        (output_dir / "second.json").hardlink_to(self.tokenizer_path)
        self.assert_collision_rejected([self.tokenizer_path, second], output_dir)
        self.assertFalse((output_dir / "sample.json").exists())

    def test_normal_output_preserves_inputs_and_can_be_regenerated(self) -> None:
        originals = {path: path.read_bytes() for path in (self.tokenizer_path, self.flores_path)}
        output_dir = self.root / "results"
        for _ in range(2):
            with unittest.mock.patch.object(sys, "argv", self.argv([self.tokenizer_path], output_dir)):
                with unittest.mock.patch("sys.stdout", io.StringIO()):
                    self.assertEqual(measure_flores.main(), 0)
            report = json.loads((output_dir / "sample.json").read_text(encoding="utf-8"))
            self.assertEqual(report["results"]["eng_Latn"], {"total": 3, "avg": 1.5, "n": 2})
            self.assertEqual(report["results"]["kor_Hang"], {"total": 5, "avg": 2.5, "n": 2})
            for path, original in originals.items():
                self.assertEqual(path.read_bytes(), original)


class PrepareFloresTest(unittest.TestCase):
    def test_reads_and_strips_devtest_lines(self) -> None:
        with TemporaryDirectory() as directory:
            archive = Path(directory) / "flores.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                for name, content in (
                    ("flores200_dataset/devtest/eng_Latn.devtest", "One. \nTwo.\n"),
                    ("flores200_dataset/dev/eng_Latn.dev", "ignored\n"),
                ):
                    data = content.encode("utf-8")
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            dataset = prepare_flores.read_devtest(archive)
        self.assertEqual(dataset, {"sentence_eng_Latn": ["One.", "Two."]})

    def test_text_digest_depends_on_every_sentence(self) -> None:
        base = {"sentence_eng_Latn": ["a", "b"], "sentence_fra_Latn": ["c"]}
        changed = {"sentence_eng_Latn": ["a", "b"], "sentence_fra_Latn": ["d"]}
        self.assertEqual(prepare_flores.text_sha256(base), prepare_flores.text_sha256(dict(reversed(base.items()))))
        self.assertNotEqual(prepare_flores.text_sha256(base), prepare_flores.text_sha256(changed))


if __name__ == "__main__":
    unittest.main()
