# Copyright 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

"""Tests for the partial-decoding measurement script."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tokenizers import ByteLevelBPETokenizer, Tokenizer, UniversalByteLevelBPETokenizer

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import measure_decoding  # noqa: E402

REPLACEMENT = "�"
SENTENCES = [
    "前文的内容很长继续生成中文",
    "Hi 빙고 🩷💚, 😀 and ภาษาไทย.",
    "ཀུན་མཁྱེན། ᱥᱟᱱᱛᱟᱲᱤ ⵜⴰⵎⴰⵣⵉⵖⵜ",
    "Plain ASCII text only.",
]


def base_symbol_tokenizer(directory: Path, universal: bool = True) -> Path:
    """A byte-level tokenizer without merges: every base symbol is a token."""
    tokenizer = UniversalByteLevelBPETokenizer() if universal else ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(["x"], vocab_size=512 if universal else 256, show_progress=False)
    path = directory / ("base" if universal else "byte_level") / "tokenizer.json"
    path.parent.mkdir()
    tokenizer.save(str(path))
    return path


def trained_tokenizer(directory: Path) -> Path:
    tokenizer = UniversalByteLevelBPETokenizer()
    tokenizer.train_from_iterator(SENTENCES * 20, vocab_size=600, show_progress=False)
    path = directory / "trained" / "tokenizer.json"
    path.parent.mkdir()
    tokenizer.save(str(path))
    return path


class HelpersTest(unittest.TestCase):
    def test_symbol_starts(self) -> None:
        self.assertEqual(measure_decoding.symbol_starts("A한😀é", universal=True), [0, 1, 3, 7, 9])
        self.assertEqual(measure_decoding.symbol_starts("A한😀é", universal=False), [0, 1, 4, 8, 10])

    def test_slice_is_exact(self) -> None:
        starts = [0, 1, 3, 7, 9]
        text = "A한😀é"
        self.assertTrue(measure_decoding.slice_is_exact("한😀é", text, starts, 1))
        self.assertFalse(measure_decoding.slice_is_exact(REPLACEMENT + "😀é", text, starts, 1))
        self.assertTrue(measure_decoding.slice_is_exact(REPLACEMENT + "😀é", text, starts, 2))
        self.assertTrue(measure_decoding.slice_is_exact(2 * REPLACEMENT + "é", text, starts, 4))
        self.assertFalse(measure_decoding.slice_is_exact("😀é", text, starts, 2))
        self.assertFalse(measure_decoding.slice_is_exact("X😀é", text, starts, 2))

    def test_windows_from_misaligned_and_aligned_starts(self) -> None:
        with TemporaryDirectory() as directory:
            tokenizer = Tokenizer.from_file(str(base_symbol_tokenizer(Path(directory))))
        prompt, continuation = "前文的内容很长", "继续生成中文"
        ids = tokenizer.encode(prompt + continuation).ids
        k = len(tokenizer.encode(prompt).ids)
        self.assertEqual(k, 14)
        # Five tokens back is the second byte of a code unit; six tokens back is a character boundary.
        self.assertNotEqual(measure_decoding.window_stream(tokenizer, ids, k, k - 5), continuation)
        self.assertEqual(measure_decoding.window_stream(tokenizer, ids, k, k - 6), continuation)
        self.assertNotEqual(measure_decoding.final_text(tokenizer, ids, k, k - 1), continuation)
        self.assertEqual(measure_decoding.final_text(tokenizer, ids, k, k - 2), continuation)

    def test_final_text_includes_a_trailing_replacement_character(self) -> None:
        with TemporaryDirectory() as directory:
            tokenizer = Tokenizer.from_file(str(base_symbol_tokenizer(Path(directory))))
        ids = tokenizer.encode("a�").ids
        self.assertEqual(measure_decoding.final_text(tokenizer, ids, 1, 0), "�")


class MeasureTest(unittest.TestCase):
    def test_byte_level_counts(self) -> None:
        with TemporaryDirectory() as directory:
            path = base_symbol_tokenizer(Path(directory), universal=False)
            _, result = measure_decoding.measure_language((str(path), "mul_Mixd", SENTENCES))
        counts = result["counts"]
        self.assertEqual(counts["round_trip"], len(SENTENCES))
        for key in ("decode_stream", "window_back5", "window_aligned", "final_back1", "final_aligned"):
            self.assertEqual(counts[key], counts["streams"], key)
        self.assertEqual(counts["slices_plain"], counts["slices"])
        self.assertEqual(counts["slices_decode_slice"], counts["slices"])

    def test_empty_stream_summaries(self) -> None:
        with TemporaryDirectory() as directory:
            path = base_symbol_tokenizer(Path(directory))
            for flores in ({}, {"eng_Latn": []}, {"eng_Latn": ["", "a"]}):
                with self.subTest(flores=flores):
                    report = measure_decoding.measure(path, flores, workers=1)
                    self.assertEqual(report["totals"]["streams"], 0)
                    self.assertEqual(report["aligned_window_tokens"], {"median": None, "p99": None, "max": None})
                    self.assertEqual(report["totals"]["sentences"], sum(map(len, flores.values())))

    def test_worker_counts_are_deterministic(self) -> None:
        with TemporaryDirectory() as directory:
            path = base_symbol_tokenizer(Path(directory))
            flores = {"mul_Mixd": SENTENCES, "eng_Latn": SENTENCES[-1:]}
            serial = measure_decoding.measure(path, flores, workers=1)
            parallel = measure_decoding.measure(path, flores, workers=2)
        self.assertEqual(serial, parallel)

    def test_counts(self) -> None:
        with TemporaryDirectory() as directory:
            for path in (base_symbol_tokenizer(Path(directory)), trained_tokenizer(Path(directory))):
                with self.subTest(tokenizer=path.parent.name):
                    lang, result = measure_decoding.measure_language((str(path), "mul_Mixd", SENTENCES))
                    counts = result["counts"]
                    self.assertEqual(lang, "mul_Mixd")
                    self.assertEqual(counts["sentences"], len(SENTENCES))
                    self.assertEqual(counts["round_trip"], len(SENTENCES))
                    self.assertEqual(counts["streams"], len(SENTENCES))
                    for key in ("decode_stream", "window_aligned", "final_aligned"):
                        self.assertEqual(counts[key], counts["streams"], key)
                    self.assertEqual(counts["slices_decode_slice"], counts["slices"])
                    self.assertGreater(counts["slices_inside_character"], 0)
                    self.assertLess(counts["slices_plain"], counts["slices"])
                    self.assertEqual(len(result["aligned_window_tokens"]), counts["streams"])

    def test_command_line(self) -> None:
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            tokenizer = base_symbol_tokenizer(directory)
            flores = directory / "flores.json"
            flores.write_text(json.dumps({"sentence_mul_Mixd": SENTENCES, "sentence_eng_Latn": SENTENCES[-1:]}),
                              encoding="utf-8")
            output = directory / "out"
            command = [sys.executable, str(SCRIPTS / "measure_decoding.py"), str(tokenizer), "--flores", str(flores),
                       "--output-dir", str(output), "--workers", "1"]
            subprocess.run(command, check=True, capture_output=True)
            report = json.loads((output / "base.json").read_text(encoding="utf-8"))
            self.assertEqual(report["decoder"], "UniversalByteLevel")
            self.assertEqual(report["languages"], 2)
            self.assertEqual(report["totals"]["sentences"], len(SENTENCES) + 1)
            self.assertEqual(report["totals"]["window_aligned"], report["totals"]["streams"])
            self.assertEqual(set(report["results"]), {"eng_Latn", "mul_Mixd"})
            named = directory / "named.json"
            named.write_bytes(tokenizer.read_bytes())
            clash = subprocess.run([sys.executable, str(SCRIPTS / "measure_decoding.py"), str(named), "--flores",
                                    str(flores), "--output-dir", str(directory), "--workers", "1"],
                                   capture_output=True, text=True)
            self.assertNotEqual(clash.returncode, 0)
            self.assertIn("would overwrite input files", clash.stderr)
            original = tokenizer.read_bytes()
            os.link(tokenizer, output / "base.json.tmp")
            (output / "base.json.tmp").replace(output / "base.json")
            clash = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(clash.returncode, 0)
            self.assertIn("would overwrite input files", clash.stderr)
            self.assertEqual(tokenizer.read_bytes(), original)
            invalid_workers = subprocess.run(command + ["--workers", "0"], capture_output=True, text=True)
            self.assertNotEqual(invalid_workers.returncode, 0)
            self.assertIn("--workers must be at least 1", invalid_workers.stderr)


if __name__ == "__main__":
    unittest.main()
