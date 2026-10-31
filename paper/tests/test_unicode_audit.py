"""Checks of the Unicode audit script and of its committed results."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import unicode_audit as audit  # noqa: E402

RESULTS = ROOT / "results" / "unicode_audit.json"
REPORT = ROOT / "results" / "unicode_audit.md"


class HelperTest(unittest.TestCase):
    def test_scalar_universe(self) -> None:
        self.assertEqual(audit.SCALAR_COUNT, 1_112_064)

    def test_parsers(self) -> None:
        self.assertEqual(audit._parse_range("0041..005A"), (0x41, 0x5A))
        self.assertEqual(audit._parse_range("00E9"), (0xE9, 0xE9))
        self.assertEqual(audit._parse_sequence("0041 0301"), "A\u0301")
        self.assertEqual(audit._parse_sequence("  "), "")

    def test_coverage_groups_can_overlap(self) -> None:
        category = bytearray(audit.MAX_CODE_POINT + 1)
        age = bytearray(audit.MAX_CODE_POINT + 1)
        category[0x41] = 1
        age[0x41] = 1
        category[0xE000] = 2
        age[0xE000] = 1
        category[0x2FFFF] = 0
        self.assertEqual(audit._coverage_groups("A", category, age), ("supported_through_unicode_16",))
        self.assertEqual(
            audit._coverage_groups("A\ue000\U0002ffff", category, age),
            ("contains_unassigned_or_reserved", "contains_private_use"),
        )

    def test_accumulator_separates_raw_and_normalized_exactness(self) -> None:
        accumulator = audit.Accumulator()
        case = audit.Case("normalization", ("nfc_changes",), "x", "A\u030a", expected_nfc="\u00c5")
        accumulator.add(case, "\u00c5", "\u00c5", None)
        metrics = accumulator.serializable()["normalization"]["all"]
        self.assertEqual(metrics["cases"], 1)
        self.assertEqual(metrics["raw_exact"], 0)
        self.assertEqual(metrics["declared_normalization_exact"], 1)
        self.assertEqual(metrics["nfc_exact"], 1)
        self.assertEqual(len(metrics["failure_examples"]), 1)


def _has_ube() -> bool:
    try:
        from tokenizers import pre_tokenizers
    except ImportError:
        return False
    return hasattr(pre_tokenizers, "UniversalByteLevel")


@unittest.skipUnless(_has_ube(), "the UBE tokenizers build is not installed")
class UbeTokenizerSettingsTest(unittest.TestCase):
    """Settings stored in a given tokenizer file must not change the audit."""

    TEXT = "한국어 text 😀"

    def tokenizer(self):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.UniversalByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.UniversalByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=600,
            special_tokens=["[PAD]", "[BOS]", "[EOS]"],
            initial_alphabet=pre_tokenizers.UniversalByteLevel.alphabet(),
            show_progress=False,
        )
        tokenizer.train_from_iterator([self.TEXT] * 10, trainer=trainer)
        return tokenizer

    def load(self, tokenizer):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(str(path))
            return audit._load_method("ube", path)

    def test_padding_truncation_and_special_tokens_are_ignored(self) -> None:
        from tokenizers import processors

        tokenizer = self.tokenizer()
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=64)
        tokenizer.enable_truncation(max_length=2)
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[("[BOS]", 1), ("[EOS]", 2)],
        )
        round_trip, declared_normalize, _ = self.load(tokenizer)
        self.assertEqual(round_trip(self.TEXT), self.TEXT)
        self.assertEqual(declared_normalize(self.TEXT), self.TEXT)

    def test_configured_normalizer_is_the_declared_normalization(self) -> None:
        from tokenizers import normalizers

        tokenizer = self.tokenizer()
        tokenizer.normalizer = normalizers.NFC()
        round_trip, declared_normalize, metadata = self.load(tokenizer)
        self.assertEqual(declared_normalize("A\u030a"), "\u00c5")
        self.assertEqual(round_trip("A\u030a"), "\u00c5")
        self.assertEqual(metadata["normalization"], {"type": "NFC"})


@unittest.skipUnless(RESULTS.is_file(), "results/unicode_audit.json not present")
class CommittedResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results = json.loads(RESULTS.read_text(encoding="utf-8"))
        cls.ube = cls.results["methods"]["ube"]
        cls.myte = cls.results["methods"]["myte"]
        cls.script = cls.results["methods"]["script"]

    def test_results_come_from_this_script_and_pinned_inputs(self) -> None:
        script = ROOT / "scripts" / "unicode_audit.py"
        self.assertEqual(self.results["audit_script_sha256"], hashlib.sha256(script.read_bytes()).hexdigest())
        self.assertEqual(self.results["design"]["unicode_version"], "17.0.0")
        self.assertEqual(self.results["design"]["scalar_count"], 1_112_064)
        self.assertEqual(self.results["design"]["sampling"], "none")
        self.assertEqual(self.results["script_bpe_commit"], audit.SCRIPT_TOK_COMMIT)
        self.assertEqual(self.results["myte_commit"], "177299ca7278a3618b8c77588e6dcd3912bad428")
        for label, source in self.results["unicode_sources"].items():
            with self.subTest(source=label):
                self.assertEqual(source["observed_sha256"], audit.DATA_SOURCES[label]["sha256"])
        self.assertEqual(self.myte["tokenizer"]["implementation_sha256"], audit.MYTE_REWRITER_SHA256)

    def test_report_is_rendered_from_the_results(self) -> None:
        self.assertEqual(REPORT.read_text(encoding="utf-8"), audit._render_report(self.results))

    def test_every_scalar_is_audited_once_per_method(self) -> None:
        groups = {
            "assigned_through_unicode_16": 155_063,
            "assigned_in_unicode_17": 4_803,
            "private_use": 137_468,
            "unassigned_or_reserved": 814_730,
        }
        for method in (self.ube, self.myte, self.script):
            scalars = method["suite_counts"]["unicode_scalars"]
            self.assertEqual(scalars["all"]["cases"], 1_112_064)
            for group, expected in groups.items():
                with self.subTest(method=method["method"], group=group):
                    self.assertEqual(scalars[group]["cases"], expected)
                    self.assertEqual(scalars[group]["errors"], 0)

    def test_ube_recovers_every_input(self) -> None:
        for suite, expected in (
            ("unicode_scalars", 1_112_064),
            ("normalization", 37_671),
            ("grapheme_break", 764),
            ("emoji", 5_225),
        ):
            with self.subTest(suite=suite):
                metrics = self.ube["suite_counts"][suite]["all"]
                self.assertEqual(metrics["cases"], expected)
                self.assertEqual(metrics["raw_exact"], expected)
                self.assertEqual(metrics["errors"], 0)

    def test_myte_counts(self) -> None:
        scalars = self.myte["suite_counts"]["unicode_scalars"]
        for group, (cases, exact) in {
            "assigned_through_unicode_16": (155_063, 154_023),
            "assigned_in_unicode_17": (4_803, 4_803),
            "private_use": (137_468, 137_468),
            "unassigned_or_reserved": (814_730, 814_730),
            "all": (1_112_064, 1_111_024),
        }.items():
            with self.subTest(group=group):
                self.assertEqual((scalars[group]["cases"], scalars[group]["raw_exact"]), (cases, exact))
        for suite, (cases, exact) in {
            "normalization": (37_671, 21_373),
            "grapheme_break": (764, 684),
            "emoji": (5_225, 5_225),
        }.items():
            with self.subTest(suite=suite):
                metrics = self.myte["suite_counts"][suite]["all"]
                self.assertEqual((metrics["cases"], metrics["raw_exact"]), (cases, exact))
        self.assertEqual(self.myte["processed_cases"], 1_155_724)

    def test_script_bpe_counts(self) -> None:
        scalars = self.script["suite_counts"]["unicode_scalars"]
        supported = scalars["assigned_through_unicode_16"]
        self.assertEqual(supported["raw_exact"], 153_943)
        self.assertEqual(supported["declared_normalization_exact"], 155_063)
        for group in ("assigned_in_unicode_17", "private_use", "unassigned_or_reserved"):
            with self.subTest(group=group):
                self.assertEqual(scalars[group]["declared_normalization_exact"], 0)
        suites = self.script["suite_counts"]
        for suite, (supported_group, supported_count, excluded_group, excluded_count) in {
            "normalization": ("supported_through_unicode_16", 37_505, "contains_unicode_17", 166),
            "grapheme_break": ("supported_through_unicode_16", 690, "contains_unassigned_or_reserved", 74),
            "emoji": ("supported_through_unicode_16", 5_158, "contains_unicode_17", 67),
        }.items():
            with self.subTest(suite=suite):
                self.assertEqual(suites[suite][supported_group]["cases"], supported_count)
                self.assertEqual(suites[suite][supported_group]["declared_normalization_exact"], supported_count)
                self.assertEqual(suites[suite][excluded_group]["cases"], excluded_count)
                self.assertEqual(suites[suite][excluded_group]["declared_normalization_exact"], 0)
                self.assertEqual(suites[suite]["all"]["cases"], supported_count + excluded_count)
        self.assertEqual(suites["normalization"]["all"]["nfc_exact"], 37_385)


if __name__ == "__main__":
    unittest.main()
