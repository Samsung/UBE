"""Tests for the MYTE codec, MYTE-UBE and the committed MYTE-UBE results.

The codec tests need ``MYTE_DIR`` set to a checkout of
https://github.com/tomlimi/MYTE at the pinned commit and are skipped otherwise.
"""

from __future__ import annotations

import json
import os
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "myte_ube_flores.json"
sys.path.insert(0, str(ROOT / "scripts"))

from myte_codec import DECOMPOSE_REL, MERGE_REL  # noqa: E402
from measure_flores import gini  # noqa: E402
from myte_ube import (  # noqa: E402
    REGION_A_BASE,
    REGION_B_BASE,
    MyteUbeCodec,
    load_official_rewriter,
    summarize,
    ube_length,
)

MYTE_DIR = Path(os.environ.get("MYTE_DIR", "MYTE"))

CASES = [
    "",
    "Hello WORLD",
    "ning",  # byte-identical to a merge-rule source
    "The quick brown fox jumps over the lazy dog.",
    "안녕하세요 세계",  # Korean, 3-byte BMP
    "中文湬湧混合 English text",  # CJK + ASCII
    "ภาษาไทย",  # Thai
    "\U0001f600\U0001f680 emoji \U00010348",  # supplementary characters
    "café naïve Über",  # 2-byte accents incl. uppercase
    "ትግርኛ",  # Ethiopic
]


def random_texts(count: int, seed: int):
    rng = random.Random(seed)
    ranges = ((0x20, 0x7E), (0x80, 0xD7FF), (0xE000, 0xFFFD), (0x10000, 0x10FFFF))
    for _ in range(count):
        chars = []
        for _ in range(rng.randrange(0, 31)):
            low, high = rng.choice(ranges)
            chars.append(chr(rng.randint(low, high)))
        yield "".join(chars)


class HelperTest(unittest.TestCase):
    def test_ube_length_routes_by_utf8_length(self) -> None:
        self.assertEqual(ube_length("aé"), 1 + 2)
        self.assertEqual(ube_length("中"), 2)
        self.assertEqual(ube_length("\U0001f600"), 4)

    def test_summarize_groups(self) -> None:
        per_language = {
            "eng_Latn": {"total": 10, "avg": 10.0},
            "zho_Hans": {"total": 20, "avg": 20.0},
            "kor_Hang": {"total": 40, "avg": 40.0},
            "bod_Tibt": {"total": 30, "avg": 30.0},
        }
        summary = summarize(per_language)
        self.assertEqual(summary["english"], 10.0)
        self.assertEqual(summary["cjk"], 30.0)
        self.assertEqual(summary["cjk_configurations"], 2)
        self.assertEqual(summary["other_bmp"], 30.0)
        self.assertEqual(summary["other_bmp_configurations"], 1)
        self.assertEqual(summary["overall"], 25.0)
        self.assertEqual(summary["total_positions"], 100)


@unittest.skipUnless((MYTE_DIR / MERGE_REL).is_file(), "set MYTE_DIR to a MYTE checkout")
class MyteUbeCodecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.codec = MyteUbeCodec.from_dir(MYTE_DIR)
        cls.myte = cls.codec.myte
        byte_rewriter = load_official_rewriter(MYTE_DIR)
        cls.official_decompose = byte_rewriter(str(MYTE_DIR / DECOMPOSE_REL))
        cls.official_merge = byte_rewriter(str(MYTE_DIR / MERGE_REL))

    def test_codec_matches_released_implementation_in_both_directions(self) -> None:
        for text in (*CASES, *random_texts(500, seed=777)):
            with self.subTest(text=text):
                raw = text.encode("utf-8")
                hex_bytes = [f"{value:02x}" for value in raw]
                encoded = self.official_merge.rewrite_bytes(self.official_decompose.rewrite_bytes(hex_bytes))
                self.assertEqual(self.myte.encode_bytes(raw), bytes.fromhex("".join(encoded)))
                decoded = self.official_decompose.rewrite_bytes(
                    self.official_merge.rewrite_bytes(encoded, reverse=True), reverse=True
                )
                self.assertEqual(self.myte.decode_bytes(self.myte.encode_bytes(raw)), bytes.fromhex("".join(decoded)))

    def test_spans_concatenate_to_the_myte_output(self) -> None:
        for text in CASES:
            with self.subTest(text=text):
                spans = self.codec.spans(text)
                self.assertEqual(b"".join(chunk for _, chunk in spans), self.myte.encode_bytes(text.encode("utf-8")))
                for is_code, chunk in spans:
                    if not is_code:
                        chunk.decode("utf-8")

    def test_symbols_rebuild_the_post_myte_byte_stream(self) -> None:
        for text in (*CASES, *random_texts(200, seed=1)):
            with self.subTest(text=text):
                symbols = self.codec.encode(text)
                self.assertEqual(self.codec.post_myte_bytes(symbols), self.myte.encode_bytes(text.encode("utf-8")))
                self.assertTrue(
                    all(
                        REGION_A_BASE <= ord(s) < REGION_A_BASE + 256 or REGION_B_BASE <= ord(s) < REGION_B_BASE + 256
                        for s in symbols
                    )
                )

    def test_decoding_equals_myte_decoding(self) -> None:
        for text in CASES:
            with self.subTest(text=text):
                expected = self.myte.decode_bytes(self.myte.encode_bytes(text.encode("utf-8"))).decode("utf-8")
                self.assertEqual(self.codec.decode(self.codec.encode(text)), expected)

    def test_plain_text_round_trips(self) -> None:
        # No character here shares its MYTE decomposition with another character.
        for text in ("", "ning", "안녕하세요 세계", "ภาษาไทย", "\U0001f600\U0001f680 emoji \U00010348", "ትግርኛ"):
            with self.subTest(text=text):
                self.assertEqual(self.codec.decode(self.codec.encode(text)), text)

    def test_unmerged_three_byte_characters_cost_two_symbols(self) -> None:
        text = "湬湧"  # rare CJK pair with no MYTE rule
        self.assertEqual(self.myte.encode_bytes(text.encode("utf-8")), text.encode("utf-8"))
        symbols = self.codec.encode(text)
        self.assertEqual([ord(s) - REGION_B_BASE for s in symbols], list(text.encode("utf-16-le")))

    def test_supplementary_character_costs_four_symbols(self) -> None:
        text = "\U00010348"
        symbols = self.codec.encode(text)
        self.assertEqual([ord(s) - REGION_B_BASE for s in symbols], list(text.encode("utf-16-le")))

    def test_routed_bytes_are_not_rewritten_by_merge_rules(self) -> None:
        # The UTF-16-LE bytes of this text spell the merge source "ning".
        text = "楮杮"
        self.assertEqual(text.encode("utf-16-le"), b"ning")
        self.assertEqual(self.myte.encode_bytes(text.encode("utf-8")), text.encode("utf-8"))
        symbols = self.codec.encode(text)
        self.assertTrue(all(REGION_B_BASE <= ord(s) < REGION_B_BASE + 256 for s in symbols))
        self.assertEqual(self.codec.decode(symbols), text)

    def test_region_order_is_preserved(self) -> None:
        symbols = "".join(
            (
                chr(REGION_A_BASE + ord("a")),
                *(chr(REGION_B_BASE + b) for b in "中".encode("utf-16-le")),
                chr(REGION_A_BASE + 0x80),
                *(chr(REGION_B_BASE + b) for b in "한".encode("utf-16-le")),
                chr(REGION_A_BASE + ord("b")),
            )
        )
        self.assertEqual(
            self.codec.post_myte_bytes(symbols),
            b"a" + "中".encode("utf-8") + b"\x80" + "한".encode("utf-8") + b"b",
        )

    def test_foreign_symbols_and_dangling_utf16_bytes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.codec.post_myte_bytes("plain text")
        with self.assertRaises(UnicodeDecodeError):
            self.codec.post_myte_bytes(chr(REGION_B_BASE + 0x6E))


@unittest.skipUnless(RESULTS.is_file(), "results/myte_ube_flores.json not present")
class CommittedResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results = json.loads(RESULTS.read_text(encoding="utf-8"))

    def test_every_sentence_is_verified(self) -> None:
        verification = self.results["verification"]
        self.assertEqual(self.results["dataset"]["sentences"], 206_448)
        for key in ("sentences_checked", "post_myte_stream_exact", "decoded_equals_released_myte_decoder"):
            self.assertEqual(verification[key], 206_448)
        self.assertEqual(self.results["myte"]["commit"], "177299ca7278a3618b8c77588e6dcd3912bad428")

    def test_table_values(self) -> None:
        expected = {
            "ube": (130.481, 91.252, 277.056, 176.088, 36_353_040),
            "utf8_bytes": (130.530, 133.665, 406.632, 196.106, 40_485_643),
            "myte_ube": (109.094, 94.610, 247.622, 141.354, 29_182_333),
            "myte": (109.140, 124.612, 349.398, 148.142, 30_583_555),
        }
        for method, (english, cjk, other_bmp, overall, total) in expected.items():
            with self.subTest(method=method):
                summary = self.results["summaries"][method]
                self.assertEqual(
                    tuple(round(summary[key], 3) for key in ("english", "cjk", "other_bmp", "overall")),
                    (english, cjk, other_bmp, overall),
                )
                self.assertEqual(summary["total_positions"], total)
                self.assertEqual((summary["cjk_configurations"], summary["other_bmp_configurations"]), (5, 7))

    def test_base_symbol_premium_gini(self) -> None:
        # prepare_flores.py validates 1,012 sentences per language,
        # so totals and means give the same premiums.
        dataset = self.results["dataset"]
        self.assertEqual((dataset["configurations"], dataset["sentences"]), (204, 204 * 1012))
        per_language = self.results["per_language"]
        expected = {"ube": "0.1500", "utf8_bytes": "0.2077", "myte_ube": "0.0925", "myte": "0.1177"}
        for method, value in expected.items():
            with self.subTest(method=method):
                english = per_language["eng_Latn"][method]
                totals = [counts[method] for language, counts in per_language.items() if language != "eng_Latn"]
                premiums = [total / english for total in totals]
                self.assertEqual(len(premiums), 203)
                # Independent pairwise population Gini; the common English divisor cancels.
                pairwise_gini = sum(abs(left - right) for left in totals for right in totals) / (
                    2 * len(totals) * sum(totals)
                )
                self.assertAlmostEqual(gini(premiums), pairwise_gini, places=12)
                self.assertEqual(f"{pairwise_gini:.4f}", value)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for row in ("| UBE | 512 | 130.481 | **91.252** | 277.056 | 176.088 | 36,353,040 | 0.1500 |",
                    "| UTF-8 bytes | 256 | 130.530 | 133.665 | 406.632 | 196.106 | 40,485,643 | 0.2077 |",
                    "| MYTE-UBE | 512 | **109.094** | 94.610 | **247.622** | **141.354** | **29,182,333** | **0.0925** |",
                    "| MYTE | 256 | 109.140 | 124.612 | 349.398 | 148.142 | 30,583,555 | 0.1177 |"):
            self.assertIn(row, readme)

    def test_comparison_with_myte(self) -> None:
        change = self.results["myte_ube_vs_myte"]
        self.assertEqual(round(change["cjk_change"] * 100, 2), -24.08)
        self.assertEqual(round(change["other_bmp_change"] * 100, 2), -29.13)
        self.assertEqual(round(change["total_positions_change"] * 100, 2), -4.58)
        self.assertEqual(
            (change["configurations_lower"], change["configurations_tied"], change["configurations_higher"]),
            (201, 3, 0),
        )


if __name__ == "__main__":
    unittest.main()
