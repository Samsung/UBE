# Copyright 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

"""The Unigram experiment runs within the repository and is documented and licensed with it."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
REPOSITORY = EXPERIMENT.parents[1]
sys.path.insert(0, str(EXPERIMENT / "scripts"))
sys.path.insert(0, str(REPOSITORY / "paper" / "scripts"))

import compare  # noqa: E402
import measure_flores  # noqa: E402
import sample_corpus  # noqa: E402

SCRIPTS = ("sample_corpus.py", "train_tokenizer.py", "compare.py")
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def readme_unigram_table() -> list[str]:
    lines = (REPOSITORY / "README.md").read_text(encoding="utf-8").splitlines()
    start = lines.index(next(line for line in lines if line.startswith("| UBE vs UTF-8 (FLORES-200 devtest)")))
    end = start
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    return lines[start:end]


class LayoutTest(unittest.TestCase):
    def test_default_paths(self) -> None:
        self.assertEqual(sample_corpus.DEFAULT_SHARES, EXPERIMENT / "mixture_shares.json")
        self.assertTrue(sample_corpus.DEFAULT_SHARES.is_file())
        self.assertEqual(measure_flores.DEFAULT_FLORES, REPOSITORY / "paper" / "data" / "flores200_devtest.json")

    def test_command_help_from_experiment_and_repository_roots(self) -> None:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        for script in SCRIPTS:
            for cwd in (EXPERIMENT, REPOSITORY):
                with self.subTest(script=script, cwd=cwd.name):
                    result = subprocess.run(
                        [sys.executable, str((EXPERIMENT / "scripts" / script).relative_to(cwd)), "--help"],
                        cwd=cwd,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=60,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout.lower())

    def test_files_carry_the_license_header(self) -> None:
        files = [EXPERIMENT / "README.md", EXPERIMENT / ".gitignore"]
        files += sorted((EXPERIMENT / "scripts").glob("*.py")) + sorted((EXPERIMENT / "tests").glob("*.py"))
        for path in files:
            with self.subTest(path=str(path.relative_to(REPOSITORY))):
                text = path.read_text(encoding="utf-8")
                self.assertIn("Copyright 2026 Samsung Electronics Co., Ltd.", text)
                self.assertIn("SPDX-License-Identifier: CC-BY-NC-SA-4.0", text)

    def test_notice_lists_the_experiments(self) -> None:
        notice = (REPOSITORY / "NOTICE").read_text(encoding="utf-8")
        self.assertIn("  experiments/ (all files)\n", notice)

    def test_documentation_links_resolve(self) -> None:
        for path in (EXPERIMENT / "README.md", REPOSITORY / "README.md"):
            text = path.read_text(encoding="utf-8")
            for target in LINK.findall(text):
                if target.startswith(("http://", "https://", "#")):
                    continue
                with self.subTest(path=str(path.relative_to(REPOSITORY)), target=target):
                    self.assertTrue((path.parent / target.split("#")[0]).exists())
        readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
        self.assertIn("(experiments/unigram/README.md)", readme)
        experiment = (EXPERIMENT / "README.md").read_text(encoding="utf-8")
        owners = set(re.findall(r"github\.com/([A-Za-z0-9_.-]+)/UBE(?![\w-])", experiment))
        self.assertEqual(owners, {"Samsung"})

    def test_readme_table_is_the_comparison_layout(self) -> None:
        # Rendering the README's own values reproduces the README table exactly.
        table = readme_unigram_table()
        rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in table[2:]]
        self.assertEqual(len(rows), 4)

        def value(cell: str) -> float:
            return float(cell.replace(compare.MINUS, "-").rstrip("%"))

        settings = [
            {
                "label": row[0],
                "unigram": {
                    "group_mean_tokens": {"change_pct": value(row[1])},
                    "english_mean_tokens": {"change_pct": value(row[3])},
                    "gini": {"change_pct": value(row[4])},
                },
                "bpe": {"group_mean_tokens": {"change_pct": value(row[2])}},
            }
            for row in rows
        ]
        self.assertEqual(compare.readme_table(settings, 39).splitlines(), table)
        self.assertIn("\n".join(table) + "\n", (EXPERIMENT / "README.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
