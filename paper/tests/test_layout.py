"""The paper analyses run within the unified tokenizer repository."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
REPOSITORY = PAPER.parent
sys.path.insert(0, str(PAPER / "scripts"))

import measure_flores  # noqa: E402
import myte_ube  # noqa: E402
import prepare_flores  # noqa: E402
import unicode_audit  # noqa: E402


class LayoutTest(unittest.TestCase):
    def test_default_paths_stay_inside_paper(self) -> None:
        flores = PAPER / "data" / "flores200_devtest.json"
        self.assertEqual(prepare_flores.DEFAULT_OUTPUT, flores)
        self.assertEqual(measure_flores.DEFAULT_FLORES, flores)
        self.assertEqual(myte_ube.DEFAULT_FLORES, flores)
        self.assertEqual(unicode_audit.DEFAULT_TRAIN_TEXT, flores)
        self.assertEqual(unicode_audit.DEFAULT_CACHE, PAPER / "data" / "unicode_17_0_0")
        self.assertEqual(myte_ube.DEFAULT_OUTPUT, PAPER / "results" / "myte_ube_flores.json")
        self.assertEqual(unicode_audit.DEFAULT_OUTPUT, PAPER / "results" / "unicode_audit.json")
        self.assertEqual(unicode_audit.DEFAULT_REPORT, PAPER / "results" / "unicode_audit.md")

    def test_documentation_uses_one_code_repository(self) -> None:
        # Every link to this project's repository names the Samsung organization, under
        # the repository's current name.
        owner = re.compile(r"github\.com/([A-Za-z0-9_.-]+)/UBE(?![\w-])")
        for path in (
            REPOSITORY / "README.md",
            REPOSITORY / "CITATION.cff",
            REPOSITORY / "bindings" / "python" / "README.md",
            REPOSITORY / "v1" / "README.md",
            REPOSITORY / "v1" / "bindings" / "python" / "README.md",
            PAPER / "README.md",
        ):
            with self.subTest(path=str(path.relative_to(REPOSITORY))):
                text = path.read_text(encoding="utf-8")
                owners = owner.findall(text)
                self.assertTrue(owners)
                self.assertNotIn("universal-byte-level-encoding", text)
                self.assertEqual(set(owners), {"Samsung"})
        self.assertIn(
            "paper/README.md", (REPOSITORY / "README.md").read_text(encoding="utf-8")
        )
        self.assertIn(
            "https://github.com/Samsung/UBE",
            (REPOSITORY / "CITATION.cff").read_text(encoding="utf-8"),
        )

    def test_local_install_inputs_and_results_exist(self) -> None:
        for relative in (
            "bindings/python/pyproject.toml",
            "LICENSE",
            "paper/requirements.txt",
            "paper/results/myte_ube_flores.json",
            "paper/results/unicode_audit.json",
            "paper/results/unicode_audit.md",
        ):
            with self.subTest(path=relative):
                self.assertTrue((REPOSITORY / relative).is_file())
        self.assertFalse((PAPER / ".git").exists())

    def test_command_help_from_paper_and_repository_roots(self) -> None:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        for script in (
            "prepare_flores.py",
            "measure_flores.py",
            "myte_ube.py",
            "unicode_audit.py",
        ):
            for cwd in (PAPER, REPOSITORY):
                with self.subTest(script=script, cwd=cwd.name):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str((PAPER / "scripts" / script).relative_to(cwd)),
                            "--help",
                        ],
                        cwd=cwd,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
