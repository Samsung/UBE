"""MYTE byte rewriting with the released maps, as used by MYTE-UBE.

MYTE rewrites UTF-8 bytes in two greedy longest-prefix passes, a decomposition
pass and a morpheme-merge pass, with the maps released by its authors. The maps
are read from a checkout of https://github.com/tomlimi/MYTE at the commit below
and are not redistributed here; their SHA-256 digests are checked on load.
``myte_ube.py`` and ``tests/test_myte_ube.py`` check this implementation
against the released ``src/rewrite_bytes.py::ByteRewriter`` in both
directions.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

MYTE_REPOSITORY = "https://github.com/tomlimi/MYTE"
MYTE_COMMIT = "177299ca7278a3618b8c77588e6dcd3912bad428"
DECOMPOSE_SHA256 = "0188cb149c3b1d7ea49c895f8852147618ec97e19c439b500b8517bec2ccf88f"
MERGE_SHA256 = "f321bfd2381905bd95ae22842ec804cfa9f8a620baded3c073cd1d0642f93cb7"

DECOMPOSE_REL = Path("byte_maps/decompose_map.json")
MERGE_REL = Path("byte_maps/merge_map.json")

_LEAF = -1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_myte_source(myte_dir: Path, *, require_git_pin: bool = True) -> dict:
    """Check the MYTE checkout and return its provenance."""
    myte_dir = myte_dir.resolve()
    decompose_path = myte_dir / DECOMPOSE_REL
    merge_path = myte_dir / MERGE_REL
    for path in (decompose_path, merge_path):
        if not path.is_file():
            raise FileNotFoundError(f"MYTE map missing: {path}")

    actual_decompose = _sha256(decompose_path)
    actual_merge = _sha256(merge_path)
    if actual_decompose != DECOMPOSE_SHA256:
        raise RuntimeError(
            "MYTE decompose map hash mismatch: "
            f"expected {DECOMPOSE_SHA256}, got {actual_decompose}"
        )
    if actual_merge != MERGE_SHA256:
        raise RuntimeError(
            "MYTE merge map hash mismatch: "
            f"expected {MERGE_SHA256}, got {actual_merge}"
        )

    git_commit: str | None = None
    proc = subprocess.run(
        ["git", "-C", str(myte_dir), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        git_commit = proc.stdout.strip()
    if require_git_pin and git_commit != MYTE_COMMIT:
        raise RuntimeError(
            f"MYTE commit mismatch: expected {MYTE_COMMIT}, got {git_commit!r}"
        )

    return {
        "repository": MYTE_REPOSITORY,
        "commit": git_commit,
        "expected_commit": MYTE_COMMIT,
        "decompose_map_sha256": actual_decompose,
        "merge_map_sha256": actual_merge,
    }


class GreedyByteRewriter:
    """Longest-prefix byte rewriter compatible with MYTE's ByteRewriter."""

    def __init__(self, rules: dict[str, str], *, reverse: bool = False):
        if reverse:
            # Dict-comprehension collision behavior intentionally matches the
            # released implementation. In the decomposition map, 125 inputs share
            # their output with another input (mostly canonically equivalent
            # characters such as K and U+212A KELVIN SIGN), so its reverse is not
            # one-to-one.
            rules = {value: key for key, value in rules.items()}

        root: dict[int, dict] = {
            value: {_LEAF: bytes((value,))} for value in range(256)
        }
        for source, target in rules.items():
            node = root
            for value in bytes.fromhex(source):
                node = node.setdefault(value, {})
            node[_LEAF] = bytes.fromhex(target)
        self._root = root

    def rewrite(self, data: bytes) -> bytes:
        output = bytearray()
        start = 0
        data_len = len(data)
        while start < data_len:
            node = self._root
            best: bytes | None = None
            best_end = start
            cursor = start
            while cursor < data_len:
                node = node.get(data[cursor])
                if node is None:
                    break
                cursor += 1
                leaf = node.get(_LEAF)
                if leaf is not None:
                    best = leaf
                    best_end = cursor
            # Identity leaves for all 256 byte values make this unreachable.
            if best is None:
                raise AssertionError(f"MYTE trie has no identity byte at {start}")
            output.extend(best)
            start = best_end
        return bytes(output)


@dataclass
class MyteCodec:
    """Two-pass MYTE encoder, optionally with the released reverse mapping."""

    decompose: GreedyByteRewriter
    merge: GreedyByteRewriter
    reverse_decompose: GreedyByteRewriter | None
    reverse_merge: GreedyByteRewriter | None
    provenance: dict

    @classmethod
    def from_dir(
        cls,
        myte_dir: Path,
        *,
        with_reverse: bool = False,
        require_git_pin: bool = True,
    ) -> "MyteCodec":
        provenance = verify_myte_source(myte_dir, require_git_pin=require_git_pin)
        decompose_rules = json.loads((myte_dir / DECOMPOSE_REL).read_text(encoding="utf-8"))
        merge_rules = json.loads((myte_dir / MERGE_REL).read_text(encoding="utf-8"))
        return cls(
            decompose=GreedyByteRewriter(decompose_rules),
            merge=GreedyByteRewriter(merge_rules),
            reverse_decompose=(
                GreedyByteRewriter(decompose_rules, reverse=True) if with_reverse else None
            ),
            reverse_merge=(
                GreedyByteRewriter(merge_rules, reverse=True) if with_reverse else None
            ),
            provenance=provenance,
        )

    def encode_bytes(self, data: bytes) -> bytes:
        return self.merge.rewrite(self.decompose.rewrite(data))

    def decode_bytes(self, data: bytes) -> bytes:
        if self.reverse_merge is None or self.reverse_decompose is None:
            raise RuntimeError("codec was loaded without reverse maps")
        return self.reverse_decompose.rewrite(self.reverse_merge.rewrite(data))
