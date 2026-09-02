from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "secret_gate.py"
SPEC = importlib.util.spec_from_file_location("secret_gate", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SecretGateTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, bytes]:
        payload = (b"sealed synthetic receipt\n" * 240_000)
        artifact = root / "artifacts" / "large.jsonl"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(payload)
        return artifact, payload

    def test_large_file_requires_exact_review_binding_and_is_scanned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact, payload = self._fixture(root)
            config = root / "configs" / "secret_gate_reviewed_large_files.json"
            config.parent.mkdir(parents=True)
            config.write_text(
                json.dumps(
                    {
                        "schema": "CarnoReviewedLargeFilesV1",
                        "files": [
                            {
                                "path": str(artifact.relative_to(root)),
                                "bytes": len(payload),
                                "sha256": hashlib.sha256(payload).hexdigest(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(MODULE.scan(root), [])
            artifact.write_bytes(payload + b"drift\n")
            findings = MODULE.scan(root)
            self.assertIn("reviewed_large_file_drift", {item["rule"] for item in findings})

    def test_unreviewed_large_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            findings = MODULE.scan(root)
            self.assertIn("oversized_unreviewed_file", {item["rule"] for item in findings})


if __name__ == "__main__":
    unittest.main()

