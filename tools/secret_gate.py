#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


FORBIDDEN_NAMES = {".env", "id_rsa", "id_ed25519"}
FORBIDDEN_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".sql", ".sqlite", ".db"}
CONTENT_RULES = {
    "private_key_header": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    "openai_style_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "assigned_secret": re.compile(
        r"(?i)\b(?:api[_-]?key|secret|password|token)\b\s*[:=]\s*['\"][A-Za-z0-9_./+=-]{12,}['\"]"
    ),
    "credentialed_database_url": re.compile(r"(?i)\b(?:postgres|mysql|mongodb)(?:ql)?://[^\s:@]+:[^\s@]+@"),
}
SKIP_PARTS = {".git", "__pycache__", ".venv", "build", "dist"}
REVIEWED_LARGE_FILE_CONFIG = Path("configs/secret_gate_reviewed_large_files.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reviewed_large_files(root: Path) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    config_path = root / REVIEWED_LARGE_FILE_CONFIG
    if not config_path.is_file():
        return {}, []
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}, [{"path": str(REVIEWED_LARGE_FILE_CONFIG), "rule": "invalid_large_file_review_config"}]
    if payload.get("schema") != "CarnoReviewedLargeFilesV1" or not isinstance(payload.get("files"), list):
        return {}, [{"path": str(REVIEWED_LARGE_FILE_CONFIG), "rule": "invalid_large_file_review_config"}]
    reviewed: dict[str, dict[str, object]] = {}
    findings: list[dict[str, object]] = []
    for entry in payload["files"]:
        path = entry.get("path") if isinstance(entry, dict) else None
        digest = entry.get("sha256") if isinstance(entry, dict) else None
        size = entry.get("bytes") if isinstance(entry, dict) else None
        valid = (
            isinstance(path, str)
            and path
            and not Path(path).is_absolute()
            and ".." not in Path(path).parts
            and isinstance(digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
            and isinstance(size, int)
            and size > 5 * 1024 * 1024
            and path not in reviewed
        )
        if not valid:
            findings.append({"path": str(REVIEWED_LARGE_FILE_CONFIG), "rule": "invalid_large_file_review_entry"})
            continue
        reviewed[path] = {"sha256": digest, "bytes": size}
    return reviewed, findings


def _scan_text(path: Path, relative: str, findings: list[dict[str, object]]) -> None:
    matches = {rule: [] for rule in CONTENT_RULES}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                for rule, pattern in CONTENT_RULES.items():
                    if pattern.search(line):
                        matches[rule].append(line_number)
    except UnicodeDecodeError:
        findings.append({"path": relative, "rule": "binary_file_requires_review"})
        return
    for rule, lines in matches.items():
        if lines:
            findings.append({"path": relative, "rule": rule, "lines": sorted(set(lines))})


def scan(root: Path) -> list[dict[str, object]]:
    reviewed, findings = _reviewed_large_files(root)
    observed_reviewed: set[str] = set()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        relative = str(path.relative_to(root))
        if path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            if path.name != ".env.example":
                findings.append({"path": relative, "rule": "forbidden_filename"})
        if path.stat().st_size > 5 * 1024 * 1024:
            review = reviewed.get(relative)
            if review is None:
                findings.append({"path": relative, "rule": "oversized_unreviewed_file"})
                continue
            observed = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            if observed != review:
                findings.append(
                    {
                        "path": relative,
                        "rule": "reviewed_large_file_drift",
                        "observed": observed,
                        "expected": review,
                    }
                )
                continue
            observed_reviewed.add(relative)
            _scan_text(path, relative, findings)
            continue
        _scan_text(path, relative, findings)
    for relative in sorted(set(reviewed) - observed_reviewed):
        findings.append({"path": relative, "rule": "reviewed_large_file_missing_or_not_large"})
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    findings = scan(root)
    payload = {
        "schema": "CarnoSecretGateV1",
        "status": "PASS" if not findings else "FAIL",
        "root": root.name,
        "finding_count": len(findings),
        "findings": findings,
        "large_file_review_config": str(REVIEWED_LARGE_FILE_CONFIG),
        "matched_values_disclosed": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
