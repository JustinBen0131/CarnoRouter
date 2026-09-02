#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_files() -> list[tuple[Path, str]]:
    files = [(ROOT / "cloud" / "aws" / "lambda_function.py", "lambda_function.py")]
    files.extend(
        (path, str(path.relative_to(ROOT / "src")))
        for path in sorted((ROOT / "src" / "carno_router").glob("*.py"))
    )
    return sorted(files, key=lambda item: item[1])


def build(output: Path) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source, destination in package_files():
            info = zipfile.ZipInfo(destination, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(info, source.read_bytes())
    receipt: dict[str, object] = {
        "schema": "CarnoAwsLambdaPackageReceiptV1",
        "status": "PASS",
        "output": str(output.resolve()),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "file_count": len(package_files()),
        "handler": "lambda_function.lambda_handler",
        "runtime": "python3.13",
        "architecture": "arm64",
        "external_state_changed": False,
    }
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "aws" / "carno_the283_lambda.zip")
    args = parser.parse_args()
    receipt = build(args.output)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

