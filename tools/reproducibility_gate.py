#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def main() -> int:
    data = ROOT / "tests" / "fixtures" / "synthetic_battles.csv"
    pricing = ROOT / "configs" / "normalized_costs.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        for output in (first, second):
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "carno_router",
                    "benchmark",
                    "--data",
                    str(data),
                    "--pricing",
                    str(pricing),
                    "--output",
                    output,
                    "--seed",
                    "1729",
                    "--bootstrap",
                    "200",
                    "--selection-bootstrap",
                    "200",
                    "--max-quality-drop",
                    "0.02",
                ],
                cwd=ROOT,
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        first_hashes = tree_hashes(Path(first))
        second_hashes = tree_hashes(Path(second))
        if first_hashes != second_hashes:
            raise SystemExit("FAIL: repeated benchmark outputs are not byte-identical")
        with data.open(encoding="utf-8", newline="") as handle:
            first_prompt = next(csv.DictReader(handle))["prompt"]
        report_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in Path(first).glob("report.*")
        )
        if first_prompt in report_text:
            raise SystemExit("FAIL: raw prompt leaked into report artifacts")
    with tempfile.TemporaryDirectory() as first_lab, tempfile.TemporaryDirectory() as second_lab:
        for output in (first_lab, second_lab):
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "build_research_autonomy_lab.py"),
                    "--output",
                    output,
                ],
                cwd=ROOT,
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        first_lab_hashes = tree_hashes(Path(first_lab))
        second_lab_hashes = tree_hashes(Path(second_lab))
        if first_lab_hashes != second_lab_hashes:
            raise SystemExit("FAIL: repeated research-autonomy outputs are not byte-identical")
        decision = (Path(first_lab) / "deployment_decision.json").read_text(encoding="utf-8")
        if '"live_agent_deployment": "BLOCKED"' not in decision:
            raise SystemExit("FAIL: research-autonomy fixture lost its deployment boundary")
    with tempfile.TemporaryDirectory() as first_stress, tempfile.TemporaryDirectory() as second_stress:
        for output in (first_stress, second_stress):
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "build_research_stress_lab.py"),
                    "--output",
                    output,
                    "--replicates",
                    "100",
                    "--root-seed",
                    "2831729",
                    "--omit-execution-receipt",
                ],
                cwd=ROOT,
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        first_stress_hashes = tree_hashes(Path(first_stress))
        second_stress_hashes = tree_hashes(Path(second_stress))
        if first_stress_hashes != second_stress_hashes:
            raise SystemExit("FAIL: repeated stress-benchmark outputs are not byte-identical")
        stress_evaluation = (Path(first_stress) / "stress_evaluation.json").read_text(encoding="utf-8")
        if '"stress_evaluator_status": "PASS"' not in stress_evaluation:
            raise SystemExit("FAIL: stress evaluator did not pass")
        oracle_hashes = []
        for stress_output in (first_stress, second_stress):
            oracle_output = Path(stress_output) / "oracle_audit"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "build_reference_oracle_audit.py"),
                    "--stress",
                    stress_output,
                    "--output",
                    str(oracle_output),
                    "--attacks-per-task",
                    "200",
                    "--root-seed",
                    "7071067",
                ],
                cwd=ROOT,
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            oracle_hashes.append(tree_hashes(oracle_output))
        if oracle_hashes[0] != oracle_hashes[1]:
            raise SystemExit("FAIL: repeated reference-oracle audits are not byte-identical")
        oracle_audit = (Path(first_stress) / "oracle_audit" / "reference_oracle_audit.json").read_text(encoding="utf-8")
        if '"status": "PASS"' not in oracle_audit or '"oracle_vs_production_disagreements": 0' not in oracle_audit:
            raise SystemExit("FAIL: independent reference oracle disagrees with the production grader")
        if '"human_grader_agreement": "NOT_RUN"' not in stress_evaluation:
            raise SystemExit("FAIL: stress benchmark manufactured a human-agreement claim")
        if '"actual_provider_accounting": "NOT_RUN"' not in stress_evaluation:
            raise SystemExit("FAIL: stress benchmark manufactured provider accounting")
    with tempfile.TemporaryDirectory() as first_aws, tempfile.TemporaryDirectory() as second_aws:
        aws_roots = (Path(first_aws), Path(second_aws))
        for aws_root in aws_roots:
            package_path = aws_root / "lambda.zip"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "build_aws_lambda_package.py"),
                    "--output",
                    str(package_path),
                ],
                cwd=ROOT,
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "validate_aws_shard_contract.py"),
                    "--package",
                    str(package_path),
                    "--output",
                    str(aws_root / "execution.json"),
                    "--core-output",
                    str(aws_root / "core.json"),
                ],
                cwd=ROOT,
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        if hashlib.sha256((aws_roots[0] / "lambda.zip").read_bytes()).hexdigest() != hashlib.sha256(
            (aws_roots[1] / "lambda.zip").read_bytes()
        ).hexdigest():
            raise SystemExit("FAIL: repeated AWS Lambda packages are not byte-identical")
        if hashlib.sha256((aws_roots[0] / "core.json").read_bytes()).hexdigest() != hashlib.sha256(
            (aws_roots[1] / "core.json").read_bytes()
        ).hexdigest():
            raise SystemExit("FAIL: repeated AWS shard-contract cores are not byte-identical")
    print(
        f"PASS: {len(first_hashes)} routing artifacts, {len(first_lab_hashes)} research-lab artifacts, "
        f"and {len(first_stress_hashes)} stress artifacts are byte-identical; reference-oracle audit byte-identical and in agreement; raw prompt absent; "
        "AWS package and 48-shard core are byte-identical; live deployment blocked; human/provider gates not run"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
