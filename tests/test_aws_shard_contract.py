from __future__ import annotations

import copy
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from carno_router.research_eval import reference_suite
from carno_router.research_stress import StressConfig, build_blinded_campaign, evaluate_blinded_campaign


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AwsShardContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.handler = _load(ROOT / "cloud" / "aws" / "lambda_function.py", "lambda_contract_test")
        cls.packager = _load(ROOT / "tools" / "build_aws_lambda_package.py", "lambda_package_test")
        cls.verifier = _load(ROOT / "tools" / "verify_aws_shard_responses.py", "lambda_verifier_test")
        suite = reference_suite()
        config = StressConfig(replicates=1, root_seed=2_831_729)
        traces, truth, _manifest = build_blinded_campaign(suite, config)
        canonical_records, canonical_evaluation = evaluate_blinded_campaign(suite, traces, truth, config)
        cls.canonical = {str(record["candidate_id"]): str(record["receipt_sha256"]) for record in canonical_records}
        cls.evaluation_hash = str(canonical_evaluation["content_sha256"])
        cls.responses = [
            cls.handler.lambda_handler(
                {
                    "schema": "CarnoAwsStressShardRequestV2",
                    "shard_index": shard_index,
                    "shard_count": 4,
                    "replicates": 1,
                    "root_seed": 2_831_729,
                    "canonical_evaluation_content_sha256": cls.evaluation_hash,
                },
                SimpleNamespace(aws_request_id="local", function_version="test"),
            )
            for shard_index in range(4)
        ]
        assembled = {
            str(item["candidate_id"]): str(item["receipt_sha256"])
            for response in cls.responses
            for item in response["deterministic"]["receipts"]
        }
        core = {
            "schema": "CarnoAwsLambdaShardContractCoreV2",
            "shard_count": 4,
            "canonical_evaluation_content_sha256": cls.evaluation_hash,
            "union_receipts_sha256": cls.verifier.content_sha256(sorted(assembled.items())),
            "expected_shards": [
                {
                    "shard_index": index,
                    "trial_count": response["deterministic"]["trial_count"],
                    "content_sha256": response["deterministic"]["content_sha256"],
                }
                for index, response in enumerate(cls.responses)
            ],
        }
        core["content_sha256"] = cls.verifier.content_sha256(core)
        cls.core = core

    def test_four_shards_are_disjoint_and_complete_for_small_campaign(self) -> None:
        observed: dict[str, str] = {}
        for response in self.responses:
            result = response["deterministic"]
            self.assertEqual(result["partition_evaluator_status"], "PASS")
            self.assertEqual(result["computation_scope"], "selected_partition_traces_only")
            self.assertEqual(result["human_grader_agreement"], "NOT_RUN")
            for item in result["receipts"]:
                self.assertNotIn(item["candidate_id"], observed)
                observed[item["candidate_id"]] = item["receipt_sha256"]
        self.assertEqual(observed, self.canonical)

    def test_every_returned_receipt_belongs_to_its_shard(self) -> None:
        for response in self.responses:
            result = response["deterministic"]
            for item in result["receipts"]:
                self.assertEqual(self.verifier.shard_for(str(item["candidate_id"]), 4), int(result["shard_index"]))

    def test_verifier_accepts_faithful_responses(self) -> None:
        verification = self.verifier.verify_responses(self.core, self.responses, self.canonical)
        self.assertEqual(verification["status"], "PASS")
        self.assertTrue(all(verification["gates"].values()))
        self.assertEqual(verification["reassembled_trial_count"], 24 * 12)

    def test_verifier_rejects_missing_repeated_tampered_and_misrouted_shards(self) -> None:
        missing = self.verifier.verify_responses(self.core, self.responses[:-1], self.canonical)
        self.assertEqual(missing["status"], "FAIL")
        self.assertEqual(missing["missing_shards"], [3])

        repeated = self.verifier.verify_responses(self.core, self.responses + [self.responses[0]], self.canonical)
        self.assertEqual(repeated["status"], "FAIL")
        self.assertEqual(repeated["repeated_shards"], [0])

        tampered = copy.deepcopy(self.responses)
        tampered[1]["deterministic"]["receipts"][0]["receipt_sha256"] = "0" * 64
        result = self.verifier.verify_responses(self.core, tampered, self.canonical)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["gates"]["every_shard_body_hash_recomputes"])
        self.assertFalse(result["gates"]["all_receipts_match_local"])

        misrouted = copy.deepcopy(self.responses)
        moved = misrouted[2]["deterministic"]["receipts"].pop()
        misrouted[0]["deterministic"]["receipts"].append(moved)
        for index in (0, 2):
            body = {key: value for key, value in misrouted[index]["deterministic"].items() if key != "content_sha256"}
            misrouted[index]["deterministic"]["content_sha256"] = self.verifier.content_sha256(body)
        result = self.verifier.verify_responses(self.core, misrouted, self.canonical)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["gates"]["every_receipt_belongs_to_its_shard"])
        self.assertFalse(result["gates"]["every_shard_hash_matches_frozen_expectation"])
        self.assertTrue(result["gates"]["every_shard_body_hash_recomputes"])

        echoed = copy.deepcopy(self.responses)
        echoed[3]["deterministic"]["receipts"] = []
        echoed[3]["deterministic"]["trial_count"] = 0
        body = {key: value for key, value in echoed[3]["deterministic"].items() if key != "content_sha256"}
        echoed[3]["deterministic"]["content_sha256"] = self.verifier.content_sha256(body)
        result = self.verifier.verify_responses(self.core, echoed, self.canonical)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["gates"]["no_missing_candidates"])
        self.assertFalse(result["gates"]["union_hash_matches_frozen_expectation"])

    def test_lambda_package_is_byte_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.zip"
            second = Path(directory) / "second.zip"
            first_receipt = self.packager.build(first)
            second_receipt = self.packager.build(second)
            self.assertEqual(first_receipt["sha256"], second_receipt["sha256"])
            self.assertEqual(hashlib.sha256(first.read_bytes()).digest(), hashlib.sha256(second.read_bytes()).digest())


if __name__ == "__main__":
    unittest.main()
