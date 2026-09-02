from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .calibration import PlattCalibrator
from .model import SparseLogisticRegression, TfidfVectorizer


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def payload_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def build_model_bundle(
    *,
    vectorizer: TfidfVectorizer,
    model: SparseLogisticRegression,
    calibrator: PlattCalibrator,
    model_fingerprint: str,
    benchmark_id: str,
    dataset_sha256: str,
    split_receipt: dict[str, object],
    threshold_selection: dict[str, object],
) -> dict[str, object]:
    """Return a provider-free, prompt-free scoring bundle."""

    payload: dict[str, object] = {
        "schema": "CarnoOfflineScoringBundleV1",
        "benchmark_id": benchmark_id,
        "model_fingerprint": model_fingerprint,
        "dataset_sha256": dataset_sha256,
        "split_strategy": split_receipt["strategy"],
        "thresholds": {
            "risk_controlled": threshold_selection["selected"]["threshold"],
            "point_estimate": threshold_selection["point_estimate"]["threshold"],
            "safety_recall_95": threshold_selection["safety_recall_95"]["threshold"],
            "balanced_accuracy": threshold_selection["balanced_accuracy"]["threshold"],
        },
        "feature_boundary": "normalized prompt text only",
        "provider_execution": False,
        "vectorizer": vectorizer.to_dict(),
        "model": model.to_dict(),
        "calibrator": calibrator.to_dict(),
    }
    payload["content_sha256"] = payload_sha256(payload)
    return payload


def load_model_bundle(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "CarnoOfflineScoringBundleV1":
        raise ValueError("unsupported Carno scoring bundle")
    expected = payload.get("content_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "content_sha256"}
    if expected != payload_sha256(unsigned):
        raise ValueError("scoring bundle content hash mismatch")
    return payload


def score_prompt(bundle: dict[str, object], prompt: str) -> float:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be non-empty text")
    vectorizer_payload = bundle.get("vectorizer")
    model_payload = bundle.get("model")
    calibrator_payload = bundle.get("calibrator")
    if not all(
        isinstance(payload, dict)
        for payload in (vectorizer_payload, model_payload, calibrator_payload)
    ):
        raise ValueError("scoring bundle components are invalid")
    vectorizer = TfidfVectorizer.from_dict(vectorizer_payload)  # type: ignore[arg-type]
    model = SparseLogisticRegression.from_dict(model_payload)  # type: ignore[arg-type]
    calibrator = PlattCalibrator.from_dict(calibrator_payload)  # type: ignore[arg-type]
    vectors = vectorizer.transform([prompt])
    return calibrator.predict(model.decision_function(vectors))[0]
