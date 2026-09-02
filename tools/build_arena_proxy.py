#!/usr/bin/env python3
"""Build a prompt-only Arena proxy cohort without reading assistant text as a feature."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_ARROW_SHA256 = "cbc3a4a44f8621bbca3f254274beed822e61fa9e2c8672d4b3e2a01b99dedaf1"
EXPECTED_SOURCE_ROWS = 33_000
EXPECTED_COHORT_ROWS = 576
EXPECTED_PROMPT_GROUPS = 570
STRONG = {"gpt-4", "claude-v1"}
WEAK = {"gpt-3.5-turbo", "claude-instant-v1"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_user_turns(conversation: list[dict[str, Any]]) -> str:
    turns = []
    for message in conversation:
        if message.get("role") == "user":
            content = unicodedata.normalize("NFKC", str(message.get("content") or ""))
            turns.append(re.sub(r"\s+", " ", content).strip())
    if not turns or any(not turn for turn in turns):
        raise ValueError("conversation lacks a non-empty user turn")
    return "\n<turn>\n".join(turns)


def nested(payload: Any, *keys: str, default: Any = None) -> Any:
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrow", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    if sha256_file(args.arrow) != EXPECTED_ARROW_SHA256:
        raise SystemExit("FAIL: Arena Arrow SHA-256 drift")
    try:
        import pyarrow.ipc as ipc  # type: ignore
    except ImportError as exc:
        raise SystemExit("pyarrow is required only for the optional Arena adapter") from exc
    with args.arrow.open("rb") as handle:
        table = ipc.open_stream(handle).read_all()
    if table.num_rows != EXPECTED_SOURCE_ROWS:
        raise SystemExit(f"FAIL: expected {EXPECTED_SOURCE_ROWS} source rows, got {table.num_rows}")

    output_rows: list[dict[str, object]] = []
    for index, row in enumerate(table.to_pylist()):
        model_a = str(row.get("model_a") or "")
        model_b = str(row.get("model_b") or "")
        if not ((model_a in STRONG and model_b in WEAK) or (model_b in STRONG and model_a in WEAK)):
            continue
        if row.get("winner") not in {"model_a", "model_b"} or row.get("language") != "English":
            continue
        moderation = row.get("openai_moderation") or {}
        toxic = row.get("toxic_chat_tag") or {}
        if bool(moderation.get("flagged")):
            continue
        if bool(nested(toxic, "roberta-large", "flagged", default=False)):
            continue
        if bool(nested(toxic, "t5-large", "flagged", default=False)):
            continue
        if float(nested(toxic, "t5-large", "score", default=0.0)) > 0.5:
            continue
        prompt_a = normalize_user_turns(row["conversation_a"])
        prompt_b = normalize_user_turns(row["conversation_b"])
        if prompt_a != prompt_b:
            raise SystemExit(f"FAIL: A/B user-turn mismatch at source row {index}")
        prompt_hash = hashlib.sha256(prompt_a.encode("utf-8")).hexdigest()
        winner_model = model_a if row["winner"] == "model_a" else model_b
        strong_model = model_a if model_a in STRONG else model_b
        weak_model = model_a if model_a in WEAK else model_b
        strong_wins = winner_model == strong_model
        timestamp = datetime.fromtimestamp(float(row["tstamp"]), tz=timezone.utc)
        output_rows.append(
            {
                "example_id": f"arena-{row['question_id']}-{index}",
                "group_id": prompt_hash,
                "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                "prompt": prompt_a,
                "cheap_quality": 0 if strong_wins else 1,
                "strong_quality": 1 if strong_wins else 0,
                "cheap_input_tokens": 0,
                "cheap_output_tokens": 0,
                "strong_input_tokens": 0,
                "strong_output_tokens": 0,
                "cheap_latency_ms": "",
                "strong_latency_ms": "",
                "model_pair": f"{strong_model}/{weak_model}",
            }
        )
    group_count = len({row["group_id"] for row in output_rows})
    if len(output_rows) != EXPECTED_COHORT_ROWS or group_count != EXPECTED_PROMPT_GROUPS:
        raise SystemExit(
            f"FAIL: expected {EXPECTED_COHORT_ROWS} rows/{EXPECTED_PROMPT_GROUPS} groups; "
            f"got {len(output_rows)}/{group_count}"
        )
    fields = list(output_rows[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    receipt = {
        "schema": "CarnoArenaProxyCohortReceiptV1",
        "source_sha256": EXPECTED_ARROW_SHA256,
        "source_rows": EXPECTED_SOURCE_ROWS,
        "cohort_rows": len(output_rows),
        "prompt_groups": group_count,
        "strong_wins": sum(int(row["strong_quality"]) for row in output_rows),
        "weak_wins": sum(int(row["cheap_quality"]) for row in output_rows),
        "feature": "normalized user turns only",
        "assistant_text_used": False,
        "network_access": False,
        "tier_policy": {"strong": sorted(STRONG), "weak": sorted(WEAK)},
        "toxicity_policy": {
            "openai_moderation_flagged": False,
            "roberta_large_flagged": False,
            "t5_large_flagged": False,
            "t5_large_max_score": 0.5
        },
        "claims_boundary": "Human-preference proxy under a frozen heuristic tier policy; not absolute model quality.",
        "output_sha256": sha256_file(args.output),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
