from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_COLUMNS = {
    "example_id",
    "group_id",
    "timestamp",
    "prompt",
    "cheap_quality",
    "strong_quality",
    "cheap_input_tokens",
    "cheap_output_tokens",
    "strong_input_tokens",
    "strong_output_tokens",
}
OPTIONAL_COLUMNS = {"cheap_latency_ms", "strong_latency_ms", "model_pair"}
FORBIDDEN_COLUMN_MARKERS = {
    "answer",
    "response",
    "completion",
    "winner",
    "provider_key",
    "api_key",
    "assistant_text",
}


def _parse_timestamp(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _bounded_float(value: str, name: str, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed) or not low <= parsed <= high:
        raise ValueError(f"{name} must be finite and in [{low},{high}]")
    return parsed


def _nonnegative_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _optional_nonnegative_float(value: str | None, name: str) -> float | None:
    if value is None or not value.strip():
        return None
    return _bounded_float(value, name, 0.0, float("inf"))


@dataclass(frozen=True)
class BenchmarkExample:
    example_id: str
    group_id: str
    timestamp: datetime
    prompt: str
    cheap_quality: float
    strong_quality: float
    cheap_input_tokens: int
    cheap_output_tokens: int
    strong_input_tokens: int
    strong_output_tokens: int
    cheap_latency_ms: float | None
    strong_latency_ms: float | None
    model_pair: str | None

    @property
    def strong_is_better(self) -> int:
        return int(self.strong_quality > self.cheap_quality)


@dataclass(frozen=True)
class SplitBundle:
    train: tuple[BenchmarkExample, ...]
    calibration: tuple[BenchmarkExample, ...]
    test: tuple[BenchmarkExample, ...]
    receipt: dict[str, object]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_examples(path: str | Path) -> tuple[BenchmarkExample, ...]:
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if not REQUIRED_COLUMNS.issubset(fields):
            missing = sorted(REQUIRED_COLUMNS - fields)
            raise ValueError("dataset is missing required columns: " + ", ".join(missing))
        forbidden = sorted(
            field
            for field in fields
            if any(marker in field.lower() for marker in FORBIDDEN_COLUMN_MARKERS)
        )
        if forbidden:
            raise ValueError(
                "dataset contains answer/label leakage columns: " + ", ".join(forbidden)
            )
        unknown = sorted(fields - REQUIRED_COLUMNS - OPTIONAL_COLUMNS)
        if unknown:
            raise ValueError("dataset contains unknown columns: " + ", ".join(unknown))

        rows: list[BenchmarkExample] = []
        seen_ids: set[str] = set()
        for line_number, raw in enumerate(reader, start=2):
            example_id = (raw.get("example_id") or "").strip()
            group_id = (raw.get("group_id") or "").strip()
            prompt = raw.get("prompt") or ""
            if not example_id or not group_id:
                raise ValueError(f"line {line_number}: identifiers must be non-empty")
            if example_id in seen_ids:
                raise ValueError(f"duplicate example_id: {example_id}")
            seen_ids.add(example_id)
            if not prompt.strip():
                raise ValueError(f"line {line_number}: prompt is empty")
            rows.append(
                BenchmarkExample(
                    example_id=example_id,
                    group_id=group_id,
                    timestamp=_parse_timestamp(raw["timestamp"]),
                    prompt=prompt,
                    cheap_quality=_bounded_float(
                        raw["cheap_quality"], "cheap_quality", 0.0, 1.0
                    ),
                    strong_quality=_bounded_float(
                        raw["strong_quality"], "strong_quality", 0.0, 1.0
                    ),
                    cheap_input_tokens=_nonnegative_int(
                        raw["cheap_input_tokens"], "cheap_input_tokens"
                    ),
                    cheap_output_tokens=_nonnegative_int(
                        raw["cheap_output_tokens"], "cheap_output_tokens"
                    ),
                    strong_input_tokens=_nonnegative_int(
                        raw["strong_input_tokens"], "strong_input_tokens"
                    ),
                    strong_output_tokens=_nonnegative_int(
                        raw["strong_output_tokens"], "strong_output_tokens"
                    ),
                    cheap_latency_ms=_optional_nonnegative_float(
                        raw.get("cheap_latency_ms"), "cheap_latency_ms"
                    ),
                    strong_latency_ms=_optional_nonnegative_float(
                        raw.get("strong_latency_ms"), "strong_latency_ms"
                    ),
                    model_pair=(raw.get("model_pair") or "").strip() or None,
                )
            )
    if len(rows) < 12:
        raise ValueError("benchmark requires at least 12 examples")
    return tuple(rows)


def group_time_split(
    rows: tuple[BenchmarkExample, ...],
    train_fraction: float = 0.70,
    calibration_fraction: float = 0.15,
) -> SplitBundle:
    if not 0 < train_fraction < 1 or not 0 < calibration_fraction < 1:
        raise ValueError("split fractions must be in (0,1)")
    if train_fraction + calibration_fraction >= 1:
        raise ValueError("train + calibration fractions must leave a test split")

    group_times: dict[str, datetime] = {}
    for row in rows:
        prior = group_times.get(row.group_id)
        group_times[row.group_id] = min(prior, row.timestamp) if prior else row.timestamp
    ordered_groups = sorted(group_times, key=lambda gid: (group_times[gid], gid))
    if len(ordered_groups) < 9:
        raise ValueError("benchmark requires at least nine distinct groups")

    train_end = max(1, int(len(ordered_groups) * train_fraction))
    calibration_end = max(
        train_end + 1,
        int(len(ordered_groups) * (train_fraction + calibration_fraction)),
    )
    calibration_end = min(calibration_end, len(ordered_groups) - 1)
    group_sets = {
        "train": set(ordered_groups[:train_end]),
        "calibration": set(ordered_groups[train_end:calibration_end]),
        "test": set(ordered_groups[calibration_end:]),
    }
    if any(
        group_sets[left] & group_sets[right]
        for left, right in (("train", "calibration"), ("train", "test"), ("calibration", "test"))
    ):
        raise RuntimeError("group leakage across splits")

    assigned: dict[str, list[BenchmarkExample]] = {name: [] for name in group_sets}
    for row in sorted(rows, key=lambda item: (item.timestamp, item.example_id)):
        matches = [name for name, groups in group_sets.items() if row.group_id in groups]
        if len(matches) != 1:
            raise RuntimeError("row group did not map to exactly one split")
        assigned[matches[0]].append(row)
    if any(not assigned[name] for name in assigned):
        raise RuntimeError("split unexpectedly contains no rows")

    receipt = {
        "schema": "CarnoGroupTimeSplitReceiptV1",
        "strategy": "earliest_group_timestamp",
        "fractions": {
            "train": train_fraction,
            "calibration": calibration_fraction,
            "test": 1.0 - train_fraction - calibration_fraction,
        },
        "row_counts": {name: len(values) for name, values in assigned.items()},
        "group_counts": {name: len(group_sets[name]) for name in group_sets},
        "group_overlap": 0,
        "feature_columns": ["prompt"],
        "prohibited_feature_classes": sorted(FORBIDDEN_COLUMN_MARKERS),
    }
    return SplitBundle(
        train=tuple(assigned["train"]),
        calibration=tuple(assigned["calibration"]),
        test=tuple(assigned["test"]),
        receipt=receipt,
    )
