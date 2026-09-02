#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROMPTS = (
    (
        "easy",
        "Rewrite this short note politely: please send the agenda before lunch.",
        0.96,
        0.96,
    ),
    (
        "lookup",
        "Give a one-sentence definition of kinetic energy for a high-school student.",
        0.92,
        0.90,
    ),
    (
        "calculation",
        "Calculate the uncertainty propagation for a product of three measured quantities and explain each term.",
        0.58,
        0.91,
    ),
    (
        "debugging",
        "Find the race condition in a concurrent queue consumer and propose a deterministic regression test.",
        0.32,
        0.88,
    ),
    (
        "specialist",
        "Explain how detector-response regularization changes bias and variance in an unfolding closure test.",
        0.42,
        0.94,
    ),
    (
        "ambiguous",
        "List two plausible interpretations of this underspecified request and state what evidence would distinguish them.",
        0.62,
        0.70,
    ),
)


FIELDS = [
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
    "cheap_latency_ms",
    "strong_latency_ms",
    "model_pair",
]


def generate(path: Path, groups: int = 60) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for group_index in range(groups):
            category, base_prompt, cheap_quality, strong_quality = PROMPTS[group_index % len(PROMPTS)]
            for variant in range(2):
                suffix = (
                    f" Context identifier {group_index}; provide a compact answer."
                    if variant == 0
                    else f" For case {group_index}, show the reasoning checks without revealing hidden chain of thought."
                )
                prompt = base_prompt + suffix
                word_count = len(prompt.split())
                timestamp = start + timedelta(days=group_index, minutes=variant)
                cheap_adjustment = 0.02 if (group_index + variant) % 11 == 0 else 0.0
                strong_adjustment = -0.03 if (group_index + 2 * variant) % 17 == 0 else 0.0
                writer.writerow(
                    {
                        "example_id": f"synthetic-{group_index:03d}-{variant}",
                        "group_id": f"synthetic-group-{group_index:03d}",
                        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                        "prompt": prompt,
                        "cheap_quality": f"{min(1.0, cheap_quality + cheap_adjustment):.3f}",
                        "strong_quality": f"{max(0.0, strong_quality + strong_adjustment):.3f}",
                        "cheap_input_tokens": word_count + 4,
                        "cheap_output_tokens": 48 + (group_index % 9),
                        "strong_input_tokens": word_count + 4,
                        "strong_output_tokens": 58 + (group_index % 11),
                        "cheap_latency_ms": 22 + (group_index % 7) * 2 + variant,
                        "strong_latency_ms": 118 + (group_index % 13) * 4 + 2 * variant,
                        "model_pair": "synthetic-cheap/synthetic-strong",
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "synthetic_battles.csv",
    )
    parser.add_argument("--groups", type=int, default=60)
    args = parser.parse_args()
    if args.groups < 12:
        parser.error("--groups must be at least 12")
    generate(args.output, args.groups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
