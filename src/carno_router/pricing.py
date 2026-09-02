from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any


MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class ModelPrice:
    fixed_per_request: Decimal
    input_per_million_tokens: Decimal
    output_per_million_tokens: Decimal

    def estimate(self, input_tokens: int, output_tokens: int) -> Decimal:
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        return (
            self.fixed_per_request
            + self.input_per_million_tokens * Decimal(input_tokens) / MILLION
            + self.output_per_million_tokens * Decimal(output_tokens) / MILLION
        )


@dataclass(frozen=True)
class PricingSnapshot:
    snapshot_id: str
    currency: str
    claims_boundary: str
    models: dict[str, ModelPrice]

    @classmethod
    def load(cls, path: str | Path) -> "PricingSnapshot":
        payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != "CarnoPricingSnapshotV1":
            raise ValueError("unsupported pricing schema")
        raw_models = payload.get("models")
        if not isinstance(raw_models, dict) or len(raw_models) < 2:
            raise ValueError("pricing snapshot requires at least two models")
        models: dict[str, ModelPrice] = {}
        for model_id, raw in sorted(raw_models.items()):
            if not isinstance(raw, dict) or not str(model_id).strip():
                raise ValueError("invalid model price entry")
            try:
                price = ModelPrice(
                    fixed_per_request=Decimal(str(raw["fixed_per_request"])),
                    input_per_million_tokens=Decimal(str(raw["input_per_million_tokens"])),
                    output_per_million_tokens=Decimal(str(raw["output_per_million_tokens"])),
                )
            except (KeyError, ArithmeticError) as exc:
                raise ValueError(f"invalid price for {model_id}") from exc
            if min(
                price.fixed_per_request,
                price.input_per_million_tokens,
                price.output_per_million_tokens,
            ) < 0:
                raise ValueError("prices must be non-negative")
            models[str(model_id)] = price
        snapshot_id = str(payload.get("snapshot_id") or "").strip()
        currency = str(payload.get("currency") or "").strip()
        boundary = str(payload.get("claims_boundary") or "").strip()
        if not snapshot_id or not currency or not boundary:
            raise ValueError("pricing snapshot metadata is incomplete")
        return cls(snapshot_id, currency, boundary, models)

    def estimate(
        self, model_id: str, input_tokens: int, output_tokens: int
    ) -> Decimal:
        try:
            price = self.models[model_id]
        except KeyError as exc:
            raise ValueError(f"unknown model: {model_id}") from exc
        return price.estimate(input_tokens, output_tokens)
