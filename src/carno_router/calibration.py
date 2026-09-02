from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from .model import sigmoid


@dataclass
class PlattCalibrator:
    epochs: int = 600
    learning_rate: float = 0.08
    l2: float = 0.001
    slope: float = 1.0
    intercept: float = 0.0
    constant: float | None = None

    def fit(self, logits: list[float], labels: list[int]) -> "PlattCalibrator":
        if len(logits) != len(labels) or not logits:
            raise ValueError("calibration inputs and labels must align")
        classes = set(labels)
        if not classes.issubset({0, 1}):
            raise ValueError("calibration labels must be binary")
        if len(classes) == 1:
            self.constant = (sum(labels) + 1.0) / (len(labels) + 2.0)
            self.slope = 0.0
            self.intercept = math.log(self.constant / (1.0 - self.constant))
            return self
        self.constant = None
        self.slope = 1.0
        prevalence = (sum(labels) + 0.5) / (len(labels) + 1.0)
        self.intercept = math.log(prevalence / (1.0 - prevalence))
        for epoch in range(self.epochs):
            gradient_slope = 0.0
            gradient_intercept = 0.0
            for logit, label in zip(logits, labels):
                error = sigmoid(self.slope * logit + self.intercept) - label
                gradient_slope += error * logit
                gradient_intercept += error
            rate = self.learning_rate / math.sqrt(1.0 + epoch / 30.0)
            scale = 1.0 / len(labels)
            self.slope -= rate * (
                gradient_slope * scale + self.l2 * self.slope
            )
            self.intercept -= rate * gradient_intercept * scale
        return self

    def predict(self, logits: list[float]) -> list[float]:
        if self.constant is not None:
            return [self.constant for _ in logits]
        return [sigmoid(self.slope * logit + self.intercept) for logit in logits]

    def fingerprint(self) -> str:
        payload = {
            "method": "platt_scaling",
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "slope": round(self.slope, 12),
            "intercept": round(self.intercept, 12),
            "constant": None if self.constant is None else round(self.constant, 12),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "CarnoPlattCalibratorV1",
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "slope": self.slope,
            "intercept": self.intercept,
            "constant": self.constant,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PlattCalibrator":
        if payload.get("schema") != "CarnoPlattCalibratorV1":
            raise ValueError("unsupported calibrator schema")
        calibrator = cls(
            epochs=int(payload["epochs"]),
            learning_rate=float(payload["learning_rate"]),
            l2=float(payload["l2"]),
        )
        calibrator.slope = float(payload["slope"])
        calibrator.intercept = float(payload["intercept"])
        constant = payload.get("constant")
        calibrator.constant = None if constant is None else float(constant)
        return calibrator
