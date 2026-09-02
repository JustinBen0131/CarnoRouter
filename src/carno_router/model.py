from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
SparseVector = dict[int, float]


def _features(
    text: str,
    *,
    word_ngram_max: int,
    char_ngram_min: int | None,
    char_ngram_max: int | None,
) -> list[str]:
    words = [token.lower() for token in TOKEN_RE.findall(text)]
    unigrams = [f"w:{word}" for word in words]
    word_ngrams: list[str] = []
    for width in range(2, word_ngram_max + 1):
        word_ngrams.extend(
            f"w{width}:" + "_".join(words[start : start + width])
            for start in range(max(0, len(words) - width + 1))
        )
    character_ngrams: list[str] = []
    if char_ngram_min is not None and char_ngram_max is not None:
        for word in words:
            padded = f" {word} "
            for width in range(char_ngram_min, char_ngram_max + 1):
                character_ngrams.extend(
                    f"c{width}:{padded[start:start + width]}"
                    for start in range(max(0, len(padded) - width + 1))
                )
    structural = [
        f"len_bucket:{min(len(words) // 16, 12)}",
        f"char_bucket:{min(len(text) // 80, 12)}",
        f"question_marks:{min(text.count('?'), 3)}",
    ]
    return unigrams + word_ngrams + character_ngrams + structural


@dataclass
class TfidfVectorizer:
    max_features: int = 4096
    word_ngram_max: int = 2
    char_ngram_min: int | None = 3
    char_ngram_max: int | None = 5
    vocabulary: dict[str, int] | None = None
    idf: list[float] | None = None

    def __post_init__(self) -> None:
        if self.max_features < 1:
            raise ValueError("max_features must be positive")
        if not 1 <= self.word_ngram_max <= 3:
            raise ValueError("word_ngram_max must be in [1,3]")
        if (self.char_ngram_min is None) != (self.char_ngram_max is None):
            raise ValueError("character n-gram bounds must both be present or absent")
        if self.char_ngram_min is not None:
            if not 2 <= self.char_ngram_min <= int(self.char_ngram_max or 0) <= 6:
                raise ValueError("character n-gram bounds must satisfy 2 <= min <= max <= 6")

    def _features(self, document: str) -> list[str]:
        return _features(
            document,
            word_ngram_max=self.word_ngram_max,
            char_ngram_min=self.char_ngram_min,
            char_ngram_max=self.char_ngram_max,
        )

    def fit(self, documents: list[str]) -> "TfidfVectorizer":
        if not documents:
            raise ValueError("vectorizer requires training documents")
        document_frequency: Counter[str] = Counter()
        for document in documents:
            document_frequency.update(set(self._features(document)))
        selected = sorted(
            document_frequency,
            key=lambda token: (-document_frequency[token], token),
        )[: self.max_features]
        self.vocabulary = {token: index for index, token in enumerate(selected)}
        self.idf = [
            math.log((1.0 + len(documents)) / (1.0 + document_frequency[token])) + 1.0
            for token in selected
        ]
        return self

    def transform_one(self, document: str) -> SparseVector:
        if self.vocabulary is None or self.idf is None:
            raise RuntimeError("vectorizer is not fitted")
        counts = Counter(self._features(document))
        vector: SparseVector = {}
        for token, count in counts.items():
            index = self.vocabulary.get(token)
            if index is not None:
                vector[index] = (1.0 + math.log(count)) * self.idf[index]
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm:
            vector = {index: value / norm for index, value in vector.items()}
        return vector

    def transform(self, documents: list[str]) -> list[SparseVector]:
        return [self.transform_one(document) for document in documents]

    def to_dict(self) -> dict[str, object]:
        if self.vocabulary is None or self.idf is None:
            raise RuntimeError("vectorizer is not fitted")
        return {
            "schema": "CarnoTfidfVectorizerV1",
            "max_features": self.max_features,
            "word_ngram_max": self.word_ngram_max,
            "char_ngram_min": self.char_ngram_min,
            "char_ngram_max": self.char_ngram_max,
            "vocabulary": self.vocabulary,
            "idf": self.idf,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TfidfVectorizer":
        if payload.get("schema") != "CarnoTfidfVectorizerV1":
            raise ValueError("unsupported vectorizer schema")
        raw_vocabulary = payload.get("vocabulary")
        raw_idf = payload.get("idf")
        if not isinstance(raw_vocabulary, dict) or not isinstance(raw_idf, list):
            raise ValueError("invalid vectorizer payload")
        vectorizer = cls(
            max_features=int(payload["max_features"]),
            word_ngram_max=int(payload["word_ngram_max"]),
            char_ngram_min=(
                None if payload.get("char_ngram_min") is None else int(payload["char_ngram_min"])
            ),
            char_ngram_max=(
                None if payload.get("char_ngram_max") is None else int(payload["char_ngram_max"])
            ),
        )
        vectorizer.vocabulary = {str(token): int(index) for token, index in raw_vocabulary.items()}
        vectorizer.idf = [float(value) for value in raw_idf]
        if len(vectorizer.vocabulary) != len(vectorizer.idf):
            raise ValueError("vectorizer vocabulary and idf lengths differ")
        return vectorizer


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponent = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(max(value, -700.0))
    return exponent / (1.0 + exponent)


@dataclass
class SparseLogisticRegression:
    epochs: int = 160
    learning_rate: float = 0.8
    l2: float = 0.02
    weights: list[float] | None = None
    bias: float = 0.0

    def fit(
        self, vectors: list[SparseVector], labels: list[int], feature_count: int
    ) -> "SparseLogisticRegression":
        if len(vectors) != len(labels) or not vectors:
            raise ValueError("model inputs and labels must be non-empty and aligned")
        if set(labels) != {0, 1}:
            raise ValueError("training labels must contain both classes")
        positives = sum(labels)
        negatives = len(labels) - positives
        class_weight = {
            0: len(labels) / (2.0 * negatives),
            1: len(labels) / (2.0 * positives),
        }
        self.weights = [0.0] * feature_count
        self.bias = math.log((positives + 0.5) / (negatives + 0.5))
        for epoch in range(self.epochs):
            gradient: dict[int, float] = {}
            bias_gradient = 0.0
            for vector, label in zip(vectors, labels):
                logit = self.bias + sum(
                    self.weights[index] * value for index, value in vector.items()
                )
                error = (_sigmoid(logit) - label) * class_weight[label]
                bias_gradient += error
                for index, value in vector.items():
                    gradient[index] = gradient.get(index, 0.0) + error * value
            rate = self.learning_rate / math.sqrt(1.0 + epoch / 12.0)
            scale = 1.0 / len(labels)
            for index in range(feature_count):
                data_gradient = gradient.get(index, 0.0) * scale
                self.weights[index] -= rate * (
                    data_gradient + self.l2 * self.weights[index]
                )
            self.bias -= rate * bias_gradient * scale
        return self

    def decision_function(self, vectors: list[SparseVector]) -> list[float]:
        if self.weights is None:
            raise RuntimeError("model is not fitted")
        return [
            self.bias
            + sum(self.weights[index] * value for index, value in vector.items())
            for vector in vectors
        ]

    def fingerprint(self, vectorizer: TfidfVectorizer) -> str:
        if self.weights is None or vectorizer.vocabulary is None or vectorizer.idf is None:
            raise RuntimeError("model and vectorizer must be fitted")
        payload = {
            "vocabulary": sorted(vectorizer.vocabulary.items()),
            "idf": [round(value, 12) for value in vectorizer.idf],
            "weights": [round(value, 12) for value in self.weights],
            "bias": round(self.bias, 12),
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "vectorizer": {
                "max_features": vectorizer.max_features,
                "word_ngram_max": vectorizer.word_ngram_max,
                "char_ngram_min": vectorizer.char_ngram_min,
                "char_ngram_max": vectorizer.char_ngram_max,
            },
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, object]:
        if self.weights is None:
            raise RuntimeError("model is not fitted")
        return {
            "schema": "CarnoSparseLogisticRegressionV1",
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "weights": self.weights,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SparseLogisticRegression":
        if payload.get("schema") != "CarnoSparseLogisticRegressionV1":
            raise ValueError("unsupported model schema")
        raw_weights = payload.get("weights")
        if not isinstance(raw_weights, list):
            raise ValueError("invalid model weights")
        model = cls(
            epochs=int(payload["epochs"]),
            learning_rate=float(payload["learning_rate"]),
            l2=float(payload["l2"]),
        )
        model.weights = [float(value) for value in raw_weights]
        model.bias = float(payload["bias"])
        return model


def sigmoid(value: float) -> float:
    return _sigmoid(value)
