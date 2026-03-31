from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class StandardScaler:
    mean_: np.ndarray
    scale_: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "StandardScaler":
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1e-8] = 1.0
        return cls(mean_=mean, scale_=scale)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (features - self.mean_) / self.scale_


@dataclass
class LogisticModel:
    scaler: StandardScaler
    weights: np.ndarray
    bias: float

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        learning_rate: float = 0.05,
        epochs: int = 800,
        l2_penalty: float = 0.005,
    ) -> "LogisticModel":
        scaler = StandardScaler.fit(features)
        x = scaler.transform(features)
        y = labels.astype(float)

        weights = np.zeros(x.shape[1], dtype=float)
        bias = 0.0

        positive_rate = float(np.clip(y.mean(), 1e-4, 1.0 - 1e-4))
        positive_weight = 0.5 / positive_rate
        negative_weight = 0.5 / (1.0 - positive_rate)

        sample_weights = np.where(y > 0.5, positive_weight, negative_weight)

        for _ in range(epochs):
            logits = x @ weights + bias
            predictions = sigmoid(logits)
            error = (predictions - y) * sample_weights

            grad_w = (x.T @ error) / len(x) + l2_penalty * weights
            grad_b = float(error.mean())

            weights -= learning_rate * grad_w
            bias -= learning_rate * grad_b

        return cls(scaler=scaler, weights=weights, bias=bias)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = self.scaler.transform(features)
        return sigmoid(x @ self.weights + self.bias)

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "LogisticModel":
        scaler = StandardScaler(
            mean_=np.asarray(payload["mean"], dtype=float),
            scale_=np.asarray(payload["scale"], dtype=float),
        )
        return cls(
            scaler=scaler,
            weights=np.asarray(payload["weights"], dtype=float),
            bias=float(payload["bias"]),
        )


@dataclass
class DecisionStump:
    feature_index: int
    threshold: float
    left_value: float
    right_value: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        mask = features[:, self.feature_index] <= self.threshold
        return np.where(mask, self.left_value, self.right_value)

    def to_dict(self) -> dict[str, float]:
        return {
            "feature_index": self.feature_index,
            "threshold": self.threshold,
            "left_value": self.left_value,
            "right_value": self.right_value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DecisionStump":
        return cls(
            feature_index=int(payload["feature_index"]),
            threshold=float(payload["threshold"]),
            left_value=float(payload["left_value"]),
            right_value=float(payload["right_value"]),
        )


@dataclass
class GradientBoostedStumpModel:
    scaler: StandardScaler
    base_score: float
    learning_rate: float
    stumps: list[DecisionStump]

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        n_estimators: int = 60,
        learning_rate: float = 0.15,
        max_bins: int = 16,
        min_leaf_size: int = 8,
    ) -> "GradientBoostedStumpModel":
        scaler = StandardScaler.fit(features)
        x = scaler.transform(features)
        y = labels.astype(float)

        positive_rate = float(np.clip(y.mean(), 1e-4, 1.0 - 1e-4))
        base_score = float(np.log(positive_rate / (1.0 - positive_rate)))
        logits = np.full(len(x), base_score, dtype=float)
        stumps: list[DecisionStump] = []

        for _ in range(n_estimators):
            probabilities = sigmoid(logits)
            residuals = y - probabilities
            stump = _fit_best_stump(x, residuals, max_bins=max_bins, min_leaf_size=min_leaf_size)
            if stump is None:
                break
            logits += learning_rate * stump.predict(x)
            stumps.append(stump)

        return cls(
            scaler=scaler,
            base_score=base_score,
            learning_rate=learning_rate,
            stumps=stumps,
        )

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = self.scaler.transform(features)
        logits = np.full(len(x), self.base_score, dtype=float)
        for stump in self.stumps:
            logits += self.learning_rate * stump.predict(x)
        return sigmoid(logits)

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist(),
            "base_score": self.base_score,
            "learning_rate": self.learning_rate,
            "stumps": [stump.to_dict() for stump in self.stumps],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "GradientBoostedStumpModel":
        scaler = StandardScaler(
            mean_=np.asarray(payload["mean"], dtype=float),
            scale_=np.asarray(payload["scale"], dtype=float),
        )
        return cls(
            scaler=scaler,
            base_score=float(payload["base_score"]),
            learning_rate=float(payload["learning_rate"]),
            stumps=[DecisionStump.from_dict(item) for item in payload["stumps"]],
        )


def _fit_best_stump(
    features: np.ndarray,
    residuals: np.ndarray,
    *,
    max_bins: int,
    min_leaf_size: int,
) -> DecisionStump | None:
    best_stump: DecisionStump | None = None
    best_error = float("inf")

    for feature_index in range(features.shape[1]):
        column = features[:, feature_index]
        unique_values = np.unique(column)
        if len(unique_values) <= 1:
            continue
        effective_min_leaf_size = max(1, min(min_leaf_size, len(column) // 3))

        if len(unique_values) > max_bins:
            quantiles = np.linspace(0.1, 0.9, max_bins)
            thresholds = np.unique(np.quantile(unique_values, quantiles))
        else:
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

        for threshold in thresholds:
            left_mask = column <= threshold
            right_mask = ~left_mask
            if left_mask.sum() < effective_min_leaf_size or right_mask.sum() < effective_min_leaf_size:
                continue

            left_value = float(np.mean(residuals[left_mask]))
            right_value = float(np.mean(residuals[right_mask]))
            predictions = np.where(left_mask, left_value, right_value)
            error = float(np.mean((residuals - predictions) ** 2))

            if error < best_error:
                best_error = error
                best_stump = DecisionStump(
                    feature_index=feature_index,
                    threshold=float(threshold),
                    left_value=left_value,
                    right_value=right_value,
                )

    return best_stump


def binary_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    probabilities = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
    return float(-(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities)).mean())


def binary_accuracy(labels: np.ndarray, probabilities: np.ndarray) -> float:
    predictions = (probabilities >= 0.5).astype(int)
    return float((predictions == labels).mean())
