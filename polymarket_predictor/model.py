from __future__ import annotations

from dataclasses import dataclass

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


def binary_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    probabilities = np.clip(probabilities, 1e-8, 1.0 - 1e-8)
    return float(-(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities)).mean())


def binary_accuracy(labels: np.ndarray, probabilities: np.ndarray) -> float:
    predictions = (probabilities >= 0.5).astype(int)
    return float((predictions == labels).mean())
