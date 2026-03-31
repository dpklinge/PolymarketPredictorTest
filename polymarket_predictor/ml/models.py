from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .model import GradientBoostedStumpModel, LogisticModel, sigmoid


def _safe_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


@dataclass
class ProbabilityCalibrator:
    slope: float = 1.0
    bias: float = 0.0
    sample_count: int = 0

    @classmethod
    def fit(cls, probabilities: np.ndarray, labels: np.ndarray, *, epochs: int = 400, learning_rate: float = 0.05) -> "ProbabilityCalibrator":
        if len(probabilities) < 8 or len(np.unique(labels)) < 2:
            return cls(sample_count=int(len(probabilities)))

        x = _safe_logit(np.asarray(probabilities, dtype=float))
        y = np.asarray(labels, dtype=float)
        slope = 1.0
        bias = 0.0

        for _ in range(epochs):
            logits = slope * x + bias
            predictions = sigmoid(logits)
            error = predictions - y
            grad_slope = float(np.mean(error * x))
            grad_bias = float(np.mean(error))
            slope -= learning_rate * grad_slope
            bias -= learning_rate * grad_bias

        return cls(slope=slope, bias=bias, sample_count=int(len(probabilities)))

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        base_probabilities = np.asarray(probabilities, dtype=float)
        logits = self.slope * _safe_logit(base_probabilities) + self.bias
        calibrated = sigmoid(logits)
        blend_weight = min(0.75, self.sample_count / 100.0)
        return (blend_weight * calibrated) + ((1.0 - blend_weight) * base_probabilities)

    def to_dict(self) -> dict[str, float]:
        return {"slope": self.slope, "bias": self.bias, "sample_count": self.sample_count}

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ProbabilityCalibrator":
        if not payload:
            return cls()
        return cls(
            slope=float(payload["slope"]),
            bias=float(payload["bias"]),
            sample_count=int(payload.get("sample_count", 0)),
        )


class ModelAdapter:
    model_type = "base"

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        calibration_features: np.ndarray | None = None,
        calibration_labels: np.ndarray | None = None,
    ) -> "ModelAdapter":
        raise NotImplementedError

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass
class LogisticAdapter(ModelAdapter):
    model: LogisticModel | None = None
    calibrator: ProbabilityCalibrator = field(default_factory=ProbabilityCalibrator)
    model_type = "logistic"

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        calibration_features: np.ndarray | None = None,
        calibration_labels: np.ndarray | None = None,
    ) -> "LogisticAdapter":
        self.model = LogisticModel.fit(features, labels)
        calibration_features = calibration_features if calibration_features is not None else features
        calibration_labels = calibration_labels if calibration_labels is not None else labels
        raw_probabilities = self.model.predict_proba(calibration_features)
        self.calibrator = ProbabilityCalibrator.fit(raw_probabilities, calibration_labels)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fit.")
        return self.calibrator.predict(self.model.predict_proba(features))

    def to_dict(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model has not been fit.")
        return {"model_type": self.model_type, "payload": self.model.to_dict(), "calibrator": self.calibrator.to_dict()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LogisticAdapter":
        return cls(
            model=LogisticModel.from_dict(payload["payload"]),
            calibrator=ProbabilityCalibrator.from_dict(payload.get("calibrator")),
        )


@dataclass
class PriorAdapter(ModelAdapter):
    positive_rate: float = 0.5
    calibrator: ProbabilityCalibrator = field(default_factory=ProbabilityCalibrator)
    model_type = "prior"

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        calibration_features: np.ndarray | None = None,
        calibration_labels: np.ndarray | None = None,
    ) -> "PriorAdapter":
        del features
        self.positive_rate = float(np.clip(labels.mean(), 1e-4, 1.0 - 1e-4))
        calibration_labels = calibration_labels if calibration_labels is not None else labels
        calibration_rows = len(calibration_labels)
        base_probabilities = np.full(calibration_rows, self.positive_rate, dtype=float)
        self.calibrator = ProbabilityCalibrator.fit(base_probabilities, calibration_labels)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        raw = np.full(features.shape[0], self.positive_rate, dtype=float)
        return self.calibrator.predict(raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "payload": {"positive_rate": self.positive_rate},
            "calibrator": self.calibrator.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PriorAdapter":
        return cls(
            positive_rate=float(payload["payload"]["positive_rate"]),
            calibrator=ProbabilityCalibrator.from_dict(payload.get("calibrator")),
        )


@dataclass
class BoostedTreesAdapter(ModelAdapter):
    model: GradientBoostedStumpModel | None = None
    calibrator: ProbabilityCalibrator = field(default_factory=ProbabilityCalibrator)
    model_type = "boosted_trees"

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        calibration_features: np.ndarray | None = None,
        calibration_labels: np.ndarray | None = None,
    ) -> "BoostedTreesAdapter":
        self.model = GradientBoostedStumpModel.fit(features, labels)
        calibration_features = calibration_features if calibration_features is not None else features
        calibration_labels = calibration_labels if calibration_labels is not None else labels
        raw_probabilities = self.model.predict_proba(calibration_features)
        self.calibrator = ProbabilityCalibrator.fit(raw_probabilities, calibration_labels)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fit.")
        return self.calibrator.predict(self.model.predict_proba(features))

    def to_dict(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model has not been fit.")
        return {"model_type": self.model_type, "payload": self.model.to_dict(), "calibrator": self.calibrator.to_dict()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BoostedTreesAdapter":
        return cls(
            model=GradientBoostedStumpModel.from_dict(payload["payload"]),
            calibrator=ProbabilityCalibrator.from_dict(payload.get("calibrator")),
        )


def create_model(model_type: str) -> ModelAdapter:
    normalized = model_type.strip().lower()
    if normalized == "logistic":
        return LogisticAdapter()
    if normalized == "boosted_trees":
        return BoostedTreesAdapter()
    if normalized == "prior":
        return PriorAdapter()
    raise ValueError(f"Unsupported model type: {model_type}")


def load_model(payload: dict[str, Any]) -> ModelAdapter:
    model_type = payload["model_type"]
    if model_type == "logistic":
        return LogisticAdapter.from_dict(payload)
    if model_type == "boosted_trees":
        return BoostedTreesAdapter.from_dict(payload)
    if model_type == "prior":
        return PriorAdapter.from_dict(payload)
    raise ValueError(f"Unsupported model type in bundle: {model_type}")
