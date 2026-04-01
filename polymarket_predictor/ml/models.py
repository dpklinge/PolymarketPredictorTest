from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression as _CalibLR

from .model import GradientBoostedStumpModel, LogisticModel


def _safe_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


@dataclass
class ProbabilityCalibrator:
    slope: float = 1.0
    bias: float = 0.0
    sample_count: int = 0

    @classmethod
    def fit(cls, probabilities: np.ndarray, labels: np.ndarray) -> "ProbabilityCalibrator":
        if len(probabilities) < 8 or len(np.unique(labels)) < 2:
            return cls(sample_count=int(len(probabilities)))

        x = _safe_logit(np.asarray(probabilities, dtype=float)).reshape(-1, 1)
        y = np.asarray(labels, dtype=int)
        clf = _CalibLR(C=1e6, solver="lbfgs", random_state=0, max_iter=1000)
        clf.fit(x, y)
        return cls(slope=float(clf.coef_[0][0]), bias=float(clf.intercept_[0]), sample_count=int(len(probabilities)))

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        base_probabilities = np.asarray(probabilities, dtype=float)
        calibrated = _sigmoid(self.slope * _safe_logit(base_probabilities) + self.bias)
        blend_weight = min(0.75, self.sample_count / 100.0)
        return (blend_weight * calibrated) + ((1.0 - blend_weight) * base_probabilities)


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
        cal_features = calibration_features if calibration_features is not None else features
        cal_labels = calibration_labels if calibration_labels is not None else labels
        raw_probabilities = self.model.predict_proba(cal_features)
        self.calibrator = ProbabilityCalibrator.fit(raw_probabilities, cal_labels)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fit.")
        return self.calibrator.predict(self.model.predict_proba(features))


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
        cal_labels = calibration_labels if calibration_labels is not None else labels
        base_probabilities = np.full(len(cal_labels), self.positive_rate, dtype=float)
        self.calibrator = ProbabilityCalibrator.fit(base_probabilities, cal_labels)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        raw = np.full(features.shape[0], self.positive_rate, dtype=float)
        return self.calibrator.predict(raw)


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
        cal_features = calibration_features if calibration_features is not None else features
        cal_labels = calibration_labels if calibration_labels is not None else labels
        raw_probabilities = self.model.predict_proba(cal_features)
        self.calibrator = ProbabilityCalibrator.fit(raw_probabilities, cal_labels)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fit.")
        return self.calibrator.predict(self.model.predict_proba(features))


def create_model(model_type: str) -> ModelAdapter:
    normalized = model_type.strip().lower()
    if normalized == "logistic":
        return LogisticAdapter()
    if normalized == "boosted_trees":
        return BoostedTreesAdapter()
    if normalized == "prior":
        return PriorAdapter()
    raise ValueError(f"Unsupported model type: {model_type}")
