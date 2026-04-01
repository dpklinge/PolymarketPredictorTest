from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as _GBT
from sklearn.linear_model import LogisticRegression as _LR
from sklearn.preprocessing import StandardScaler as _Scaler


class LogisticModel:
    def __init__(self, scaler: _Scaler, clf: _LR) -> None:
        self._scaler = scaler
        self._clf = clf

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
        scaler = _Scaler()
        x = scaler.fit_transform(features)
        clf = _LR(
            C=1.0 / l2_penalty,
            class_weight="balanced",
            max_iter=max(epochs, 1000),
            solver="lbfgs",
            random_state=0,
        )
        clf.fit(x, labels)
        return cls(scaler, clf)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = self._scaler.transform(features)
        return self._clf.predict_proba(x)[:, 1]


class GradientBoostedStumpModel:
    def __init__(self, scaler: _Scaler, clf: _GBT) -> None:
        self._scaler = scaler
        self._clf = clf

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        n_estimators: int = 60,
        learning_rate: float = 0.15,
        min_leaf_size: int = 8,
    ) -> "GradientBoostedStumpModel":
        scaler = _Scaler()
        x = scaler.fit_transform(features)
        effective_min_leaf = max(1, min(min_leaf_size, len(features) // 4))
        clf = _GBT(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=1,
            min_samples_leaf=effective_min_leaf,
            random_state=0,
        )
        clf.fit(x, labels)
        return cls(scaler, clf)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = self._scaler.transform(features)
        return self._clf.predict_proba(x)[:, 1]
