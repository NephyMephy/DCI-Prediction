from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class ScoreScaler:
    mean_: float = 0.0
    std_: float = 1.0

    def fit(self, values: Iterable[float]) -> "ScoreScaler":
        array = np.asarray(list(values), dtype=float)
        array = array[np.isfinite(array)]
        if array.size == 0:
            self.mean_ = 0.0
            self.std_ = 1.0
            return self

        self.mean_ = float(array.mean())
        self.std_ = float(array.std(ddof=0)) if array.size > 1 else 1.0
        if self.std_ == 0:
            self.std_ = 1.0
        return self

    def transform(self, values: Iterable[float]) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        return (array - self.mean_) / self.std_

    def inverse_transform(self, values: Iterable[float]) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        return array * self.std_ + self.mean_
