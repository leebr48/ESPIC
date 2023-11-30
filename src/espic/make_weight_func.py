"""Defines ``ChargeWeightFunc``, which allows charges to be weighted appropriately for deposition on the spatial grid."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FArray = NDArray[np.float64]


class ChargeWeightFunc:
    def __init__(
        self,
        distances: FArray,
        spatial_grid_delta: float,
    ) -> None:
        self.distances = distances
        self.spatial_grid_delta = spatial_grid_delta

    def zeroth_order(self) -> FArray:
        return (
            1
            / self.spatial_grid_delta
            * (self.distances < self.spatial_grid_delta / 2).astype(int)
        )
