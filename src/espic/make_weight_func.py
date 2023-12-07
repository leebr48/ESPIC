"""
Defines ``ChargeWeightFunc``, which allows charges to be
weighted appropriately for deposition on the spatial grid.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FArray = NDArray[np.float64]


class ChargeWeightFunc:
    """
    Specify a weight function. This can be used to "split" charges
    between nodes on the spatial grid in a systematic way.

    Parameters
    ----------
    distances
        Distances from charges to the nearest node on the spatial
        grid.
    spatial_grid_delta
        Spacing of the (uniform) spatial grid.
    """

    def __init__(
        self,
        distances: FArray,
        spatial_grid_delta: float,
    ) -> None:
        self.distances = distances
        self.spatial_grid_delta = spatial_grid_delta

    def zeroth_order(self) -> FArray:
        """
        Apply the simplest possible weight function - charges are placed
        on the spatial grid node nearest to them, without being "split".

        Returns
        -------
            Weights to distribute charges onto the spatial grid.
        """
        return (
            1
            / self.spatial_grid_delta
            * (self.distances < self.spatial_grid_delta / 2).astype(int)
        )
