"""
Defines ``Uniform1DGrid``, which sets up a one-dimensional spatial grid,
and ``Uniform2DGrid``, which sets up a two-dimensional spatial grid.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np
from numpy.typing import NDArray

FArray = NDArray[np.float64]


class Uniform1DGrid:
    """
    Creates a one-dimensional spatial grid with uniform spacing.

    Parameters
    ----------
    num_points
        Number of grid points.
    x_min
        Least extent of the grid.
    x_max
        Greatest extent of the grid.
    """

    def __init__(
        self,
        num_points: int = 100,
        x_min: float = -1,
        x_max: float = 1,
    ) -> None:
        self.num_points = num_points
        self.x_min = x_min
        self.x_max = x_max

    @cached_property
    def grid(self) -> FArray:
        """The coordinates of the spatial grid nodes."""
        return np.linspace(self.x_min, self.x_max, self.num_points)

    @cached_property
    def size(self) -> int:
        """Number of elements in ``grid``."""
        return self.grid.size

    @cached_property
    def shape(self) -> tuple[int, ...]:
        """Dimension of ``grid``."""
        return self.grid.shape

    @cached_property
    def delta(self) -> float:
        """Spacing between grid points."""
        return self.grid[1] - self.grid[0]


class Uniform2DGrid:
    """
    Creates a two-dimensional spatial grid with uniform spacing.

    Parameters
    ----------
    num_points
        Number of grid points for each dimension.
    x_min
        Least extent of the first grid.
    x_max
        Greatest extent of the first grid.
    y_min
        Least extent of the second grid.
    y_max
        Greatest extent of the second grid.
    """

    def __init__(
        self,
        num_points: int = 100,
        x_min: float = -1,
        x_max: float = 1,
        y_min: float = -1,
        y_max: float = 1,
    ) -> None:
        self.num_points = num_points
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    @cached_property
    def x_grid(self) -> FArray:
        """The coordinates of the spatial nodes of the first grid."""
        return np.linspace(self.x_min, self.x_max, self.num_points)

    @cached_property
    def y_grid(self) -> FArray:
        """The coordinates of the spatial nodes of the second grid."""
        return np.linspace(self.y_min, self.y_max, self.num_points)

    @cached_property
    def grid(self) -> list[FArray]:
        """The coordinates of the spatial grid nodes in 2D."""
        return np.meshgrid(self.x_grid, self.y_grid, indexing="ij")

    @cached_property
    def size(self) -> int:
        """Number of elements in ``grid``."""
        return self.x_grid.size * self.y_grid.size

    @cached_property
    def shape(self) -> tuple[int, ...]:
        """Dimensions of each axis of ``grid``."""
        return (self.x_grid.shape[0], self.y_grid.shape[0])

    @cached_property
    def delta(self) -> float:
        """Spacing between grid points."""
        return self.x_grid[1] - self.x_grid[0]
