"""Implements the Particle class, which holds data relevant to an individual particle."""

import numpy as np
from dataclasses import dataclass

# FIXME test!!!
@dataclass(eq=False)
class Particle():
    charge: int
    mass: float
    position: np.ndarray
    velocity: np.ndarray
