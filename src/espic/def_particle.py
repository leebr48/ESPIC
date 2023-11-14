"""Implements the Particle class, which holds data relevant to an individual particle."""

import numpy as np
from dataclasses import dataclass

@dataclass
class Particle():
    charge: int 
    mass: float # FIXME positive
    position: np.ndarray # FIXME 1D, 1-3 elements!
    velocity: np.ndarray # FIXME 1D, 1-3 elements!
