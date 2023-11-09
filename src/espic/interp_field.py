"""Defines InterpolateField, which allows for the conversion of the potential on the spatial grid to the electric field at an arbitrary point in space."""

import numpy as np
from scipy.interpolate import RectBivariateSpline


class InterpolateField:
    def __init__(self, x_grid, y_grid, phi_on_grid):
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.phi_on_grid = phi_on_grid

    def interpolated_phi(self):
        return RectBivariateSpline(self.x_grid, self.y_grid, self.phi_on_grid)

    def calc_field(self, x_eval, y_eval):
        derx = -1 * self.interpolated_phi().ev(x_eval, y_eval, dx=1, dy=0)
        dery = -1 * self.interpolated_phi().ev(x_eval, y_eval, dx=0, dy=1)
        return np.asarray([derx, dery])
    '''
    def calc_field(self, x_eval, y_eval):
        derx = self.interpolated_phi().ev(x_eval, y_eval, dx=1, dy=0)
        dery = self.interpolated_phi().ev(x_eval, y_eval, dx=0, dy=1)
        return np.asarray([derx, dery])

    def calc_field(self, x_eval, y_eval):
        dery = -1 * self.interpolated_phi().ev(x_eval, y_eval, dx=1, dy=0)
        derx = -1 * self.interpolated_phi().ev(x_eval, y_eval, dx=0, dy=1)
        return np.asarray([derx, dery])

    def calc_field(self, x_eval, y_eval):
        dery = self.interpolated_phi().ev(x_eval, y_eval, dx=1, dy=0)
        derx = self.interpolated_phi().ev(x_eval, y_eval, dx=0, dy=1)
        return np.asarray([derx, dery])
    '''


    # FIXME should we have this in a 'helper function' file?
    def calc_gradient_2D(self, x_eval, y_eval, h=1e-3):
        # Oddly, 2D gradients with SciPy splines are difficult, so we must implement one.
        # FIXME implement other differencing schemes?
        phi = self.interpolated_phi()
        # Apply the center differencing scheme
        derx = (phi.ev(x_eval + h, y_eval) - phi.ev(x_eval - h, y_eval)) / (2 * h)
        dery = (phi.ev(x_eval, y_eval + h) - phi.ev(x_eval, y_eval - h)) / (2 * h)
        return np.asarray((derx, dery))
    '''
    def calc_field(self, x_eval, y_eval):
        gradient = self.calc_gradient_2D(x_eval, y_eval)
        return -1 * gradient
    '''
