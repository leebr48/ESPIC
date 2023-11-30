"""Test field interpolation routine"""

import numpy as np

from espic.interp_field import InterpolatedField
from espic.make_grid import Uniform1DGrid


def test_interpolate_field_1D_null_field():
    # Simplest possible test - 1D, zero electric field
    x_grid = Uniform1DGrid(num_points=10, x_min=0, x_max=1)
    phi_on_grid = np.ones(x_grid.size)
    interpolated_field = InterpolatedField([x_grid], phi_on_grid)
    evaluated_field = interpolated_field.evaluate([0.5])
    target_field = np.asarray([0])

    assert np.allclose(evaluated_field, target_field)


def test_interpolate_field_1D_linear_potential():
    # 1D test with constant electric field
    x_grid = Uniform1DGrid(num_points=10, x_min=0, x_max=1)
    phi_on_grid = 3 * x_grid.grid
    interpolated_field = InterpolatedField([x_grid], phi_on_grid)
    pts = [0, 0.5, 1]
    evaluated_field = interpolated_field.evaluate(pts)
    target_field = np.asarray([-3] * 3)

    assert np.allclose(evaluated_field, target_field)


def test_interpolate_field_2D_null_field():
    # 2D test with zero electric field
    x_grid = Uniform1DGrid(num_points=10, x_min=0, x_max=1)
    y_grid = x_grid
    phi_on_grid = np.ones((x_grid.size, y_grid.size))
    interpolated_field = InterpolatedField([x_grid, y_grid], phi_on_grid)
    evaluated_field = interpolated_field.evaluate([0.5] * 2)
    target_field = np.asarray([0] * 2)

    assert np.allclose(evaluated_field, target_field)


def test_interpolate_field_2D_linear_potential():
    # 2D test with constant electric field
    x_grid = Uniform1DGrid(num_points=100, x_min=0, x_max=1)
    y_grid = x_grid
    xx, yy = np.meshgrid(x_grid.grid, y_grid.grid, indexing="ij")
    zz1 = xx
    zz2 = yy
    interpolated_field1 = InterpolatedField([x_grid, y_grid], zz1)
    interpolated_field2 = InterpolatedField([x_grid, y_grid], zz2)
    pts = [0.5, 0.5]
    evaluated_field1 = interpolated_field1.evaluate(pts)
    evaluated_field2 = interpolated_field2.evaluate(pts)
    target_field1 = np.asarray([-1, 0])
    target_field2 = np.asarray([0, -1])

    assert np.allclose(evaluated_field1, target_field1)
    assert np.allclose(evaluated_field2, target_field2)


def test_interpolate_field_2D_parabolic_potential():
    # 2D test with parabolic potential - check numerical accuracy
    x_grid = Uniform1DGrid(num_points=1000, x_min=-2, x_max=2)
    y_grid = x_grid
    xx, yy = np.meshgrid(x_grid.grid, y_grid.grid, indexing="ij")
    zz = xx**2 + yy**2
    interpolated_field = InterpolatedField([x_grid, y_grid], zz)
    pts = [[0, 0], [-1, 1], [1.5, -1.5], [0.5, 1]]
    evaluated_field = interpolated_field.evaluate(pts)
    target_field = np.asarray([[0, 0], [2, -2], [-3, 3], [-1, -2]])

    assert np.allclose(evaluated_field, target_field)


def test_interpolate_field_2D_semiparabolic_potential():
    # 2D test with semiparabolic potential - check numerical accuracy
    x_grid = Uniform1DGrid(num_points=1000, x_min=-2, x_max=2)
    y_grid = x_grid
    xx, yy = np.meshgrid(x_grid.grid, y_grid.grid, indexing="ij")
    zz = xx**2 - 2 * yy**2
    interpolated_field = InterpolatedField([x_grid, y_grid], zz)
    pts = [[0, 0], [-1, 1], [1.5, -1.5], [0.5, 1]]
    evaluated_field = interpolated_field.evaluate(pts)
    target_field = np.asarray([[0, 0], [2, 4], [-3, -6], [-1, 4]])

    assert np.allclose(evaluated_field, target_field)


def test_interpolate_field_asymmetric_grids_messy_potential():
    # 2D test with somewhat nasty potential - check numerical accuracy
    x_grid = Uniform1DGrid(num_points=1000, x_min=-2, x_max=2)
    y_grid = Uniform1DGrid(num_points=2000, x_min=-3, x_max=4)
    xx, yy = np.meshgrid(x_grid.grid, y_grid.grid, indexing="ij")
    zz = xx**2 - 2 * yy**2 + 3 * xx * yy
    interpolated_field = InterpolatedField([x_grid, y_grid], zz)
    pts = [[1, 1.5], [-0.5, 0], [-1, 1]]
    evaluated_field = interpolated_field.evaluate(pts)
    target_field = np.asarray([[-6.5, 3], [1, 1.5], [-1, 7]])

    assert np.allclose(evaluated_field, target_field)


def test_interpolate_field_updater():
    # 2D test with semiparabolic potential - check you can update the potential properly
    x_grid = Uniform1DGrid(num_points=1000, x_min=-2, x_max=2)
    y_grid = x_grid
    xx, yy = np.meshgrid(x_grid.grid, y_grid.grid, indexing="ij")
    zz = xx**2 - 2 * yy**2
    interpolated_field = InterpolatedField([x_grid, y_grid], zz)
    new_zz = 2 * xx**2 - 3 * yy**2
    interpolated_field.update_potential(new_zz)
    pts = [[0, 0], [-1, 1], [1.5, -1.5], [0.5, 1]]
    evaluated_field = interpolated_field.evaluate(pts)
    target_field = np.asarray([[0, 0], [4, 6], [-6, -9], [-2, 6]])

    assert np.allclose(evaluated_field, target_field)
