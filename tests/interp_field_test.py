"""Test field interpolation routine"""

import numpy as np

from espic.interp_field import InterpolatedField


def test_interpolate_field_1D_null_field():
    x_vec = np.linspace(0, 1, num=10)
    phi_on_grid = np.ones(x_vec.size)
    interpolated_field = InterpolatedField([x_vec], phi_on_grid)
    evaluated_field = interpolated_field.evaluate([0.5])
    target_field = np.asarray([0])

    assert np.allclose(evaluated_field, target_field)


def test_interpolate_field_1D_linear_potential():
    x_vec = np.linspace(0, 1, num=10)
    phi_on_grid = 3 * x_vec
    interpolated_field = InterpolatedField([x_vec], phi_on_grid)
    pts = [0, 0.5, 1]
    evaluated_field = interpolated_field.evaluate(pts)
    target_field = np.asarray([-3] * 3)

    assert np.allclose(evaluated_field, target_field)


def test_interpolate_field_2D_null_field():
    x_vec = np.linspace(0, 1, num=10)
    y_vec = x_vec
    phi_on_grid = np.ones((x_vec.size, y_vec.size))
    interpolated_field = InterpolatedField([x_vec, y_vec], phi_on_grid)
    evaluated_field = interpolated_field.evaluate([0.5] * 2)
    target_field = np.asarray([0] * 2)

    assert np.allclose(evaluated_field, target_field)


def test_interpolate_field_2D_linear_potential():
    x_vec = np.linspace(0, 1, num=100)
    y_vec = x_vec
    xx, yy = np.meshgrid(x_vec, y_vec, indexing="ij")
    zz1 = xx
    zz2 = yy
    interpolated_field1 = InterpolatedField([x_vec, y_vec], zz1)
    interpolated_field2 = InterpolatedField([x_vec, y_vec], zz2)
    pts = [0.5, 0.5]
    evaluated_field1 = interpolated_field1.evaluate(pts)
    evaluated_field2 = interpolated_field2.evaluate(pts)
    target_field1 = np.asarray([-1, 0])
    target_field2 = np.asarray([0, -1])

    assert np.allclose(evaluated_field1, target_field1)
    assert np.allclose(evaluated_field2, target_field2)


def test_interpolate_field_2D_parabolic_potential():
    x_vec = np.linspace(-2, 2, num=1000)
    y_vec = x_vec
    xx, yy = np.meshgrid(x_vec, y_vec, indexing="ij")
    zz = xx**2 + yy**2
    interpolated_field = InterpolatedField([x_vec, y_vec], zz)
    pts = [[0, 0], [-1, 1], [1.5, -1.5], [0.5, 1]]
    evaluated_field = interpolated_field.evaluate(pts)
    target_field = np.asarray([[0, 0], [2, -2], [-3, 3], [-1, -2]])

    assert np.allclose(evaluated_field, target_field)


def test_interpolate_field_2D_semiparabolic_potential():
    x_vec = np.linspace(-2, 2, num=1000)
    y_vec = x_vec
    xx, yy = np.meshgrid(x_vec, y_vec, indexing="ij")
    zz = xx**2 - 2 * yy**2
    interpolated_field = InterpolatedField([x_vec, y_vec], zz)
    pts = [[0, 0], [-1, 1], [1.5, -1.5], [0.5, 1]]
    evaluated_field = interpolated_field.evaluate(pts)
    target_field = np.asarray([[0, 0], [2, 4], [-3, -6], [-1, 4]])

    assert np.allclose(evaluated_field, target_field)


def test_interpolate_field_messy_potential():
    x_vec = np.linspace(-2, 2, num=1000)
    y_vec = x_vec
    xx, yy = np.meshgrid(x_vec, y_vec, indexing="ij")
    zz = xx**2 - 2 * yy**2 + 3 * xx * yy
    interpolated_field = InterpolatedField([x_vec, y_vec], zz)
    pts = [[1, 1.5], [-0.5, 0], [-1, 1]]
    evaluated_field = interpolated_field.evaluate(pts)
    target_field = np.asarray([[-6.5, 3], [1, 1.5], [-1, 7]])

    assert np.allclose(evaluated_field, target_field)
