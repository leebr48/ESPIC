"""Test field interpolation routine"""

import numpy as np
from espic.interp_field import InterpolateField

def test_interpolate_field_2D_null_field():
    x_vec = np.linspace(0, 1, num=10)
    y_vec = x_vec
    phi_on_grid = np.ones((x_vec.size, y_vec.size)) # Constant potential, so should give zero electric field
    interpolated_field = InterpolateField(y_vec, x_vec, phi_on_grid) # Order switched due to convention difference
    evaluated_field = interpolated_field.calc_field(0.55, 0.55)
    target_field = np.asarray([0, 0])

    assert np.allclose(evaluated_field, target_field)

def test_interpolate_field_2D_linear_potential():
    x_vec = np.linspace(0, 1, num=100)
    y_vec = x_vec
    xx, yy = np.meshgrid(x_vec, y_vec)
    zz1 = xx
    zz2 = yy
    interpolated_field1 = InterpolateField(y_vec, x_vec, zz1) # Order switched due to convention difference
    interpolated_field2 = InterpolateField(y_vec, x_vec, zz2)
    evaluated_field1 = interpolated_field1.calc_field(0.55, 0.55)
    evaluated_field2 = interpolated_field2.calc_field(0.55, 0.55)
    target_field1 = np.asarray([-1, 0])
    target_field2 = np.asarray([0, -1])

    assert np.allclose(evaluated_field1, target_field1)
    assert np.allclose(evaluated_field2, target_field2)

def test_interpolate_field_2D_parabolic_potential():
    x_vec = np.linspace(-2, 2, num=1000)
    y_vec = x_vec
    xx, yy = np.meshgrid(x_vec, y_vec)
    zz = xx**2 + yy**2
    interpolated_field = InterpolateField(y_vec, x_vec, zz) # Order switched due to convention difference
    evaluated_field1 = interpolated_field.calc_field(0, 0)
    evaluated_field2 = interpolated_field.calc_field(-1, 1)
    evaluated_field3 = interpolated_field.calc_field(1.5, -1.5)
    evaluated_field4 = interpolated_field.calc_field(0.5, 1)
    target_field1 = np.asarray([0, 0])
    target_field2 = np.asarray([2, -2])
    target_field3 = np.asarray([-3, 3])
    target_field4 = np.asarray([-1, -2])
    
    assert np.allclose(evaluated_field1, target_field1)
    assert np.allclose(evaluated_field2, target_field2)
    assert np.allclose(evaluated_field3, target_field3)
    assert np.allclose(evaluated_field4, target_field4)

def test_interpolate_field_2D_semiparabolic_potential():
    x_vec = np.linspace(-2, 2, num=1000)
    y_vec = x_vec
    xx, yy = np.meshgrid(x_vec, y_vec)
    zz = xx**2 - 2 * yy**2
    interpolated_field = InterpolateField(y_vec, x_vec, zz) # Order switched due to convention difference
    evaluated_field1 = interpolated_field.calc_field(0, 0)
    evaluated_field2 = interpolated_field.calc_field(-1, 1)
    evaluated_field3 = interpolated_field.calc_field(1.5, -1.5)
    evaluated_field4 = interpolated_field.calc_field(0.5, 1)
    target_field1 = np.asarray([0, 0])
    target_field2 = np.asarray([2, 4])
    target_field3 = np.asarray([-3, -6])
    target_field4 = np.asarray([-1, 4])
    
    assert np.allclose(evaluated_field1, target_field1)
    assert np.allclose(evaluated_field2, target_field2)
    assert np.allclose(evaluated_field3, target_field3)
    assert np.allclose(evaluated_field4, target_field4)
