"""Test field interpolation routine"""

import numpy as np
from espic.interp_field import InterpolateField

def test_interpolate_field_1D_null_field():
    x_vec = np.linspace(0, 1, num=10)
    phi_on_grid = np.ones(x_vec.size)
    interpolated_field = InterpolateField([x_vec], phi_on_grid)
    evaluated_field = interpolated_field.ev([0.5])
    target_field = np.asarray([0])

    assert np.allclose(evaluated_field, target_field)

def test_interpolate_field_2D_null_field():
    x_vec = np.linspace(0, 1, num=10)
    y_vec = x_vec
    phi_on_grid = np.ones((x_vec.size, y_vec.size))
    interpolated_field = InterpolateField([x_vec, y_vec], phi_on_grid)
    evaluated_field = interpolated_field.ev([0.5, 0.5])
    target_field = np.asarray([0, 0])

    assert np.allclose(evaluated_field, target_field)

def test_interpolate_field_2D_linear_potential():
    indexing1 = 'xy'
    indexing2 = 'ij'
    x_vec = np.linspace(0, 1, num=100)
    y_vec = x_vec
    xx1, _ = np.meshgrid(x_vec, y_vec, indexing=indexing1)
    xx2, _ = np.meshgrid(x_vec, y_vec, indexing=indexing2)
    zz1 = xx1
    zz2 = xx2
    interpolated_field1 = InterpolateField([x_vec, y_vec], zz1, indexing=indexing1)
    interpolated_field2 = InterpolateField([x_vec, y_vec], zz2, indexing=indexing2)
    evaluated_field1 = interpolated_field1.ev([0.5, 0.5])
    evaluated_field2 = interpolated_field2.ev([0.5, 0.5])
    target_field = np.asarray([-1, 0])

    assert np.allclose(evaluated_field1, target_field)
    assert np.allclose(evaluated_field2, target_field)

def test_interpolate_field_2D_parabolic_potential():
    indexing1 = 'xy'
    indexing2 = 'ij'
    x_vec = np.linspace(-2, 2, num=1000)
    y_vec = x_vec
    xx1, yy1 = np.meshgrid(x_vec, y_vec, indexing=indexing1)
    xx2, yy2 = np.meshgrid(x_vec, y_vec, indexing=indexing2)
    zz1 = xx1**2 + yy1**2
    zz2 = xx2**2 + yy2**2
    interpolated_field1 = InterpolateField([x_vec, y_vec], zz1, indexing=indexing1)
    interpolated_field2 = InterpolateField([x_vec, y_vec], zz2, indexing=indexing2)
    evaluated_field11 = interpolated_field1.ev([0, 0])
    evaluated_field12 = interpolated_field1.ev([-1, 1])
    evaluated_field13 = interpolated_field1.ev([1.5, -1.5])
    evaluated_field14 = interpolated_field1.ev([0.5, 1])
    evaluated_field21 = interpolated_field2.ev([0, 0])
    evaluated_field22 = interpolated_field2.ev([-1, 1])
    evaluated_field23 = interpolated_field2.ev([1.5, -1.5])
    evaluated_field24 = interpolated_field2.ev([0.5, 1])
    target_field1 = np.asarray([0, 0])
    target_field2 = np.asarray([2, -2])
    target_field3 = np.asarray([-3, 3])
    target_field4 = np.asarray([-1, -2])

    print(evaluated_field12)
    print(evaluated_field22)
    quit()

    print(evaluated_field11)
    print(target_field1)
    print(evaluated_field12)
    print(target_field2)
    print(evaluated_field13)
    print(target_field3)
    print(evaluated_field14)
    print(target_field4)

    assert np.allclose(evaluated_field11, evaluated_field21)
    assert np.allclose(evaluated_field12, evaluated_field22)
    assert np.allclose(evaluated_field13, evaluated_field23)
    assert np.allclose(evaluated_field14, evaluated_field24)
    assert np.allclose(evaluated_field11, target_field1)
    assert np.allclose(evaluated_field12, target_field2)
    assert np.allclose(evaluated_field13, target_field3)
    assert np.allclose(evaluated_field14, target_field4)
    assert np.allclose(evaluated_field21, target_field1)
    assert np.allclose(evaluated_field22, target_field2)
    assert np.allclose(evaluated_field23, target_field3)
    assert np.allclose(evaluated_field24, target_field4)

'''

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
'''

if __name__ == '__main__':
    test_interpolate_field_1D_null_field()
    test_interpolate_field_2D_null_field()
    test_interpolate_field_2D_linear_potential()
    test_interpolate_field_2D_parabolic_potential()
