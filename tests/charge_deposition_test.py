"""Test charge deposition routine"""

import numpy as np

from espic.deposit_charge import ChargeDeposition
from espic.make_grid import Uniform2DGrid


def test_1d_deposit_grid():
    # Test 1D charge deposition when all charges are on grid points and all charges are equal.
    cd = ChargeDeposition()
    xarr = cd.grid.grid
    qarr = np.ones(len(xarr))

    rho = cd.deposit(qarr, xarr)
    err = np.abs(rho - 1 / cd.grid.delta * np.ones(len(rho)))
    np.testing.assert_allclose(err, np.zeros(len(err)), atol=1e-9)


def test_1d_deposit_half_grid():
    # Test 1D charge deposition when all charges are on half grid points and all charges are equal.
    cd = ChargeDeposition()
    epsilon = 1e-9
    xarr = np.array(
        [
            1 / (2 + epsilon) * (cd.grid.grid[i] + cd.grid.grid[i + 1])
            for i in range(len(cd.grid.grid) - 1)
        ],
    )
    qarr = np.ones(len(xarr))

    rho = cd.deposit(qarr, xarr)
    true_rho = 49.5 * np.ones(len(rho))
    true_rho[0] = 0
    true_rho[-1] = 0

    err = np.abs(rho - true_rho)
    np.testing.assert_allclose(err, np.zeros(len(err)), atol=1e-9)


def test_2d_deposit_grid():
    # Test 2D charge deposition when all charges are on grid points and all charges are equal.
    cd = ChargeDeposition(grid=Uniform2DGrid())
    pos_arr = cd.return_coords(cd.grid)
    q_arr = np.ones(len(pos_arr))

    rho = cd.deposit(q_arr, pos_arr)

    err = np.abs(rho - 49.5 * np.ones(rho.shape))
    np.testing.assert_allclose(err, np.zeros(err.shape), atol=1e-9)


def test_2d_deposit_single():
    # Test 2D charge deposition when there is a single charge on a single grid point on the corner.
    cd = ChargeDeposition(grid=Uniform2DGrid())
    pos_arr = cd.return_coords(cd.grid)
    q_arr = np.zeros(len(pos_arr))
    q_arr[0] = 1

    rho = cd.deposit(q_arr, pos_arr)
    true_rho = np.zeros(rho.shape)
    true_rho[0][0] = 49.5
    err = np.abs(rho - true_rho)
    np.testing.assert_allclose(err, np.zeros(err.shape), atol=1e-9)


# def test_2d_deposit_half_grid():
#     # Test 2D charge deposition when all charges are on half grid points and all charges are equal.
#     cd = ChargeDeposition(grid=Uniform2DGrid())
#     x_grid = cd.grid.x_grid
#     y_grid = cd.grid.y_grid

#     x_half_grid = np.array(
#         [
#             1 / (2) * (x_grid[i] + x_grid[i + 1])
#             for i in range(len(x_grid) - 1)
#         ],
#     )

#     y_half_grid = np.array(
#         [
#             1 / (2) * (y_grid[i] + y_grid[i + 1])
#             for i in range(len(y_grid) - 1)
#         ],
#     )

#     half_grid = np.meshgrid(x_half_grid, y_half_grid)

#     pos_arr = cd.return_coords(half_grid)
#     q_arr = np.ones(len(pos_arr))

#     rho = cd.deposit(q_arr,pos_arr)

#     return rho
