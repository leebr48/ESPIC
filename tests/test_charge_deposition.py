"""Test charge deposition routine"""

import numpy as np
from espic.deposit_charge import ChargeDeposition


def test_deposit_grid():
    # Test charge deposition when all charges are on grid points and all charges are equal.
    cd = ChargeDeposition()
    xarr = cd.grid
    qarr = np.ones(len(xarr))

    rho = cd.deposit(qarr, xarr)
    err = np.abs(rho - 1 / cd.delta * np.ones(len(rho)))
    np.testing.assert_allclose(err, np.zeros(len(err)), atol=1e-9)


def test_deposit_half_grid():
    # Test charge deposition when all charges are on half grid points and all charges are equal.
    cd = ChargeDeposition()
    epsilon = 1e-9
    xarr = np.array(
        [
            1 / (2 + epsilon) * (cd.grid[i] + cd.grid[i + 1])
            for i in range(len(cd.grid) - 1)
        ],
    )
    qarr = np.ones(len(xarr))

    rho = cd.deposit(qarr, xarr)
    true_rho = 49.5 * np.ones(len(rho))
    true_rho[0] = 0
    true_rho[-1] = 0

    err = np.abs(rho - true_rho)
    np.testing.assert_allclose(err, np.zeros(len(err)), atol=1e-9)
