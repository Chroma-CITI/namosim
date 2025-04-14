import numpy as np

from namoros import utils


def test_euler_to_quat():
    quat = utils.euler_to_quat(0, 0, 90)
    assert np.allclose(quat, (0, 0, 0.70710678, 0.70710678))


def test_quat_to_euler():
    euler = utils.quat_to_euler((0, 0, 0.70710678, 0.70710678))
    assert np.allclose(euler, (0, 0, 90))
