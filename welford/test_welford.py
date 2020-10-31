import numpy as np
from welford import Welford


def test_init():
    w = Welford(dim=2)
    assert w.count == 0
    assert np.alltrue(w.mean == 0)
    assert np.alltrue(np.isnan(w.var_s))
    assert np.alltrue(np.isnan(w.var_p))

    a = np.array([[0]])
    w = Welford(a)
    assert w.count == 1
    assert np.allclose(w.mean, np.array([0]))
    assert np.alltrue(np.isnan(w.var_s))
    assert np.allclose(w.var_p, np.array([0]))

    a = np.array([[0], [1]])
    w = Welford(a)
    assert w.count == 2
    assert np.allclose(w.mean, np.array([0.5]))
    assert np.allclose(w.var_s, np.array([0.5]))
    assert np.allclose(w.var_p, np.array([0.25]))

    a = np.array([[0, 100], [1, 110], [2, 120], [3, 130], [4, 140]])
    w = Welford(a)
    assert w.count == 5
    assert np.allclose(w.mean, np.array([2, 120]))
    assert np.allclose(w.var_s, np.array([2.5, 250]))
    assert np.allclose(w.var_p, np.array([2, 200]))
