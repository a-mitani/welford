import numpy as np
from welford import Welford


def test_init():
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

    a = np.arange(60).reshape(3, 4, 5)
    w = Welford(a)
    assert w.mean.shape == w.var_s.shape == w.var_p.shape == (4, 5)
    assert np.allclose(w.mean, np.mean(a, axis=0))
    assert np.allclose(w.var_s, np.var(a, axis=0, ddof=1))
    assert np.allclose(w.var_p, np.var(a, axis=0, ddof=0))


def test_add():
    w = Welford()
    w.add(np.array([0, 100]))
    assert np.allclose(w.mean, np.array([0, 100]))
    assert np.alltrue(np.isnan(w.var_s))
    assert np.allclose(w.var_p, np.array([0, 0]))

    w.add(np.array([1, 110]))
    assert np.allclose(w.mean, np.array([0.5, 105]))
    assert np.allclose(w.var_s, np.array([0.5, 50]))
    assert np.allclose(w.var_p, np.array([0.25, 25]))

    w = Welford()
    w.add(np.array([[0, 100, 1000], [2, 220, 2200]]))
    w.add(np.array([[1, 110, 1100], [2, 220, 2200]]))
    assert np.allclose(w.mean, np.array([[0.5, 105, 1050], [2, 220, 2200]]))
    assert np.allclose(w.var_s, np.array([[0.5, 50, 5000], [0, 0, 0]]))
    assert np.allclose(w.var_p, np.array([[0.25, 25, 2500], [0, 0, 0]]))


def test_add_all():
    w = Welford()
    a = np.array([[0, 100], [1, 110], [2, 120], [3, 130], [4, 140]])
    w.add_all(a)
    assert w.count == 5
    assert np.allclose(w.mean, np.array([2, 120]))
    assert np.allclose(w.var_s, np.array([2.5, 250]))
    assert np.allclose(w.var_p, np.array([2, 200]))

    w = Welford()
    a = np.array([[0, 100]])
    w.add_all(a)
    assert w.count == 1
    assert np.allclose(w.mean, np.array([0, 100]))
    assert np.alltrue(np.isnan(w.var_s))
    assert np.allclose(w.var_p, np.array([0, 0]))

    a = np.array([[1, 110], [2, 120], [3, 130], [4, 140]])
    w.add_all(a)
    assert w.count == 5
    assert np.allclose(w.mean, np.array([2, 120]))
    assert np.allclose(w.var_s, np.array([2.5, 250]))
    assert np.allclose(w.var_p, np.array([2, 200]))

    w = Welford()
    a = np.array([[[0, 100, 1000], [2, 220, 2200]], [[1, 110, 1100], [2, 220, 2200]]])
    w.add_all(a)
    assert w.count == 2
    assert np.allclose(w.mean, np.array([[0.5, 105, 1050], [2, 220, 2200]]))
    assert np.allclose(w.var_s, np.array([[0.5, 50, 5000], [0, 0, 0]]))
    assert np.allclose(w.var_p, np.array([[0.25, 25, 2500], [0, 0, 0]]))


def test_rollback():
    a = np.array([[0, 100]])
    w = Welford(a)

    a = np.array([1, 110])
    w.add(a)
    assert np.allclose(w.mean, np.array([0.5, 105]))
    assert np.allclose(w.var_s, np.array([0.5, 50]))
    assert np.allclose(w.var_p, np.array([0.25, 25]))

    w.rollback()
    a = np.array([2, 120])
    w.add(a)
    assert w.count == 2
    assert np.allclose(w.mean, np.array([1, 110]))
    assert np.allclose(w.var_s, np.array([2, 200]))
    assert np.allclose(w.var_p, np.array([1, 100]))

    a = np.array([[2, 120], [3, 130]])
    w.add_all(a)
    w.rollback()
    assert w.count == 2
    assert np.allclose(w.mean, np.array([1, 110]))
    assert np.allclose(w.var_s, np.array([2, 200]))
    assert np.allclose(w.var_p, np.array([1, 100]))

    w = Welford()
    w.add(np.array([[0, 100, 1000], [2, 220, 2200]]))
    w.add(np.array([[1, 110, 1100], [2, 220, 2200]]))
    w.rollback()
    w.add(np.array([[2, 120, 1200], [2, 220, 2200]]))
    assert np.allclose(w.mean, np.array([[1.0, 110, 1100], [2, 220, 2200]]))
    assert np.allclose(w.var_s, np.array([[2.0, 200, 20000], [0, 0, 0]]))
    assert np.allclose(w.var_p, np.array([[1, 100, 10000], [0, 0, 0]]))


def test_merge():
    a = np.array([[0, 100], [1, 110], [2, 120], [3, 130], [4, 140]])
    wa = Welford(a)
    b = np.array([[5, 150], [6, 160], [7, 170]])
    wb = Welford(b)
    wa.merge(wb)
    assert wa.count == 8
    assert np.allclose(wa.mean, np.array([3.5, 135]))
    assert np.allclose(wa.var_s, np.array([6, 600]))
    assert np.allclose(wa.var_p, np.array([5.25, 525]))

    wa.rollback()
    c = np.array([[5, 150], [6, 160], [7, 170], [8, 180], [9, 190], [10, 200]])
    wc = Welford(c)
    wa.merge(wc)
    assert wa.count == 11
    assert np.allclose(wa.mean, np.array([5, 150]))
    assert np.allclose(wa.var_s, np.array([11, 1100]))
    assert np.allclose(wa.var_p, np.array([10, 1000]))

    a = np.array([[0, 100]])
    wa = Welford(a)
    b = np.array([[1, 110]])
    wb = Welford(b)
    wa.merge(wb)
    assert wa.count == 2
    assert np.allclose(wa.mean, np.array([0.5, 105]))
    assert np.allclose(wa.var_s, np.array([0.5, 50]))
    assert np.allclose(wa.var_p, np.array([0.25, 25]))

    wa = Welford(np.array([[[0, 100, 1000], [2, 220, 2200]]]))
    wb = Welford(np.array([[[1, 110, 1100], [2, 220, 2200]]]))
    wa.merge(wb)
    assert wa.count == 2
    assert np.allclose(wa.mean, np.array([[0.5, 105, 1050], [2, 220, 2200]]))
    assert np.allclose(wa.var_s, np.array([[0.5, 50, 5000], [0, 0, 0]]))
    assert np.allclose(wa.var_p, np.array([[0.25, 25, 2500], [0, 0, 0]]))
