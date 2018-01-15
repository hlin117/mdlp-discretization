import itertools
import numpy as np
import scipy.sparse

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal

from mdlp.discretization import MDLP
from mdlp._mdlp import slice_entropy, find_cut


def test_slice_entropy():
    y = np.array([0, 0, 0, 1, 1, 0, 1, 3, 1, 1])
    entropy1, k1 = slice_entropy(y, 0, 3)
    entropy2, k2 = slice_entropy(y, 3, 10)

    assert_equal(entropy1, 0, "Entropy was not calculated correctly.")
    assert_equal(k1, 1, "Incorrect number of classes found.")
    assert_almost_equal(entropy2, 0.796311640173813,
                        err_msg="Entropy was not calculated correctly.")
    assert_equal(k2, 3, "Incorrect number of classes found.")

def test_find_cut():
    y = np.array([0, 0, 0, 0, 1, 0, 1, 1])
    k = find_cut(y, 0, len(y))
    assert_equal(4, k)

def test_find_cut_no_cut():
    y = np.array([0, 0, 0, 0])
    k = find_cut(y, 0, len(y))
    assert_equal(-1, k)

def test_fit_transform_scale():
    expected = [
        [0, 0],
        [0, 0],
        [1, 0],
        [2, 0],
    ]

    X = np.array([
        [0.1, 0.1],
        [0.2, 0.4],
        [0.3, 0.2],
        [0.4, 0.3]
    ])
    y = np.array([0, 0, 1, 2])
    for i in range(10):
        scaled_disc = MDLP(shuffle=False).fit_transform(X / 10**i, y)
        assert_array_equal(expected, scaled_disc)

def test_fit_transform_translate():
    expected = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1]).reshape(-1, 1)

    X = np.arange(9, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1])
    transformed = MDLP(shuffle=False).fit_transform(X, y)
    assert_array_equal(expected, transformed)

    # translating data does not affect discretization result
    translated = MDLP(shuffle=False).fit_transform(X - 5, y)
    assert_array_equal(expected, translated)

def test_coerce_list():
    expected = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1]).reshape(-1, 1)

    X = [[i] for i in range(9)]
    y = [0, 0, 0, 0, 1, 0, 1, 1, 1]
    transformed = MDLP(shuffle=False).fit_transform(X, y)
    assert_array_equal(expected, transformed)

    np_X = np.arange(9).reshape(-1, 1)
    np_y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1])
    np_transformed = MDLP(shuffle=False).fit_transform(np_X, np_y)
    assert_array_equal(expected, np_transformed)

def test_drop_collapsed_features_dense():
    expected = [
        [0, 0],
        [0, 0],
        [1, 1],
        [2, 2],
    ]

    X = np.array([
        [0.1, 0.1, 0.1, 0.1, 0.1],
        [0.4, 0.2, 0.4, 0.2, 0.4],
        [0.2, 0.3, 0.2, 0.3, 0.2],
        [0.3, 0.4, 0.3, 0.4, 0.3]
    ])
    y = np.array([0, 0, 1, 2])
    disc = MDLP(drop_collapsed_features=True, shuffle=False).fit_transform(X, y)
    assert_array_equal(expected, disc)

def test_sparse_input():
    expected = [
        [0, 0],
        [0, 0],
        [1, 0],
        [2, 0],
    ]

    dense_X = np.array([
        [0.1, 0.1],
        [0.2, 0.4],
        [0.3, 0.2],
        [0.4, 0.3]
    ])
    X = scipy.sparse.csr_matrix(dense_X)
    y = np.array([0, 0, 1, 2])
    disc = MDLP(shuffle=False).fit_transform(X, y)
    assert_array_equal(expected, disc.toarray())

def test_drop_collapsed_features_sparse():
    expected = [
        [0, 0],
        [0, 0],
        [1, 1],
        [2, 2],
    ]

    dense_X = np.array([
        [0.1, 0.1, 0.1, 0.1, 0.1],
        [0.4, 0.2, 0.4, 0.2, 0.4],
        [0.2, 0.3, 0.2, 0.3, 0.2],
        [0.3, 0.4, 0.3, 0.4, 0.3]
    ])
    X = scipy.sparse.csr_matrix(dense_X)
    y = np.array([0, 0, 1, 2])
    disc = MDLP(drop_collapsed_features=True, shuffle=False).fit_transform(X, y)
    assert_array_equal(expected, disc.toarray())

def test_multiprocessing():
    expected = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 2, 0, 2, 0],
    ]

    X = np.array([
        [0.1, 0.1, 0.1, 0.1, 0.1],
        [0.4, 0.2, 0.4, 0.2, 0.4],
        [0.2, 0.3, 0.2, 0.3, 0.2],
        [0.3, 0.4, 0.3, 0.4, 0.3]
    ])
    y = np.array([0, 0, 1, 2])
    disc = MDLP(n_jobs=3, shuffle=False).fit_transform(X, y)
    assert_array_equal(expected, disc)
