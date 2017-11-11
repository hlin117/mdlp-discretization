import numpy as np

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal

from mdlp.discretization import MDLP
from mdlp._mdlp import slice_entropy, find_cut

from util import load_iris_test_result


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

def test_mdlp_iris():
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data
    y = iris.target
    mdlp = MDLP(shuffle=False)
    transformed = mdlp.fit_transform(X, y)
    expected = load_iris_test_result()
    assert_array_equal(transformed, expected,
                       err_msg="MDLP output is inconsistent with previous runs.")
