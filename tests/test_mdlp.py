import numpy as np
import random

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
    x = np.array([0., 1., 2., 3., 4., 5., 6., 7.])
    y = np.array([0, 0, 0, 0, 1, 0, 1, 1])
    labels_map = {0: {0}, 1: {0}, 2: {0}, 3: {0}, 4: {1}, 5: {0}, 6: {1}, 7: {1}}
    k = find_cut(x, y, labels_map, 0, len(y))
    assert_equal(4, k)

def test_find_cut_no_cut():
    x = np.array([0., 1., 2., 3.])
    y = np.array([0, 0, 0, 0])
    labels_map = {0: {0}, 1: {0}, 2: {0}, 3: {0}}
    k = find_cut(x, y, labels_map, 0, len(y))
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

def test_mdlp_iris_with_shuffle():
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data
    y = iris.target
    expected = load_iris_test_result()
    for _ in range(100):
      inds = range(X.shape[0])
      random.shuffle(inds)
      X_shuff = X[inds]
      y_shuff = y[inds]
      transformed_shuff = MDLP(shuffle=False).fit_transform(X_shuff, y_shuff)
      assert_array_equal(
          transformed_shuff,
          expected[inds],
          err_msg="discretization is not row-order independent"
      )
