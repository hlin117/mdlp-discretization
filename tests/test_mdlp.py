import itertools
import pytest
import numpy as np

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

def test_e2e():
  X = np.array([
    [0.1, 0.1],
    [0.2, 1.1],
    [0.3, 0.2],
    [1.5, .9]
  ])
  y = np.array([0, 0, 0, 1])
  transformed = MDLP(shuffle=False).fit_transform(X, y)
  expected = np.array([
    [0, 0],
    [0, 0],
    [0, 0],
    [1, 0],
  ])
  assert_array_equal(transformed, expected)

@pytest.mark.skip(reason='demonstrating behavior to be fixed')
def test_e2e_duplicates():
  X = np.array([
    [0.1],
    [0.2],
    [0.3],
    [0.3],
    [0.3],
    [0.4]
  ])
  y = np.array([0, 0, 0, 1, 1, 1])
  transformed = MDLP(shuffle=False).fit_transform(X, y)
  expected = np.array([
    [0],
    [0],
    [0],
    [0],
    [0],
    [1],
  ])
  missed = 0
  for p in itertools.permutations(range(X.shape[0])):
    inds = list(p)
    X_shuff = X[inds]
    y_shuff = y[inds]
    expected_shuff = expected[inds]
    transformed = MDLP(shuffle=False).fit_transform(X_shuff, y_shuff)
    try:
      assert_array_equal(expected_shuff, transformed)
    except:
      missed += 1
  assert missed == 0, missed
