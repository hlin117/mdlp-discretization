import itertools
import numpy as np
import pytest

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

def test_fit_transform():
  X = np.array([
    [0.1, 0.1],
    [0.2, 0.4],
    [0.3, 0.2],
    [0.4, 0.3]
  ])
  y = np.array([0, 0, 1, 2])
  transformed = MDLP(shuffle=False).fit_transform(X, y)
  expected = [
    [0, 0],
    [0, 0],
    [1, 0],
    [2, 0],
  ]
  assert_array_equal(transformed, expected)

  # discretization is invariant under rescaling of the data
  scaled_disc = MDLP(shuffle=False).fit_transform(10 * X, y)
  assert_array_equal(scaled_disc, expected)

  # discretization is invariant under translation of the data
  translated_disc = MDLP(shuffle=False).fit_transform(X - 5, y)
  assert_array_equal(translated_disc, expected)

  # discretization is invariant under renaming of the labels
  relabeled_disc = MDLP(shuffle=False).fit_transform(X, y + 1)
  assert_array_equal(relabeled_disc, expected)
