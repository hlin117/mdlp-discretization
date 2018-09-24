# Title          : discretization.py
# Author         : Henry Lin <hlin117@gmail.com>
# License        : BSD 3 clause
#==============================================================================

from math import floor, log10

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import (
    check_array,
    check_X_y,
    column_or_1d,
    check_random_state,
)
from sklearn.utils.validation import check_is_fitted
from mdlp._mdlp import MDLPDiscretize


# def normalize(cut_points, _range, precision):
#     # if len(cut_points) == 0:
#         # return cut_points
#     # _range = np.max(col) - np.min(col)
#     multiplier = 10**(-floor(log10(_range))) / precision
#     return (cut_points * multiplier).astype(np.int) / multiplier


class MDLP(BaseEstimator, TransformerMixin):
    """Bins continuous values using MDLP "expert binning" method.

    Implements the MDLP discretization algorithm from Usama Fayyad's
    paper "Multi-Interval Discretization of Continuous-Valued
    Attributes for Classification Learning". Given the class labels
    for each sample, this transformer attempts to discretize a
    continuous attribute by minimizing the entropy at each interval.

    Parameters
    ----------
    continuous_features : 
        - None (default): All features are treated as continuous for discretization.
        - array of indices: Array of continous feature indices.
        - mask: Array of length n_features and with dtype=bool.

        If `X` is a 1-D array, then continuous_features is neglected.

    min_depth : int (default=0)
        The minimum depth of the interval splitting. Overrides
        the MDLP stopping criterion. If the entropy at a given interval
        is found to be zero before `min_depth`, the algorithm will stop.

    random_state : int (default=None)
        Seed of pseudo RNG to use when shuffling the data. Affects the
        outcome of MDLP if there are multiple samples with the same
        continuous value, but with different class labels.

    min_split : float (default=1e-3)
        The minmum size to split a bin

    dtype : np.dtype (default=np.int)
        The dtype of the transformed X

    Attributes
    ----------
    continuous_features_ : array-like of type int
        Similar to continous_features. However, for 2-D arrays, this
        attribute cannot be None.

    cut_points_ : dict of type {int : np.array}
        Dictionary mapping indices to a numpy array. Each
        numpy array is a sorted list of cut points found from
        discretization.

    dimensions_ : int
        Number of dimensions to input `X`. Either 1 or 2.

    Examples
    --------
    ```
        >>> from mdlp.discretization import MDLP
        >>> from sklearn.datasets import load_iris
        >>> iris = load_iris()
        >>> X = iris.data
        >>> y = iris.target
        >>> mdlp = MDLP()
        >>> conv_X = mdlp.fit_transform(X, y)

        `conv_X` will be the same shape as `X`, except it will contain
        integers instead of continuous attributes representing the results
        of the discretization process.

        To retrieve the explicit intervals of the discretization of, say,
        the third column (index 2), one can do

        >>> mdlp.cat2intervals(conv_X, 2)

        which would return a list of tuples `(a, b)`. Each tuple represents
        the contnuous interval (a, b], where `a` can be `float("-inf")`,
        and `b` can be `float("inf")`.
    ```
    """

    def __init__(self, continuous_features=None, min_depth=0, random_state=None, 
                 min_split=1e-3, dtype=np.int):
        # Parameters
        # self.continuous_features = None
        self.min_depth = min_depth
        self.random_state = random_state
        self.min_split = min_split
        self.continuous_features = continuous_features
        self.dtype = dtype

        # Attributes
        self.continuous_features_ = None
        self.cut_points_ = None
        self.mins_ = None
        self.maxs_ = None

    def fit(self, X, y):
        """Finds the intervals of interest from the input data.

        Parameters
        ----------
        X : The array containing features to be discretized. Continuous
            features should be specified by the `continuous_features`
            attribute if `X` is a 2-D array.

        y : A list or array of class labels corresponding to `X`.

        continuous_features : (default None) a list of indices that you want to discretize
                              or a list (or array) of bools indicating the continuous features
        """
        X = check_array(X, force_all_finite=True, ensure_2d=True, dtype=np.float64)
        y = column_or_1d(y)
        y = check_array(y, ensure_2d=False, dtype=np.int)
        X, y = check_X_y(X, y)

        if len(X.shape) != 2:
            raise ValueError("Invalid input dimension for `X`. "
                             "Input shape is expected to be 2D, but is {0}".format(X.shape))

        state = check_random_state(self.random_state)
        perm = state.permutation(len(y))
        X = X[perm]
        y = y[perm]

        if self.continuous_features is None:
            self.continuous_features_ = np.arange(X.shape[1])
        else:
            continuous_features = np.array(self.continuous_features)
            if continuous_features.dtype == np.bool:
                continuous_features = np.arange(len(continuous_features))[continuous_features]
            else:
                continuous_features = continuous_features.astype(np.int, casting='safe')
                assert np.max(continuous_features) < X.shape[1] and np.min(continuous_features) >= 0
            self.continuous_features_ = continuous_features

        self.cut_points_ = [None] * X.shape[1]
        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)

        for index in self.continuous_features_:
            col = X[:, index]
            cut_points = MDLPDiscretize(col, y, self.min_depth, self.min_split)
            self.cut_points_[index] = cut_points
            # self.cut_points_[index] = normalize(cut_points, maxs[index] - mins[index], self.precision)

        self.mins_ = mins
        self.maxs_ = maxs
        return self

    def transform(self, X):
        """Discretizes values in X into {0, ..., k-1}.

        `k` is the number of bins the discretizer creates from a continuous
        feature.
        """
        X = check_array(X, force_all_finite=True, ensure_2d=False)
        check_is_fitted(self, "cut_points_")

        output = X.copy()
        for i in self.continuous_features_:
            output[:, i] = np.searchsorted(self.cut_points_[i], X[:, i])
        return output.astype(self.dtype)

    def cat2intervals(self, X, index):
        """Converts categorical data into intervals.

        Parameters
        ----------
        X : The discretized array

        index: which feature index to convert
        """

        cp_indices = X[:, index]
        return self.assign_intervals(cp_indices, index)

    def cts2cat(self, col, index):
        """Converts each continuous feature from index `index` into
        a categorical feature from the input column `col`.
        """
        return np.searchsorted(self.cut_points_[index], col)

    def assign_intervals(self, cp_indices, index):
        """Assigns the cut point indices `cp_indices` (representing
        categorical features) into a list of intervals.
        """

        # Case for a 1-D array
        cut_points = self.cut_points_[index]
        if cut_points is None:
            raise ValueError("The given index %d has not been discretized!")
        non_zero_mask = cp_indices[cp_indices - 1 != -1].astype(int) - 1
        fronts = np.zeros(cp_indices.shape)
        fronts[cp_indices == 0] = float("-inf")
        fronts[cp_indices != 0] = cut_points[non_zero_mask]

        n_cuts = len(cut_points)
        backs = np.zeros(cp_indices.shape)
        non_n_cuts_mask = cp_indices[cp_indices != n_cuts].astype(int)
        backs[cp_indices == n_cuts] = float("inf")
        backs[cp_indices != n_cuts] = cut_points[non_n_cuts_mask]

        return [(front, back) for front, back in zip(fronts, backs)]

    # def
