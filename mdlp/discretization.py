# Title          : discretization.py
# Author         : Henry Lin <hlin117@gmail.com>
# License        : BSD 3 clause
#==============================================================================

from __future__ import division

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

import numpy as np
import scipy


class MDLP(BaseEstimator, TransformerMixin):
    """Bins continuous values using MDLP "expert binning" method.

    Implements the MDLP discretization algorithm from Usama Fayyad's
    paper "Multi-Interval Discretization of Continuous-Valued
    Attributes for Classification Learning". Given the class labels
    for each sample, this transformer attempts to discretize a
    continuous attribute by minimizing the entropy at each interval.

    Parameters
    ----------
    continuous_features : array-like of type int (default=None)
        A list of indices indicating which columns should be discretized.

        If `X` is a 1-D array, then continuous_features should be None.
        Otherwise, for a 2-D array, defaults to `np.arange(X.shape[1])`.

    drop_collapsed_features : bool (default=False)
        When set to `True` will remove single-bin features when transforming
        `X`.

    min_depth : int (default=0)
        The minimum depth of the interval splitting. Overrides
        the MDLP stopping criterion. If the entropy at a given interval
        is found to be zero before `min_depth`, the algorithm will stop.

    shuffle : bool (default=True)
        Deprecated, and will be removed from future versions of this module.

        Please see random_state. (Note that this parameter overrides the
        random_state parameter. Thus, setting shuffle=False will override
        the affect of random_state.)

    random_state : int (default=None)
        Seed of pseudo RNG to use when shuffling the data. Affects the
        outcome of MDLP if there are multiple samples with the same
        continuous value, but with different class labels.

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

    >>> from discretization import MDLP
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
    """

    def __init__(self, continuous_features=None, min_depth=0, shuffle=True,
                 drop_collapsed_features=False, random_state=None):
        # Parameters
        self.continous_features = continuous_features
        self.min_depth = min_depth
        self.random_state = random_state
        self.shuffle = shuffle
        self.drop_collapsed_features = drop_collapsed_features

        # Attributes
        self.continuous_features_ = continuous_features
        self.cut_points_ = None
        self.dimensions_ = None
        self.__collapsed_features_count = 0

    def fit(self, X, y):
        """Finds the intervals of interest from the input data.

        Parameters
        ----------
        X : The array containing features to be discretized. Continuous
            features should be specified by the `continuous_features`
            attribute if `X` is a 2-D array.

        y : A list or array of class labels corresponding to `X`.
        """
        X = check_array(X, accept_sparse=True, force_all_finite=True, \
                        ensure_2d=False, dtype=np.float64)
        y = column_or_1d(y)
        y = check_array(y, ensure_2d=False, dtype=np.int64)
        X, y = check_X_y(X, y, accept_sparse=True)

        self.dimensions_ = len(X.shape)

        if self.dimensions_ > 2:
            raise ValueError("Invalid input dimension for `X`. Input shape is"
                             "{0}".format(X.shape))

        if not self.shuffle:
            import warnings
            warnings.warn("Shuffle parameter will be removed in the future.",
                          DeprecationWarning)
        else:
            state = check_random_state(self.random_state)
            perm = state.permutation(len(y))
            X = X[perm]
            y = y[perm]

        if self.dimensions_ == 2:
            if self.continuous_features_ is None:
                self.continuous_features_ = np.arange(X.shape[1])

            self.cut_points_ = dict()

            for index, col in enumerate(X.T):
                if index not in self.continuous_features_:
                    continue
                if scipy.sparse.issparse(col):
                    col = col.toarray()[0]
                cut_points = MDLPDiscretize(col, y, self.min_depth)
                self.cut_points_[index] = cut_points
        else:
            if self.continuous_features_ is not None:
                raise ValueError("Passed in a 1-d column of continuous features, "
                                 "but continuous_features is not None")
            self.continuous_features_ = None
            cut_points = MDLPDiscretize(X, y, self.min_depth)
            self.cut_points_ = cut_points

        self.__collapsed_features_count = \
            len([cp for cp in self.cut_points_.values() if cp.size == 0])

        return self

    def transform(self, X, y=None):
        """Discretizes values in X into {0, ..., k-1}.

        `k` is the number of bins the discretizer creates from a continuous
        feature.
        """
        X = check_array(X, accept_sparse=True, force_all_finite=True, ensure_2d=False)
        check_is_fitted(self, "cut_points_")
        if self.dimensions_ == 1:
            output = np.searchsorted(self.cut_points_, X)
        else:
            output = self.__transform_2d(X)
        return output

    def __transform_2d(self, X):
        if self.drop_collapsed_features:
            new_shape = (X.shape[0], X.shape[1] - self.__collapsed_features_count)
            output = self.__make_output(X, new_shape)
            output_col = 0
            for input_col in range(X.shape[1]):
                if input_col in self.continuous_features_:
                    if self.cut_points_[input_col].size > 0:
                        output[:, output_col] = \
                            np.searchsorted(self.cut_points_[input_col], \
                                            self.__get_col(X, input_col))
                        output_col += 1
                else:
                    output[:, output_col] = self.__get_col(X, input_col)
                    output_col += 1
        else:
            output = X.copy()
            for i in self.continuous_features_:
                output[:, i] = np.searchsorted(self.cut_points_[i], self.__get_col(X, i))
        return output

    def __make_output(self, X, shape):
        if scipy.sparse.issparse(X):
            output = scipy.sparse.dok_matrix(shape, dtype=X.dtype)
        else:
            output = np.ndarray(shape=shape, dtype=X.dtype)
        return output

    def __get_col(self, X, col_index):
        if scipy.sparse.issparse(X):
            col = X[:, col_index].toarray()
        else:
            col = X[:, col_index]
        return col

    def cat2intervals(self, X, index=None):
        """Converts a categorical feature into a list of intervals.
        """
        # TODO: Throw warning if `self.dimensions_` == 1 and index is not None
        if self.dimensions_ == 1:
            return self._assign_intervals(X, index)
        elif self.dimensions_ == 2 and index is None:
            raise ValueError("Index of `X` to be discretized needs to be "
                             "specified.")
        else:
            cp_indices = X.T[index]
            return self._assign_intervals(cp_indices, index)

    def cts2cat(self, col, index=None):
        """Converts each continuous feature from index `index` into
        a categorical feature from the input column `col`.
        """
        if self.dimensions_ == 1:
            return np.searchsorted(self.cut_points_, col)
        if self.dimensions_ == 2 and index is None:
            raise ValueError("Index of `X` to be discretized needs to be "
                             "specified.")
        return np.searchsorted(self.cut_points_[index], col)

    def _assign_intervals(self, cp_indices, index):
        """Assigns the cut point indices `cp_indices` (representing
        categorical features) into a list of intervals.
        """

        # Case for a 1-D array
        if self.dimensions_ == 1:
            cut_points = self.cut_points_
        else:
            cut_points = self.cut_points_[index]

        non_zero_mask = cp_indices[cp_indices - 1 != -1].astype(int) - 1
        fronts = np.zeros(cp_indices.shape)
        fronts[cp_indices == 0] = float("-inf")
        fronts[cp_indices != 0] = cut_points[non_zero_mask]

        numCuts = len(cut_points)
        backs = np.zeros(cp_indices.shape)
        non_numCuts_mask = cp_indices[cp_indices != numCuts].astype(int)
        backs[cp_indices == numCuts] = float("inf")
        backs[cp_indices != numCuts] = cut_points[non_numCuts_mask]

        return [(front, back) for front, back in zip(fronts, backs)]
