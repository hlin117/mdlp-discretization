import numpy as np
cimport numpy as np
from scipy.stats import entropy
from libc.math cimport log, pow
from libcpp.set cimport set as stdset
#from numpy.math cimport INFINITY

def MDLPDiscretize(col, y, bint shuffle, int min_depth):
    """Performs the discretization process on X and y
    """
    # Shuffle array, and then reorder them
    if shuffle:
        shuffled_order = np.random.permutation(len(y))
        col = col[shuffled_order]
        y = y[shuffled_order]

    order = np.argsort(col)
    col = col[order]
    y = y[order]

    cut_points = stdset[int]()

    def get_cut(ind):
        return (col[ind-1] + col[ind]) / 2

    # Now we do a depth first search to create cut_points
    num_samples = len(col)
    search_intervals = list()
    search_intervals.append((0, num_samples, 0))
    while len(search_intervals) > 0:
        start, end, depth = search_intervals.pop()

        k = find_cut(y, start, end)

        # Need to see whether the "front" and "back" of the interval need
        # to be float("-inf") or float("inf")
        if (k == -1) or (depth >= min_depth and reject_split(y, start, end, k)):
            front = float("-inf") if (start == 0) else get_cut(start)
            back = float("inf") if (end == num_samples) else get_cut(end)

            if front == back: continue  # Corner case
            if front != float("-inf"): cut_points.add(front)
            if back != float("inf"): cut_points.add(back)
            continue

        search_intervals.append((start, k, depth + 1))
        search_intervals.append((k, end, depth + 1))

    cut_points = np.array([num for num in cut_points])
    cut_points = np.sort(cut_points)
    return cut_points

def slice_entropy(np.ndarray[np.int64_t, ndim=1] y, int start, int end):
    """Returns the entropy of the given slice of y. Also returns the
    number of classes within the interval.

    NOTE: The entropy function scipy provides uses the natural log,
    not log base 2.
    """
    counts = np.bincount(y[start:end])
    vals = np.true_divide(counts, end - start)
    return entropy(vals), np.sum(vals != 0)

cdef bint reject_split(np.ndarray[np.int64_t, ndim=1] y, int start, int end, int k):
    """Using the minimum description length principal, determines
    whether it is appropriate to stop cutting.
    """

    cdef float N = <float> (end - start)
    entropy1, k1 = slice_entropy(y, start, k)
    entropy2, k2 = slice_entropy(y, k, end)
    whole_entropy, k = slice_entropy(y, start, end)

    # Calculate the final values
    cdef:
        float part1 = 1 / N * ((start - k) * entropy1 + (end - k) * entropy2)
        float gain = whole_entropy - part1
        float entropy_diff = k * whole_entropy - k1 * entropy1 - k2 * entropy2
        float delta = log(pow(3, k) - 2) - entropy_diff
    return gain <= 1 / N * (log(N - 1) + delta)

cdef int find_cut(np.ndarray[np.int64_t, ndim=1] y, int start, int end):
    """Finds the best cut between the specified interval. The cut returned is
    an index k. If k split the array into two sub arrays A and B, then
    k is the last index of A. In other words, let start == 0 and end == n.
    We might have the array

        [0, 1, 2, 3, ..., k - 1] [k, k + 2, ..., n]

    Therefore, k is in the range {1, 2, ..., n}. (If k == n, then array B
    is simply of length 1.)

    If k = -1, then no good cut point was found.
    """

    # Want to probe for the best partition _entropy in a "smart" way
    # Input is the splitting index, to create partitions [start, ind)
    # and [ind, end).
    cdef:
        int length = end - start
        float prev_entropy = np.inf #INFINITY
        int k = -1
        int ind
        float first_half, second_half, curr_entropy
    for ind in range(start + 1, end):

        # I choose not to use a `min` function here for this optimization.
        if y[ind-1] == y[ind]:
            continue

        # Finds the partition entropy, and see if this entropy is minimum
        first_half = (<float> (ind - start)) / length \
                            * slice_entropy(y, start, ind)[0]
        second_half = (<float> (end - ind)) / length \
                             * slice_entropy(y, ind, end)[0]
        curr_entropy = first_half + second_half

        if prev_entropy > curr_entropy:
            prev_entropy = curr_entropy
            k = ind

    return k  # NOTE: k == -1 if there is no good cut

