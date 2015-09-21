import numpy as np
cimport numpy as np

ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef SIZE_t* LEVEL

#cdef class SearchLevel:
#    cdef int start, end, depth
#
#    def inline __cinit__(self, start, end, depth):
#        self.start = start
#        self.end = end
        #self.depth = depth

