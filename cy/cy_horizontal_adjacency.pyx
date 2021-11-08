from cython import boundscheck, wraparound, initializedcheck
from libc.stdint cimport uint8_t, uint32_t
from cython.parallel cimport prange
import numpy as np

@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
def cy_get_horizontal_adjacency(uint8_t[:, ::1] cells):
    cdef int nrows = cells.shape[0]
    cdef int ncols = cells.shape[1]
    cdef uint32_t[:, ::1] adjacency_horizontal = np.zeros((nrows, ncols), dtype=np.uint32)
    cdef int x, y, span
    for y in prange(nrows, nogil=True, schedule="static"):
        span = 0
        for x in reversed(range(ncols)):
            if cells[y, x] > 0:
                span += 1
            else:
                span = 0
            adjacency_horizontal[y, x] = span
    return np.array(adjacency_horizontal, copy=False)