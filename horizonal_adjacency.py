import cv2 as cv
img = cv.imread("mask.png", 0)

from time import time
import numpy as np
import numba as nb

from cy.cy_horizontal_adjacency import cy_get_horizontal_adjacency

def get_horizontal_adjacency(cells):
    adjacency_horizontal = np.zeros(cells.shape, dtype=int)
    for y in range(cells.shape[0]):
        span = 0
        for x in reversed(range(cells.shape[1])):
            if cells[y, x] > 0:
                span += 1
            else:
                span = 0
            adjacency_horizontal[y, x] = span
    return adjacency_horizontal

@nb.jit('void(uint8[:,::1], int32[:,::1])', parallel=True)
def nb_get_horizontal_adjacency(cells, result):
    for y in nb.prange(cells.shape[0]):
        span = 0
        for x in range(cells.shape[1]-1,-1,-1):
            if cells[y, x] > 0:
                span += 1
            else:
                span = 0
            result[y, x] = span
    return

if __name__ == "__main__":
    ts = time()
    result1 = get_horizontal_adjacency(img)
    te = time()
    print(te-ts) # 3.675
    
    ts = time()
    result2 = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
    nb_get_horizontal_adjacency(img, result2)
    te = time()
    print(te-ts) # 0.002
    
    ts = time()
    result3 = cy_get_horizontal_adjacency(img)
    te = time()
    print(te-ts) # 0.005

    np.testing.assert_array_equal(result1, result2)
    np.testing.assert_array_equal(result1, result3)