import numpy as np


# Please replace "MatricNumber" with your actual matric number here and in the filename
def A1_A0307665X(x, y):
    """
    Input type
    :x type: numpy.ndarray
    :y type: numpy.ndarray

    Return type
    :euclidean_dist type: numpy.ndarray
    :manhattan_dist type: numpy.ndarray
   
    """

    # your code goes here
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x or y is not a 1D array")

    if len(x) != len(y):
        raise ValueError("x and y are not the same length")
    
    diff = x - y

    euclidean_dist = np.sqrt(np.sum(diff ** 2))
    manhattan_dist = np.sum(np.abs(diff))

    euclidean_dist = np.round(euclidean_dist, 2)
    manhattan_dist = np.round(manhattan_dist, 2)

    # return in this order
    return euclidean_dist, manhattan_dist


x = np.array([1, 2, 5])
y = np.array([7, 0, 3])
A1_A0307665X(x, y)

