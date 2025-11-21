import numpy as np
from numpy.linalg import inv

# Please replace "MatricNumber" with your actual matric number here and in the filename
def A1_A0307946U(X, y):
    """
    Input type
    :X type: numpy.ndarray
    :y type: numpy.ndarray

    Return type
    :InvXTX type: numpy.ndarray
    :w type: numpy.ndarray
   
    """

    # your code goes here
    InvXTX = inv(X.T @ X)
    w = InvXTX @ X.T @ y

    # return in this order
    return InvXTX, w

# test
A = np.array([[1, 1], [4, 2], [4, 6], [3, -6], [0, -10]])
B = np.array([[-3], [2], [1], [5], [4]])
C = np.array([[1, 1], [1, -1], [1, 0]])
D = np.array([[1], [0], [2]])

print(A1_A0307946U(A,B))
print(A1_A0307946U(C,D))
