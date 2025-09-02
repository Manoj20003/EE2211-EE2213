import numpy as np


# Please replace "MatricNumber" with your actual matric number here and in the filename
def A1_A0307665X(X, y):
    """
    Input type
    :X type: numpy.ndarray
    :y type: numpy.ndarray

    Return type
    :InvXTX type: numpy.ndarray
    :w type: numpy.ndarray
   
    """

    # your code goes here
    InvXTX = np.linalg.inv(X.T @ X)
    w = InvXTX @ (X.T @ y)


    return InvXTX, w


