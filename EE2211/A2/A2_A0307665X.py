import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

# Please replace "MatricNumber" with your actual matric number here and in the filename
def A2_A0307665X(N):
    """
    Input type
    :N type: int

    Return type
    :X_train type: numpy.ndarray of size (number_of_training_samples, 4)
    :y_train type: numpy.ndarray of size (number_of_training_samples,)
    :X_test type: numpy.ndarray of size (number_of_test_samples, 4)
    :y_test type: numpy.ndarray of size (number_of_test_samples,)
    :Ytr type: numpy.ndarray of size (number_of_training_samples, 3)
    :Yts type: numpy.ndarray of size (number_of_test_samples, 3)
    :Ptrain_list type: List[numpy.ndarray]
    :Ptest_list type: List[numpy.ndarray]
    :w_list type: List[numpy.ndarray]
    :error_train_array type: numpy.ndarray
    :error_test_array type: numpy.ndarray
    """
    # your code goes here
    X_train, X_test, y_train, y_test = train_test_split(load_iris().data, load_iris().target, test_size=0.6, random_state=N)

    encoder = OneHotEncoder(sparse_output=False)
    Ytr = encoder.fit_transform(y_train.reshape(-1, 1))
    Yts = encoder.transform(y_test.reshape(-1, 1))
    Ptrain_list = []
    Ptest_list = []
    w_list = []
    error_train_array = np.zeros(10, dtype=int)
    error_test_array = np.zeros(10, dtype=int)
    lambda_ = 0.0001

    for degree in range(1, 11):
        pf = PolynomialFeatures(degree=degree)
        P_tr = pf.fit_transform(X_train)
        P_te = pf.transform(X_test)

        Ptrain_list.append(P_tr)
        Ptest_list.append(P_te)

        n_tr, p = P_tr.shape


        if n_tr > p:
            XtX = P_tr.T @ P_tr
            W = np.linalg.inv(XtX + lambda_ * np.eye(p)) @ (P_tr.T @ Ytr)
        else:
            K = P_tr @ P_tr.T
            W = P_tr.T @ (np.linalg.inv(K + lambda_ * np.eye(n_tr)) @ Ytr)

        w_list.append(W)

        train_pred = np.argmax(P_tr @ W, axis=1)
        test_pred = np.argmax(P_te @ W, axis=1)

        error_train_array[degree - 1] = int(np.sum(train_pred != y_train))
        error_test_array[degree - 1] = int(np.sum(test_pred != y_test))

    # return in this order
    
    return X_train, y_train, X_test, y_test, Ytr, Yts, Ptrain_list, Ptest_list, w_list, error_train_array, error_test_array


