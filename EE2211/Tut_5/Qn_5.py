import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv



def Qn_5():

    df = pd.read_csv("government_expenditure_on_education.csv")
    df = pd.DataFrame(df)

    X = np.array([[1]* len(df), df['year'].tolist()]).T
    y = df['total_expenditure_on_education'].tolist()
    y = np.array(y).reshape(-1,1)

    w = inv(X.T@X) @ X.T @ y

    y_new = np.array([1, 2021]) @ w

    print(y_new)
    return y_new

Qn_5()