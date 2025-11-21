import numpy as np

def gd_scalar(grad, w0, lr=0.1, steps=10):
    w = w0
    for i in range(steps):
        w = w - lr * grad(w)
        print(i, w)
    return w
grad = lambda w: np.sin(2*w)
gd_scalar(grad, w0=3.0, lr=0.1, steps=5)
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def poly_reg(X, y, order, Xtest, ytest=None, lam=0):
    poly = PolynomialFeatures(order)
    P = poly.fit_transform(X)

    if lam == 0:
        w = np.linalg.lstsq(P, y, rcond=None)[0]
    else:
        I = np.eye(P.shape[1])
        w = np.linalg.inv(P.T@P + lam*I) @ P.T @ y

    # Training MSE
    ypred = P @ w
    train_mse = np.mean((ypred - y)**2)

    # Test MSE
    if ytest is not None:
        Ptest = poly.transform(Xtest)
        ytest_pred = Ptest @ w
        test_mse = np.mean((ytest_pred - ytest)**2)
        return w, train_mse, test_mse

    return w, train_mse
import numpy as np

def split_mse(X, Y, threshold):
    left = X < threshold
    right = ~left

    Y1, Y2 = Y[left], Y[right]
    m1, m2 = np.mean(Y1), np.mean(Y2)

    left_mse = np.mean((Y1 - m1)**2) if len(Y1)>0 else 0
    right_mse = np.mean((Y2 - m2)**2) if len(Y2)>0 else 0
    overall = (len(Y1)*left_mse + len(Y2)*right_mse) / len(Y)

    return left_mse, right_mse, overall
def best_split(X, Y):
    thresholds = np.unique(X)
    best_thr, best_err = None, 1e18

    for t in thresholds:
        _,_,err = split_mse(X,Y,t)
        if err < best_err:
            best_thr, best_err = t, err

    return best_thr, best_err
import numpy as np

def solve_system(A, b):
    # Check consistency
    rA = np.linalg.matrix_rank(A)
    rAb = np.linalg.matrix_rank(np.column_stack((A,b)))

    if rA == rAb:
        print("Exact solution exists.")
        sol = np.linalg.lstsq(A,b,rcond=None)[0]
    else:
        print("No exact solution — using left inverse.")
        sol = np.linalg.inv(A.T@A) @ A.T @ b

    return sol
import numpy as np

def pearson(X, Y):
    X = np.array(X)
    Y = np.array(Y)

    Xc = X - X.mean()
    Yc = Y - Y.mean()

    return (Xc*Yc).sum() / (len(X)*X.std()*Y.std())
def poly_params(k, d):
    from math import comb
    return comb(k + d, d)
def kmeans(X, k, steps=10):
    import numpy as np
    np.random.seed(0)

    centers = X[np.random.choice(len(X), k, replace=False)]
    for _ in range(steps):
        labels = np.argmin(((X[:,None,:]-centers[None,:,:])**2).sum(axis=2), axis=1)
        for i in range(k):
            centers[i] = X[labels==i].mean(axis=0)
    return centers, labels
def law_total_probability(priors, conditionals):
    # priors = [P(A), P(B)]
    # conditionals = [P(X|A), P(X|B)]
    return sum(p*c for p,c in zip(priors, conditionals))
def entropy(p):
    import numpy as np
    p = np.array(p)
    p = p[p>0]
    return -(p*np.log2(p)).sum()

def info_gain(parent, left, right):
    N = len(parent)
    return entropy(parent) - (len(left)/N)*entropy(left) - (len(right)/N)*entropy(right)
def metrics(tp, fp, fn, tn):
    acc = (tp+tn)/(tp+fp+fn+tn)
    fpr = fp/(fp+tn)
    fnr = fn/(fn+tp)
    return acc, fpr, fnr
import sympy as sp
import numpy as np

def auto_gd(cost, vars, w0, lr=0.1, steps=10):
    grads = [sp.diff(cost, v) for v in vars]
    f_grads = [sp.lambdify(vars, g, 'numpy') for g in grads]

    w = np.array(w0, float)
    for i in range(steps):
        g = np.array([fg(*w) for fg in f_grads])
        w = w - lr * g
        print(i, w)
    return w
