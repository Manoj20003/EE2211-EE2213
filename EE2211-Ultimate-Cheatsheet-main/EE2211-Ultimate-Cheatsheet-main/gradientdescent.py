import numpy as np

def gradient_descent_scalar(gradient, w0, lr=0.1, steps=20):
    """
    gradient: function that returns dC/dw at a given w
    w0: initial value
    lr: learning rate
    steps: number of iterations
    """
    w = w0
    history = []

    for i in range(steps):
        grad = gradient(w)
        w = w - lr * grad
        history.append((i, w, grad))

    return history


# ------------------------
# Example: C(w) = sin^2(w)
# dC/dw = sin(2w)
# ------------------------

grad = lambda w: np.sin(2*w)

result = gradient_descent_scalar(grad, w0=3.0, lr=0.1, steps=10)

for step, w, g in result:
    print(f"step {step:2d}: w = {w:.6f}, gradient = {g:.6f}")





import numpy as np

def gradient_descent_vector(gradient, w0, lr=0.1, steps=20):
    """
    gradient: function returning gradient vector ∇C(w)
    w0: initial vector, numpy array
    """
    w = w0.astype(float)
    history = []

    for i in range(steps):
        grad = gradient(w)
        w = w - lr * grad
        history.append((i, w.copy(), grad.copy()))

    return history


# ------------------------
# Example:
# C(x,y) = x^2 + x*y^2
# gradient = [2x + y^2, 2xy]
# ------------------------

def grad(v):
    x, y = v
    return np.array([
        2*x + y**2,
        2*x*y
    ])

w0 = np.array([3.0, 2.0])

result = gradient_descent_vector(grad, w0, lr=0.2, steps=5)

for step, w, g in result:
    print(f"step {step:2d}: w = {w}, gradient = {g}")





import numpy as np
import sympy as sp

def auto_gradient_descent(cost_expr, variables, w0, lr=0.1, steps=20):
    # Automatic gradient symbols
    grads = [sp.diff(cost_expr, var) for var in variables]
    
    # lambdify cost + grads for numeric evaluation
    f_cost = sp.lambdify(variables, cost_expr, 'numpy')
    f_grads = [sp.lambdify(variables, g, 'numpy') for g in grads]

    w = np.array(w0, dtype=float)
    history = []

    for i in range(steps):
        grad_vals = np.array([g(*w) for g in f_grads], dtype=float)
        w = w - lr * grad_vals
        history.append((i, w.copy(), grad_vals.copy()))

    return history


# -----------------------------
# Example: C(x,y) = x^2 + x*y^2
# -----------------------------
x, y = sp.symbols('x y')
C = x**2 + x*y**2

initial = [3.0, 2.0]

res = auto_gradient_descent(C, [x, y], initial, lr=0.2, steps=5)

for step, w, grad in res:
    print(f"step {step:2d}: w = {w}, grad = {grad}")
