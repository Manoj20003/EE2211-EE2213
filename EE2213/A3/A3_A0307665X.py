import numpy as np
import cvxpy as cp

# Please replace "StudentMatriculationNumber" with your actual matric number in the filename
# Please do NOT change the function names in this file.
# Filename should be: A3_StudentMatriculationNumber.py (replace “StudentMatriculationNumber” with your own your student matriculation number).






def optimize_shipments(supply, demand, cost_matrix):
    """
    Problem 2: Logistics Optimization
    
    Inputs:
    :supply: list of int
        List of factory capacities [China, India, Brazil]
    :demand: list of int
        List of market demands [Singapore, US, Germany, Japan]
    :cost_matrix: 2D list (3x4)
        3x4 matrix where cost_matrix[i][j] is cost from factory i to market j
        Rows correspond to factories [China, India, Brazil].
        Columns correspond to markets [Singapore, US, Germany, Japan].
        
    Returns:
    :minimal_cost: float
        The total minimized transportation cost.
    :shipment_matrix: numpy.ndarray
        3x4 array of integers where shipment_matrix[i, j] is units 
        shipped from factory i to market j.
        Rows correspond to factories [China, India, Brazil].
        Columns correspond to markets [Singapore, US, Germany, Japan].
    """

    """
    China: c
    India: i
    Brazil: b
    Singapore: s
    US: u
    Germany: g 
    Japan: j

    c <= 50
    i <= 30
    b <= 40
    s >= 20
    u >= 45
    g >= 25
    j >= 30


    min = 10cs + 25cu + 30cg + 20cj + 12is + 32iu + 25ig + 22ij + 35bs + 20bu + 15bg + 40bj

    """

    supply = np.array(supply)
    demand = np.array(demand)
    C = np.array(cost_matrix)

    S, D = C.shape


    x = cp.Variable((S, D), integer=True)

    cost = cp.Minimize(cp.sum(cp.multiply(C, x)))

    constraints = [
        x >= 0  
    ]
    
    for i in range(S):
        constraints.append(cp.sum(x[i, :]) <= supply[i])

    for j in range(D):
        constraints.append(cp.sum(x[:, j]) >= demand[j])


    prob = cp.Problem(cost, constraints)
    
    prob.solve()


    shipment_matrix = np.rint(x.value).astype(int)
    minimal_cost = float(prob.value)
 
##    print("Minimal Cost:", minimal_cost)
##    print("Shipment Matrix:\n", shipment_matrix)
    # Replace the following with actual return values
    return minimal_cost, shipment_matrix


def gradient_descent(learning_rate, num_iters):
    """
    Problem 2: Gradient Descent

    Inputs:
    :learning_rate: float
        The learning rate for gradient descent. Value between 0 and 0.2.
    :num_iters: int
        Number of gradient descent iterations.

    Returns:
    :w_out: numpy.ndarray
        Array of length num_iters containing updated w values at each step.
    :f_out: numpy.ndarray
        Array of length num_iters containing f(w) = 1 + (w - 5)^2 at each step.
    """

    w = 3.5
    w_out = np.zeros(num_iters, dtype=float)
    f_out = np.zeros(num_iters, dtype=float)


    for t in range(num_iters):
        grad = 2.0 * (w - 5.0)
        w = w - learning_rate * grad
        w_out[t] = w
        f_out[t] = 1.0 + (w - 5.0) ** 2

    return w_out, f_out



