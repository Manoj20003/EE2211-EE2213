#import libraries
import cvxpy as cp #For so,ving linear programs
import numpy as np #For numerical computations (used for plotitng here)

#1. Define variannle
x_A = cp.Variable(name="product_A",integer=True) #integer variable 
x_B = cp.Variable(name="product_B",integer=True) #integer variable

#2. Define objective
objective = cp.Maximize(40 * x_A + 30 * x_B)

#3. Define constraints
constraints = [
    x_A >= 0,                   #non-negativity for A
    x_B >= 0,                   #non-negativity for B
    2 * x_A + 1 * x_B <= 100,   #Machine time limit
    x_B >= 10,                  #Minimum product B
    x_A + x_B <= 40             #Production Capacity
    ]

#4. Create the CVXPY optimization problem by combining the objective and the constraints
prob = cp.Problem(objective, constraints)

#5. Solve the LP problem
prob.solve()

#6. Results
xA_opt = x_A.value #Get the optimal value of decision variable x_A
XB_opt = x_B.value #Get the optimal value of decision variable x_b
max_profit = prob.value #Get the maximum profit from the objective function
problem_status = prob.status #Get the solution status

print(max_profit)