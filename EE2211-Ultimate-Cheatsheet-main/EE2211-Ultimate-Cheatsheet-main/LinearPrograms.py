# Import libraries
import cvxpy as cp  # For solving linear programs
import numpy as np
import matplotlib.pyplot as plt


# 1. Define variables
x_1 = cp.Variable()   # defines a continuous decision variable x_1.
x_2 = cp.Variable()   # defines a continuous decision variable x_2.
x_3 = cp.Variable()   # defines a continuous decision variable x_3.
## DECLARE MORE VARIABLES IF NEEDED

# 2. Define objective
objective = cp.Maximize( 7 * x_1 + 2 * x_2 + 6 * x_3)  # Objective function to maximize

# 3. Define constraints
constraints = [
    x_1 >= 0,
    x_2 >= 0,
    x_3 >= 0,
    2 *x_1 + x_2 + 3 * x_3 <= 14,
    -1 * x_1 + 4 * x_2 + x_3 >= 3,
    3* x_1 - 2 * x_2 + 2 *x_3 <= 12,
    1* x_1 + 5 * x_2 + 2 *x_3 <= 20,
    1* x_1 + 1 * x_2 + 3 *x_3 >= 5,
]

# 4. Create a CVXPY problem object
prob = cp.Problem(objective, constraints)

# 5. Solve the LP problem
prob.solve()

# 6. Results
x1_opt = x_1.value   # Get the optimal value of decision variable x_1
x2_opt = x_2.value   # Get the optimal value of decision variable x_2
x3_opt = x_3.value   # Get the optimal value of decision variable x_3
## ADD THE MORE VARIABLES HERE IF USED 
min_obj = prob.value # Get the minimum from the objective function
problem_status = prob.status # Get the solution status
status = prob.status
obj_opt = prob.value

# Output results
# print("\nSolution Status:", problem_status)
# print("x1:", x1_opt)
# print("x2:", x2_opt)
# print("x3:", x3_opt)
# ## ADD THE MORE VARIABLES HERE IF USED
# print("Minimum objective value:", min_obj)

if status in ["infeasible", "infeasible_inaccurate"]:
    print("Result: ❌ The LP is infeasible (no solution satisfies all constraints).")

elif status in ["unbounded", "unbounded_inaccurate"]:
    print("Result: ⚠️ The LP is unbounded (objective can grow infinitely).")

elif status in ["optimal", "optimal_inaccurate"]:
    print("Result: ✅ Feasible and bounded (optimal solution found).")
    print(f"x1 = {x1_opt}")
    print(f"x2 = {x2_opt}")
    print(f"x3 = {x3_opt}")
    print("Maximum objective value:", obj_opt)

else:
    print("⚠️ Solver returned an unexpected status. Interpretation uncertain.")

print("==================================\n")



X1, X2 = np.meshgrid(np.linspace(0, 10, 400), np.linspace(0, 10, 400))

# Feasibility mask
feasible = (
    (X1 >= 0) &
    (X2 >= 0) &
    (X1 + X2 <= 5) &
    (2*X1 + X2 >= 12) &
    (-X1 + 2*X2 <= 4) &
    (X1 - X2 <= 3)
)

# Plot feasible region
plt.figure(figsize=(8, 8))
plt.imshow(feasible.astype(int), extent=[0, 10, 0, 10],
           origin='lower', cmap='Greens', alpha=0.4)

# Plot constraint boundary lines
x = np.linspace(0, 10, 400)
plt.plot(x, 5 - x, label="x1 + x2 = 5")
plt.plot(x, 12 - 2*x, label="2x1 + x2 = 12")
plt.plot(x, (x + 4)/2, label="-x1 + 2x2 = 4")
plt.plot(x, x - 3, label="x1 - x2 = 3")

# Plot optimal point
plt.scatter(x1_opt, x2_opt, color='red', s=100, label="Optimal Solution")

plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Feasible Region & Optimal Point")
plt.legend()
plt.show()
plt.savefig("feasible_region.png")