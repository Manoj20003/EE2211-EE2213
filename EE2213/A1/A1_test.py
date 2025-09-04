import numpy as np
import A1_A0307665X as grading  # Make sure this filename matches the student's submission

# Define test inputs
x = np.array([1, 2, 5])
y = np.array([7, 0, 3])

# Call the student function
try:
    euclidean_dist, manhattan_dist = grading.A1_A0307665X(x, y)

    print("Euclidean Distance:", euclidean_dist[0])   
    print("Manhattan Distance:", manhattan_dist[0])   

except Exception as e:
    print("Error during grading:", str(e))


## Expected output: 
# Euclidean Distance: 6.63
# Manhattan Distance: 10