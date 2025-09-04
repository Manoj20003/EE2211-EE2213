import numpy as np
import A2_A0307665X as grading  # Make sure this filename matches the student's submission

# Define test inputs   
road_map = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}
city_coordinates = {
    'A': (0, 0),
    'B': (1, 0),
    'C': (1, 1),
    'D': (2, 1)
}
start_city = 'A'
destination_city = 'D'  
# Call the student function
try:
    shortest_path, total_cost = grading.A2_A0307665X(road_map, city_coordinates, start_city, destination_city)

    print("Shortest Path:", shortest_path)   
    print("Total Cost:", total_cost)    
except Exception as e:
    print("Error during grading:", str(e))
    print("Please check the implementation of the function.")


## Expected output:
# Shortest Path: ['A', 'B', 'C', 'D']
# Total Cost: 4.0
