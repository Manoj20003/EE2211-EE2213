import numpy as np

# Please replace "MatricNumber" with your actual matric number here and in the filename
def A2_MatricNumber(road_map, city_coordinates, start_city, destination_city):
    """
    Input types:
    :road_map type: dict
        Dictionary representing the weighted graph. 
        Example: {'A': {'B': 5, 'C': 10}, 'B': {'D': 2}, ...}

    :city_coordinates type: dict
        Dictionary mapping city names to (x, y) coordinates. 
        Example: {'A': (0, 0), 'B': (3, 4), 'C': (6, 0), ...}

    :start_city type: str
    :destination_city type: str

    Return types:
    :shortest_path type: list
        List of city names from start to destination (inclusive).

    :total_cost type: float
        Total path cost (sum of weights along the path).
    """

    # Heuristic function: Euclidean distance between two cities
    def heuristic(city1, city2):
        coord1 = np.array(city_coordinates[city1])
        coord2 = np.array(city_coordinates[city2])
        return np.linalg.norm(coord1 - coord2)

    # Hint 1: You may want to use a dictionary to store the best known cost (g-score) and path for each city.
    # Example structure (you may use a different structure if you prefer): node_data = {city: {'dist': ..., 'prev': [...]}}

    # Hint 2: You will need a priority queue (e.g., min-heap) to explore cities in order of their f = g + h values.
    # You can use heapq.heappush() and heapq.heappop() for priority queue operations.

    # Initialize the data structures here
    # ...

    # Main A* search loop goes here
    # ...

    # If no path is found, return None and np.inf
    # (Do not change or modify this line.)
    return None, np.inf  
