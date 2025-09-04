import numpy as np
import heapq

# Please replace "MatricNumber" with your actual matric number here and in the filename
def A2_A0307665X(road_map, city_coordinates, start_city, destination_city):
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

    # Hint 1: You may want to use a dictionary to store the best known cost (g-score) and path for each city.
    # Example structure (you may use a different structure if you prefer): node_data = {city: {'dist': ..., 'prev': [...]}}

    # Hint 2: You will need a priority queue (e.g., min-heap) to explore cities in order of their f = g + h values.
    # You can use heapq.heappush() and heapq.heappop() for priority queue operations.

    # Heuristic function: Euclidean distance between two cities
    def heuristic(city1, city2):
        coord1 = np.array(city_coordinates[city1])
        coord2 = np.array(city_coordinates[city2])
        return np.linalg.norm(coord1 - coord2)  

    # Initialize the data structures here
    node_data = {city:{'dist': np.inf, 'prev': []} for city in road_map}
    node_data[start_city]['dist'] = 0
    visited = set()

    priority_queue = []
    heapq.heappush(priority_queue, (0 + heuristic(start_city, destination_city), start_city))
    

    # Main A* search loop goes here
    while priority_queue:
        f, city = heapq.heappop(priority_queue)

        if city == destination_city:
            path = []
            while city:
                path.append(city)
                city = node_data[city]['prev']
            return path[::-1], node_data[destination_city]['dist']
        
        if city not in visited:
            visited.add(city)
        
        for neighbour, cost in road_map[city].items():
            if city in visited:
                continue
            tentative_g = node_data[city]['dist'] + cost

            if tentative_g < node_data[neighbour]['dist']:
                node_data[neighbour]['dist'] = tentative_g
                node_data[neighbour]['prev'] = city
                heapq.heappush(priority_queue, (tentative_g + heuristic(neighbour, destination_city), neighbour))

        




    # If no path is found, return None and np.inf
    # (Do not change or modify this line.)
    return None, np.inf  
