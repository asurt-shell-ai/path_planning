import matplotlib.pyplot as plt
import numpy as np
from itertools import permutations
from A_star import *

corner_points = np.load("./corner_points.npy")
map = np.load("./manual_map.npy")

road_points = np.load("./test1_5points.npy")

road_points = np.concatenate((corner_points, road_points))
print(road_points)
path_adj_list = [
    [1, 6],
    [0, 12, 2],
    [1, 5, 14],
    [14, 13],

    [12, 7, 5],
    [2, 8, 4],

    [0, 15, 9],
    [15, 4, 8],
    [7, 5, 10],

    [6, 16],
    [16, 8, 11],
    [13, 10],



    [1, 4],
    [3, 11],
    [2, 3],
    [6, 7],
    [9, 10]
]



""" BRUTE FORCE THE HELL OUT OF IT """

best_path = []
best_per_dist = np.inf
graph_points = road_points
print(graph_points)
# generates all the permutation of the graph points
permutations = list(permutations(
    range(len(corner_points), len(road_points))))

print(permutations)
for permutation in permutations:
    per_distance = 0
    per_path = []

    print(permutation)

    # looping on every two consecutive points
    for i in range(len(permutation) - 1):
        # using A* to find the optimal rout between two points

        print(i, permutation[i], permutation[i+1], graph_points.shape[0])
        find_obj = Find_Path(
            permutation[i], permutation[i+1], graph_points, path_adj_list)

        per_path += [road_points[i]
                     for i in find_obj.A_star()]

        # the distance travelled in the path
        per_distance += find_obj.path_distance

    # getting the best optimal path by comparing distances
    if best_per_dist > per_distance:
        best_per_dist = per_distance
        best_path = per_path

path = np.array(best_path)

print(path)
np.save("./test1_path.npy", path)

plt.plot(path[:, 0], path[:, 1])
plt.scatter(road_points[:, 0], road_points[:, 1], c="red")
plt.scatter(map[:, 0], map[:, 1], c="green", alpha=.1)
plt.scatter(corner_points[:, 0], corner_points[:, 1], c="blue")

plt.grid()
plt.show()