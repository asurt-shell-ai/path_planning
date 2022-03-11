from matplotlib import pyplot as plt
import numpy as np


class Find_Path (object):
    def __init__(self, graph_points: np.array):
        self.graph_points = graph_points
        self.n_nodes = self.graph_points.shape[0]
        self.prob_matrix = np.zeros((self.n_nodes, self.n_nodes))
        self.pheremone_matrix = np.zeros((self.n_nodes, self.n_nodes))

        a = graph_points.reshape(self.n_nodes, 1, 2)
        b = a.reshape(1, self.n_nodes, 2)
        self.distance_matrix = np.linalg.norm(a-b, axis=2)

        self.best_route = []
        self.best_route_indices = []


    """ ACO Implementation """

    def __get_max_prob_index(self, root_index, visited, beta=5, alpha=1):

        # list contains the probability of each unvisited node
        probs = []

        # we want the pheremones of the nodes we'didnt visit yet
        root_pheremones = np.multiply(
            self.pheremone_matrix[root_index], np.logical_xor(visited, 1))

        # if all the pheremones are zeros, it means that we are still in the first Iteration
        if np.all(root_pheremones == 0):
            return -1

        # pher ** alpha
        pher_pow_alpha = np.power(root_pheremones, alpha)

        # ( 1 / dist ) ** beta
        dists_pow_beta = np.power(
            np.divide(1, self.distance_matrix[root_index]), beta)

        # since the dist can be zero so 1/dist can be inf , so we replace it with 0
        dists_pow_beta[dists_pow_beta == np.inf] = 0

        # sum of all unvisited nodes over (pher_i ** alpha) * ((1 / dists_i) ** beta)
        s = np.sum(np.multiply(pher_pow_alpha, dists_pow_beta))

        # filling the probability list
        for pher, dist in zip(pher_pow_alpha, dists_pow_beta):
            probs.append((pher * dist) / s)

        # retunring the index of the max probability
        return probs.index(max(probs))

    def __update_pheremones(self, distances_travelled, paths, Q=100, row=.5):
        for k in range(len(paths)):
            for i in range(len(paths[k]) - 1):

                node_i = paths[k][i]
                after_node_i = paths[k][i+1]

                # as the graph is undirceted
                self.pheremone_matrix[node_i][after_node_i] = row * \
                    self.pheremone_matrix[node_i][after_node_i] + \
                    Q/distances_travelled[k]
                self.pheremone_matrix[after_node_i][node_i] = row * \
                    self.pheremone_matrix[after_node_i][node_i] + \
                    Q/distances_travelled[k]

    def ant_colony(self):
        # like MAX_INT in c++
        best_total_distance = 1000000

        n_iterations = self.n_nodes * 3
        k_ants = self.n_nodes * 3

        for i in range(n_iterations):

            # to store the total pherimones secreted by k_ants
            distances_travelled = []
            # contains all the paths in the iteration
            paths = []
            for k in range(k_ants):
                # Evaluates array of Falses
                visited = np.zeros(self.n_nodes)

                # Distance travelled by the kth_ant
                distance_travelled = 0

                # a random root choosen from the graph
                root = np.random.default_rng().choice(self.graph_points, 1, replace=False)
                root_index = np.where(
                    np.all(self.graph_points == root, axis=1))[0][0]

                # The root is marked as visited
                visited[root_index] = True

                # ndArray of all the nodes excpet the root as it's visited
                nodes_not_visited = np.delete(self.graph_points, root_index, 0)

                # add the root node to the path
                path_indices = [root_index]

                iterator_root = root
                iterator_root_index = root_index
                while not np.all(visited == True):

                    max_prob_index = self.__get_max_prob_index(
                        iterator_root_index, visited)
                    # if it's the first iteration
                    if max_prob_index == -1:
                        # a random root choosen from neighbour_nodes
                        # which is garanteed to be not visited
                        new_root = np.random.default_rng().choice(nodes_not_visited, 1, replace=False)

                        # the indexing is based on graph_points
                        new_root_index = np.where(
                            np.all(self.graph_points == new_root, axis=1))[0][0]

                    # if it's not the first iteration
                    else:
                        # making the next root with maximum prob in the probability list
                        # The index is based on Graph Points
                        new_root_index = max_prob_index
                        new_root = self.graph_points[new_root_index]

                    # The new_root is marked as visited
                    visited[new_root_index] = True

                    # taking the index of the root based on neighbours array
                    new_root_index_neighbour = np.where(
                        np.all(nodes_not_visited == new_root, axis=1))[0][0]

                    # ndArray of all the nodes excpet the new_root
                    nodes_not_visited = np.delete(
                        nodes_not_visited, new_root_index_neighbour, 0)

                    # add the new_root node to the path
                    path_indices.append(new_root_index)

                    # distance travelled by the ant
                    distance_travelled += self.distance_matrix[iterator_root_index][new_root_index]

                    # Iterating
                    iterator_root = new_root
                    iterator_root_index = new_root_index

                # appending the index of the root node to make a cycle
                path_indices.append(root_index)

                # adding the distance from the last node to the root
                distance_travelled += self.distance_matrix[iterator_root_index][root_index]

                # Update the paths
                paths.append(path_indices)

                # update the distances travelled
                distances_travelled.append(distance_travelled)

                if distance_travelled < best_total_distance:
                    best_total_distance = distance_travelled
                    self.best_route_indices = path_indices

                # print(distances_travelled)

            # print(paths)
            # print(distance_travelled)

            # update the pheremone Matrix
            self.__update_pheremones(distances_travelled, paths)

            # print(pheremone_matrix)

        print(f"{best_total_distance = }")
        self.best_route = np.array([self.graph_points[i]
                                   for i in self.best_route_indices])
        return self.best_route

    def plot_best_route(self):
        if not len(self.best_route):
            print("you should find the path first\nuse ant_colony()")
            return
        # scatter plot of the graph points
        plt.scatter(self.graph_points[:, 0], self.graph_points[:, 1])
        plt.plot(self.best_route[:, 0], self.best_route[:, 1])
        plt.show()

    def _g(self, i, s) -> int:
        # print(i, s)
        # Base Case : s is an empty set
        if not len(s):
            return self.distance_matrix[i][0]

        return min([self.distance_matrix[i][k] + self._g(k, s-{k}) for k in s])

    # SUPER SLOW but finds the distance of the optimal path
    def DP_optimal_distance(self):
        return self._g(0, set(range(self.n_nodes))-{0})


# TEST
# graph_points = np.array(
#     [[0.,    0.],
#      [114.28714842,   41.98759603],
#      [62.47783741,  -10.16037304],
#      [24.5058101,  179.34217451],
#      [-9.51912637,   86.66461683],
#      [-107.49794568,   23.53735607],
#      [91.98201947,  153.39087436],
#      [130.74459844,   95.76204485],
#      [100.11681872,   -9.67705624],
#      [-97.97803427, -116.85725875],
#      [12.76266999,  200.86528354],
#      [35.42322761,  -86.24626489],
#      [-15.17293957,  -39.18455774],
#      [50.71204359,  -43.96724648],
#      [-165.73686143,  -40.45509785],
#      [-59.11389879,  -75.68251789],
#      [-65.86307249,  -51.20886232],
#      [-118.65056562,   58.30787596],
#      [0.25090536,   49.73685338],
#      [95.02087717,   63.66352072]]
# )

# a = Find_Path(graph_points)
# print(a.ant_colony())
# a.plot_best_route()

# use it if you have a super computer 
# print(a.DP_optimal_distance()) 
