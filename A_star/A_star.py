import numpy as np
import matplotlib.pyplot as plt
from pygame import init


class Find_Path():
    def __init__(self, start_index: int, end_index: int, graph_points: np.array, adj_list: list):
        self.start_index = start_index
        self.end_index = end_index
        self.graph_points = graph_points
        self.adj_list = adj_list

        self.path = []
        self.path_distance = 0

        # no. of nodes in the graph points
        self.n_nodes = self.graph_points.shape[0]

        # using proadcasating to calculate the cost matrix
        a = self.graph_points.reshape(self.n_nodes, 1, 2)
        b = a.reshape(1, self.n_nodes, 2)
        self.cost_matrix = np.linalg.norm(a-b, axis=2)

        """
        !if we make l = [[]]*5
        here we made one list with 5 copies of its reference
        so please DON'T USE IT
        """
        self.adj_list_with_cost = [[] for i in range(self.n_nodes)]

        for i in range(self.n_nodes):
            for j in adj_list[i]:
                self.adj_list_with_cost[i].append((j, self.cost_matrix[i][j]))

    def A_star(self) -> list:
        # calculating the direct distance from all the points to the end point
        heuristic_list = np.linalg.norm(
            self.graph_points-self.graph_points[self.end_index], axis=1)

        # list contains all the visited nodes and its parent
        done = [-1]*self.n_nodes

        # [node_index, from_node_index, total_cost, combined heuristic = h[child] + c[parent]]
        # it should be priority queue but u know python
        pq = [[self.start_index, None, 0, heuristic_list[self.start_index]]]

        # [node_index, total_cost]
        cur_node = [self.start_index, 0]
        while cur_node[0] != self.end_index:

            # the indices of the cur node and the its prev node
            cur_index, prev_index = pq.pop(0)[:2]
            done[cur_index] = prev_index

            for next_edge in self.adj_list_with_cost[cur_node[0]]:
                # edge = [node_index, cost from cur_node]

                # if we are visiting an already visited node then continue
                if next_edge[0] in done:
                    continue

                # the cost of visiting the next edge = the cost of visiting the current node + the cost of the edge
                next_edge_cost = next_edge[1] + cur_node[1]

                # if the next edge is still in the pq
                next_edge_index = [i for i, v in enumerate(
                    pq) if v[0] == next_edge[0]]
                if next_edge_index:
                    # if the new cost < the cost in the pq
                    if next_edge_cost < pq[next_edge_index[0]][2]:
                        pq[next_edge_index[0]][1] = cur_node[0]
                        pq[next_edge_index[0]][2] = next_edge_cost
                        pq[next_edge_index[0]][3] = next_edge_cost + \
                            heuristic_list[next_edge[0]]
                else:
                    # exploring a new node which is not in the pq
                    pq.append([next_edge[0], cur_node[0], next_edge_cost,
                               next_edge[1] + heuristic_list[next_edge[0]]])

            # sort based on the combined heuristic value
            pq = sorted(pq, key=lambda x: x[3])

            # the cur_node will be the node with the least heuristic
            cur_node = (pq[0][0], pq[0][2])

            # if the node which we are going to visit is the goal, then we will append it to done
            if cur_node[0] == self.end_index:
                cur_index, prev_index = pq[0][:2]
                done[cur_index] = prev_index

        # constructing the path
        self.path = [self.end_index]
        i = self.end_index
        while done[i] != None:
            self.path_distance += self.cost_matrix[i][done[i]]
            self.path.append(done[i])
            i = done[i]

        return list(reversed(self.path))
