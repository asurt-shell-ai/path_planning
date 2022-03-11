from email import iterators
from itertools import permutations
from tkinter import W
import numpy as np
import pandas as pd
import pygame
import pygame.gfxdraw
from A_star import *

# initialize pygame
pygame.init()

# constants
WINDOW_SIZE = (500, 500)
WIDTH, HEIGHT = WINDOW_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 100, 0)
GREY = (128, 128, 128)
RED = (255, 0, 0)
ROAD_WIDTH = 50  # Just an Arbitrary Number
BLOCK_LENGTH = int((HEIGHT - 3 * ROAD_WIDTH)/2)  # BOOM MATH !!!

# variable controls what appears on the screen
# set the screen size to be 500px x 500px
screen = pygame.display.set_mode(WINDOW_SIZE)

# sets the window title
pygame.display.set_caption("TEST")

# Game Variables
is_running = True  # The state of the game
corner_points = []
path = []
min_path_distance = 0


# Draws a 2x2 grid
def draw_grid(screen):
    global ROAD_WIDTH, BLOCK_LENGTH, corner_points

    # DRAW Blocks
    for x in range(ROAD_WIDTH, WIDTH, ROAD_WIDTH+BLOCK_LENGTH):
        for y in range(ROAD_WIDTH, HEIGHT, ROAD_WIDTH+BLOCK_LENGTH):
            pygame.draw.rect(screen, GREEN, (x, y, BLOCK_LENGTH, BLOCK_LENGTH))

    # Draw corner points
    for coordinates in corner_points:
        pygame.draw.circle(screen, BLUE, coordinates, 5)


def draw_graph_points(screen):
    # print(graph_points, end="\n----------\n")
    for point_pos in graph_points:
        pygame.draw.circle(screen, RED, point_pos, 3)


def set_corner_points():
    global corner_points

    MID_ROAD = int(ROAD_WIDTH/2)
    for x in range(MID_ROAD, WIDTH, BLOCK_LENGTH+ROAD_WIDTH):
        for y in range(MID_ROAD, WIDTH, BLOCK_LENGTH+ROAD_WIDTH):
            corner_points.append([x, y])


def draw_path(screen, path):
    if path:
        for i in range(len(path)-1):
            pygame.draw.line(screen, WHITE, path[i], path[i+1], 2)


set_corner_points()
# Don't think about it
corner_points = [
    [25, 25],
    [25, 250],
    [25, 475],
    [250, 25],
    [250, 250],
    [250, 475],
    [475, 25],
    [475, 250],
    [475, 475]
]
# TEST 1
t1_graph_points = [
    [25, 85],
    [25, 318],
    [248, 131],
    [473, 134]
]

t1_adj_list = [
    [9, 3],
    [9, 4, 10],
    [10, 5],
    [0, 11, 6],
    [1, 11, 5, 7],
    [4, 2, 8],
    [3, 12],
    [12, 8, 4],
    [7, 5],

    [0, 1],
    [1, 2],
    [3, 4],
    [6, 7]
]
# TEST 2
t2_graph_points = [
    [25, 85],
    [25, 163],
    [25, 318],
    [100, 23],
    [176, 249],
    [248, 131],
    [329, 29],
    [473, 134]
]

t2_adj_list = [
    [12, 9],
    [10, 13, 11],
    [11, 5],
    [12, 14, 15],
    [13, 14, 5, 7],
    [2, 4, 8],
    [15, 16],
    [16, 4, 8],
    [5, 7],


    [0, 10],
    [9, 1],
    [1, 2],
    [3, 0],
    [4, 1],
    [3, 4],
    [6, 3],
    [6, 7]
]

# game loop
graph_points = corner_points + t1_graph_points
while is_running:

    # checks for an event
    for event in pygame.event.get():

        # if the exit button is pressed
        if event.type == pygame.QUIT:
            is_running = False

        # by every Mouse click the position is stored in graph_points
        elif event.type == pygame.MOUSEBUTTONDOWN:
            clickPos = pygame.mouse.get_pos()
            graph_points.append(list(clickPos))

        elif event.type == pygame.KEYDOWN:
            # if ENTER is Pressed
            if event.key == pygame.K_RETURN:

                """ BRUTE FORCE THE HELL OUT OF IT """

                best_path = []
                best_per_dist = np.inf
                # generates all the permutation of the graph points
                for permutation in list(permutations(t1_graph_points)):
                    per = list(permutation)
                    per_distance = 0
                    per_path = []

                    # adding the corner points to the graph
                    per_graph_points = np.array(corner_points + per)

                    # looping on every two consecutive points
                    for i in range(len(corner_points), len(per) + len(corner_points) - 1):
                        # using A* to find the optimal rout between two points
                        find_obj = Find_Path(
                            i, i+1, per_graph_points, t1_adj_list)

                        per_path += [graph_points[i]
                                     for i in find_obj.A_star()]
                        
                        # the distance travelled in the path
                        per_distance += find_obj.path_distance
                        
                    # getting the best optimal path by comparing distances
                    if best_per_dist > per_distance:
                        best_per_dist = per_distance
                        best_path = per_path

                # السلام عليكم ورحمة الله وبركاته 
                path = best_path
                print(path)

            # if C key is pressed, graph points will be cleared
            elif event.key == pygame.K_c:
                graph_points.clear()

    """ DRAW """
    # Fill the background with black
    screen.fill(GREY)

    # Draw GRID
    draw_grid(screen)

    # draw graph_points
    draw_graph_points(screen)

    # draw path
    draw_path(screen, path)

    # update the screen with the changes
    pygame.display.update()


# quits the game if is_running == False
pygame.quit()
