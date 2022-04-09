from tkinter import W
import numpy as np
import pandas as pd
import pygame
import pygame.gfxdraw
from path_planning import *

# initaialize pygame
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
ROAD_WIDTH = 70  # Just an Arbitrary Number
BLOCK_LENGTH = int((HEIGHT - 3 * ROAD_WIDTH)/2)  # BOOOM MATH !!!

# variable controls what appeares on the screen
# set the screen size to be 500px x 500px
screen = pygame.display.set_mode(WINDOW_SIZE, pygame.NOFRAME)

# sets the window title
pygame.display.set_caption("TEST")

# Game Variables
is_running = True  # The state of the game
graph_points = []
corner_points = []
path = []


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
    if len(path):
        pygame.draw.polygon(screen, WHITE, path, 2)


set_corner_points()
# game loop
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
                # finds the path of the graph points using ant colony
                # Then casts the np array into a list
                """ Change this line to work with different Algorithm """
                path = Find_Path(np.array(graph_points)).ant_colony().tolist()

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
