# import helper
import sys
import os
sys.path.append(os.getcwd())
import glob
import json
import numpy as np
from util.TrainLevelHelper import *
import GetLevelZ

X,z,i2s = GetLevelZ.generate_training_level()

training_3d = []
for troom in X:
    nptroom = np.array(troom)
    training_3d.append(np.stack([np.where(nptroom == i, 1, 0) for i in range(14)], axis=0))
    



def evaluate(room):
    '''
    Input
    room: 2D array of a generated room

    First check if there are at least two doors, so this room is beatable, then check if there's a path connecting this two doors with an A* agent
    '''
    door_coor = np.transpose(np.where(room == 8))
    monster_coor = np.transpose(np.where(room == 4))
    door_count = 1
    if len(door_coor) < 2:
        return 0, 0

    #Check for numbers of actual doors
    actual_door = [door_coor[0]]
    for i in range(len(door_coor) - 1):
        room[door_coor[i][0]][door_coor[i][1]] = 2
        if not next_to(door_coor[i], door_coor[i + 1]):
            actual_door.append(door_coor[i + 1])
            door_count += 1
    room[door_coor[-1][0]][door_coor[-1][1]] = 2

    if door_count < 2:
        return 0, 0

    # for monster in monster_coor:
    #     # print(monster)
    #     room[monster[0]][monster[1]] = 2
    #Check if there's a path between two doors
    paths = []
    shortest_path = []
    for start in range(len(actual_door)):
        s = (actual_door[start][0], actual_door[start][1])
        for end in range(start + 1, len(actual_door)):
            e = (actual_door[end][0], actual_door[end][1])
            # print(s, e)
            path = astar(room, s, e)
            # print(path)
            if path:
                # steps += path[1]
                # print(path)
                paths.append(path)
                shortest_path.append(len(path))
                # print(shortest_path)
    # print(shortest_path)
    if len(shortest_path) > 0:
        # print(shortest_path)
        path_i = 0
        shortest = shortest_path[0]
        monster_encountered = 0
        for i, path in enumerate(shortest_path):
            # print(i, path, shortest)
            # print(path < shortest)
            if path < shortest:
                path_i = i
        # print(paths[path_i])
        for monster in monster_coor:
            for coor in paths[path_i]:
                if (monster[0], monster[1]) == coor:
                    monster_encountered += 1
        return len(paths[path_i]), monster_encountered
    return 0, 0

    
def average_L2_distance(room, training_rooms):
    '''
    Input
    room: 2d array / json of a generated room Shape: [row, column]
    training_rooms: this argument is unused and is an artifact of an older implementation
    average_L2_distance: the average L2 distance between the room generated and every room in the training dataset

    Remove the two lines which turn the room into numpy if they are already numpy array
    '''
    num_training = len(training_rooms)
    
    room = np.array(room)
    room_oh = np.stack([np.where(room == i, 1, 0) for i in range(14)], axis=0)
    
    L2 = 0
    for training_room in training_3d:
        L2 += np.linalg.norm(room_oh - training_room)
    average_L2_distance = L2 / num_training
    return average_L2_distance

def next_to(coor1, coor2):
    '''Check if the two coor are next to each other'''
    for i in range(-1, 2):
        for j in range(-1, 2):
            if abs(i) != abs(j):
                # print(coor1[0] + i, coor1[1] + j)
                if coor1[0] + i == coor2[0] and coor1[1] + j == coor2[1]:
                    return True
    return False

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    steps = 0
    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0 and steps < 5000:
        steps += 1
        # Get the current node
        current_node = open_list[0]
        # print("current", current_node.position)
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)
        # print("CLOSE LIST")
        # for list in closed_list:
            # print(list.position)
        # print("==========")
        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 2 and maze[node_position[0]][node_position[1]] != 4:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:
            flag = True
            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    flag = False
                # else:
                    # print(child.position)

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    flag = False

            # Add the child to the open list
            if flag:
                open_list.append(child)
    return False

def print_room(room):
    '''Print out the room'''
    for row in room:
        for column in row:
            print(column, end=" ")
        print()


# def get_training_data(path_to_folder):
#     '''load all the json file in a folder'''
#     for level in glob.glob("{0}/*json".format(path_to_folder)):
#         with open(level) as file:
#             data = json.load(file)
#     return data


# maze = [[2, 2, 2, 1, 2, 2, 2, 2, 2, 2],
        # [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        # [1, 2, 3, 3, 3, 2, 2, 2, 2, 2],
        # [1, 1, 3, 2, 3, 2, 2, 2, 2, 2],
        # [2, 2, 3, 2, 3, 2, 2, 2, 2, 2],
        # [2, 2, 2, 2, 2, 4, 4, 3, 4, 2],
        # [2, 2, 2, 2, 2, 2, 2, 2, 4, 2],
        # [2, 2, 2, 2, 3, 2, 4, 4, 4, 2],
        # [2, 2, 2, 2, 3, 4, 2, 2, 2, 2],
        # [2, 2, 2, 2, 3, 3, 3, 2, 2, 2],
        # [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        # [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        # [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        # [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        # [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        # [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]

# start = (2, 1)
# end = (4, 5)
# path = astar(maze, start, end)
# maze = np.array(maze)
# print(evaluate(maze))
# print(path)

def get_all_training_levels_list():
    str2idx = {"-" : 0, "W" : 1, "F" : 2, "B" : 3, "M" : 4, "P" : 5, "O" : 6, "I" : 7, "D" : 8, "S" : 9, "L" : 10, "V" : 11, "#" : 12, "U" : 13}
    level_path = "./../../GameGAN-Processed/Rooms"
    all_lvls = get_lvls_Z(level_path, str2idx)
    # void_strip = [0]*16
    # five_void_strips = [void_strip]*5
    # return [lvl + five_void_strips for lvl in all_lvls]
    return all_lvls