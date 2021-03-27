'''
Loads the training levels and converts them to integer representation.
generate_training_level() returns a list of said levels, the number of different tile types, and the representation conversion dictionary.
'''

import os
import sys
from util.TrainLevelHelper import *
import toml
import numpy as np

def generate_training_level():
    path_parts = os.getcwd().split("\\")[:-1]
    path = ""
    for i in path_parts:
        path = path + i + "/"
    path = path + "GANTrain/config/LevelConfigZ.tml"
    parsed_toml = toml.load(path)

    level_path=parsed_toml["LevelPath"]
    level_width=parsed_toml["LevelWidth"]
    compressed=parsed_toml["Compressed"]
    
    
    
    # 0 = "-" = VOID = black
    # 1 = "W" = WALL = indigo
    # 2 = "F" = FLOOR = beige
    # 3 = "B" = BLOCK = orange
    # 4 = "M" = MONSTER = red
    # 5 = "P" = ELEMENT (LAVA, WATER) = blue
    # 6 = "O" = ELEMENT + FLOOR (LAVA/BLOCK, WATER/BLOCK) = turquoise
    # 7 = "I" = ELEMENT + BLOCK = cyan
    # 8 = "D" = DOOR = brown
    # 9 = "S" = STAIR = grey
    #10 = "L" = unknown = purple
    #11 = "V" = unknown = pink
    #12 = "#" = unknown = white
    #13 = "U" = unknown = lime
    
    str2idx = {"-" : 0, "W" : 1, "F" : 2, "B" : 3, "M" : 4, "P" : 5, "O" : 6, "I" : 7, "D" : 8, "S" : 9, "L" : 10, "V" : 11, "#" : 12, "U" : 13}
    idx2str = {v : k for k,v in str2idx.items()}
    
    lvls_list = get_lvls_Z(level_path, str2idx)
    X = np.array(lvls_list).astype(np.uint8)
    
    return X, len(str2idx), idx2str

