'''
python ShowLevelZFancy.py <filepath> <format> will use matplotlib to display a generated level
<format> = 0 corresponds to json file
<format> = 1 corresponds to string representation (the format the original training levels are given in)
<format> = 2 corresponds to integer representation (same as above but with 0 instead of '-', 1 instead of 'W', etc)
If providing a relative filepath, it should be given relative to "ZeldaGAN-LSI/search".
'''

import os
import sys
import matplotlib.pyplot as plt
from matplotlib import colors
import json
import numpy as np

str2idx = {"-" : 0, "W" : 1, "F" : 2, "B" : 3, "M" : 4, "P" : 5, "O" : 6, "I" : 7, "D" : 8, "S" : 9, "L" : 10, "V" : 11, "#" : 12, "U" : 13}

# idx2str = {v:k for k,v in str2idx.items()}

sprite_dict = {}
for i in range(14):
    pic = plt.imread("./sprites/{}.png".format(i))
    if pic.shape == (15,15,4):
        pic = pic[:,:,:3]
    sprite_dict[i] = pic
    

    
    
    
    
if __name__ == "__main__":
    filename, mode = sys.argv[1], int(sys.argv[2])
    
    if mode == 0:
        with open(filename) as f:
            empty = json.load(f)
    else:
        f = open(filename, 'r')
        empty = []
        lvl=f.readlines()
        empty = []
        for l in lvl:
            if len(l.strip()) > 0:
                if mode == 1:
                    empty.append([str2idx[ch] for ch in l.strip()])
                else:
                    empty.append(l.strip())
    
    unconcatenated = [[sprite_dict[i] for i in line] for line in empty]
    
    half_concatenated = []
    for line in unconcatenated:
        half_concatenated.append(np.concatenate(line, axis=1))
    
    concatenated = np.concatenate(half_concatenated, axis=0)
    
    plt.imshow(concatenated)
    plt.axis('off')
    plt.tight_layout()
    plt.show()