'''
File utilised by run_searchZ.py to control the illumination algorithm.
'''

import os
import sys
sys.path.append(os.getcwd()[:-6] + "GANTrain")
from util import SearchHelper
import pathlib
# os.environ['CLASSPATH']=os.path.join(str(pathlib.Path().absolute()),"Mario.jar")
#os.environ['CLASSPATH'] = "/home/tehqin/Projects/MarioGAN-LSI/Mario.jar"


import pandas as pd
import numpy as np
from numpy.linalg import eig
import torch
#import torchvision.utils as vutils
from torch.autograd import Variable

import toml
import json
import numpy
import util.models.dcgan as dcgan
import torch
#import torchvision.utils as vutils
from torch.autograd import Variable
import json
import numpy
import util.models.dcgan as dcgan
import math
import random
from collections import OrderedDict
import csv
from algorithmsZ import *
from util.SearchHelper import *
import evaluate #kevin's

# from jnius import autoclass
# MarioGame = autoclass('engine.core.MarioGame')
# Agent = autoclass('agents.robinBaumgarten.Agent')

"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-c','--config', help='path of experiment config file',required=True)
opt = parser.parse_args()
"""

batch_size =1
nz = 16
record_frequency=1000

if not os.path.exists("logs"):
    os.mkdir("logs")

global EliteMapConfig
EliteMapConfig=[]

all_lvls = evaluate.get_all_training_levels_list()

import sys



    
def eval_zelda(ind, visualize):
    level = ind.level[0][:11,:]
    num_steps, num_enemies_killed = evaluate.evaluate(level)
    if num_steps == 0:
        steps_bc = 0
    else:
        steps_bc = int(num_steps/5) + 1
    L2_dist = int(evaluate.average_L2_distance(level, all_lvls) > 8.61)
    
    
    total_enemies = np.count_nonzero(level == 4)
    num_blocks = np.count_nonzero(level == 3)
    if num_blocks == 0:
        blocks_bc = 0 
    else:
        blocks_bc = int(num_blocks/5) + 1
        
    num_water = np.count_nonzero(level == 5) + np.count_nonzero(level == 6) + np.count_nonzero(level == 7)
    if num_water == 0:
        water_bc = 0
    else:
        water_bc = int(num_water/5) + 1
        
    ind.features = (steps_bc, num_enemies_killed, total_enemies, blocks_bc, water_bc, L2_dist)
    
    completable = int(num_steps > 0)
    ind.statsList = [completable]
    
    return completable



def run_trial(num_to_evaluate,algorithm_name,algorithm_config,elite_map_config,trial_name,model_path,visualize):
    feature_ranges=[]
    column_names=['emitterName', 'latentVector', 'completable']
    bc_names=[]
    for bc in elite_map_config["Map"]["Features"]:
        feature_ranges.append((bc["low"],bc["high"]))
        column_names.append(bc["name"])
        bc_names.append(bc["name"])

    if(trial_name.split('_')[1]=="ZeldaBC"):
        feature_map = FeatureMap(num_to_evaluate, feature_ranges, resolutions=(9,9,9,9,9,2))
    else:
        sys.exit('unknown BC name. Exiting the program.')

    
    if algorithm_name=="CMAME":
        print("Start Running CMAME")
        mutation_power=algorithm_config["mutation_power"]
        population_size=algorithm_config["population_size"]
        initial_population = algorithm_config["initial_population"] 
        emitter_type = algorithm_config["emitter_type"] 
        algorithm_instance=CMA_ME_Algorithm(mutation_power,initial_population, num_to_evaluate,population_size,feature_map,trial_name,column_names,bc_names, emitter_type) 
    elif algorithm_name=="MAPELITES":
        print("Start Running MAPELITES")
        mutation_power=algorithm_config["mutation_power"]
        initial_population=algorithm_config["initial_population"]
        algorithm_instance=MapElitesAlgorithm(mutation_power, initial_population, num_to_evaluate, feature_map,trial_name,column_names,bc_names)
    elif algorithm_name=="RANDOM":
        print("Start Running RANDOM")
        algorithm_instance=RandomGenerator(num_to_evaluate,feature_map,trial_name,column_names,bc_names)
    
    simulation=1
    while algorithm_instance.is_running():
        ind = algorithm_instance.generate_individual()

        ind.level=gan_generate(ind.param_vector,batch_size,nz,model_path)
        ind.fitness = eval_zelda(ind,visualize)

        algorithm_instance.return_evaluated_individual(ind)
        
        if simulation % 100 == 0:
            print(str(simulation)+"/"+str(num_to_evaluate)+" simulations finished")
        simulation=simulation+1
    
    # print("Total = {}\nWinners = {}\nWin Rate = {}\nLow L2s = {}\nLow L2 Rate = {}".format(algorithm_instance.individuals_evaluated, algorithm_instance.winners, algorithm_instance.winners/algorithm_instance.individuals_evaluated, algorithm_instance.low_L2s, algorithm_instance.low_L2s/algorithm_instance.individuals_evaluated))
    algorithm_instance.all_records.to_csv("logs/"+trial_name+"_all_simulations.csv")



def start_search(sim_number,trial_index,experiment_toml,model_path,visualize):
    experiment_toml=experiment_toml["Trials"][trial_index]
    trial_toml=toml.load(experiment_toml["trial_config"])
    NumSimulations=trial_toml["num_simulations"]
    AlgorithmToRun=trial_toml["algorithm"]
    AlgorithmConfig=toml.load(trial_toml["algorithm_config"])
    global EliteMapConfig
    EliteMapConfig=toml.load(trial_toml["elite_map_config"])
    TrialName=trial_toml["trial_name"]+"_sim"+str(sim_number)
    run_trial(NumSimulations,AlgorithmToRun,AlgorithmConfig,EliteMapConfig,TrialName,model_path,visualize)
    print("Finished One Trial")
	


