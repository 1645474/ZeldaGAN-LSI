'''
Loads an already-trained model (from the /saved folder) and generates a level. The level is then saved as a json (same folder), and evaluated. WIll generate multiple random levels if --num_levels > 1
e.g.
python myrunZ.py --nz <whatever latent vector size the model was trained on> --state_epochs x --state_seed y --seed 6
Where the model you want to load is at "/saved/netG_epoch_x_y.pth"
'''

from __future__ import print_function

import argparse
import json
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import models.dcgan as dcgan
import os
import GetLevelZ
import json
import ShowLevelZFancy
import evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--nz', type=int, default=32, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--niter', type=int, default=5000, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')

parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--problem', type=int, default=0, help='Level examples')
parser.add_argument('--json', default=None, help='Json file')
parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')

parser.add_argument('--model_epochs', type=int, default=1999, help='Use this and model_seed to specify which model to load from the saved folder')
parser.add_argument('--model_seed', type=int, default=6, help='Use this and model_epochs to specify which model to load from the saved folder')
parser.add_argument('--num_levels', type=int, default=1, help='Number of levels to generate')

opt = parser.parse_args()


map_size = 16
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
X, z_dims, index2str=GetLevelZ.generate_training_level()
# z_dims = 14
n_extra_layers = int(opt.n_extra_layers)
n_levels = int(opt.num_levels)
state_epochs = int(opt.model_epochs)
state_seed = int(opt.model_seed)



netG = dcgan.DCGAN_G(map_size, nz, z_dims, ngf, ngpu, n_extra_layers)


loaded_state_dict = torch.load("./saved/netG_epoch_{}_{}.pth".format(state_epochs, state_seed))
netG.load_state_dict(loaded_state_dict)
netG.eval()


device = torch.device("cpu")

all_lvls = evaluate.get_all_training_levels_list()
results = []
total_L2 = 0
total_steps = 0
total_completable = 0
total_enemies = 0

with torch.no_grad():
    for i in range(n_levels):
        fixed_noise = torch.randn(1, nz, 1, 1, device=device)
        fake = netG(fixed_noise).detach().cpu()
        
        base = torch.zeros(map_size)
        for d in range(z_dims):
            slice = fake[0][d]
            base = torch.where(slice != 0, torch.full((map_size,map_size), float(d)), base)
        
        random_id_num =  random.randint(0, 1000000)
        f=open('saved/{}_{}_{}.json'.format(state_epochs, state_seed, random_id_num),"w")
        f.write(json.dumps(base.numpy().tolist()))
        f.close()    
        
        base = base.numpy()[:11,:]
        num_steps, num_enemies = evaluate.evaluate(base)
        L2 = evaluate.average_L2_distance(base, all_lvls)
        results.append((random_id_num, num_steps, num_enemies, L2))
        total_L2 += L2
        total_steps += num_steps
        total_enemies += num_enemies
        if num_steps:
            total_completable += 1
    
print(results)
if total_completable > 0:
    print("Mean L2 = {}\nMean number of steps = {}\nMean enemies killed = {}\nCompletability rate = {}".format(total_L2/n_levels, total_steps/total_completable, total_enemies/total_completable, total_completable/n_levels))