'''
    this is a simple test file
'''
import sys
import argparse

sys.path.append('model')
sys.path.append('utils')

from utils_SH import *

# other modules
import os
from os import listdir
from os.path import isfile, join
import numpy as np

from torch.autograd import Variable
from torchvision.utils import make_grid
import torch
import time
import cv2
import pickle
from defineHourglass_1024_gray_skip_matchFeature import *

# ---------------- create normal for rendering half sphere ------
img_size = 256
x = np.linspace(-1, 1, img_size)
z = np.linspace(1, -1, img_size)
x, z = np.meshgrid(x, z)

mag = np.sqrt(x**2 + z**2)
valid = mag <=1
y = -np.sqrt(1 - (x*valid)**2 - (z*valid)**2)
x = x * valid
y = y * valid
z = z * valid
normal = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
normal = np.reshape(normal, (-1, 3))
#-----------------------------------------------------------------

modelFolder = 'trained_model/'

# load model
my_network_512 = HourglassNet(16)
my_network = HourglassNet_1024(my_network_512, 16)
my_network.load_state_dict(torch.load(os.path.join(modelFolder, 'trained_model_1024_03.t7')))
my_network.cuda()
my_network.train(False)


lightFolder = 'data/example_light/'

def get_sh(file):
    img = cv2.imread(file)
    row, col, _ = img.shape
    img = cv2.resize(img, (1024, 1024))
    Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    inputL = Lab[:,:,0]
    inputL = inputL.astype(np.float32)/255.0
    inputL = inputL.transpose((0,1))
    inputL = inputL[None,None,...]
    inputL = Variable(torch.from_numpy(inputL).cuda())

    for i in range(7):
        sh = np.loadtxt(os.path.join(lightFolder, 'rotate_light_{:02d}.txt'.format(i)))
        sh = sh[0:9]
        sh = sh * 0.7

        # rendering half-sphere
        sh = np.squeeze(sh)

        #----------------------------------------------
        #  rendering images using the network
        #----------------------------------------------
        sh = np.reshape(sh, (1,9,1,1)).astype(np.float32)
        sh = Variable(torch.from_numpy(sh).cuda())
        outputImg, _, outputSH, _  = my_network(inputL, sh, 0)
        return outputSH


def execute(path, latents, save_dir):

    if latents is not None:
        file = open(latents, 'rb')
        sg2latents = pickle.load(file)
        sh = np.zeros((len(sg2latents['Filenames']), 1, 9, 1, 1))
        for idx, f in enumerate(sg2latents['Filenames']):
            file = path + '/' + f
            file = file.split(".")[0] + ".png"
            sh[idx] = get_sh(file).detach().cpu().numpy()

    else:
        sh = []
        for f in listdir(path):
            file = join(path, f)
            if isfile(file) and file.endswith(".png"):
                sh.append(get_sh(file).detach().cpu())

        sh = np.array(sh)

    np.save(save_dir + '/light.npy', sh)


if __name__ == '__main__':

    pars = argparse.ArgumentParser(description='Obtain lightning from the images in a folder')
    pars.add_argument('path', type=str, help='Image folder.')
    pars.add_argument('--latents', type=str, default=None, help='sg2latents.pickle path to get the order of the files read to match with it.')
    pars.add_argument('--save_dir', type=str, default="./result", help='directory where the resulting lights will be saved as lights.npy.')

    args = pars.parse_args()

    path = args.path
    latents = args.latents
    save_dir = args.save_dir

    execute(path, latents, save_dir)
