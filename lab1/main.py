### Importing libraries ###

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torchvision.utils import save_image

from lab1 import *
from utils import *

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time


### Argument parsing ###

parser = argparse.ArgumentParser(description="A Temporal Neural Network Simulator")

parser.add_argument('--mode',        type=int, default=1,\
                    help='3 modes available: 0 - SNL column simulation; 1 - RNL Column Simulation; 2 - ECVT network simulation')

args = parser.parse_args()


### Weight Update Parameters ###

ucapture  = 1/2
usearch   = 1/1024
ubackoff  = 1/2

### Column Layer Parameters ###

inputsize = 26
rfsize    = 3
stride    = 1
nprev     = 2
neurons   = 12
theta     = 4

### Voter Layer Parameters ###

rows_v    = 24
cols_v    = 24
nprev_v   = 12
classes_v = 10
thetav_lo = 1/32
thetav_hi = 15/32
tau_eff   = 2

### Enabling CUDA support for GPU ###
cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

### MNIST dataset loading and preprocessing ###

train_loader = DataLoader(MNIST('./data', True, download=True, transform=transforms.Compose(
                                                                    [
                                                                     transforms.ToTensor(),
                                                                    #  transforms.Grayscale(),
# torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) → Tensor
                                                                     IntensityTranslation(8)
                                                                    ]
                                                                                       )
                               ),
                          batch_size=1,
                          shuffle=False
                         )

test_loader = DataLoader(MNIST('./data', False, download=True, transform=transforms.Compose(
                                                                    [
                                                                     transforms.ToTensor(),
                                                                    #  transforms.Grayscale(),
                                                                     IntensityTranslation(8)
                                                                    ]
                                                                                       )
                               ),
                          batch_size=1,
                          shuffle=False
                         )


######################## Step-No-Leak Column Simulation ##################
if args.mode == 0:

    weights_save = 1

    ### Layer Initialization ###

    clayer = TNNColumnLayer(28, 28, 1, 2, 12, 400, ntype="snl", device=device)

    if cuda:
        clayer.cuda()


    ### Training ###

    print("Starting column training")
    for epochs in range(1):
        start = time.time()

        for idx, (data,target) in enumerate(train_loader):
            if idx == 10000:
                break
            print("Sample: {0}\r".format(idx), end="")

            if cuda:
                data                    = data.cuda()
                target                  = target.cuda()

            out1, layer_in1, layer_out1 = clayer(data[0].permute(1,2,0))
            clayer.weights = clayer.stdp(layer_in1, layer_out1, clayer.weights, ucapture, usearch, ubackoff)

            endt                   = time.time()
            print("                                 Time elapsed: {0}\r".format(endt-start), end="")

        end   = time.time()
        print("Column training done in ", end-start)


    ### Display and save weights as images ###

    if weights_save == 1:

        image_list = []
        for i in range(12):
            temp = clayer.weights[i].reshape(56,28)
            image_list.append(temp)

        out = torch.stack(image_list, dim=0).unsqueeze(1)
        save_image(out, 'column_visweights_snl.png', nrow=6)


    ### Testing and computing metrics ###

    table    = torch.zeros((12,10))
    pred     = torch.zeros(10)
    totals   = torch.zeros(10)

    print("Starting testing")
    start    = time.time()

    for idx, (data,target) in enumerate(test_loader):
        print("Sample: {0}\r".format(idx), end="")

        if cuda:
            data                    = data.cuda()
            target                  = target.cuda()

        out1, layer_in1, layer_out1 = clayer(data[0].permute(1,2,0))
        out = torch.flatten(out1)

        arg = torch.nonzero(out != float('Inf'))

        if arg.shape[0] != 0:
            table[arg[0].long(), target[0]] += 1

        endt = time.time()
        print("                                 Time elapsed: {0}\r".format(endt-start), end="")

    end = time.time()
    print("Testing done in ", end-start)

    print("Confusion Matrix:")
    print(table)

    maxval   = torch.max(table, 1)[0]
    totals   = torch.sum(table, 1)
    pred     = torch.sum(maxval)
    covg_cnt = torch.sum(totals)

    print("Purity: ", pred/covg_cnt)
    print("Coverage: ", covg_cnt/(idx+1))


####################### Ramp-No-Leak Column Simulation #######################
elif args.mode == 1:

    weights_save = 1

    ### Layer Initialization ###
    # filter layer
    #flayer = OnOffCenterFilter(28, 3, 1, wres=3, device = device)
    clayer = TNNColumnLayer(28, 28, 1, 2, 12, 400, ntype="rnl", device=device, w_init="normal")
    # def __init__(self, inputsize, rfsize, stride, nprev, q, theta, wres=3, w_init="half", ntype="rnl",                  device="cpu"):
    # def __init__(self, inputsize, rfsize, stride, wres=3, device="cpu"):

    
    if cuda:
        clayer.cuda()
        # flayer.cuda()


    ### Training ###

    print("Starting column training")
    for epochs in range(1):
        start = time.time()
        image_list = []
        input_image_list = []
        for idx, (data,target) in enumerate(train_loader):
            if idx == 10000:
                break
            print("Sample: {0}\r".format(idx), end="")

            if cuda:
                data                    = data.cuda()
                target                  = target.cuda()
            # filteredLayer = flayer(data[0].permute(1,2,0))

            
            if idx < 12:
                # image_list.append(filteredLayer.permute(2,0,1).reshape(52,26))
                input_image_list.append(data[0].squeeze())
            if idx == 12:
                # out = 256 - torch.stack(image_list, dim=0).unsqueeze(1) * 256/8
                # save_image(out, 'input_images_seletion.png', nrow=6)
                input_out = 256 - torch.stack(input_image_list, dim=0).unsqueeze(1) * 256/8
                save_image(input_out, 'greyscale_images_seletion.png', nrow=6)

            # TODO instead of sending to clayer, send to flayer then to clayer
            out1, layer_in1, layer_out1 = clayer(data[0].permute(1,2,0))
            clayer.weights = clayer.stdp(layer_in1, layer_out1, clayer.weights, ucapture, usearch, ubackoff)

            endt                   = time.time()
            print("                                 Time elapsed: {0}\r".format(endt-start), end="")

        end   = time.time()
        print("Column training done in ", end-start)


    ### Display and save weights as images ###

    if weights_save == 1:

        image_list = []
        for i in range(12):
            temp = clayer.weights[i].reshape(56,28)
            image_list.append(temp)

        out = torch.stack(image_list, dim=0).unsqueeze(1)
        save_image(out, 'column_visweights_rnl.png', nrow=6)


    ### Testing and computing metrics ###

    table    = torch.zeros((12,10))
    pred     = torch.zeros(10)
    totals   = torch.zeros(10)

    print("Starting testing")
    start    = time.time()

    for idx, (data,target) in enumerate(test_loader):
        print("Sample: {0}\r".format(idx), end="")

        if cuda:
            data                    = data.cuda()
            target                  = target.cuda()
        # filteredLayer = flayer(data[0].permute(1,2,0))
        out1, layer_in1, layer_out1 = clayer(data[0].permute(1,2,0))
        out = torch.flatten(out1)

        arg = torch.nonzero(out != float('Inf'))

        if arg.shape[0] != 0:
            table[arg[0].long(), target[0]] += 1

        endt = time.time()
        print("                                 Time elapsed: {0}\r".format(endt-start), end="")

    end = time.time()
    print("Testing done in ", end-start)

    print("Confusion Matrix:")
    print(table)

    maxval   = torch.max(table, 1)[0]
    totals   = torch.sum(table, 1)
    pred     = torch.sum(maxval)
    covg_cnt = torch.sum(totals)

    print("Purity: ", pred/covg_cnt)
    print("Coverage: ", covg_cnt/(idx+1))


############################## Online Learning ##############################
elif args.mode == 2:

    inc_learn   = 1
    breakpoint1 = 60000
    interval1   = 1000
    breakpoint2 = 10000
    interval2   = 1000

    ### Layer Initialization ###

    flayer = OnOffCenterFilter(28, 3, 1, wres=3, device = device)
    clayer = TNNColumnLayer(inputsize, rfsize, stride, nprev, neurons, theta, ntype="rnl", device=device)
    vlayer = DualTNNVoterTallyLayer(rows_v, cols_v, nprev_v, classes_v, thetav_lo, thetav_hi, tau_eff,\
                                    device=device)

    if cuda:
        flayer.cuda()
        clayer.cuda()
        vlayer.cuda()


    ### Training ###

    for epochs in range(1):
        start = time.time()
        error1 = 0
        error2 = 0
        errorlist1 = []
        errorlist2 = []

        for idx, (data,target) in enumerate(train_loader):
            if idx == breakpoint1:
                break
            print("Sample: {0}\r".format(idx+1), end="")

            if cuda:
                data                    = data.cuda()
                target                  = target.cuda()

            if (idx+1) > 20000:
                data                    = torch.transpose(data,2,3)

            if (idx+1) > 50000:
                if (target[0]%2) == 0:
                    target[0] = target[0] + 1
                else:
                    target[0] = target[0] - 1

            filteredLayer = flayer(data[0].permute(1,2,0))
            out1, layer_in1, layer_out1 = clayer(filteredLayer)
            pred, voter_in, _           = vlayer(out1)

            if torch.argmax(pred) != target[0]:
                error1 += 1
                error2 += 1

            clayer.weights = clayer.stdp(layer_in1, layer_out1, clayer.weights, ucapture, usearch, ubackoff)
            vlayer.weights = vlayer.stdp(target, voter_in, vlayer.weights)

            if (idx+1)%interval1 == 0:
                errorlist1.append(error1/(idx+1))
                errorlist2.append(error2/interval1)
                error2 = 0

            endt = time.time()
            print("                                                     Time elapsed: {0}\r".format(endt-start), end="")

        end = time.time()
        print("Training for {0} samples done in {1}".format(idx, end-start))
        print("Training Accuracy for {1} epochs: {0}%".format((breakpoint1-error1)*100/breakpoint1, epochs+1))


    ### Testing ###

    error3 = 0
    start = time.time()

    for idx, (data,target) in enumerate(test_loader):
        if idx == breakpoint2:
            break
        print("Sample: {0}\r".format(idx+1+breakpoint1), end="")

        if cuda:
            data                    = data.cuda()
            target                  = target.cuda()

        data                    = torch.transpose(data,2,3)
        if (target[0]%2) == 0:
            target[0] = target[0] + 1
        else:
            target[0] = target[0] - 1

        filteredLayer = flayer(data[0].permute(1,2,0))
        out1, layer_in1, layer_out1 = clayer(filteredLayer)
        pred, voter_in, _           = vlayer(out1)

        if torch.argmax(pred) != target[0]:
            error1 += 1
            error2 += 1
            error3 += 1

        if inc_learn == 1:
            clayer.weights = clayer.stdp(layer_in1, layer_out1, clayer.weights, ucapture, usearch, ubackoff)
            vlayer.weights = vlayer.stdp(target, voter_in, vlayer.weights)

        if (idx+1)%interval2 == 0:
            errorlist1.append(error1/(idx+1+breakpoint1))
            errorlist2.append(error2/interval2)
            error2 = 0

        endt = time.time()
        print("                                                     Time elapsed: {0}\r".format(endt-start), end="")

    end = time.time()
    print("Testing for {0} samples done in {1}".format(idx, end-start))
    print("Testing Accuracy: {0}%".format((breakpoint2-error3)*100/breakpoint2))


    ### Storing image ###
    plt.figure(figsize=(10,4))

    dx = [i for i in range(1,int(breakpoint1/interval1+breakpoint2/interval2+1))]
    plt.plot(dx, errorlist2, color='red', linestyle='solid', linewidth=1.5)

    plt.xlabel("Samples (x1000)")
    plt.ylabel("Error Rate")
    plt.xticks(np.arange(1,int(breakpoint1/interval1+breakpoint2/interval2+1),2))
    plt.legend(["ECVT"])
    plt.savefig("online_learning.png")
