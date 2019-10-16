import argparse

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import seaborn as sb
import json
import argparse
from functions import define_dir, define_loaders, check_device, define_nn, train_model, test_model, save_check
# Set default for all variables

input_dir = "flowers"
gpu_mode = True

arch = "densenet121"
hidden_layer1 = 512
drop = 0.2
lr = 0.003

epoch_input = 5
target_file = "classifier_check.pth"

try:
    default = input("Do you want to continue with the default parameters? Enter yes or no (lowercase): ")
except:
    default = "no"

if default == 'yes':
# continue with the default parameters
    input_dir = "flowers"
    gpu_mode = True

    arch = "densenet121"
    hidden_layer1 = 512
    drop = 0.2
    lr = 0.003

    epoch_input = 5
    target_file = "classifier_check.pth"
# User inputs
else: 
    try:
        input_dir = input("Directory consisting of train, valid and test folder - default 'flowers': ")
    except:
        print ('Not a valid input, the program continues with the default value.')
        input_dir = "flowers"
    try:
       arch = input("Choose the pretrained Network you want to work with: densenet121 or vgg16 - default densenet121: ")
    except:
        print ('Not a valid input, the program continues with the default value.')
        arch = "densenet121"
    try:
        gpu_mode = bool(input("If GPU available type in True, otherwise False - default True: "))
    except:
        print ('Not a valid input, the program continues with the default value.')
        gpu_mode = True
    try:
        hidden_layer1 = int(input("Choose the prefered number of hidden units for the first layer - default 512: "))
    except:
        print ('Not a valid input, the program continues with the default value.')
        hidden_layer1 = 512
    try:
        drop = float(input("Choose a dropour rate for the model - default 0.2: " ))
    except:    
        print ('Not a valid input, the program continues with the default value.')
        drop = 0.2
    try:
        lr = float(input("Choose a learning rate for the model - default 0.003: " ))
    except:    
        print ('Not a valid input, the program continues with the default value.')
        lr = 0.003
    try:
        epoch_input = int(input("Choose the number of epochs the model should perform - default 5: " ))
    except:    
        print ('Not a valid input, the program continues with the default value.')
        epoch_input = 8
    try:
        target_file = float(input("Choose the filename to save the trained model - default 'classifier_check.pth: " ))
    except:    
        print ('Not a valid input, the program continues with the default value.')
        target_file = "classifier_check.pth"
# argparse didn't gave any opportunity to make an input and left all with None. If you know what I did wrong with that, please let me know :-)
#parser = argparse.ArgumentParser(description= "Neural Network Training Script")

#parser.add_argument('--input_dir',type=str, help="Directory consisting of train, valid and test folder - default 'flowers'")
#parser.add_argument('--gpu_mode',type=bool, help="If GPU available type in True, otherwise False - default True")
#parser.add_argument('--arch',type=str, action='store', help="Choose the pretrained Network you want to work with: densenet121 or vgg16 - default densenet121")
#parser.add_argument('--hidden_layer1',type=int, action='store', help="Choose the prefered number of hidden units for the first layer - default 512")
#parser.add_argument('--drop',type=float, action='store', help="Choose a dropour rate for the model - default 0.2")
#parser.add_argument('--lr',type=float, action='store', help="Choose a learning rate for the model - default 0.003")
#parser.add_argument('--epoch_input',type=int, action='store', help="Choose the number of epochs the model should perform - default 5")
#parser.add_argument('--target_file',type=str, action='store', help="Choose the filename to save the trained model - default 'classifier_check.pth")

#args = parser.parse_args()

# Overwrite default parameters if entered in the command line
#if args.input_dir:
#    input_dir = args.input_dir
#if args.gpu_mode:
#    gpu_mode = args.gpu_mode
#if args.arch:
#    arch = args.arch
#if args.hidden_layer1:
#    hidden_layer1 = args.hidden_layer1
#if args.drop:
#    drop = args.drop
#if args.lr:
#    lr = args.lr
#if args.epoch_input:
#    epoch_input = args.epch_input
#if args.target_file:
#    target_file = args.target_file
    

# functions
# call define_dir
train_dir, valid_dir, test_dir = define_dir(input_dir)
# call define_loaders
trainloader, validloader, testloader, train_dataset = define_loaders(train_dir, valid_dir, test_dir)
# call check_device
device = check_device(gpu_mode)
# call define_nn
model, criterion, optimizer = define_nn(arch, hidden_layer1, drop, lr)
# model to gpu/cpu
model.to(device)
# call train_model
model, optimizer = train_model(model, criterion, optimizer, epoch_input, trainloader, validloader, device)
# call test_model
test_model(testloader, model, device, criterion) 
# call save_check
save_check(target_file, model, train_dataset, arch, hidden_layer1, drop, lr)