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
from functions import check_pred_device, load_checkpoint, process_image, imshow, predict, create_frame, predict_flower, define_nn, label_mapping


# Set default for all needed variables
#label_file = 'cat_to_name.json'
#gpu_mode = True
#filepath = "classifier_check.pth"
#image_path = "flowers/valid/100/image_07895.jpg"
#k = 5
# inputs
# ask for default
try:
    default = input("Do you want to continue with the default parameters? Enter yes or no (lowercase): ")
except:
    default = "no"

if default == 'yes':
    label_file = 'cat_to_name.json'
    gpu_mode = True
    filepath = "classifier_check.pth"
    image_path = "flowers/valid/100/image_07895.jpg"
    k = 5
# User inputs
else: 
    try:
        label_file = input("Please enter the name of the file containing the labels for the data - default 'cat_to_name.json': ")
    except:
        print ('Not a valid input, the program continues with the default value.')
        label_file = 'cat_to_name.json'
    try:
        k = int(input("Choose how many of the most probable guesses you want to see - default 5: "))
    except:
        print ('Not a valid input, the program continues with the default value.')
        k = 5
    try:
        gpu_mode = bool(input("Please select if GPU is available. Tpye True or False - default True: "))
    except:
        print ('Not a valid input, the program continues with the default value.')
        gpu_mode = True
    try:
        filepath = input("Enter the path to the file the trained neural network is stored - default 'classifier_check.pth': ")
    except:
        print ('Not a valid input, the program continues with the default value.')
        filepath = "classifier_check.pth"
    try:
        image_path = input("Please type in the path to the image you want to classifie - default 'flowers/valid/100/image_07895.jpg': " )  
    except:    
        print ('Not a valid input, the program continues with the default value.')
        image_path = "flowers/valid/100/image_07895.jpg"
                    
                          



# parser.add_argument gave no chance to enter any input in the console, so I work with input() in try/except blocks.
# If you have any idea why the parser solution didn't worked please let me know :-)
#parser = argparse.ArgumentParser(description= "Neural Network Prediction Script")

#parser.add_argument('-label_file',type=str, help="Name of the file containing the labels for the data - default 'cat_to_name.json'")
#parser.add_argument('-gpu_mode',type=bool, help="If GPU available type in True, otherwise False - default True")
#parser.add_argument('-filepath',type=str, action='store', help="Enter the path to the file the trained neural network is stored - default 'classifier_check.pth'")
#parser.add_argument('-image_path',type=str, action='store', help="Please type in the path to the image you want to classifie - default 'flowers/valid#/100/image_07895.jpg'")
#parser.add_argument('-k',type=int, action='store', help="Choose how much of the most probable guesses you want to see - default 5")
#
#args = parser.parse_args()
#print(args)
#if args.label_file:
#    label_file = args.label_file
#if args.gpu_mode:
#    gpu_mode = args.gpu_mode
#if args.filepath:
#    filepath = args.filepath
#if args.image_path:
#    image_path = args.image_path
#if args.k:
#    k = args.k


# call check_pre_device
pred_device = check_pred_device(gpu_mode)
# call load_checkpoint. Now available are: model, criterion, optimizer

checkpoint = torch.load(filepath)
arch = checkpoint['architecture']
hidden_layer1 = checkpoint['hidden1']
drop = checkpoint["dropout"]
lr = checkpoint['learnrate']
model, criterion, optimizer = define_nn(arch, hidden_layer1, drop, lr)
model.class_to_idx = checkpoint["class_to_idx"]
model.load_state_dict(checkpoint['state_dict'])
model.to(pred_device)
# Till here check, all data is loaded to the script

#Call label_mapping
labels_dict = label_mapping(label_file)
predict_flower(image_path, pred_device, model, k, labels_dict)
