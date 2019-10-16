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


# Input: Directory;  Output: Train_directory, Valid_directory, Test_directory
def define_dir(data_dir):
    data_dir = data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    return train_dir, valid_dir, test_dir

# Input: Train_directory, Valid_directory, Test_directory
# Outout: Trainloader, Validloader, Testloader, train_dataset
def define_loaders(train_dir, valid_dir, test_dir):
    train_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.RandomRotation(25),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    return trainloader, validloader, testloader, train_dataset

# Input: File with labels;  Output: Dictionary with Labels/Folders
def label_mapping(label_file='cat_to_name.json'):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name
    
#Input: gpu_mode True/False;  Output: device to continue
def check_device(gpu_mode=True):
    device = torch.device("cuda" if gpu_mode == True else "cpu")
    return device

# Defines the Neural Network. Input: architecture as string, hidden_layer1, drop, lr
# Output: model, criterion, optimizer
def define_nn(architecture="densenet121", hidden1=512, dropout=0.2, learnrate=0.003):
    
    arch_dict = {"densenet121":1024,
               "vgg16":25088}
    # First define which network is used. Possible are densenet121 and vgg16.
    if architecture == "densenet121":
        model = models.densenet121(pretrained=True)
    elif architecture == "vgg16":
        model = models.vgg16(pretrained=True)
    else:
        # IF the input is different than densenet121 and vgg16, the program works with the default densenet121
        print("{} is not included. The program will continue with the default model 'densenet121'".format(architecture))
        model = models.densenet121(pretrained=True)
        
    # freeze the parameters so they're not effected by backprop
    for param in model.parameters():
        param.requires_grad = False
    
    # define the classifier
    model.classifier = nn.Sequential(nn.Linear(arch_dict[architecture], hidden1),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(hidden1, 256),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(256, 102),
                                nn.LogSoftmax(dim=1))
    
    # define the criterion
    criterion = nn.NLLLoss()
    
    # train the classifier parameters
    optimizer = optim.Adam(model.classifier.parameters(), learnrate)
    
    return model, criterion, optimizer



# Trains the Neural Network on the given dataset trainloader and validates the results in validloader
# Inputs model, criterion, optimizer are given by define_nn, epoch is set. 
# Output: model, optimizer   <- both trained on the dataset
def train_model(model, criterion, optimizer, epoch_input, trainloader, validloader, device ):

    steps = 0
    print_every = 20
    running_loss = 0
    for epoch in range(epoch_input):
    
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            # validation with validloader / just do this every print_every time
            if steps % print_every == 0:
                # turning the model in evaluation mode
                model.eval()
                valid_running_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                    
                        # Output, Loss and Running-Loss
                        valid_out = model.forward(inputs)
                        valid_loss = criterion(valid_out, labels)
                    
                        valid_running_loss += valid_loss.item()
                    
                        # Accuracy
                        valid_ps = torch.exp(valid_out)
                        top_p, top_class = valid_ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    # print validation results after the for loop and turn model back in train mode
                    print ("Epoch: ", (epoch+1),"/",epoch_input)
                    print ("Training Loss: ", (running_loss/print_every))
                    print ("Validation Loss: ", (valid_running_loss/len(validloader)))
                    print ("Validation Accuracy: ", (accuracy/len(validloader)))
                    running_loss = 0
                    model.train()
        
    return model, optimizer
    
# Input testloader
# Output: prints the results of the test
def test_model(test_data, model, device, criterion):
    test_running_loss = 0
    test_accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_data:
            inputs, labels = inputs.to(device), labels.to(device)
        
            # Outout, Loss and Running-Loss
            test_out = model.forward(inputs)
            test_loss = criterion(test_out, labels)
        
            test_running_loss += test_loss.item()
        
            # Accuracy
            test_ps = torch.exp(test_out)
            ttop_p, ttop_class = test_ps.topk(1, dim=1)
            equals = ttop_class == labels.view(*ttop_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        print ("Test Loss: ", (test_running_loss/len(test_data)))
        print ("Test Accuracy: ", (test_accuracy/len(test_data)))
        
# Input: Path the model sould be saved to
def save_check(target_file, model, train_dataset, arch, hidden_layer1, drop, lr):
    model.class_to_idx = train_dataset.class_to_idx
    model.cpu()
    classifier_check = {"architecture": arch,
                    "hidden1":hidden_layer1,
                    "dropout":drop,
                    "learnrate": lr,
                    "state_dict": model.state_dict(),
                    "class_to_idx": train_dataset.class_to_idx}

    torch.save(classifier_check, target_file)
    
# checks if gpu_mode is True for the prediction process
def check_pred_device(gpu_mode=True):
    device = torch.device("cuda" if gpu_mode == True else "cpu")
    return device

# loads the checkpoint from the given filepath and stores it to the selected device 
def load_checkpoint(filepath, device):
    checkpoint = torch.load(filepath)
    arch = checkpoint['architecture']
    hidden_layer1 = checkpoint['hidden1']
    drop = checkpoint["dropout"]
    lr = checkpoint['learnrate']
    model, criterion, optimizer = define_nn(arch, hidden_layer1, drop, lr)
    model.class_to_idx = checkpoint["class_to_idx"]
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    
# stores the given image to the right format    
def process_image(image):
    
    im_image = Image.open(image)
    im_width, im_height = im_image.size
    
    if im_height < im_width:
        im_height = 256
        im_image.thumbnail((10000, im_height), Image.ANTIALIAS)
    else:
        im_width = 256
        im_image.thumbnail((im_width, 10000), Image.ANTIALIAS)
    
    left = (256 - 224)/2
    upper = (256 - 224)/2
    right = left + 224
    lower = upper + 224
    
    im_image = im_image.crop((left, upper, right, lower))
    
    np_image = np.array(im_image)/255
    
    mean = [0.485, 0.456, 0.406]
    stdv = [0.229, 0.224, 0.225]
    
    np_image = (np_image - mean)/ stdv
    output_image = np_image.transpose((2,0,1))
    
    return output_image

# prints the image to the console
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

# Inputs are the image_path, the selected model and how much topk should print
# Returns a np.array with probabilities of guesses and converted classes
def predict(image_path, pred_device, model, k):

        # convert image to the correct format
    np_img = process_image(image_path)
    tensor_img = torch.tensor(np_img)
    tensor_img = tensor_img.float()
    # for error => batch size one
    tensor_img = tensor_img.unsqueeze(0)
    tensor_img = tensor_img.to(pred_device)
    with torch.no_grad():
        model.eval()
        img_output = model.forward(tensor_img)
        img_ps = torch.exp(img_output)
        top_p, top_class = img_ps.topk(k)
        top_p, top_class = top_p.cpu(), top_class.cpu()
        prob_img = top_p.numpy()
        prob_img = prob_img.tolist()[0]
        indices_img = top_class.numpy()
        indices_img = indices_img.tolist()[0]
        label_dict = model.class_to_idx
        
        iterating = {val: key for key, val in label_dict.items()}
        classes = [iterating [item] for item in indices_img]
        classes = np.array (classes) #converting to Numpy array 
      
                
        return prob_img, classes
    
# creates pandas daraframe from the given probabilitys, classes and the flower_names from the dictionary
def create_frame(classes, probs, labels_dict):
    flower_names = []
    for n in classes:
        flower_names.append(labels_dict[n])

    pred_items = {"Class": pd.Series(classes), "Probabilitys": pd.Series(probs), "Flowername": pd.Series(flower_names)}
    pred_data = pd.DataFrame(pred_items)
    pred_data = pred_data.set_index('Class')
    return pred_data

# predicts the flowername of a given image
# Outputs the
def predict_flower(image_path, pred_device, model, k, labels_dict):
    
    probs, classes = predict(image_path, pred_device, model, k)
    pred_data = create_frame(classes, probs, labels_dict)
    print(pred_data)