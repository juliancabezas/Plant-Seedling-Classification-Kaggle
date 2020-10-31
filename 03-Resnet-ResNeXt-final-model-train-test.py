###################################
# Julian Cabezas Pena
# Big Data Analysis and Project
# University of Adelaide
# Project 2
# Trains the ResNet and ResNeXt on the complete train database and generates a Kaggle submission file
####################################

# Necessary packages
import os
from os import listdir
from PIL import Image
import pandas as pd
import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as func
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

#------------------------------------
#Parameters

# Hyperparameters of the neural network

batch_size = 10 # Mini batch size (number of images)
n_epoch_f = 20 # Number of epochs of the first stage of training
n_epoch_uf = 20 # Number of epochs of the second stage of training
learning_rate_decrease = 0.1 # Factor to multiply the learning rate from the stage 1 to the stage 2 of training

# Learning rate, the best models used 0.001 as i nitial learning rate
learning_rate = 0.001 

# Root directory
root_dir = './'

# Pick the model. WSL (weakly supervised learning) means Instagram pretrained
# Uncomment to pick a specific model

model = 'Resnet_101_IN'
#model = 'Resnext_32x8d_101_IN'
#model = 'Resnext_32x8d_101_IN_WSL'
#model = 'Resnext_32x16d_101_IN_WSL'

#----------------------------------------------


# Name of the model configuration
model_name = model + '_' + str(n_epoch_f) + 'e_' + str(batch_size) + 'b_' + 'adam_lr' + str(learning_rate).replace('0.','') + '_final'
model_name


# Generate dataset to read it in Pytorch, the init, len and getitem are required by pytorch
class SeedlingClassificationData(Dataset):
    def __init__(self, data_table, root_dir, subset=False, transform=None):
        self.data_table = data_table
        self.root_dir = root_dir
        self.transform = transform
    
    # Returns the amount of images in the dataset
    def __len__(self):
        return len(self.data_table)
    
    def __getitem__(self, idx):
        # Get the path of the image based on the index and the provided dataframe
        img_path = os.path.join(self.root_dir, self.data_table.iloc[idx, 0]) 

        # open the image to image
        image = Image.open(img_path).convert('RGB')

        # Get the label from the tird column of the dataframe
        label = self.data_table.iloc[idx, 2]

        # Apply transformations
        image = self.transform(image)

        return image, int(label)


# Get classes (class) names
classes = os.listdir(root_dir + 'data/' + 'train/')
classes = sorted(classes)

# Create a dictionary
classes_num_dic = dict(zip(range(len(classes)), classes))

# Empty list to store the train data directories
train_list = []
id_counter = 0

# loop though the species
for species in classes:

    # List files in each directory
    files = os.listdir(root_dir + 'data/' + 'train/' + species + '/')

    path = root_dir + 'data/' + 'train/' + species + '/'

    # List the files in the directory
    files_list = os.listdir(path)

    # Generate the training list
    for filename in files_list:
        train_list.append([species +'/'+filename, species, id_counter])
    
    id_counter = id_counter + 1
    
# Generate a dataframe with the training data
train_df = pd.DataFrame(train_list, columns=['file', 'species', 'species_id',]) 

# Here the train data is not splitted into validation and training
train_data = train_df

# Read the sample submission file
sample_submission = pd.read_csv(root_dir + 'data/' + 'sample_submission.csv')
sample_submission['species_id'] = 0


# Set up data augmentation techniques for the training data
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, 10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# For the validation and testing data we will only resize and crop the images to 224 pexels
val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create the test and train datasets
train_set = SeedlingClassificationData(data_table = train_data, root_dir = root_dir + 'data/' + 'train/', transform = train_transform)
test_set = SeedlingClassificationData(sample_submission, root_dir = root_dir + 'data/' + 'test/', transform = val_test_transform)

# Put the datasets in the dataloaders
trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

# Set up the device (GPU or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load the pre trained model
if model == 'Resnet_101_IN':
    net = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=True)
elif model == 'Resnext_32x8d_101_IN':
    net = torch.hub.load('pytorch/vision:v0.6.0', 'resnext101_32x8d', pretrained=True)
elif model == 'Resnext_32x8d_101_IN_WSL':
    net = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
elif model == 'Resnext_32x16d_101_IN_WSL':
    net = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')

# First stage: Train the last fully connected layer

# We will froze the pre trained parameters of the neural network for the first epochs
for parameter in net.parameters():
    parameter.requires_grad = False

# replace the last fully connected layer with the classifier
num_ftrs = net.fc.in_features
net.fc = torch.nn.Linear(num_ftrs, len(classes))

# Put neural net into the GPU
net = net.to(device)

# Set up criterion (cross entropy loss)
criterion = torch.nn.CrossEntropyLoss()

# We will use Adam optimazer to train the NN faster
optimizer = torch.optim.Adam(net.fc.parameters(), lr=learning_rate)

# The scheduler will make the ls go down by a factor of 0.1 every 7 epochs
learning_rate_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Set cuda as device if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Empty list to store the results
epoch_full = []
acc_full = []
loss_full = []

# Train with the dataset multiple times (n epoch)
for epoch in range(1,n_epoch_f+1):  

    print("Epoch:",epoch)

    # Initializing the training variables
    running_loss_full = 0.0
    correct_total_train = 0
    nbatch = 0
    nsamples_train = 0

    print('Training the neural network...')
    # Go though all the training data
    for i, data in enumerate(trainloader, 0):
        print(i, end='\r')

        # Get the inputs (images) and labels (target variable)
        inputs, labels = data
        # Put them in the graphics card
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Set the gradients to zero
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs)

        # Calculate the  gradients (backward)
        loss = criterion(outputs, labels)
        loss.backward()
        # Backpropagation
        optimizer.step()

        # Measure the loss
        running_loss_full += loss.item() 
        nbatch = nbatch + 1


    print('Getting Training accuracy...')
    # Get the accuracy in the training data
    with torch.no_grad():
        for data in trainloader:
            images_train, labels_train = data
            images_train = images_train.to(device)
            labels_train = labels_train.to(device)
            # Generate prediction without gradient update and calculate the correct matches
            outputs_train = net(images_train)
            _, predicted_train = torch.max(outputs_train, 1)
            correct_train = (predicted_train == labels_train).float().sum()
            correct_total_train = correct_total_train + correct_train.item()
            nsamples_train = nsamples_train + len(labels_train)  

    train_accuracy = 100.0 * correct_total_train / nsamples_train
    print("Train Accuracy:", train_accuracy)

    # Store the accuracies and loss in the lists
    acc_full.append(train_accuracy)
    loss_full.append(running_loss_full / nbatch)
    epoch_full.append(epoch)

    # Decrease the learning rate using the scheduler
    learning_rate_scheduler.step()

    print("Loss:", running_loss_full / nbatch)
    print('---------------------------')

print("Training of the freeze net ready!")

# Second stage of training

# Train the whole neural network
for parameter in net.parameters():
    parameter.requires_grad = True

# Same optimizer, but the learning rate is multiplier by a learning_rate_decrease factor (0.1)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate*learning_rate_decrease)
learning_rate_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train with the dataset multiple times (n epoch)
for epoch in range(1,n_epoch_uf+1):  

    print("Epoch:",epoch)

    # Initializing the training variables
    running_loss_full = 0.0
    correct_total_train = 0
    nbatch = 0
    nsamples_train = 0

    print('Training the neural network...')
    # Go though all the training data
    for i, data in enumerate(trainloader, 0):
        print(i, end='\r')

        # Get the inputs (images) and labels (target variable)
        inputs, labels = data
        # Put them in the graphics card
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Set the gradients to zero
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs)

        # Calculate the  gradients (backward)
        loss = criterion(outputs, labels)
        loss.backward()
        # Backpropagation
        optimizer.step()

        # Measure the loss
        running_loss_full += loss.item() 
        nbatch = nbatch + 1


    print('Getting Training accuracy...')
    # Get the accuracy in the training data
    with torch.no_grad():
        for data in trainloader:
            images_train, labels_train = data
            images_train = images_train.to(device)
            labels_train = labels_train.to(device)
            # Generate prediction without gradient update and calculate the correct matches
            outputs_train = net(images_train)
            _, predicted_train = torch.max(outputs_train, 1)
            correct_train = (predicted_train == labels_train).float().sum()
            correct_total_train = correct_total_train + correct_train.item()
            nsamples_train = nsamples_train + len(labels_train)  

    train_accuracy = 100.0 * correct_total_train / nsamples_train
    print("Train Accuracy:", train_accuracy)

    # Store the accuracies and loss in the lists
    acc_full.append(train_accuracy)
    loss_full.append(running_loss_full / nbatch)
    epoch_full.append(epoch)

    # Decrease the learning rate using the scheduler
    learning_rate_scheduler.step()

    print("Loss:", running_loss_full / nbatch)
    print('---------------------------')

print("Training of the unfreeze net ready!")

# Create pandas dataset and store the results in a csv
dic = {'epoch':epoch_full,'train_accuracy':acc_full,'loss':loss_full}
df_grid_search = pd.DataFrame(dic)
df_grid_search.to_csv(root_dir + 'results/' + model_name + '_' + 'train_acc' + '.csv')

# Build the Kaggle submission
predicted_list = []

# Use the neural network to predict on the test data
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        predicted_list.append(predicted.cpu().numpy())


# Generate a single numpy array
predicted_numpy = np.concatenate(predicted_list)

# Build the submission file for Kaggle
sample_submission['species_id'] = predicted_numpy
sample_submission["species_id"].replace(classes_num_dic, inplace=True)
sample_submission = sample_submission.drop(columns='species')
sample_submission = sample_submission.rename(columns={'species_id':'species'})

# write it in the kaggle submissions folder
sample_submission.to_csv(root_dir + 'results/kaggle_submissions/' 'submission_kaggle_'+ model_name + '.csv', index=False)
