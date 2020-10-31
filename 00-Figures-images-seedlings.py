###################################
# Julian Cabezas Pena
# Big Data Analysis and Project
# University of Adelaide
# Project 2
# Figures of Plant Seedling dataset and species distribution in the dataset
####################################

# Importing the necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image
import seaborn as sns

# Set the style of the seaborn graphs
sns.set_style("whitegrid")

# Root dirirectory path
root_dir = './'

# -----------------------------------------
# Species distribution plot


# Get classes (class) names
classes = os.listdir(root_dir + 'data/' + 'train/')
classes = sorted(classes)

# generate an empty list for the training data
train_list = []
id_counter = 0

# loop though the 12 species
for species in classes:

    # List files in each directory
    files = os.listdir(root_dir + 'data/' + 'train/' + species + '/')

    # Get the path of every image and put it in the train list
    path = root_dir + 'data/' + 'train/' + species + '/'


    # List the files in the directory
    files_list = os.listdir(path)

    for filename in files_list:
        train_list.append([species +'/'+filename, species, id_counter])

    
    id_counter = id_counter + 1

# generate a dataframe with the train data paths
train_df = pd.DataFrame(train_list, columns=['file', 'species', 'species_id',]) 

# Generate a countplot with the distribution of the species
ax = sns.countplot(y="species", orient = 'h', color='darkblue',data=train_df,order = train_df['species'].value_counts().index).set(xlabel='Number of Images',ylabel='Species')
plt.savefig('./figures/species_freq.pdf',bbox_inches='tight')
plt.clf()

# -----------------------------------------------
# Examples of images plot

# Get classes (class) names
classes = os.listdir(root_dir + 'data/' + 'train/')
classes = sorted(classes)

images_list= []

# loop though the 12 species
for species in classes:

    # List a single file per directory, in this time the one in position 11
    image_path = os.listdir(root_dir + 'data/' + 'train/' + species + '/')[11]

    # Get the actual path
    image_path = root_dir + 'data/' + 'train/' + species + '/' + image_path

    # open the image to image
    image = Image.open(image_path).convert('RGB')

    # Store them in a list
    images_list.append(image)


# Make a grid plot with 6 rows and 2 columns for the 12 images
fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(16, 6))

counter = 0

# Build the plot
for row in ax:
    for col in row:

        # Het the image
        im = images_list[counter]

        # Show the image in the particular row and col
        col.imshow(im)
        col.axis('off')

        # Put the title (class name)
        col.set_title(classes[counter],fontsize=12)
        counter += 1

print('Made figure with images per class')

plt.axis('off')

# Save the image
plt.savefig('./figures/samples_images.pdf',bbox_inches='tight')


