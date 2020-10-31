###################################
# Julian Cabezas Pena
# Deep Learning Fundamentals
# University of Adelaide
# Assingment 2
# Figures of the accuracies and loss curves of the models using different data augmentatiobn methods
####################################

# Import the libraries to do graphs
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set the style of the seaborn graphs
sns.set_style("whitegrid")

#-----------------------------------------
# Plot of the learning rate validation
print('Making learning rate graph')

# Route where the learning rate 
val_results_path = './results/learning_rate_tuning/'

# List the files of the validation
files = os.listdir(val_results_path)

# Generate a database for each model

# Start with Resnet101

# get the files starning with vgg19
files_model = list(filter(lambda k: 'Resnet_101' in k, files))

# Get the files and genenrate a dataframe with the compiled data
counter = 1
for i, csv_file in enumerate(files_model):

    # Read the csv
    path = val_results_path + '/' + csv_file
    val_result = pd.read_csv(path)

    val_result['epoch'] = np.arange(start=1,stop=41,step=1)
    

    # Put good names for the graph
    if '001' in csv_file: 
        val_result['Learning Rate'] = 'LR=0.001 - 0.0001'
    elif '003' in csv_file: 
        val_result['Learning Rate'] = 'LR=0.003 - 0.0003'
    elif '005' in csv_file: 
        val_result['Learning Rate'] = 'LR=0.005 - 0.0005'

    if i== 0:
        val_result_resnet101 = val_result
    else:
        val_result_resnet101 = pd.concat([val_result_resnet101,val_result])


# Melt to generate a single graph with the training and validation accuracies
val_result_resnet101_acc=pd.melt(val_result_resnet101[['epoch','train_accuracy','val_accuracy','Learning Rate']],id_vars= ['epoch','Learning Rate'],value_vars=['train_accuracy','val_accuracy'])

# Rename variables for the graph
val_result_resnet101_acc = val_result_resnet101_acc.rename(columns={'variable': 'Accuracy'},)
val_result_resnet101_acc['Accuracy'] = val_result_resnet101_acc['Accuracy'].str.replace('train_accuracy','Training').str.replace('val_accuracy','Validation')

# Now the Resnext101-32x8d IN

# get the files starning with vgg19
files_model = list(filter(lambda k: 'Resnext_Imagenet_32_8d' in k, files))

# Get the files and genenrate a dataframe with the compiled data
counter = 1
for i, csv_file in enumerate(files_model):

    # Read the csv
    path = val_results_path + '/' + csv_file
    val_result = pd.read_csv(path)

    val_result['epoch'] = np.arange(start=1,stop=41,step=1)
    

    # Put good names for the graph
    if '001' in csv_file: 
        val_result['Learning Rate'] = 'LR=0.001 - 0.0001'
    elif '003' in csv_file: 
        val_result['Learning Rate'] = 'LR=0.003 - 0.0003'
    elif '005' in csv_file: 
        val_result['Learning Rate'] = 'LR=0.005 - 0.0005'

    if i== 0:
        val_result_resnext_imagenet_32_8d = val_result
    else:
        val_result_resnext_imagenet_32_8d = pd.concat([val_result_resnext_imagenet_32_8d,val_result])


# Melt to generate a single graph with the training and validation accuracies
val_result_resnext_imagenet_32_8d_acc=pd.melt(val_result_resnext_imagenet_32_8d[['epoch','train_accuracy','val_accuracy','Learning Rate']],id_vars= ['epoch','Learning Rate'],value_vars=['train_accuracy','val_accuracy'])

# Rename variables for the graph
val_result_resnext_imagenet_32_8d_acc = val_result_resnext_imagenet_32_8d_acc.rename(columns={'variable': 'Accuracy'},)
val_result_resnext_imagenet_32_8d_acc['Accuracy'] = val_result_resnext_imagenet_32_8d_acc['Accuracy'].str.replace('train_accuracy','Training').str.replace('val_accuracy','Validation')

# Now the Resnext101-32x8d IN-WSL

# get the files starning with vgg19
files_model = list(filter(lambda k: 'Resnext_32_8d' in k, files))

# Get the files and genenrate a dataframe with the compiled data
counter = 1
for i, csv_file in enumerate(files_model):

    # Read the csv
    path = val_results_path + '/' + csv_file
    val_result = pd.read_csv(path)

    val_result['epoch'] = np.arange(start=1,stop=41,step=1)
    

    # Put good names for the graph
    if '001' in csv_file: 
        val_result['Learning Rate'] = 'LR=0.001 - 0.0001'
    elif '003' in csv_file: 
        val_result['Learning Rate'] = 'LR=0.003 - 0.0003'
    elif '005' in csv_file: 
        val_result['Learning Rate'] = 'LR=0.005 - 0.0005'

    if i== 0:
        val_result_resnext_32_8d = val_result
    else:
        val_result_resnext_32_8d = pd.concat([val_result_resnext_32_8d,val_result])


val_result_resnext_32_8d

# Melt to generate a single graph with the training and validation accuracies
val_result_resnext_32_8d_acc=pd.melt(val_result_resnext_32_8d[['epoch','train_accuracy','val_accuracy','Learning Rate']],id_vars= ['epoch','Learning Rate'],value_vars=['train_accuracy','val_accuracy'])

# Rename variables for the graph
val_result_resnext_32_8d_acc = val_result_resnext_32_8d_acc.rename(columns={'variable': 'Accuracy'},)
val_result_resnext_32_8d_acc['Accuracy'] = val_result_resnext_32_8d_acc['Accuracy'].str.replace('train_accuracy','Training').str.replace('val_accuracy','Validation')

# Now the Resnext101-32x16d IN-WSL

# get the files starning with vgg19
files_model = list(filter(lambda k: 'Resnext_32_16d' in k, files))

# Get the files and genenrate a dataframe with the compiled data
counter = 1
for i, csv_file in enumerate(files_model):

    # Read the csv
    path = val_results_path + '/' + csv_file
    val_result = pd.read_csv(path)

    val_result['epoch'] = np.arange(start=1,stop=41,step=1)
    

    # Put good names for the graph
    if '001' in csv_file: 
        val_result['Learning Rate'] = 'LR=0.001 - 0.0001'
    elif '003' in csv_file: 
        val_result['Learning Rate'] = 'LR=0.003 - 0.0003'
    elif '005' in csv_file: 
        val_result['Learning Rate'] = 'LR=0.005 - 0.0005'

    if i== 0:
        val_result_resnext_32_16d = val_result
    else:
        val_result_resnext_32_16d = pd.concat([val_result_resnext_32_16d,val_result])



# Melt to generate a single graph with the training and validation accuracies
val_result_resnext_32_16d_acc=pd.melt(val_result_resnext_32_16d[['epoch','train_accuracy','val_accuracy','Learning Rate']],id_vars= ['epoch','Learning Rate'],value_vars=['train_accuracy','val_accuracy'])

# Rename variables for the graph
val_result_resnext_32_16d_acc = val_result_resnext_32_16d_acc.rename(columns={'variable': 'Accuracy'},)
val_result_resnext_32_16d_acc['Accuracy'] = val_result_resnext_32_16d_acc['Accuracy'].str.replace('train_accuracy','Training').str.replace('val_accuracy','Validation')


# order of the learning rates
hue_order = ['LR=0.001 - 0.0001','LR=0.003 - 0.0003','LR=0.005 - 0.0005'] 

# Make a grid plot with training and validation accuracy and loss
fig, ax =plt.subplots(1,4)

# Go graph by braph
sns.lineplot(x="epoch", y="value",style="Accuracy", hue="Learning Rate",hue_order=hue_order,data=val_result_resnet101_acc,ax=ax[0]).set(ylabel='Accuracy (%)',title='ResNet-101\n(IN pretrained)',ylim=(10, 100))
sns.lineplot(x="epoch", y="value",style="Accuracy", hue="Learning Rate",hue_order=hue_order,data=val_result_resnext_imagenet_32_8d_acc,ax=ax[1]).set(ylabel='',title='ResNeXt-101 32x8d\n(IN pretrained)',ylim=(10, 100))
ax[1].get_legend().remove()
sns.lineplot(x="epoch", y="value",style="Accuracy", hue="Learning Rate",hue_order=hue_order,data=val_result_resnext_32_8d_acc,ax=ax[2]).set(ylabel='',title='ResNeXt-101 32x8d\n(Ins+IN pretrained)',ylim=(10, 100))
ax[2].get_legend().remove()
sns.lineplot(x="epoch", y="value",style="Accuracy", hue="Learning Rate",hue_order=hue_order,data=val_result_resnext_32_16d_acc,ax=ax[3]).set(ylabel='',title='ResNeXt-101 32x16d\n(Ins+IN pretrained)',ylim=(10, 100))
ax[3].get_legend().remove()
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
fig.set_size_inches(15, 5)

# Save the graph as pdf
fig.savefig('./figures/lr_4models.pdf',bbox_inches='tight')  
plt.clf()
print('Ready!')

#-------------------------------------
# Graph of loss vs epoch
print('Making the loss curve plot')

# order of the learning rates
hue_order = ['LR=0.001 - 0.0001','LR=0.003 - 0.0003','LR=0.005 - 0.0005'] 

# Make a grid plot with training and validation accuracy and loss
fig, ax =plt.subplots(1,4)

val_result_resnet101

#Go graph by braph
sns.lineplot(x="epoch", y="loss", hue="Learning Rate",hue_order=hue_order,data=val_result_resnet101,ax=ax[0]).set(ylabel='Loss',title='ResNet-101\n(IN pretrained)',ylim=(0, 3))
sns.lineplot(x="epoch", y="loss", hue="Learning Rate",hue_order=hue_order,data=val_result_resnext_imagenet_32_8d,ax=ax[1]).set(ylabel='',title='ResNeXt-101 32x8d\n(IN pretrained)',ylim=(0, 3))
ax[1].get_legend().remove()
sns.lineplot(x="epoch", y="loss", hue="Learning Rate",hue_order=hue_order,data=val_result_resnext_32_8d,ax=ax[2]).set(ylabel='',title='ResNeXt-101 32x8d\n(Ins+IN pretrained)',ylim=(0, 3))
ax[2].get_legend().remove()
sns.lineplot(x="epoch", y="loss", hue="Learning Rate",hue_order=hue_order,data=val_result_resnext_32_16d,ax=ax[3]).set(ylabel='',title='ResNeXt-101 32x16d\n(Ins+IN pretrained)',ylim=(0, 3))
ax[3].get_legend().remove()
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
fig.set_size_inches(15, 5)

# Save the graph as pdf
fig.savefig('./figures/loss_4models.pdf',bbox_inches='tight')  
plt.clf()
print('Ready!')
















