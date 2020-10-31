# Plant seedling classification
Using Instagram and ImageNet pretrained deep convolutional networks for seedling species image classification

## Environment

This code was tested under a Linux 64 bit OS (Ubuntu 18.04 LTS), using Python 3.7.7

## How to run this code:

In order to use this repo:

1. Install Miniconda or Anaconda
2. Add conda forge to your list of channels

In the terminal run:

    conda config --add channels conda-forge

3. Create a environment using the environment.yml file included in this repo:

Open a terminal in the folder were the environment.yml file is (a1785086_Code_Project2) and run:

    conda env create -f environment.yml --name environment

4. Download the train and test data from https://www.kaggle.com/c/plant-seedlings-classification/data (including the sample submission) and put it in the dta folder

5. Make sure the folder structure of the project is as follows

a1785086_Code_Project2
├── data
    ├── train
    ├── test
    ├── sample_submission.csv
├── results
    ├── kaggle_submission
    ├── learning_rate_tuning
├── figures
├── 00-Figures-images-seedlings.py
├── 01-Resnet-ResNeXt-training-learning-rate-tuning.py
├── 02-ResNeXt-101-32x16d-finetuning.py
├── 03-Resnet-ResNeXt-final-model-train-test.py
├── 04-Graphs-learning_loss_accuracies.py
├── README.txt
└── ...

5.  Run the code in the conda environment: Open a terminal in the a1785086_Code_Project2  and run 
	
	conda activate house_regression
	python <filename.py>


Note:
The codes are preceded by a number from 00 to 04, and do:
00: Figures for the distribution of species in the dataset and figures of the images contained in the train dataset
01: Trains the Resnet and ResNeXt architectures on the train data and evaluate in the validation data, the hyperparameters are in the beggining of the file and can be changed to tune them
02; Trains the ResNeXt-101 32x16d architecture  and evaluates on the validation data, includes the option to add extra data transformations and to modify the hyperparameters
03: Trains the final models using the total train data (without splitting train and validation data), generates predictions on the test data for Kaggle submission
04: Generates Figures of the adjusting of the learning rate hyperparameters (train-val accuracy and loss curves)


Alternatevely,  run the codes in your IDE of preference, (I recommend VS Code with the Python extension), using the root folder of the directory (a1785086_Code_Project1) as working directory to make the relative paths work.

Note: Alternatevely, for 2 and 3 you can build your own environment following the package version contained in requirements.yml file

